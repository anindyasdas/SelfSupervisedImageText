import torch.nn as nn
import torch
import random
from models.utils1 import sample_z

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class RNNDText(nn.Module):

    def __init__(self, emb_dim, vocab_size, hid_dim, n_layers, dropout, sos, eos, pad, device=device):
        super(RNNDText, self).__init__()

        self.emb_dim =  emb_dim
        self.vocab_size = vocab_size
        self.hid_dim =  hid_dim
        self.n_layers = n_layers
        self.dropout =  dropout
        self.sos = sos
        self.eos = eos
        self.pad = pad
        self.device=device

        assert self.sos is not None
        assert self.eos is not None

        self.embeddings = nn.Embedding(self.vocab_size, self.emb_dim, padding_idx=self.pad)

        self.out = nn.Linear(self.hid_dim * 2, self.vocab_size)

        # Initialize the RNN
        self.encoder = nn.LSTM(self.emb_dim,
                           self.hid_dim,
                           self.n_layers,
                           dropout=self.dropout,
                           batch_first=True,
                           bidirectional=True)

        self.decoder = nn.LSTM(self.emb_dim,
                           self.hid_dim * 2,
                           self.n_layers,
                           dropout=self.dropout,
                           batch_first=True,
                           bidirectional=False)



    def forward(self,
                text_length,
                batch_positions=None,
                cell=None,
                hidden=None,
                pass_type = 'generate',
                teacher_forcing_prob=0.0,
                batch_size=None
                ):

        if pass_type == 'generate':

            assert hidden is not None
            if teacher_forcing_prob > 0.0:
                assert batch_positions is not None
                assert len(batch_positions[0]) == max(text_length)
                batch_size = len(batch_positions)
            # else:
                # assert batch_size is not None

            return self.generate(hidden, batch_size, batch_positions, teacher_forcing_prob, text_length, cell=cell)

        elif pass_type == 'encode':
            assert batch_positions is not None
            return self.encode(batch_positions, text_length)


    def encode(self, batch_positions, text_length):

        embedded = self.embeddings(batch_positions)

        sorted_lens, sorted_idx = torch.sort(text_length, descending=True)
        forwards_sorted = embedded[sorted_idx]
        _, sortedsorted_idx = sorted_idx.sort()
        packed = torch.nn.utils.rnn.pack_padded_sequence(forwards_sorted, sorted_lens, batch_first=True)
        h, _ = self.encoder(packed)
        h_tmp, _ = torch.nn.utils.rnn.pad_packed_sequence(h, batch_first=True)
        h_t = torch.max(h_tmp, 1)[0]
        h_t = h_t[sortedsorted_idx]


        return h_t,


    def generate(self, hidden, batch_size, batch_positions, teacher_forcing_prob, text_length, cell=None):
        if batch_size is None:
            batch_size = len(hidden)
        step_emb = self.embeddings(torch.LongTensor([self.sos]).repeat(batch_size).to(self.device))
        hidden_ = torch.zeros((self.n_layers, batch_size, self.hid_dim * 2) ).to(self.device)
        hidden_[0] = hidden[0]
        hidden = hidden_

        if cell is None:
            cell = torch.zeros_like(hidden).to(self.device)

        max_length = max(text_length)
        argmax_indices = torch.zeros(max_length, batch_size).to(self.device)
        hidden_outputs = torch.zeros(max_length, batch_size, self.hid_dim * 2).to(self.device)
        outputs = torch.zeros(max_length, batch_size, self.vocab_size).to(self.device)

        for t in range(1, max_length):
            step_emb = step_emb.view(batch_size,1,self.emb_dim)
            output, (hidden, cell) = self.decoder(step_emb, (hidden, cell))

            hidden_outputs[t] = hidden[-1]
            logits = self.out(output)
            outputs[t] = logits.squeeze()

            argmax_index = logits.view(batch_size,-1).max(1)[1]
            argmax_indices[t] = argmax_index

            teacher_force = random.random() < teacher_forcing_prob
            if teacher_force:
                step_emb = self.embeddings(batch_positions[:,t])
            else:
                step_emb = self.embeddings(argmax_index)

        # text_length = self.get_length(argmax_indices)
        hidden_outputs = hidden_outputs.transpose(1,0)
        sorted_lens, sorted_idx = torch.sort(text_length, descending=True)
        hidden_sorted = hidden_outputs[sorted_idx]
        _, sortedsorted_idx = sorted_idx.sort()
        packed = torch.nn.utils.rnn.pack_padded_sequence(hidden_sorted, sorted_lens, batch_first=True)
        h_tmp, _ = torch.nn.utils.rnn.pad_packed_sequence(packed, batch_first=True)
        h_t = torch.max(h_tmp, 1)[0].squeeze()


        return h_t, outputs.transpose(0,1), argmax_indices.transpose(0,1)


    # def get_length(self, indices):
    #     indices = indices.transpose(0, 1)
    #     self.indices_np = indices.clone()
    #     self.indices_np[:,-1] = self.eos
    #     seq_len = torch.argmax(self.indices_np == self.eos, 1)
    #     seq_len += 1
    #     return seq_len

class AutoEncoderD(nn.Module):

    def __init__(self, config, embeddings=None):
        super(AutoEncoderD, self).__init__()

        self.rnn = RNNDText(**config)

    def forward(self, batch_positions, text_length, teacher_forcing_prob=0.0):
        batch_size = len(batch_positions)
        h = self.rnn(pass_type='encode',
                     batch_positions=batch_positions,
                     text_length=text_length)
        h, o, i = self.rnn(pass_type ='generate',
                           batch_positions=batch_positions,
                           hidden=h,
                           teacher_forcing_prob=teacher_forcing_prob,
                           text_length=text_length,
                           batch_size=batch_size)
        return o, i


    def store_model(self, path):

        state = {
            'state_dict': self.state_dict(),
        }
        print("dumping new best model to " + str( path))
        torch.save(state,  path)

    def load_model(self, path):
        """
        Load model from file
        :param best:
        :return:
        """

        checkpoint = torch.load(path, map_location=lambda storage, loc: storage)
        self.load_state_dict(checkpoint['state_dict'])



class RNNText(nn.Module):

    def __init__(self, emb_dim, vocab_size, hid_dim, n_layers, dropout, sos, eos, device, vae):
        super(RNNText, self).__init__()

        #
        # self.emb_dim =  config['embedding_dim']
        # self.vocab_size = config['vocab_size']
        # self.hid_dim =  config['rnn_hidden_dim']
        # self.n_layers = config['rnn_layers']
        # self.dropout =  config['rnn_dropout']
        # self.sos = config['sos']
        # self.eos = config['eos']

        self.emb_dim =  emb_dim
        self.vocab_size = vocab_size
        self.hid_dim =  hid_dim
        self.n_layers = n_layers
        self.dropout =  dropout
        self.sos = sos
        self.eos = eos
        self.device=device
        self.vae=vae

        assert self.sos is not None
        assert self.eos is not None

        # if embeddings:
        #     self.embeddings = embeddings
        # else:
        self.embeddings = nn.Embedding(self.vocab_size, self.emb_dim, padding_idx=0)

        self.out = nn.Linear(self.hid_dim, self.vocab_size)

        # Initialize the RNN
        self.rnn = nn.LSTM(self.emb_dim,
                           self.hid_dim,
                           self.n_layers,
                           dropout=self.dropout,
                           batch_first=True,
                           bidirectional=False)

        if self.vae:
            self.vae_transform = nn.Sequential(nn.Linear(self.hid_dim, self.hid_dim * 2),
                                               nn.Tanh())


    def forward(self,
                text_length,
                batch_positions=None,
                cell=None,
                hidden=None,
                pass_type = 'generate',
                teacher_forcing_prob=0.0,
                batch_size=None
                ):

        if pass_type == 'generate':

            assert hidden is not None
            if teacher_forcing_prob > 0.0:
                assert batch_positions is not None
                assert len(batch_positions[0]) == max(text_length)
                batch_size = len(batch_positions)
            # else:
                # assert batch_size is not None

            return self.generate(hidden, batch_size, batch_positions, teacher_forcing_prob, text_length, cell=cell)

        elif pass_type == 'encode':
            assert batch_positions is not None
            return self.encode(batch_positions, text_length)

    def generate(self, hidden, batch_size, batch_positions, teacher_forcing_prob, text_length, cell=None):
        if batch_size is None:
            batch_size = len(hidden)
        step_emb = self.embeddings(torch.LongTensor([self.sos]).repeat(batch_size).to(self.device))
        hidden_ = torch.zeros((self.n_layers,batch_size, self.hid_dim)).to(self.device)
        hidden_[0] = hidden
        hidden = hidden_

        if cell is None:
            cell = torch.zeros_like(hidden).to(self.device)

        max_length = max(text_length)
        argmax_indices = torch.zeros(max_length, batch_size).to(self.device)
        hidden_outputs = torch.zeros(max_length, batch_size, self.hid_dim).to(self.device)
        outputs = torch.zeros(max_length, batch_size, self.vocab_size).to(self.device)

        for t in range(1, max_length):
            step_emb = step_emb.view(batch_size,1,self.emb_dim)
            output, (hidden, cell) = self.rnn(step_emb, (hidden, cell))

            hidden_outputs[t] = hidden[-1]
            logits = self.out(output)
            outputs[t] = logits.squeeze()

            argmax_index = logits.view(batch_size,-1).max(1)[1]
            argmax_indices[t] = argmax_index

            teacher_force = random.random() < teacher_forcing_prob
            if teacher_force:
                step_emb = self.embeddings(batch_positions[:,t])
            else:
                step_emb = self.embeddings(argmax_index)

        text_length = self.get_length(argmax_indices)
        hidden_outputs = hidden_outputs.transpose(1,0)
        sorted_lens, sorted_idx = torch.sort(text_length, descending=True)
        hidden_sorted = hidden_outputs[sorted_idx]
        _, sortedsorted_idx = sorted_idx.sort()
        packed = torch.nn.utils.rnn.pack_padded_sequence(hidden_sorted, sorted_lens, batch_first=True)
        h_tmp, _ = torch.nn.utils.rnn.pad_packed_sequence(packed, batch_first=True)
        h_t = torch.max(h_tmp, 1)[0].squeeze()

        if self.vae:
            z = self.vae_transform(h_t)
            mu = z[:,:h_t.size()[1]]
            log_var = z[:,h_t.size()[1]:]

            if self.training:
                h_t = sample_z(mu, log_var, self.device)
            else:
                h_t = mu
        else:
            mu, log_var = None, None

        return h_t, outputs.transpose(0,1), argmax_indices.transpose(0,1), mu, log_var

    def encode(self, batch_positions, text_length):

        embedded = self.embeddings(batch_positions)

        sorted_lens, sorted_idx = torch.sort(text_length, descending=True)
        forwards_sorted = embedded[sorted_idx]
        _, sortedsorted_idx = sorted_idx.sort()
        packed = torch.nn.utils.rnn.pack_padded_sequence(forwards_sorted, sorted_lens, batch_first=True)
        h, _ = self.rnn(packed)
        h_tmp, _ = torch.nn.utils.rnn.pad_packed_sequence(h, batch_first=True)
        h_t = torch.max(h_tmp, 1)[0]
        h_t = h_t[sortedsorted_idx]

        if self.vae:
            z = self.vae_transform(h_t)
            mu = z[:,:h_t.size()[1]]
            log_var = z[:,h_t.size()[1]:]

            if self.training:
                h_t = sample_z(mu, log_var, self.device)
            else:
                h_t = mu

        else:
            mu, log_var = None, None

        return h_t, mu, log_var

    def get_length(self, indices):
        indices = indices.transpose(0, 1)
        self.indices_np = indices.clone()
        self.indices_np[:,-1] = self.eos
        seq_len = torch.argmax(self.indices_np == self.eos, 1)
        seq_len += 1
        return seq_len

class AutoEncoder(nn.Module):

    def __init__(self, config, embeddings=None):
        super(AutoEncoder, self).__init__()

        self.rnn = RNNText(config, vae=False)

    def forward(self, batch_positions, text_length, teacher_forcing_prob=0.0):
        batch_size = len(batch_positions)
        h = self.rnn(pass_type='encode',
                     batch_positions=batch_positions,
                     text_length=text_length)
        h, o, i = self.rnn(pass_type ='generate',
                           batch_positions=batch_positions,
                           hidden=h,
                           teacher_forcing_prob=teacher_forcing_prob,
                           text_length=text_length,
                           batch_size=batch_size)
        return o, i


    def store_model(self, path):

        state = {
            'state_dict': self.state_dict(),
        }
        print("dumping new best model to " + str( path))
        torch.save(state,  path)

    def load_model(self, path):
        """
        Load model from file
        :param best:
        :return:
        """

        checkpoint = torch.load(path, map_location=lambda storage, loc: storage)
        self.load_state_dict(checkpoint['state_dict'])



