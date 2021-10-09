import torch.nn as nn
import torch
import random
from models.utils import sample_z



class RNNDText(nn.Module):

    def __init__(self, emb_dim, vocab_size, hid_dim, n_layers, dropout, sos, eos, pad, device='cpu'):
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
                           batch_first=True, #only for input and output does not have any impact on cell state or hidden state
                           bidirectional=True)

        self.decoder = nn.LSTM(self.emb_dim,
                           self.hid_dim * 2,
                           self.n_layers,
                           dropout=self.dropout,
                           batch_first=True, #only for input and output does not have any impact on cell state or hidden state
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

        embedded = self.embeddings(batch_positions) #Batch size * padded sentence len* embedding dim each word
        #in order to use packed representation the embeddings need to be sorted based one length
        sorted_lens, sorted_idx = torch.sort(text_length, descending=True)
        forwards_sorted = embedded[sorted_idx]#sort the embedding based on length
        _, sortedsorted_idx = sorted_idx.sort()#sorting the sorted index, to figure out unsorted index
        packed = torch.nn.utils.rnn.pack_padded_sequence(forwards_sorted, sorted_lens, batch_first=True)#reduces computation packed sequence is a tuple of two 
        #lists . One list contain sequences , where sequences are interleaved by time space.Other list contains
        #the batch size at each time step. first list format (sequences, embedding dimention)#(batch first) is used if the input to the fuction has batch as first dimention
        h, _ = self.encoder(packed) #Inputs: input, (h_0, c_0) when hidden and cell not provided default taken as 0 vectors as in this case
        #Outputs: output, (h_n, c_n), if input is packed output is also packed which needs to be unpacked
        #tensor containing the output features (h_t) from the last layer(if multiple layers are used) of the LSTM, for each time step
        #output for packed has a dimention(all sequences, num_directions(forward and backward for bidirectional) * hidden_size)
        #h_n of shape (num_layers * num_directions, batch, hidden_size): tensor containing the hidden state for t = seq_len
        #c_n of shape (num_layers * num_directions, batch, hidden_size): tensor containing the cell state for t = seq_len.
        h_tmp, _ = torch.nn.utils.rnn.pad_packed_sequence(h, batch_first=True)
        #returns unpacked output and sequence length vector, unpacked output has dimention(batch, sequence, num_directions * hidden_size)
        h_t = torch.max(h_tmp, 1)[0] #taking the maximum among all timesteps
        h_t = torch.mean(h_tmp, 1)
        h_t = h_t[sortedsorted_idx]#sequence are sorted back to unsorted format


        return h_t


    def generate(self, hidden, batch_size, batch_positions, teacher_forcing_prob, text_length, cell=None):
        if batch_size is None:
            batch_size = len(hidden)
        step_emb = self.embeddings(torch.LongTensor([self.sos]).repeat(batch_size).to(self.device)) #starting symbol embeddings for each sentences in batch, shape (batch_size,embedding dim)
        hidden_ = torch.zeros((self.n_layers, batch_size, self.hid_dim * 2) ).to(self.device) #shape (no_layer, batch_size, hidden_dim*2)
        #hidden has a shape of (batch_size, hidden*2)
        hidden_[0] = hidden[0] #replacing for first layer entry , in this case only single layer
        hidden = hidden_ #shape is (layer_no, batchsize, hidden*2), this is the encoded value from encoder in the given format, which will be input as hidden state to decoder at first timestep

        if cell is None:
            cell = torch.zeros_like(hidden).to(self.device)#shape is (layer_no, batchsize, hidden*2)#initializing cell to zero

        max_length = max(text_length)
        argmax_indices = torch.zeros(max_length, batch_size).to(self.device) # to strore maximum indices for entire sequence length
        hidden_outputs = torch.zeros(max_length, batch_size, self.hid_dim * 2).to(self.device)
        outputs = torch.zeros(max_length, batch_size, self.vocab_size).to(self.device) #stores the output for entire sequence length
        argmax_indices[0]= torch.LongTensor([self.sos]).repeat(batch_size).to(self.device)
        hidden_outputs[0]= hidden[-1]

        for t in range(1, max_length): #Note here we can#t pack the sequence as we are generating one by one, as we don#t have input beforehand
            step_emb = step_emb.view(batch_size,1,self.emb_dim) #(batch, sequence length, embedding_dim) Notebatch first is used in decoder
            output, (hidden, cell) = self.decoder(step_emb, (hidden, cell))#hidden and cell usually have multiple entries each for number of layers and directions ; shape (num_layers * num_directions, batch, hidden_size)

            hidden_outputs[t] = hidden[-1] #here it means take hidden from last layer, in this case  only one layer
            logits = self.out(output) #probability of a particular word
            outputs[t] = logits.squeeze() #removes dimention of 1

            argmax_index = logits.view(batch_size,-1).max(1)[1] #maximum around axis 1, returns value tensor and index tensor, taking the index tensor
            argmax_indices[t] = argmax_index

            teacher_force = random.random() < teacher_forcing_prob
            if teacher_force:
                step_emb = self.embeddings(batch_positions[:,t])
            else:
                step_emb = self.embeddings(argmax_index) #obtaining the next input which is embedding based on predicted logits

        # text_length = self.get_length(argmax_indices)
        hidden_outputs = hidden_outputs.transpose(1,0)
        #sorted_lens, sorted_idx = torch.sort(text_length, descending=True)
        #hidden_sorted = hidden_outputs[sorted_idx]
        #_, sortedsorted_idx = sorted_idx.sort()
        #packed = torch.nn.utils.rnn.pack_padded_sequence(hidden_sorted, sorted_lens, batch_first=True)
        #h_tmp, _ = torch.nn.utils.rnn.pad_packed_sequence(packed, batch_first=True)
        #h_t = torch.max(h_tmp, 1)[0].squeeze()


        return outputs.transpose(0,1), argmax_indices.transpose(0,1)


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
        o, i = self.rnn(pass_type ='generate',
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

        embedded = self.embeddings(batch_positions)  #Batch size * padded sentence len* embedding dim each word

        sorted_lens, sorted_idx = torch.sort(text_length, descending=True)#sort based on decreasing length
        forwards_sorted = embedded[sorted_idx] #sort the embedding based on length
        _, sortedsorted_idx = sorted_idx.sort()#sorting the sorted index, to figure out unsorted index
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




if __name__ == '__main__':


    # we take a random dataset
    text = [
    'this is a dog',
    'this is a cat',
    'cat is nice',
    'dog is nice',
    'this is nice',
    'is this a dog',
    'is this a cat',
    'is this a nice dog',
    'is this a nice cat',
    'this is a nice cat',
    'this is a nice dog',
    ]

    # we create a dictionary and a list
    vocab_dict = {}
    vocab_list = []

    # we define special tokens
    special_tokens = ['<PAD>', '<S>', '</S>']

    # we add our special tokens to our vocabulary
    for st in special_tokens:
        if st not in vocab_dict:
            vocab_list.append(st)
            vocab_dict[st] = len(vocab_dict)

    # we tokenize our sentences
    data_set = []


    # we loop through our dataset, tokenize the sentences (Here I just tokenize on white space) and add each of the tokens
    # to our vocabulary if its not in yet
    for s in text:
        # tokenization
        toks = s.split()
        # a new data point of positions for sentences
        data_point = []

        # we prepend the start token
        data_point.append(vocab_dict['<S>'])

        # loop through all the tokens in the sentence and add token to vocab if not in yet
        for tok in toks:
            if tok not in vocab_dict:
                vocab_list.append(tok)
                vocab_dict[tok] = len(vocab_dict)
            data_point.append(vocab_dict[tok])

        # add the stop token
        data_point.append(vocab_dict['</S>'])

        # get the length of the sentences
        sentence_length = len(data_point)



        #add data point to data_set
        data_set.append({'sentence':data_point, 'length':sentence_length})




    batch_size = 3
    embedding_dim = 5
    vocab_size = len(vocab_dict)

    config = {  'emb_dim': embedding_dim,
                'hid_dim': 4,
                'n_layers': 1,
                'dropout': 0.0,
                'vocab_size': vocab_size,
                'sos': vocab_dict['<S>'],
                'eos': vocab_dict['</S>'],
                'pad': vocab_dict['<PAD>'],
             }

    ae = AutoEncoderD(config)

    # text_length = torch.randint(low=1, high=max_text_length, size=(batch_size,))
    #
    # batch_positions = torch.randint(vocab_size, (batch_size,max_text_length))

    optimizer = torch.optim.Adam(ae.parameters())

    criterion = torch.nn.CrossEntropyLoss(ignore_index=vocab_dict['<PAD>'])

    # h = ae(pass_type = 'encode', batch_positions=batch_positions, text_length=text_length)
    #
    # h, o = ae(pass_type = 'generate', batch_positions=batch_positions, hidden=h, teacher_forcing_prob=0.0, text_length=text_length, batch_size=batch_size)

    from random import shuffle
    import copy


    for e in range(1):

        shuffle(data_set)

        for i in range(0, len(data_set), batch_size):

            batch = data_set[i:i+batch_size]

            x = []
            l = []

            max_length = 0

            for elems in batch:
                x.append(copy.deepcopy(elems['sentence']))
                l.append(elems['length'])
                max_length = max(elems['length'], max_length)

            # if the sentences are not yet max length then we pad it
            for x_ in x:
                while len(x_) < max_length:

                    x_.append(vocab_dict['<PAD>'])

            try:
                x = torch.tensor(x)
            except:
                print('')
            l = torch.tensor(l)

            out, i = ae(x,l)

            loss = criterion(out[:, 1:, :].contiguous().view(-1, out.shape[2]), x[:, 1:].flatten())

            loss.backward()

            torch.nn.utils.clip_grad_norm_(ae.parameters(), 5.00)

            optimizer.step()

            print(loss)



    print ('done')
