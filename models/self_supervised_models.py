import torch
from torch import nn
from models.image_models import initialize_torchvision_model
from models.text_models import RNNText
from models.adversarial_models import Discriminator, D_NET64
from models.stack_gan2.model import G_NET
from torch.nn.modules import padding

# TODO:
# Better introduce some kind of subclassing for the supervised and unsupervised models since they use the same methods
# like 'encode_image(self, x)', 'image_text(self, image) and also have the same components self.image_decoder,
# self. RNNText, etc.

class CycleGAN(nn.Module):

    def __init__(self, vocab, cfg, device):
        super(CycleGAN, self).__init__()

        self.vocab = vocab
        self.vocab_size = self.vocab.vocab_size()
        self.sos = self.vocab.sos_pos()
        self.eos = self.vocab.eos_pos()
        self.pad = self.vocab.pad_pos()

        self.device = device

        if cfg.IMAGE.VAE:
            image_multiplier = 2
        else:
            image_multiplier = 1

        self.cfg = cfg


        self.image_encoder, self.image_input_size = initialize_torchvision_model(cfg.IMAGE.ENCODER_NAME,
                                                                      cfg.IMAGE.DIMENSION * image_multiplier,
                                                                      cfg.IMAGE.FIX_ENCODER,
                                                                                 use_pretrained = cfg.IMAGE.PRETRAINED_ENCODER,
                                                                                 vae=cfg.IMAGE.VAE,
                                                                                 device=self.device)
        self.image_decoder = G_NET()
        # self.image_decoder = SimpleImageDecoder(cfg.IMAGE.DIMENSION)

        self.text_encoder_decoder = RNNText(emb_dim=cfg.TEXT.EMBEDDING_DIM,
                                            vocab_size=self.vocab_size,
                                            hid_dim=cfg.TEXT.DIMENSION,
                                            n_layers=cfg.TEXT.N_LAYERS,
                                            dropout=cfg.TEXT.DROPOUT,
                                            vae=cfg.TEXT.VAE,
                                            sos=self.sos,
                                            eos=self.eos,
                                            device=device)

        self.adv_text = Discriminator(emb_dim=cfg.TEXT.DIMENSION,
                                      dis_layers=cfg.TEXT.ADV.LAYERS,
                                      dis_hid_dim=cfg.TEXT.ADV.DIM,
                                      dis_dropout=cfg.TEXT.ADV.DROPOUT,
                                      dis_input_dropout=cfg.TEXT.ADV.INPUT_DROPOUT,
                                      noise=cfg.TEXT.ADV.NOISE,
                                      device=device
                                      )

        self.adv_image_enc = Discriminator(emb_dim=cfg.IMAGE.DIMENSION,
                                      dis_layers=cfg.IMAGE.ADV.LAYERS,
                                      dis_hid_dim=cfg.IMAGE.ADV.DIM,
                                      dis_dropout=cfg.IMAGE.ADV.DROPOUT,
                                      dis_input_dropout=cfg.IMAGE.ADV.INPUT_DROPOUT,
                                      noise=cfg.IMAGE.ADV.NOISE,
                                      device=device)

        self.adv_image = D_NET64(df_dim=cfg.GAN.DF_DIM,
                                 ef_dim=cfg.GAN.EMBEDDING_DIM,
                                 conditional=cfg.GAN.B_CONDITION)

        self.pad = padding.ConstantPad2d(80, 0.0)


    def encode_image(self, x):

        # enc = self.image_encoder(x)
        #
        # if self.cfg.IMAGE.VAE:
        #     z = self.vae_transform(enc)
        #     mu = z[:,:enc.size[1]//2]
        #     log_var = z[:,enc.size[1]//2:]
        #
        #     if self.training:
        #         enc = sample_z(mu, log_var)
        #     else:
        #         enc = mu
        # else:
        #     mu, log_var = None, None
        #
        # return enc, mu, log_var
        return self.image_encoder(x)

    def encode_text(self, batch_positions, text_length):
        return self.text_encoder_decoder(pass_type='encode', batch_positions=batch_positions, text_length=text_length)

    def decode_image(self, x):
        return self.image_decoder(x)

    def decode_text(self, encoding, batch_positions=None, text_length=None, batch_size=None, teacher_forcing_prob=0.0):

        if text_length is None:
            text_length = torch.LongTensor([30]).repeat(len(encoding))

        encoding, logits, indices, mu_text, log_var_text = self.text_encoder_decoder(pass_type = 'generate',
                                            batch_positions=batch_positions,
                                            hidden=encoding,
                                            teacher_forcing_prob=teacher_forcing_prob,
                                            text_length=text_length,
                                            batch_size=batch_size)

        return encoding, logits, indices, mu_text, log_var_text

    def text_image_text(self, batch_positions, text_length, teacher_forcing_prob=0.0):

        text_encoding, mu_text, log_var_text = self.encode_text(batch_positions,text_length)

        image, image_norm = self.decode_image(text_encoding)

        padded_image = self.pad(image)

        image_encoding, mu_img, log_var_img  = self.encode_image(padded_image)

        re_text_encoding, text_output, indices, _, _ = self.decode_text(image_encoding,
                                                                  text_length=text_length,
                                                                  teacher_forcing_prob=teacher_forcing_prob)

        return text_output, text_encoding, image, image_norm, image_encoding, re_text_encoding, mu_text, log_var_text


    def image_text_image(self, image):

        image_encoding, mu_img, log_var_img  = self.encode_image(image)

        text_encoding, text_output, indices, mu_text, log_var_text = self.decode_text(image_encoding)

        re_image, image_norm = self.decode_image(text_encoding)

        return re_image, image_encoding, text_encoding, image_norm, mu_text, log_var_text

    def image_text(self, image):

        image_encoding, mu_img, log_var_img = self.encode_image(image)

        text_encoding, text_output, indices, mu_text, log_var_text = self.decode_text(image_encoding)

        return indices

    def text_image(self, batch_positions, text_length):

        text_encoding, mu_text, log_var_text = self.encode_text(batch_positions,text_length)

        image, image_norm = self.decode_image(text_encoding)

        return image

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

