import torch.nn as nn


class ImageToTextModel(nn.Module):

    def __init__(self, vocab, cfg, device):
        super(ImageToTextModel, self).__init__()

        self.vocab = vocab
        self.vocab_size = self.vocab.vocab_size()
        self.sos = self.vocab.sos_pos()
        self.eos = self.vocab.eos_pos()
        self.pad = self.vocab.pad_pos()

        self.image_encoder, self.image_input_size = initialize_torchvision_model(cfg.IMAGE.ENCODER_NAME,
                                                                      cfg.IMAGE.DIMENSION,
                                                                      cfg.IMAGE.FIX_ENCODER,
                                                                      cfg.IMAGE.PRETRAINED_ENCODER)

        self.text_encoder_decoder = RNNText(emb_dim=cfg.TEXT.EMBEDDING_DIM,
                                            vocab_size=self.vocab_size,
                                            hid_dim=cfg.TEXT.DIMENSION,
                                            n_layers=cfg.TEXT.N_LAYERS,
                                            dropout=cfg.TEXT.DROPOUT,
                                            sos=self.sos,
                                            eos=self.eos,
                                            device=device,
                                            vae = cfg.TEXT.VAE)

    def encode_image(self, x):
        return self.image_encoder(x)

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

    def image_text(self, image, text_length=None):

        image_encoding, mu_img, log_var_img = self.encode_image(image)

        text_encoding, text_output, indices, mu_text, log_var_text = self.decode_text(image_encoding, text_length=text_length)

        return text_output, indices

    def forward(self, x, text_length=None):
        return self.image_text(x, text_length=text_length)

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


class TextToImageModel(nn.Module):

    def __init__(self, vocab, cfg, device):
        super(TextToImageModel, self).__init__()

        self.vocab = vocab
        self.vocab_size = self.vocab.vocab_size()
        self.sos = self.vocab.sos_pos()
        self.eos = self.vocab.eos_pos()
        self.pad = self.vocab.pad_pos()

        self.image_decoder = G_NET()

        self.text_encoder_decoder = RNNText(emb_dim=cfg.TEXT.EMBEDDING_DIM,
                                            vocab_size=self.vocab_size,
                                            hid_dim=cfg.TEXT.DIMENSION,
                                            n_layers=cfg.TEXT.N_LAYERS,
                                            dropout=cfg.TEXT.DROPOUT,
                                            sos=self.sos,
                                            eos=self.eos,
                                            device=device,
                                            vae=False)

        self.adv_image = D_NET64(df_dim=cfg.GAN.DF_DIM,
                                 ef_dim=cfg.GAN.EMBEDDING_DIM,
                                 conditional=cfg.GAN.B_CONDITION)

    def encode_text(self, batch_positions, text_length):
        return self.text_encoder_decoder(pass_type='encode', batch_positions=batch_positions, text_length=text_length)

    def decode_image(self, x):
        return self.image_decoder(x)

    def text_image(self, batch_positions, text_length):
        text_encoding, mu_text, log_var_text = self.encode_text(batch_positions,text_length)

        image, image_norm = self.decode_image(text_encoding)

        return image, text_encoding, mu_text, log_var_text

    def forward(self, batch_positions, text_length):
        image, text_encoding, mu_text, log_var_text = self.text_image(batch_positions, text_length)
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


#################################################################

# Currently only the above classes are used. They should be almost
# identical in their behaviour with the classes below.
#
# There are some small differences like
#         if cfg.IMAGE.VAE:
#             image_multiplier = 2
#         else:
#             image_multiplier = 1
# which is only the Image2Text model.
#
# The classes below might have an outdated interface so I recommend using
# ImageToTextModel and TextToImageModel.
# TODO: check differences and remove what is not needed anymore

#################################################################

import torch
from torch import nn
from models.image_models import  initialize_torchvision_model
# from models.image_models import SimpleImageDecoder, initialize_torchvision_model
from models.text_models import RNNText
from models.adversarial_models import Discriminator, D_NET64
from models.stack_gan2.model import G_NET
from torch.nn.modules import padding



class Image2Text(nn.Module):

    def __init__(self, vocab, cfg, device):
        super(Image2Text, self).__init__()

        self.vocab = vocab
        self.vocab_size = self.vocab.vocab_size()
        self.sos = self.vocab.sos_pos()
        self.eos = self.vocab.eos_pos()

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


        self.pad = padding.ConstantPad2d(80, 0.0)


    def encode_image(self, x):

        return self.image_encoder(x)


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


    def image_text(self, image):

        image_encoding, mu_img, log_var_img  = self.encode_image(image)

        text_encoding, text_output, indices, mu_text, log_var_text = self.decode_text(image_encoding)

        return text_encoding, text_output, indices, image_encoding, mu_img, log_var_img


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






class Text2Image(nn.Module):

    def __init__(self, vocab, cfg, device):
        super(Text2Image, self).__init__()

        self.vocab = vocab
        self.vocab_size = self.vocab.vocab_size()
        self.sos = self.vocab.sos_pos()
        self.eos = self.vocab.eos_pos()

        self.device = device

        self.cfg = cfg

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


        self.adv_image = D_NET64(df_dim=cfg.GAN.DF_DIM,
                                 ef_dim=cfg.GAN.EMBEDDING_DIM,
                                 conditional=cfg.GAN.B_CONDITION)



    def encode_text(self, batch_positions, text_length):
        return self.text_encoder_decoder(pass_type='encode', batch_positions=batch_positions, text_length=text_length)

    def decode_image(self, x):
        return self.image_decoder(x)


    def text_image(self, batch_positions, text_length):

        text_encoding, mu_text, log_var_text = self.encode_text(batch_positions,text_length)

        image, image_norm = self.decode_image(text_encoding)

        return image, text_encoding, mu_text, log_var_text

        # return re_image, image_encoding, text_encoding, image_norm, mu_text, log_var_text

    def forward(self, batchpositions, text_length):
        return self.text_image(batchpositions, text_length)

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

