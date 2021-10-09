from models.utils import get_activation_function
import torch
from torch import nn
from models.stack_gan2.model import conv3x3

class GaussianNoise(nn.Module):
    def __init__(self, device, stddev=0.2):
        super(GaussianNoise, self).__init__()
        self.stddev = stddev
        self.device = device

    def forward(self, din):
        if self.training:
            noise = torch.autograd.Variable(torch.randn(din.size())).to(self.device)
            # noise = noise.cuda() if self.CUDA else noise
            return din + noise * self.stddev
        return din

class Discriminator(nn.Module):

    def __init__(self, emb_dim, dis_layers, dis_hid_dim, dis_dropout, dis_input_dropout, noise,device):
        super(Discriminator, self).__init__()


        self.emb_dim = emb_dim
        self.dis_layers = dis_layers
        self.dis_hid_dim = dis_hid_dim
        self.dis_dropout = dis_dropout
        self.dis_input_dropout = dis_input_dropout
        self.noise = noise
        self.device = device

        # self.emb_dim = params.emb_dim
        # self.dis_layers = params.dis_layers
        # self.dis_hid_dim = params.dis_hid_dim
        # self.dis_dropout = params.dis_dropout
        # self.dis_input_dropout = params.dis_input_dropout

        layers = [nn.Dropout(self.dis_input_dropout)]
        if self.noise:
            layers.append(GaussianNoise(device))
        for i in range(self.dis_layers ):
            input_dim = self.emb_dim if i == 0 else self.dis_hid_dim
            output_dim = 1 if i == self.dis_layers else self.dis_hid_dim
            layers.append(nn.Linear(input_dim, output_dim))
            if i < self.dis_layers:
                layers.append(get_activation_function('LeakyRelu'))
                layers.append(nn.Dropout(self.dis_dropout))
        # layers.append(nn.Sigmoid())

        self.final = nn.Linear(input_dim, 1)

        self.sigmoid = nn.Sigmoid()
        self.layers = nn.Sequential(*layers)

    def forward(self, x):
        assert x.dim() == 2 and x.size(1) == self.emb_dim

        logits = self.layers(x)

        output = self.sigmoid(self.final(logits))

        return output.view(-1), logits
        # return self.layers(x).view(-1)




def Block3x3_leakRelu(in_planes, out_planes):
    block = nn.Sequential(
        conv3x3(in_planes, out_planes),
        nn.BatchNorm2d(out_planes),
        nn.LeakyReLU(0.2, inplace=True)
    )
    return block


# Downsale the spatial size by a factor of 2
def downBlock(in_planes, out_planes):
    block = nn.Sequential(
        nn.Conv2d(in_planes, out_planes, 4, 2, 1, bias=False),
        nn.BatchNorm2d(out_planes),
        nn.LeakyReLU(0.2, inplace=True)
    )
    return block


# Downsale the spatial size by a factor of 16
def encode_image_by_16times(ndf):
    encode_img = nn.Sequential(
        # --> state size. ndf x in_size/2 x in_size/2
        nn.Conv2d(3, ndf, 4, 2, 1, bias=False),
        nn.LeakyReLU(0.2, inplace=True),
        # --> state size 2ndf x x in_size/4 x in_size/4
        nn.Conv2d(ndf, ndf * 2, 4, 2, 1, bias=False),
        nn.BatchNorm2d(ndf * 2),
        nn.LeakyReLU(0.2, inplace=True),
        # --> state size 4ndf x in_size/8 x in_size/8
        nn.Conv2d(ndf * 2, ndf * 4, 4, 2, 1, bias=False),
        nn.BatchNorm2d(ndf * 4),
        nn.LeakyReLU(0.2, inplace=True),
        # --> state size 8ndf x in_size/16 x in_size/16
        nn.Conv2d(ndf * 4, ndf * 8, 4, 2, 1, bias=False),
        nn.BatchNorm2d(ndf * 8),
        nn.LeakyReLU(0.2, inplace=True)
    )
    return encode_img


# For 64 x 64 images
class D_NET64(nn.Module):
    def __init__(self, df_dim, ef_dim, conditional):
        super(D_NET64, self).__init__()
        # self.df_dim = cfg.GAN.DF_DIM
        # self.ef_dim = cfg.GAN.EMBEDDING_DIM
        self.df_dim = df_dim
        self.ef_dim = ef_dim
        self.conditional = conditional

        self.define_module()


    def define_module(self):
        ndf = self.df_dim
        efg = self.ef_dim
        self.img_code_s16 = encode_image_by_16times(ndf)

        self.sigmoid = nn.Sigmoid()

        self.logits = nn.Sequential(
            nn.Conv2d(ndf * 8, 256, kernel_size=4, stride=4),

            )
        self.output = nn.Linear(256, 1)
        if self.conditional:
            self.jointConv = Block3x3_leakRelu(ndf * 8 + efg, ndf * 8)
            self.uncond_logits = nn.Sequential(
                nn.Conv2d(ndf * 8, 1, kernel_size=4, stride=4), nn.Sigmoid())

    def forward(self, x_var, c_code=None):
        x_code = self.img_code_s16(x_var)

        if self.conditional and c_code is not None:
            c_code = c_code.view(-1, self.ef_dim, 1, 1)
            c_code = c_code.repeat(1, 1, 4, 4)
            # state size (ngf+egf) x 4 x 4
            h_c_code = torch.cat((c_code, x_code), 1)
            # state size ngf x in_size x in_size
            h_c_code = self.jointConv(h_c_code)
        else:
            h_c_code = x_code

        logits = self.logits(h_c_code)

        logits = logits.squeeze()

        output = self.sigmoid(self.output(logits))
        if self.conditional:
            out_uncond = self.uncond_logits(x_code)
            return [output.view(-1), out_uncond.view(-1)]
        else:
            return [output.view(-1)], logits
