from __future__ import print_function
from __future__ import division
import torch
import torch.nn as nn
import torchvision
from torchvision import models
from models.utils import sample_z
print("PyTorch Version: ",torch.__version__)
print("Torchvision Version: ",torchvision.__version__)

### Torchvision models ###

model_names = ['resnet', 'alexnet', 'vgg', 'squeezenet', 'densenet', 'inception']

###################This is identitx function for NN###########
class Identity(nn.Module):
    def __init__(self):
        super(Identity, self).__init__()
        
    def forward(self, x):
        return x
#################################################################

def set_parameter_requires_grad(model, feature_extracting):
    if feature_extracting:
        for param in model.parameters():
            param.requires_grad = False


class ImageEncoder(nn.Module):
    def __init__(self, model, device, vae=False):
        super(ImageEncoder, self).__init__()

        self.model = model
        self.vae = vae
        self.device=device

        # if self.vae:
        #     self.vae_transform = nn.Sequential(nn.Linear(self.hid_dim, self.hid_dim * 2),
        #                                            nn.Tanh())

    def forward(self, x):
        enc = self.model(x)
        #print('encoder size:', enc.size())

        if self.vae:
            # z = self.vae_transform(enc)
            mu = enc[:,:enc.size()[1]//2]
            #print('mu size:', mu.size())
            log_var = enc[:,enc.size()[1]//2:]
            #print('log size:', log_var.size())

            if self.training:
                #print('training mode; taking samples')
                enc = sample_z(mu, log_var, self.device)
            else:
                #print('testing mode; taking mean')
                enc = mu
        else:
            mu, log_var = None, None
        #print('new encode size:', enc.size())

        return enc, mu, log_var


def initialize_torchvision_model(model_name, output_dim, feature_extract, device, use_pretrained=True, vae=False):
    # Initialize these variables which will be set in this if statement. Each of these
    #   variables is model specific.
    model_ft = None
    input_size = 0

    if model_name == "resnet18":
        """ Resnet18
        """
        model_ft = models.resnet18(pretrained=use_pretrained)
        set_parameter_requires_grad(model_ft, feature_extract)
        model_ft.avgpool = Identity()##############average pool replaced by identity####
        #num_ftrs = model_ft.fc.in_features
        num_ftrs = model_ft.fc.in_features*7*7 # as average pooling is replaced by Identity, infeatures are multipleid with kernel size
        if vae: #vae changes: reparameterization at vae size down samples by 2 so we multiply by 2 to maintain consistency in the pipeline
            model_ft.fc = nn.Linear(num_ftrs, output_dim*2) #vae changes
        else:#vae chages
            model_ft.fc = nn.Linear(num_ftrs, output_dim) #vae changes
        #model_ft.fc = nn.Linear(num_ftrs, output_dim) #vae changes
        input_size = 224
        
    elif model_name == "resnet152":
        """ Resnet152
        """
        model_ft = models.resnet152(pretrained=use_pretrained)
        set_parameter_requires_grad(model_ft, feature_extract)
        num_ftrs = model_ft.fc.in_features
        if vae: #vae changes: reparameterization at vae size down samples by 2 so we multiply by 2 to maintain consistency in the pipeline
            model_ft.fc = nn.Linear(num_ftrs, output_dim*2) #vae changes
        else:#vae chages
            model_ft.fc = nn.Linear(num_ftrs, output_dim) #vae changes
        #model_ft.fc = nn.Linear(num_ftrs, output_dim) #vae changes
        input_size = 224

    elif model_name == "alexnet":
        """ Alexnet
        """
        model_ft = models.alexnet(pretrained=use_pretrained)
        set_parameter_requires_grad(model_ft, feature_extract)
        num_ftrs = model_ft.classifier[6].in_features
        model_ft.classifier[6] = nn.Linear(num_ftrs, output_dim)
        input_size = 224

    elif model_name == "vgg":
        """ VGG11_bn
        """
        model_ft = models.vgg11_bn(pretrained=use_pretrained)
        set_parameter_requires_grad(model_ft, feature_extract)
        num_ftrs = model_ft.classifier[6].in_features
        model_ft.classifier[6] = nn.Linear(num_ftrs, output_dim)
        input_size = 224

    elif model_name == "squeezenet":
        """ Squeezenet
        """
        model_ft = models.squeezenet1_0(pretrained=use_pretrained)
        set_parameter_requires_grad(model_ft, feature_extract)
        model_ft.classifier[1] = nn.Conv2d(512, output_dim, kernel_size=(1, 1), stride=(1, 1))
        model_ft.num_classes = output_dim
        input_size = 224

    elif model_name == "densenet":
        """ Densenet
        """
        model_ft = models.densenet121(pretrained=use_pretrained)
        set_parameter_requires_grad(model_ft, feature_extract)
        num_ftrs = model_ft.classifier.in_features
        model_ft.classifier = nn.Linear(num_ftrs, output_dim)
        input_size = 224

    elif model_name == "inception":
        """ Inception v3
        Be careful, expects (299,299) sized images and has auxiliary output
        """
        model_ft = models.inception_v3(pretrained=use_pretrained)
        set_parameter_requires_grad(model_ft, feature_extract)
        # Handle the auxilary net
        num_ftrs = model_ft.AuxLogits.fc.in_features
        model_ft.AuxLogits.fc = nn.Linear(num_ftrs, output_dim)
        # Handle the primary net
        num_ftrs = model_ft.fc.in_features
        model_ft.fc = nn.Linear(num_ftrs, output_dim)
        input_size = 299

    else:
        print("Invalid model name, exiting...")
        exit()


    model_ft = ImageEncoder(model_ft, vae=vae, device=device)

    return model_ft, input_size

