from __future__ import print_function
from __future__ import division
import os
import errno
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.autograd import Variable
import torchvision
import time
from data.datasets1 import BirdsDataset1, FlowersDataset1
from torchvision import transforms
import torchvision.models as torch_models
from config import cfg
from models.stack_gan2.model1 import encoder_resnet1, G_NET1, G_NET, D_NET_TEXT1, MAP_NET_IT22, D_NET_IMAGE1, MAP_NET_TI22
import models.text_auto_models1 as text_models
from data.resultwriter import ResultWriter
from models.utils1 import Logger
from tensorboardX import SummaryWriter
from tensorboardX import summary
from tensorboardX import FileWriter
import datetime
import dateutil.tz
import sys
from pathlib import Path
from tempfile import mkdtemp
import pickle
from copy import deepcopy
print("PyTorch Version: ",torch.__version__)
print("Torchvision Version: ",torchvision.__version__)

#print('sys.argv[1]',sys.argv[1])
#print('sys.argv[2]',sys.argv[2])
#print('sys.argv[3]',sys.argv[3])
#print(os.getcwd())
torch.set_num_threads(5)

def norm_ip(img, min1, max1):
    img = img.clamp_(min=min1, max=max1)
    img = img.add_(-min1).div_(max1 - min1 + 1e-5)
    return img

def norm_range(t, range1=None):
    if range1 is not None:
        img1 = norm_ip(t, range1[0], range1[1])
    else:
        img1 = norm_ip(t, float(torch.min(t)), float(torch.max(t)))
    return img1

def mkdir_p(path):
    try:
        os.makedirs(path)
    except OSError as exc:  # Python >2.5
        if exc.errno == errno.EEXIST and os.path.isdir(path):
            pass
        else:
            raise

def adjust_padding(cap, len1):
    cap = cap.numpy()
    len1 = len1.numpy()
    max_len = max(len1)
    temp=[]
    for i in cap:
        j = i[0:max_len]
        temp.append(j)
    cap =torch.LongTensor(temp)
    len1= torch.LongTensor(len1)
    return cap, len1

###########################################################################
###########################################################################
####################### variables#######################################
##########################################################################
########################################################################

encoder_path='/home/das/dev/unsup/saved_models/flowers/encG_175800_128.pth' #image encoder
dec_path='/home/das/dev/unsup/saved_models/flowers/netG_175800_128.pth' #image decoder
text_autoencoder_path = '//home/das/dev/unsup/saved_models/flowers/AutoEncoderDglove100_flowerFalse203.pt' # text auto encoder
IT_GEN_PATH = 'netGIT_33000.pth' # netGIT_100.pth #this is Image t TEXt embedding generator , if not restarted
#from previous fails should be empty
IT_DIS_PATH= 'netDIT.pth' #netDIT.pth
TI_GEN_PATH= '' #netGTI_100.pth
TI_DIS_PATH= '' #netDTI.pth

dset='flowers'#'birds'

if dset=='birds':
    glove_file='glove.6B'
else:
    glove_file='glove.6B_flowers'

embedding_matrx_path =glove_file + '/' +'emtrix.obj'
vocab_i2t_path = glove_file + '/' + 'vocab_i2t.obj'
vocab_t2i_path = glove_file + '/' + 'vocab_t2i.obj'


model_name = "AutoEncoderD"
embedding_dim = 100
hidden_dim = 100

# Top level data directory
data_dir = sys.argv[1]

# Batch size for training (change depending on how much memory you have)
batch_size = 32
# Number of epochs to train for
num_epochs = 500#150
#restart_epoch = 1 #restartting from failed step

# Flag for feature extracting. When False, we finetune the whole model, else we only extract features

# mean and std calculated fort the dataset
mean_r= 0.5
mean_g= 0.5
mean_b= 0.5
std_r= 0.5
std_g= 0.5
std_b= 0.5
lr1 = 1e-3
lr2 = 1e-3
wt = 1e-5
##################################################


# Detect if we have a GPU available
#device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
gpu_id = '5'
torch.cuda.set_device(int(gpu_id))
s_gpus = gpu_id.split(',')
gpus = [int(ix) for ix in s_gpus]

######################################################################
#########################################################################

#############################################################
now = datetime.datetime.now(dateutil.tz.tzlocal())
timestamp = now.strftime('%Y_%m_%d_%H_%M_%S')
#log_dir = 'output/%s_%s' %(cfg.DATASET_NAME, timestamp)
log_dir =  'output/flowers_2020_06_26_21_30_38'
mkdir_p(log_dir)
model_dir = os.path.join(log_dir, 'modeldir')
mkdir_p(model_dir)

sys.stdout = Logger('{}/run.log'.format(log_dir))
print("###############output folder###############################")
print(os.path.join(os.getcwd(),log_dir))
###############validation set dir#####################
img_dir_val = os.path.join(log_dir, 'imgdirval')
img_txt_dir_val = os.path.join(log_dir, 'imgtxtdirval')
txt_img_dir_val = os.path.join(log_dir, 'txtimgdirval')
results_writer_img_val = ResultWriter(img_dir_val)
results_writer_imgtxt_val = ResultWriter(img_txt_dir_val)
results_writer_txtimg_val = ResultWriter(txt_img_dir_val)
img_input = os.path.join(log_dir, 'img_input')
img_output = os.path.join(log_dir, 'img_out')
results_writer_input = ResultWriter(img_input)
results_writer_output = ResultWriter(img_output)



#######################################################################
########################################################################
########################################################################
################## Functions and Class defination#########################
#####################################################################



class SimpleAutoencoder(nn.Module):
    #############image autoencoder##############################
    def __init__(self, encoder, decoder):
        super(SimpleAutoencoder, self).__init__()
        self.encoder = encoder
        self.decoder = decoder

    def forward(self, x):
        x, mu, sigma = self.encoder(x)
        #x, _, _ = self.encoder(x)
        x, _ = self.decoder(x)
        return x, mu, sigma



def weights_init(m): # to inotialize weigts of model
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        nn.init.orthogonal(m.weight.data, 1.0)
    elif classname.find('BatchNorm') != -1:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)
    elif classname.find('Linear') != -1:
        nn.init.orthogonal(m.weight.data, 1.0)
        if m.bias is not None:
            m.bias.data.fill_(0.0)
            
def load_params(model, new_param):
    for p, new_p in zip(model.parameters(), new_param):
        p.data.copy_(new_p)


def copy_G_params(model):
    flatten = deepcopy(list(p.data for p in model.parameters()))
    return flatten


def define_optimizers(netGIT, netDIT,netGTI, netDTI, path):
    optimizerDIT = optim.Adam(netDIT.parameters(),
                         lr=cfg.TRAIN.DISCRIMINATOR_LR,
                         betas=(0.5, 0.999))
      
    # G_opt_paras = []
    # for p in netG.parameters():
    #     if p.requires_grad:
    #         G_opt_paras.append(p)
    optimizerGIT = optim.Adam(netGIT.parameters(),
                            lr=cfg.TRAIN.GENERATOR_LR,
                            betas=(0.5, 0.999))
    
    #optimizerDTI = optim.Adam(netDTI.parameters(),
     #                    lr=cfg.TRAIN.DISCRIMINATOR_LR,
      #                   betas=(0.5, 0.999))
    
    optimizerDTI = optim.RMSprop(netDTI.parameters(), lr=5e-5)
      
    # G_opt_paras = []
    # for p in netG.parameters():
    #     if p.requires_grad:
    #         G_opt_paras.append(p)
    #optimizerGTI = optim.Adam(netGTI.parameters(),
     #                       lr=cfg.TRAIN.GENERATOR_LR,
      #                      betas=(0.5, 0.999))
    optimizerGTI = optim.RMSprop(netGTI.parameters(), lr=5e-5)
    
    
    count = 0
    if IT_GEN_PATH != '':
        # example cfg.TRAIN.NET_G = 
        Gpath = os.path.join(path, IT_GEN_PATH)
        print('loading optimizer from ', Gpath)
        checkpoint = torch.load(Gpath)
        optimizerGIT.load_state_dict(checkpoint['optimizer'])
        optimizerGIT = optimizerToDevice(optimizerGIT)
        istart = IT_GEN_PATH.rfind('_') + 1
        iend = IT_GEN_PATH.rfind('.')
        count = IT_GEN_PATH[istart:iend]
        count = int(count) + 1
        
    if IT_DIS_PATH != '':
            Dpath = os.path.join(path, IT_DIS_PATH)
            checkpoint = torch.load(Dpath)
            print('loading optimizer from ', Dpath)
            optimizerDIT.load_state_dict(checkpoint['optimizer'])
            optimizerDIT = optimizerToDevice(optimizerDIT)
    
    if TI_GEN_PATH != '':
        # example cfg.TRAIN.NET_G = 
        Gpath = os.path.join(path, TI_GEN_PATH)
        print('loading optimizer from ', Gpath)
        checkpoint = torch.load(Gpath)
        optimizerGTI.load_state_dict(checkpoint['optimizer'])
        optimizerGTI = optimizerToDevice(optimizerGTI)
        istart = TI_GEN_PATH.rfind('_') + 1
        iend = TI_GEN_PATH.rfind('.')
        #count = TI_GEN_PATH[istart:iend]
        #count = int(count) + 1
        
    if TI_DIS_PATH != '':
            Dpath = os.path.join(path, TI_DIS_PATH)
            checkpoint = torch.load(Dpath)
            print('loading optimizer from ', Dpath)
            optimizerDTI.load_state_dict(checkpoint['optimizer'])
            optimizerDTI = optimizerToDevice(optimizerDTI)
    return optimizerGIT, optimizerDIT, optimizerGTI, optimizerDTI, count
 
def load_network(path):
    ####################Image deoder################################
    netG = G_NET1()
    netG.apply(weights_init)
    netG = torch.nn.DataParallel(netG, device_ids=gpus)
    #################################################################
    #########################Image to text GEN##################################
    genIT = MAP_NET_IT22()
    genIT.apply(weights_init)
    genIT = torch.nn.DataParallel(genIT, device_ids=gpus)
    ####################################################################
    #########################Image to text Discriminator###############
    disIT = D_NET_TEXT1()
    disIT.apply(weights_init)
    disIT = torch.nn.DataParallel(disIT, device_ids=gpus)
    print(disIT)
    #########################Text to Image GEN##################################
    genTI = MAP_NET_TI22()
    genTI.apply(weights_init)
    genTI = torch.nn.DataParallel(genTI, device_ids=gpus)
    #########################text to IMAGE Discriminator###############
    disTI = D_NET_IMAGE1()
    disTI.apply(weights_init)
    disTI = torch.nn.DataParallel(disTI, device_ids=gpus)
    print(disTI)
    
    
    
    if IT_GEN_PATH != '':
        # example cfg.TRAIN.NET_G = 
        Gpath = os.path.join(path, IT_GEN_PATH)
        #print(Gpath)
        #print(genIT)
        checkpoint = torch.load(Gpath)
        genIT.load_state_dict(checkpoint['state_dict'])
        #Epath = os.path.join(path, 'encG.pth' )
        #checkpoint = torch.load(Epath)
        #enc.load_state_dict(checkpoint['state_dict'])
        
        print('Load ', IT_GEN_PATH)

        #istart = IT_GEN_PATH('_') + 1
        #iend = IT_GEN_PATH.rfind('.')
        #count = IT_GEN_PATH[istart:iend]
        #count = int(count) + 1

    if IT_DIS_PATH != '':
            Dpath = os.path.join(path, IT_DIS_PATH)
            print('Load ', IT_DIS_PATH)
            checkpoint = torch.load(Dpath)
            disIT.load_state_dict(checkpoint['state_dict'])
            
    if TI_GEN_PATH != '':
        # example cfg.TRAIN.NET_G = 
        Gpath = os.path.join(path, TI_GEN_PATH)
        checkpoint = torch.load(Gpath)
        genTI.load_state_dict(checkpoint['state_dict'])
        #Epath = os.path.join(path, 'encG.pth' )
        #checkpoint = torch.load(Epath)
        #enc.load_state_dict(checkpoint['state_dict'])
        
        print('Load ', TI_GEN_PATH)

        #istart = IT_GEN_PATH('_') + 1
        #iend = IT_GEN_PATH.rfind('.')
        #count = IT_GEN_PATH[istart:iend]
        #count = int(count) + 1

    if TI_DIS_PATH != '':
            Dpath = os.path.join(path, TI_DIS_PATH)
            print('Load ', TI_DIS_PATH)
            checkpoint = torch.load(Dpath)
            disTI.load_state_dict(checkpoint['state_dict'])

    

    #if cfg.CUDA:
    if True:
        netG.cuda()
        genIT.cuda()
        disIT.cuda()
        genTI.cuda()
        disTI.cuda()
        
    
    
    return netG, genIT, disIT, genTI, disTI



def loss_function(final_img,residual_img,upscaled_img,com_img,orig_img):
#size average false means return sum over all pixel points if set to true average over pixel points returned
  com_loss = nn.MSELoss(size_average=False)(orig_img, final_img)
  rec_loss = nn.MSELoss(size_average=False)(residual_img,orig_img-upscaled_img)
  
  return com_loss + rec_loss


def KL_loss(mu, logvar):
    # -0.5 * sum(1 + log(sigma^2) - mu^2 - sigma^2)
    KLD_element = mu.pow(2).add_(logvar.exp()).mul_(-1).add_(1).add_(logvar)
    KLD = torch.mean(KLD_element).mul_(-0.5)
    return KLD



        
def load_checkpoint(model, optimizer, filename):
    # Note: Input model & optimizer should be pre-defined.  This routine only updates their states.
    print("=> loading checkpoint '{}'".format(filename))
    checkpoint = torch.load(filename)
    model.load_state_dict(checkpoint['state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer'])
    return model, optimizer

def optimizerToDevice(optimizer):
    for state in optimizer.state.values():
            for k, v in state.items():
                if isinstance(v, torch.Tensor):
                    state[k] = v.cuda()
    return optimizer


def save_model(netGIT, optimizerGIT, netDIT, optimizerDIT, netGTI, optimizerGTI, netDTI, optimizerDTI, epoch, model_dir):
    #load_params(netG, avg_param_G)
    #load_params(enc, avg_param_E)
    
    
    
    
    stateGIT = {'state_dict': netGIT.state_dict(),
             'optimizer': optimizerGIT.state_dict()}
    torch.save(
        stateGIT,
        '%s/netGIT_%d.pth' % (model_dir, epoch))
    stateDIT = {'state_dict':  netDIT.state_dict(),
             'optimizer': optimizerDIT.state_dict()}
    torch.save(
            stateDIT,
            '%s/netDIT.pth' % model_dir)
    
    
    stateGTI = {'state_dict': netGTI.state_dict(),
             'optimizer': optimizerGTI.state_dict()}
    torch.save(
        stateGTI,
        '%s/netGTI_%d.pth' % (model_dir, epoch))
    stateDTI = {'state_dict':  netDTI.state_dict(),
             'optimizer': optimizerDTI.state_dict()}
    torch.save(
            stateDTI,
            '%s/netDTI.pth' % model_dir)
    
    print('Save G/Ds models...count:%d'%epoch)
        
def initialize_model(model_name, config, embeddings_matrix):
    
    model_ft= text_models.AutoEncoderD(config, embeddings_matrix)
    model_ft.cuda()
    #model_ft = model_ft.to(device)

    
    dec, genIT, disIT, genTI, disTI= load_network(model_dir)
    
    #enc = torch_models.resnet50(pretrained=True)
    #num_ftrs = enc.fc.in_features
    #enc.fc = nn.Linear(num_ftrs, 1024)
    #enc = enc.to(device)
    enc = encoder_resnet1()
    enc.cuda()
    
    print("=> loading Image encoder from '{}'".format(encoder_path))
    encoder = torch.load(encoder_path)
    enc.load_state_dict(encoder['state_dict'])
    
    
    print("=> loading Image decoder from '{}'".format(dec_path))
    decoder = torch.load(dec_path)
    dec.load_state_dict(decoder['state_dict'])
    
    
    print("=> loading text autoencoder from '{}'".format(text_autoencoder_path))
    text_autoencoder = torch.load(text_autoencoder_path)
    model_ft.load_state_dict(text_autoencoder['state_dict'])
    

    return model_ft, enc, dec, genIT, disIT, genTI, disTI



def save_results(imgs_input, imgs_output, imgs_generated, text_input, text_generated, text_output, count):
    if count != -1: #for validation will be saved in a single folder
        img_dir = os.path.join(log_dir, 'imgdir%d'%count)
        img_txt_dir = os.path.join(log_dir, 'imgtxtdir%d'%count)
        txt_img_dir = os.path.join(log_dir, 'txtimgdir%d'%count)
        results_writer_img = ResultWriter(img_dir)
        results_writer_imgtxt = ResultWriter(img_txt_dir)
        results_writer_txtimg = ResultWriter(txt_img_dir)
    else:
        img_dir = img_dir_val
        img_txt_dir = img_txt_dir_val
        results_writer_img = results_writer_img_val
        results_writer_imgtxt = results_writer_imgtxt_val
        results_writer_txtimg = results_writer_txtimg_val
        
    fg =open(os.path.join(img_dir, 'generated.txt'), 'a+')
    fo =open(os.path.join(img_dir, 'output.txt'), 'a+')
    for ii, io, ig, ti, tg, to in zip(imgs_input, imgs_output, imgs_generated, text_input, text_generated, text_output):
        ii = norm_range(ii)#normalize to (0,1)
        io = norm_range(io)#normalize to (0,1)
        ig = norm_range(ig)#normalize to (0,1)
        ii = ii.cpu()
        io = io.detach().cpu()
        if count == -1:
            results_writer_input.write_images1(ii)
            results_writer_output.write_images1(io)
        ii = ii.numpy().transpose(1,2,0) #in order to use plt.imshow the channel should be the last dimention
        io = io.numpy().transpose(1,2,0)
        ig = ig.detach().cpu().numpy().transpose(1,2,0)
        results_writer_img.write_images(io, ii)
        results_writer_imgtxt.write_image_with_text(ii, tg)
        results_writer_txtimg.write_image_with_text(ig, ti)
        print(ti,'\t',tg, file = fg)
        print(ti,'\t',to, file = fo)
    fg.close()
    fo.close()
    
 

class ImageTextTrainer(object):
    def __init__(self, model, enc, dec, genIT, disIT, genTI, disTI, dataloaders, num_epochs, log_dir):
        self.model = model
        self.enc = enc
        self.dec = dec
        self.genIT= genIT
        self.disIT = disIT
        self.genTI= genTI
        self.disTI = disTI
        self.dataloaders = dataloaders
        self.num_epochs = num_epochs
        self.criterion = nn.BCELoss()
        self.batch_size = batch_size
        self.max_epoch= num_epochs
        self.num_batches = len(self.dataloaders['train'])
        self.log_dir = log_dir
        self.tensor_board = os.path.join(self.log_dir, 'tensorboard')
        #self.model_dir = os.path.join(self.log_dir, 'modeldir')
        self.model_dir = model_dir
        mkdir_p(self.tensor_board)
        #mkdir_p(self.model_dir)
        self.writer = SummaryWriter(self.tensor_board)
        self.train_dis1 = True
        self.train_dis2 = True
        self.train_gen1 = False
        self.train_gen2 = False
        
     

   

    def train_Dnet(self, count):
        flag = count % 100
        
        criterion = self.criterion

        netDIT, optDIT = self.disIT, self.optimizerDIT
        #print('netDIT:', netDIT)
        #print('self.optimizerDIT:', self.optimizerDIT)
       
        #####################For IT Training#####################
        batch_size = self.text_embedding[0].size(0)
        real_embedding_text = self.text_embedding[0]
        #print('real text emb shape:' , real_embedding.shape)
        fake_embedding_text = self.text_embedding_fake
        #print('fake text emb shape:' , fake_embedding.shape)
        #
        netDIT.zero_grad()
        # Forward
        real_labels = self.real_labels[:batch_size]
        fake_labels = self.fake_labels[:batch_size]
        # for real
        real_logitsIT = netDIT(real_embedding_text.detach())
        fake_logitsIT = netDIT(fake_embedding_text.detach())
        #
        errD_realIT = criterion(real_logitsIT[0], real_labels)
        errD_fakeIT = criterion(fake_logitsIT[0], fake_labels)
       
        errDIT = errD_realIT + errD_fakeIT
        
        
       #######################For TI Training#################### 
        netDTI, optDTI = self.disTI, self.optimizerDTI
        #print('netDTI:', netDTI)
        #print('self.optimizerDTI:', self.optimizerDTI)
        batch_size = self.img_embedding.size(0)
        
        real_embedding_image = self.img_embedding
        #print('real image emb shape:' , real_embedding.shape)
        fake_embedding_image = self.img_embedding_fake
        #print('fake image emb shape:' , fake_embedding.shape)
        #
        netDTI.zero_grad()
        # Forward
        real_labels = self.real_labels[:batch_size]
        fake_labels = self.fake_labels[:batch_size]
        # for real
        real_logitsTI = netDTI(real_embedding_image.detach())
        fake_logitsTI = netDTI(fake_embedding_image.detach())
        #
        errD_realTI = criterion(real_logitsTI[0], real_labels)
        errD_fakeTI = criterion(fake_logitsTI[0], fake_labels)
        #errD_realTI = -torch.mean(real_logitsTI[0])
        #errD_fakeTI = torch.mean(fake_logitsTI[0])
       
        errDTI = errD_realTI + errD_fakeTI
        ###########################################################
        #if True:
        if (count +1)% self.mod != 0 :
        #if self.train_dis1 == True:
            #print("di param:")
            #for name, param in netD.named_parameters():
             #   print (name, param.data)
        # backward
            errDIT.backward()
            torch.nn.utils.clip_grad_norm_(netDIT.parameters(), 5.00)
        # update parameters
            optDIT.step()
            
        #if True:
        mod2= 5
        #if (count +1)%self.mod != 0 :
        if (count +1)%mod2 != 0 :
        #if self.train_dis2 == True:
        # backward
            errDTI.backward()
            torch.nn.utils.clip_grad_norm_(netDTI.parameters(), 5.00)
        # update parameters
            optDTI.step()
            for p in netDTI.parameters():
                p.data.clamp_(-0.01, 0.01)
        # log
        if flag == 0:
            self.writer.add_scalar('DIT_loss', errDIT.item(), count)
            self.writer.add_scalar('DTI_loss', errDTI.item(), count)
        return errDIT, errDTI

    def train_Gnet(self, count):
        
        self.genIT.zero_grad()
        errGIT_total = 0
        mod1 = self.mod
        mod2 = self.mod
        
        flag = count % 100
        batch_size = self.text_embedding[0].size(0)
        criterion= self.criterion
       
        real_labels = self.real_labels[:batch_size]
        outputs = self.disIT(self.text_embedding_fake)
        errGIT_total= criterion(outputs[0], real_labels)
        if flag == 0:
            self.writer.add_scalar('GIT_loss', errGIT_total.item(), count)
            
        if errGIT_total > 1.5:
            self.train_gen1 = True
            mod1 =1 
        elif errGIT_total < 1.0 :
            self.train_gen1 = False
            mod1 = self.mod
        
        
            
        if (count +1)% mod1 ==0 :
        #if self.train_dis1 == False or (self.train_gen1 == True) :
        #if True:
            errGIT_total.backward()
            torch.nn.utils.clip_grad_norm_(self.genIT.parameters(), 5.00)
            self.optimizerGIT.step()
            #print("GEN param:")
            #for name, param in self.gen.named_parameters():
             #   print (name, param.data)
        #####################for TI############################
        self.genTI.zero_grad()
        errGTI_total = 0
        batch_size = self.img_embedding.size(0)
       
        real_labels = self.real_labels[:batch_size]
        outputs = self.disTI(self.img_embedding_fake)
        errGTI_total= criterion(outputs[0], real_labels)
        #errGTI_total= -torch.mean(outputs[0])
        if flag == 0:
            self.writer.add_scalar('GTI_loss', errGTI_total.item(), count)
            
            
        if errGTI_total > 1.5:
            self.train_gen2= True
            mod2 = 1
        elif errGTI_total <1.0 :
            self.train_gen2 = False
            mod2 = self.mod
            
        mod2= 5
            
        #if True:
        if (count +1)% mod2  ==0:
        #if self.train_dis2 == False or (self.train_gen2 == True):
            errGTI_total.backward()
            torch.nn.utils.clip_grad_norm_(self.genTI.parameters(), 5.00)
            self.optimizerGTI.step()
            #print("GEN param:")
            #for name, param in self.gen.named_parameters():
             #   print (name, param.data)
        
        return errGIT_total, errGTI_total
    
    def evaluate(self):
        ecount=0
        ######setting in eval mode##########################
        self.enc.eval()
        self.dec.eval()
        self.model.eval()
        self.genIT.eval()
        self.genTI.eval()
        ############################################################
        nz = cfg.GAN.Z_DIM
        #fixed_noise1 = Variable(torch.FloatTensor(self.batch_size, nz).normal_(0, 1))
        fixed_noise2 = Variable(torch.FloatTensor(self.batch_size, nz).normal_(0, 1))
        #fixed_noise3 = Variable(torch.FloatTensor(self.batch_size, 20).normal_(0, 1))
        if True:
            #fixed_noise1 = fixed_noise1.cuda()
            fixed_noise2 = fixed_noise2.cuda()
            #fixed_noise3 = fixed_noise3.cuda()
        for uinputs, inputs,_, labels, captions, lengths in self.dataloaders['val']:
            with torch.no_grad():
                inp0 = uinputs[0]
                inp0= inp0.cuda()
                N = inp0.size(0)
                #n1 = fixed_noise1[:N]
                n2 = fixed_noise2[:N]
                #n3 = fixed_noise3[:N]
                captions, lengths= adjust_padding(captions, lengths)
                captions = captions.cuda()
                lengths = lengths.cuda()
            
                #self.img_embedding = self.enc(inp0)
                self.img_embedding, _, _ = self.enc(inp0)
                self.text_embedding = self.model.rnn(pass_type='encode',batch_positions=captions, text_length=lengths)

            
                #self.text_embedding_fake = self.genIT(n1, self.img_embedding.detach())
                self.text_embedding_fake = self.genIT(self.img_embedding.detach())
                temb = self.text_embedding_fake,
            
            ###################generated text from Image##################################
                length1 = [max_len]*N #taking maximum length
                length1= torch.LongTensor(length1)
                _, indices_g = self.model.rnn(pass_type ='generate', hidden=temb, text_length=length1, batch_size=N)
            ########################below verification for original encoded output#####################
                _, indices_o = self.model.rnn(pass_type ='generate', hidden=self.text_embedding, text_length=length1, batch_size=N)
            ###############We can use original image or output of image decoder#############
                #fake_imgs_o, _, _ = self.dec(n2, self.img_embedding.detach())
                fake_imgs_o = self.dec(n2, self.img_embedding.detach())
                #self.img_embedding_fake = self.genTI(n3, self.text_embedding[0].detach())
                self.img_embedding_fake = self.genTI(self.text_embedding[0].detach())
                #fake_imgs_g, _, _ = self.dec(n2, self.img_embedding_fake.detach())
                fake_imgs_g = self.dec(n2, self.img_embedding_fake.detach())
                texts_i = vocab.decode_positions(captions)
                texts_g = vocab.decode_positions(indices_g)
                texts_o = vocab.decode_positions(indices_o)
                if ecount % 1 == 0:
                    save_results(inputs[2], fake_imgs_o[2], fake_imgs_g[2], texts_i, texts_g, texts_o, -1)
                ecount = ecount +1
                del fake_imgs_o, fake_imgs_g, texts_i, texts_g, texts_o
                del self.img_embedding, self.img_embedding_fake, self.text_embedding, self.text_embedding_fake
                
        print("#################Evaluation complete#######################################")

        

    def train(self):
        #avg_param_GIT = copy_G_params(self.genIT)
        #avg_param_GTI = copy_G_params(self.genTI)
        
        

        self.optimizerGIT, self.optimizerDIT, self.optimizerGTI, self.optimizerDTI, count = define_optimizers(self.genIT, self.disIT, self.genTI, self.disTI, self.model_dir)


        self.real_labels = Variable(torch.FloatTensor(self.batch_size).fill_(1))
        self.fake_labels = Variable(torch.FloatTensor(self.batch_size).fill_(0))

        self.gradient_one = torch.FloatTensor([1.0])
        self.gradient_half = torch.FloatTensor([0.5])

        nz = cfg.GAN.Z_DIM
        noise1 = Variable(torch.FloatTensor(self.batch_size, nz))
        noise2 = Variable(torch.FloatTensor(self.batch_size, 20))
        fixed_noise1 = Variable(torch.FloatTensor(self.batch_size, nz).normal_(0, 1))
        fixed_noise2 = Variable(torch.FloatTensor(self.batch_size, nz).normal_(0, 1))
        fixed_noise3 = Variable(torch.FloatTensor(self.batch_size, 20).normal_(0, 1))
        
        ######setting in eval mode##########################
        self.enc.eval()
        self.dec.eval()
        self.model.eval()
        self.genIT.train()
        self.disIT.train()
        self.genTI.train()
        self.disTI.train()
        ############################################################
        nr_train_gen = 1
        nr_train_dis = 1
        self.mod = nr_train_gen + 2*nr_train_dis
        

        #if cfg.CUDA:
        if True:
            self.criterion.cuda()
            self.real_labels = self.real_labels.cuda()
            self.fake_labels = self.fake_labels.cuda()
            self.gradient_one = self.gradient_one.cuda()
            self.gradient_half = self.gradient_half.cuda()
            noise1, noise2, fixed_noise1, fixed_noise2, fixed_noise3 = noise1.cuda(), noise2.cuda(), fixed_noise1.cuda(), fixed_noise2.cuda(), fixed_noise3.cuda()
        start_count = count
        start_epoch = start_count // (self.num_batches)
        for epoch in range(start_epoch, self.max_epoch):
            start_t = time.time()
            errorGIT_list = []
            errorGTI_list = []
            errorDIT_list = []
            errorDTI_list = []
            i =-1

            for uinputs, inputs,_, labels, captions, lengths in self.dataloaders['train']:
                i+=1
                if i ==2 :
                    self.train_dis1= False
                    self.train_dis2= False
                elif i==3:
                    self.train_dis1 = True
                    self.train_dis2 = True
                    i= 0
                
                #######################################################
                # (0) Prepare training data
                ######################################################
                inp0 = uinputs[0]
                inp0= inp0.cuda()
                N = inp0.size(0)
                noise1_ = noise1[:N]
                noise2_ = noise2[:N]
                n1 = fixed_noise1[:N]
                #n2 = fixed_noise2[:N]
                #n3 = fixed_noise3[:N]
                #print('input shape:', N)
                captions, lengths= adjust_padding(captions, lengths)
                #print('captions shape:', captions.shape)
                #print('new cap:',captions)
                #print('new_len:', lengths)
                #print('length of dataset',len(dataloaders[phase].dataset))
                
                captions = captions.cuda()
                lengths = lengths.cuda()
                ##########################################################
                # (1) Image embedding from pretrained model
                ##########################################################
                #self.img_embedding = self.enc(inp0)
                self.img_embedding, _, _ = self.enc(inp0)
                #print('image emb shape:' , self.img_embedding.shape)
                
                ##########################################################
                # (2) Text embedding from pretrained model
                ##########################################################
                self.text_embedding = self.model.rnn(pass_type='encode',batch_positions=captions, text_length=lengths)

                #######################################################
                # (1) Generate fake text_embedding
                ######################################################
                noise1_.data.normal_(0, 1)
                #print('noise shape:', noise.shape)
                #print('image emb shape:', self.img_embedding.shape)
                #self.text_embedding_fake = self.genIT(noise1_, self.img_embedding.detach())
                self.text_embedding_fake = self.genIT(self.img_embedding.detach())
                #######################################################
                # (1) Generate fake image_embedding
                ######################################################
                noise2_.data.normal_(0, 1)
                
                #self.img_embedding_fake = self.genTI(noise2_, self.text_embedding[0].detach())
                self.img_embedding_fake = self.genTI(self.text_embedding[0].detach())
                #print('text fake emb shape:', self.text_embedding_fake.shape)
                
                
                #self.fake_imgs, self.mu, self.logvar = \
                    #self.netG(noise, self.txt_embedding.detach())

                #######################################################
                # (2) Update D network
                ######################################################
                errDIT_total, errDTI_total = self.train_Dnet(count)
                errorDIT_list.append(errDIT_total.item())
                errorDTI_list.append(errDTI_total.item())
                
                #######################################################
                # (3) Update G network: maximize log(D(G(z)))
                ######################################################
                errGIT_total, errGTI_total = self.train_Gnet(count)
                errorGIT_list.append(errGIT_total.item())
                errorGTI_list.append(errGTI_total.item())
                #for p, avg_p in zip(self.genIT.parameters(), avg_param_GIT):
                 #   avg_p.mul_(0.999).add_(0.001, p.data)
                #for p, avg_p in zip(self.genTI.parameters(), avg_param_GTI):
                 #   avg_p.mul_(0.999).add_(0.001, p.data)
                #for e, avg_e in zip(self.enc.parameters(), avg_param_E):
                 #   avg_e.mul_(0.999).add_(0.001, e.data)

                # for inception score
                
                count = count + 1

                if count % cfg.TRAIN.SNAPSHOT_INTERVAL == 0:
                #if count % 10 == 0:
                    save_model(self.genIT, self.optimizerGIT, self.disIT, self.optimizerDIT, self.genTI, self.optimizerGTI, self.disTI, self.optimizerDTI, count, self.model_dir)
                    # Save images
                    #backup_para = copy_G_params(self.netG)
                    #backup_para_E = copy_G_params(self.enc)
                    #load_params(self.netG, avg_param_G)
                    #load_params(self.enc, avg_param_E)
                    #
                    with torch.no_grad():
                        #self.text_embedding_fake = self.genIT(n2, self.img_embedding.detach())
                        self.text_embedding_fake = self.genIT(self.img_embedding.detach())
                    ###################generated text from Image##################################
                        length1 = [max_len]*N #taking maximum length
                        length1= torch.LongTensor(length1)
                        temb = self.text_embedding_fake,
                        _, indices_g = self.model.rnn(pass_type ='generate', hidden=temb, text_length=length1, batch_size=N)
                    ################################The below is just for verification of original text###########
                        _, indices_o = self.model.rnn(pass_type ='generate', hidden=self.text_embedding, text_length=length1, batch_size=N)
                    ###############We can use original image or output of image decoder#############
                    #fake_imgs_o, _, _ = self.dec(n1, self.img_embedding.detach())
                        texts_i = vocab.decode_positions(captions)
                        texts_g = vocab.decode_positions(indices_g)
                        texts_o = vocab.decode_positions(indices_o)
                    ############################################################
                        #fake_imgs_o, _, _ = self.dec(n1, self.img_embedding.detach())
                        fake_imgs_o = self.dec(n1, self.img_embedding.detach())
                        #self.img_embedding_fake = self.genTI(n3, self.text_embedding[0].detach())
                        self.img_embedding_fake = self.genTI(self.text_embedding[0].detach())
                        #fake_imgs_g, _, _ = self.dec(n1, self.img_embedding_fake.detach())
                        fake_imgs_g = self.dec(n1, self.img_embedding_fake.detach())
                    
                    
                    
                    
                        save_results(inputs[2], fake_imgs_o[2], fake_imgs_g[2], texts_i, texts_g, texts_o, count)
                        del fake_imgs_o, fake_imgs_g, texts_i, texts_g, texts_o
                        del self.img_embedding, self.img_embedding_fake, self.text_embedding, self.text_embedding_fake
                   

            end_t = time.time()
            #print('''[%d/%d][%d]
             #            Loss_D: %.2f Loss_G: %.2f Time: %.2fs
              #        '''  # D(real): %.4f D(wrong):%.4f  D(fake) %.4f
               #   % ((epoch +1), self.max_epoch, self.num_batches,
                #     errD_total.item(), errG_total.item(), end_t - start_t))
            print('''[%d/%d][%d]
                         Loss_DIT: %.2f Loss_GIT: %.2f Loss_DTI: %.2f Loss_GTI: %.2f Time: %.2fs
                      '''  # D(real): %.4f D(wrong):%.4f  D(fake) %.4f
                  % ((epoch +1), self.max_epoch, self.num_batches,
                     sum(errorDIT_list)/len(errorDIT_list), sum(errorGIT_list)/len(errorGIT_list),
                     sum(errorDTI_list)/len(errorDTI_list), sum(errorGTI_list)/len(errorGTI_list),
                     end_t - start_t))

        #save_model(self.enc, avg_param_E, self.netG, self.optimizerG, avg_param_G, self.netsD, self.optimizersD, count, self.model_dir)
        save_model(self.genIT, self.optimizerGIT, self.disIT, self.optimizerDIT, self.genTI, self.optimizerGTI, self.disTI, self.optimizerDTI, count, self.model_dir)
         
        
        
        print('####################training completed#########################')
        
    

    



    
    





##########################################################################
#########################################################################
#####################################################MAIN###############
########################################################################
# Initialize the model for this run() second one if want to load weights from previous check point
#model_ft, netG, netsD, num_Ds = initialize_model(encoder_name, feature_vector_dim, feature_extract, use_pretrained=False, vae=var_ae, use_finetuned= finetuned)
#model_ft, input_size = initialize_model(model_name, feature_vector_dim, feature_extract, use_pretrained=True, vae=var_ae, use_finetuned='checkpoint.pt')
#############################################################################
# Print the model we just instantiated
im_size = 256
#data_transforms2 = transforms.Compose([transforms.ToPILImage(), transforms.Pad(0), transforms.ToTensor(), transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
data_transforms = transforms.Compose([transforms.ToPILImage(), transforms.Pad(0), transforms.ToTensor(),transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])])
inv_normalize = transforms.Normalize(mean=[-mean_r/std_r, -mean_g/std_g, -mean_b/std_b],std=[1/std_r, 1/std_g, 1/std_b])

print("Initializing Datasets and Dataloaders...")
# Create training and validation data
if dset=='birds':
    text_datasets = {x: BirdsDataset1(os.path.join(data_dir), transform=data_transforms, split=x) for x in ['train', 'val']}
else:
    text_datasets = {x: FlowersDataset1(os.path.join(data_dir), transform=data_transforms, split=x) for x in ['train', 'val']}

ds = text_datasets['train']

vocab = ds.get_vocab_builder()
max_len = ds.max_sent_length
#############################################################
#####################################################################
print("Loading vocabulary, embedding matrix from trained text model.....")
file_ematrix = open(embedding_matrx_path, 'rb') 
file_vocab_i2t = open(vocab_i2t_path, 'rb')
file_vocab_t2i = open(vocab_t2i_path, 'rb')
embeddings_matrix = pickle.load(file_ematrix)
vocab.i2t= pickle.load(file_vocab_i2t)
vocab.t2i= pickle.load(file_vocab_t2i)
text_datasets['val'].vocab_builder.i2t =vocab.i2t
text_datasets['val'].vocab_builder.t2i = vocab.t2i
############################################################################

# Create training and validation dataloaders
dataloaders_dict = {x: DataLoader(text_datasets[x], batch_size=batch_size, shuffle=True, num_workers=0) for x in ['train', 'val']}

config = {  'emb_dim': embedding_dim,
                'hid_dim': hidden_dim//2, #birectional is used so hidden become double
                'n_layers': 1,
                'dropout': 0.0,
                'vocab_size': vocab.vocab_size(),
                'sos': vocab.sos_pos(),
                'eos': vocab.eos_pos(),
                'pad': vocab.pad_pos(),
             }

model_ft, enc, dec, genIT, disIT, genTI, disTI = initialize_model(model_name, config, embeddings_matrix)

#defining optimizers for Generator and discriminator





# Setup the loss fxn
#criterion = loss_function

IT_model = ImageTextTrainer(model_ft, enc, dec, genIT, disIT, genTI, disTI, dataloaders_dict, num_epochs, log_dir)

# Train and evaluate
IT_model.train()
IT_model.evaluate()
