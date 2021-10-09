#!/usr/bin/env python
# encoding: utf-8


import argparse
import random
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.utils.data
import torchvision.utils as vutils
from torch.autograd import Variable
from torchvision import transforms
import torchvision.models as torch_models
from torch.utils.data import DataLoader
import torch.nn.functional as F
import os
import timeit
import sys
import mmd.util as util
import dateutil.tz
import errno
import numpy as np
import datetime
from models.utils1 import Logger
from data.resultwriter import ResultWriter
import mmd.base_module as base_module
from mmd.mmd import mix_rbf_mmd2
from tensorboardX import SummaryWriter
from models.stack_gan2.model1 import encoder_resnet1, G_NET1, MAP_NET_IT2, MAP_NET_TI2
import models.text_auto_models1 as text_models
from config import cfg
from data.datasets1 import BirdsDataset1, FlowersDataset1
import pickle

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
            

# Get argument
parser = argparse.ArgumentParser()
parser = util.get_args(parser)
args = parser.parse_args()
print(args)



if torch.cuda.is_available():
    args.cuda = True
    torch.cuda.set_device(args.gpu_device)
    gpu_id = str(args.gpu_device)
    s_gpus = gpu_id.split(',')
    gpus = [int(ix) for ix in s_gpus]
    #device = torch.device("cuda")
    print("Using GPU device", torch.cuda.current_device())
else:
    device = torch.device("cpu")
    raise EnvironmentError("GPU device not available!")

args.manual_seed = 1126
np.random.seed(seed=args.manual_seed)
random.seed(args.manual_seed)
torch.manual_seed(args.manual_seed)
torch.cuda.manual_seed(args.manual_seed)
cudnn.benchmark = True

encoder_path='/home/das/dev/unsup/saved_models/flowers/encG_175800_128.pth' #image encoder
dec_path='/home/das/dev/unsup/saved_models/flowers/netG_175800_128.pth' #image decoder
text_autoencoder_path = '/home/das/dev/unsup/saved_models/flowers/AutoEncoderDglove100_flowerFalse203.pt' # text auto encoder
GEN_PATH = 'netG_77036.pth' # netG_100.pth #this is Image t TEXt embedding generator , if not restarted
#from previous fails should be empty
DIS_PATH= 'netD_77036.pth' 

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
##################################################################################
data_dir = args.dataset

#############################################################
now = datetime.datetime.now(dateutil.tz.tzlocal())
timestamp = now.strftime('%Y_%m_%d_%H_%M_%S')
#log_dir = 'output/%s_%s' %(cfg.DATASET_NAME, timestamp)
log_dir =  'output/flowers_2020_06_13_20_13_28'
mkdir_p(log_dir)
model_dir = os.path.join(log_dir, 'modeldir')
mkdir_p(model_dir)

sys.stdout = Logger('{}/run.log'.format(log_dir))
print("###############output folder###############################")
print(os.path.join(os.getcwd(),log_dir))
########################################################################
###########################################################################
log_dir1 = 'output/flowers_2020_06_27_12_15_37'#save the new result
mkdir_p(log_dir1)
txt_img_dir_val = os.path.join(log_dir1, 'txtimgdirval')
print("###############Results Printed###############################")
print(os.path.join(os.getcwd(),log_dir1))
#######################################################################
###############validation set dir#####################
#txt_img_dir_val = os.path.join(log_dir, 'txtimgdirval')
results_writer_txtimg_val = ResultWriter(txt_img_dir_val)
tensor_board = os.path.join(log_dir, 'tensorboard')
mkdir_p(tensor_board)
writer = SummaryWriter(tensor_board)
# NetG is a decoder
# input: batch_size * nz * 1 * 1
# output: batch_size * nc * image_size * image_size
class NetG(nn.Module):
    def __init__(self, decoder):
        super(NetG, self).__init__()
        self.decoder = decoder

    def forward(self, input1):
        output = self.decoder(input1)
        return output


# NetD is an encoder + decoder
# input: batch_size * nc * image_size * image_size
# f_enc_X: batch_size * k * 1 * 1
# f_dec_X: batch_size * nc * image_size * image_size
class NetD(nn.Module):
    def __init__(self, encoder, decoder):
        super(NetD, self).__init__()
        self.encoder = encoder
        self.decoder = decoder

    def forward(self, input1):
        f_enc_X = self.encoder(input1)
        f_dec_X = self.decoder(f_enc_X)

        #f_enc_X = f_enc_X.view(input.size(0), -1)
       # f_dec_X = f_dec_X.view(input.size(0), -1)
        return f_enc_X, f_dec_X


class ONE_SIDED(nn.Module):
    def __init__(self):
        super(ONE_SIDED, self).__init__()

        main = nn.ReLU()
        self.main = main

    def forward(self, input):
        output = self.main(-input)
        output = -output.mean()
        return output
    

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

def optimizerToDevice(optimizer):
    for state in optimizer.state.values():
            for k, v in state.items():
                if isinstance(v, torch.Tensor):
                    if torch.cuda.is_available():
                        state[k] = v.cuda()
                    else:
                        state[k] = v.to(device)
    return optimizer

def define_optimizers(netG, netD, path):
    optimizerD = torch.optim.Adam(netD.parameters(),
                         lr=2e-5,
                         betas=(0.5, 0.999))
      
    # G_opt_paras = []
    # for p in netG.parameters():
    #     if p.requires_grad:
    #         G_opt_paras.append(p)
    optimizerG = torch.optim.Adam(netG.parameters(),
                            lr=2e-5,
                            betas=(0.5, 0.999))
    
    #optimizerG = torch.optim.RMSprop(netG.parameters(), lr=args.lr)
    #optimizerD = torch.optim.RMSprop(netD.parameters(), lr=args.lr)
    
    count = 0
    if GEN_PATH != '':
        # example cfg.TRAIN.NET_G = 
        Gpath = os.path.join(path, GEN_PATH)
        print('loading optimizer from ', Gpath)
        checkpoint = torch.load(Gpath)
        optimizerG.load_state_dict(checkpoint['optimizer'])
        optimizerG = optimizerToDevice(optimizerG)
        istart = GEN_PATH.rfind('_') + 1
        iend = GEN_PATH.rfind('.')
        count = GEN_PATH[istart:iend]
        count = int(count) + 1
        
    if DIS_PATH != '':
            Dpath = os.path.join(path, DIS_PATH)
            checkpoint = torch.load(Dpath)
            print('loading optimizer from ', Dpath)
            optimizerD.load_state_dict(checkpoint['optimizer'])
            optimizerD = optimizerToDevice(optimizerD)
    return optimizerG, optimizerD, count

def load_network(path):
    ####################Image deoder################################
    dec = G_NET1()
    dec.apply(base_module.weights_init)
    dec = torch.nn.DataParallel(dec, device_ids=gpus)
    #################################################################
    # construct encoder/decoder modules
    #hidden_dim = args.nz
    G_decoder = MAP_NET_TI2() # This the Actual Generator 
    D_encoder = MAP_NET_IT2() #Discriminator should be an Auto encoder without noise
    D_decoder = MAP_NET_TI2()#

    netG = NetG(G_decoder)
    netD = NetD(D_encoder, D_decoder)
    one_sided = ONE_SIDED()
    print("netG:", netG)
    print("netD:", netD)
    print("oneSide:", one_sided)

    netG.apply(base_module.weights_init)
    netD.apply(base_module.weights_init)
    one_sided.apply(base_module.weights_init)
    
    
    
    if GEN_PATH != '':
        # example cfg.TRAIN.NET_G = 
        Gpath = os.path.join(path, GEN_PATH)
        checkpoint = torch.load(Gpath)
        netG.load_state_dict(checkpoint['state_dict'])
        #Epath = os.path.join(path, 'encG.pth' )
        #checkpoint = torch.load(Epath)
        #enc.load_state_dict(checkpoint['state_dict'])
        
        print('Load ', GEN_PATH)

        #istart = IT_GEN_PATH('_') + 1
        #iend = IT_GEN_PATH.rfind('.')
        #count = IT_GEN_PATH[istart:iend]
        #count = int(count) + 1

    if DIS_PATH != '':
            Dpath = os.path.join(path, DIS_PATH)
            print('Load ', DIS_PATH)
            checkpoint = torch.load(Dpath)
            netD.load_state_dict(checkpoint['state_dict'])
            
            
    if args.cuda:
        dec.cuda()
        netG.cuda()
        netD.cuda()
        one_sided.cuda()

    

    
        
    
    
    return dec, netG, netD, one_sided
         
def initialize_model(model_name, config, embeddings_matrix):
    
    model_ft= text_models.AutoEncoderD(config, embeddings_matrix)
    #model_ft = model_ft.to(device)
    model_ft.cuda()

    
    dec, gen, dis, one_sided= load_network(model_dir)
    #############################################################
    #enc = torch_models.resnet50(pretrained=True)
    #num_ftrs = enc.fc.in_features
    #enc.fc = nn.Linear(num_ftrs, 1024)
    #enc = enc.to(device)
    ################################################################
    enc = encoder_resnet1()
    enc.cuda()
    #enc = enc.to(device)
    
    print("=> loading Image encoder from '{}'".format(encoder_path))
    encoder = torch.load(encoder_path)
    enc.load_state_dict(encoder['state_dict'])
    
    
    print("=> loading Image decoder from '{}'".format(dec_path))
    decoder = torch.load(dec_path)
    dec.load_state_dict(decoder['state_dict'])
    
    
    print("=> loading text autoencoder from '{}'".format(text_autoencoder_path))
    text_autoencoder = torch.load(text_autoencoder_path)
    model_ft.load_state_dict(text_autoencoder['state_dict'])
    

    return model_ft, enc, dec, gen, dis, one_sided

def save_results(fake_imgs, text_input, count):
    if count != -1: #for validation will be saved in a single folder
        #img_dir = os.path.join(log_dir, 'imgdir%d'%count)
        #txt_img_dir = os.path.join(log_dir, 'txtimgdir%d'%count)
        txt_img_dir = os.path.join(log_dir1, 'txtimgdir%d'%count)
        #results_writer_img = ResultWriter(img_dir)
        results_writer_txtimg = ResultWriter(txt_img_dir)
        
    else:
        #img_dir = img_dir_val
        txt_img_dir = txt_img_dir_val
        #results_writer_img = results_writer_img_val
        results_writer_txtimg = results_writer_txtimg_val
    #mkdir_p(img_dir)
        
    #fg =open(os.path.join(img_dir, 'generated.txt'), 'w+')
    #fo =open(os.path.join(img_dir, 'output.txt'), 'w+')
    for ig, ti in zip(fake_imgs, text_input):
        #ii = norm_range(ii)#normalize to (0,1)
        ig = norm_range(ig)#normalize to (0,1)
        #ii = ii.cpu().numpy().transpose(1,2,0) #in order to use plt.imshow the channel should be the last dimention
        ig = ig.detach().cpu().numpy().transpose(1,2,0)
        #results_writer_img.write_images(io, ii)
        results_writer_txtimg.write_image_with_text(ig, ti)
        #print(ti,'\t',tg, file = fg)
        #print(ti,'\t',to, file = fo)
    #fg.close()
    #fo.close()
    
def save_model(netG, optimizerG, netD, optimizerD, epoch, model_dir):
    #load_params(netG, avg_param_G)
    #load_params(enc, avg_param_E)
    
    
    
    
    stateG = {'state_dict': netG.state_dict(),
             'optimizer': optimizerG.state_dict()}
    torch.save(
        stateG,
        '%s/netG_%d.pth' % (model_dir, epoch))
    stateD = {'state_dict':  netD.state_dict(),
             'optimizer': optimizerD.state_dict()}
    torch.save(
            stateD,
            '%s/netD_%d.pth' % (model_dir, epoch))
    print('Save G/Ds models...count:%d'%epoch)
        


######################################################################










# Get data
#trn_dataset = util.get_data(args, train_flag=True)

data_transforms = transforms.Compose([transforms.ToPILImage(), transforms.Pad(0), transforms.ToTensor(),transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])])


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
dataloaders_dict = {x: DataLoader(text_datasets[x], batch_size=args.batch_size, shuffle=True, num_workers=int(args.workers)) for x in ['train', 'val']}

config = {  'emb_dim': embedding_dim,
                'hid_dim': hidden_dim//2, #birectional is used so hidden become double
                'n_layers': 1,
                'dropout': 0.0,
                'vocab_size': vocab.vocab_size(),
                'sos': vocab.sos_pos(),
                'eos': vocab.eos_pos(),
                'pad': vocab.pad_pos(),
             }

model, enc, dec, netG, netD, one_sided = initialize_model(model_name, config, embeddings_matrix)




criterion = nn.MSELoss()
# sigma for MMD
base = 1.0
sigma_list = [1, 2, 4, 8, 16]
sigma_list = [sigma / base for sigma in sigma_list]

# put variable into cuda device
fixed_noise = torch.cuda.FloatTensor(args.batch_size, args.nz).normal_(0, 1)
noise = Variable(torch.FloatTensor(args.batch_size, args.nz))
one = torch.tensor(1, dtype=torch.float)


if args.cuda:
    dec.cuda()
    enc.cuda()
    model.cuda()
    netG.cuda()
    netD.cuda()
    one_sided.cuda()
    criterion.cuda()
    
    noise, fixed_noise, one = noise.cuda(), fixed_noise.cuda(), one.cuda()
    
mone = one * -1 



# setup optimizer
optimizerG, optimizerD, count = define_optimizers(netG, netD, model_dir)


lambda_MMD = 1.0
lambda_AE_X = 8.0
lambda_AE_Y = 8.0
lambda_rg = 16.0


enc.eval()
dec.eval()
model.eval()
netG.train()
netD.train()

time = timeit.default_timer()
gen_iterations = 0
#count = 0
start_count = count
start_epoch = start_count // len(dataloaders_dict['train'])
#for t in range(start_epoch, args.max_iter):
for t in range(start_epoch, args.max_iter):
    data_iter = iter(dataloaders_dict['train'])
    i = 0
    while (i < len(dataloaders_dict['train'])):
        # ---------------------------
        #        Optimize over NetD
        # ---------------------------
        for p in netD.parameters():
            p.requires_grad = True

        if gen_iterations < 25 or gen_iterations % 500 == 0:
            Diters = 100
            Giters = 1
        else:
            Diters = 5
            Giters = 1

        for j in range(Diters):
            if i == len(dataloaders_dict['train']):
                break

            # clamp parameters of NetD encoder to a cube
            # do not clamp paramters of NetD decoder!!!
            for p in netD.encoder.parameters():
                p.data.clamp_(-0.01, 0.01)

            data = data_iter.next()
            i += 1
            count += 1
            netD.zero_grad()

            uinputs, inputs,_, labels, captions, lengths = data
            
            inp0 = uinputs[0]
            #inp0= inp0.to(device)
            inp0= inp0.cuda()
            N = inp0.size(0)
            noise_ = noise[:N]
            n1 = fixed_noise[:N]
            captions, lengths= adjust_padding(captions, lengths)
            #captions = captions.to(device)
            #lengths = lengths.to(device)
            captions = captions.cuda()
            lengths = lengths.cuda()
            ##########################################################
                # (1) Image embedding from pretrained model
            ##########################################################
            
            #img_embedding = enc(inp0)
            img_embedding, _, _ = enc(inp0)
            x = img_embedding
            
            ##########################################################
                # (2) Text embedding from pretrained model
            ##########################################################
            text_embedding = model.rnn(pass_type='encode',batch_positions=captions, text_length=lengths) #text_embedding 
            #x= x[0]
            #print('x:', x.shape)

            #######################################################
            
            #######################################################
                # (1) Generate fake text_embedding
            ######################################################
            noise_.data.normal_(0, 1)
            y = netG(text_embedding[0].detach()) #img_embedding_fake
            #print('y:', x.shape)
            
            
            

            f_enc_X_D, f_dec_X_D = netD(x.detach())
            #print('f_enc_X_D:', f_enc_X_D.shape)
            #print('f_dec_X_D:', f_dec_X_D.shape)

            

            f_enc_Y_D, f_dec_Y_D = netD(y.detach())

            # compute biased MMD2 and use ReLU to prevent negative value
            mmd2_D = mix_rbf_mmd2(f_enc_X_D, f_enc_Y_D, sigma_list)
            mmd2_D = F.relu(mmd2_D)
            #print('mmd2_D:', mmd2_D)

            # compute rank hinge loss
            #print('f_enc_X_D:', f_enc_Y_D.size())
            #print('f_dec_Y_D:', f_dec_Y_D.size())
            one_side_errD = one_sided(f_enc_X_D.mean(0) - f_enc_Y_D.mean(0))

            # compute L2-loss of AE
            L2_AE_X_D = criterion(f_dec_X_D, x.detach())
            L2_AE_Y_D = criterion(f_dec_Y_D, y.detach())

            errD = torch.sqrt(mmd2_D) + lambda_rg * one_side_errD - lambda_AE_X * L2_AE_X_D - lambda_AE_Y * L2_AE_Y_D
            #print(netD.parameters())
            
            errD.backward(mone) # negative objective function
            torch.nn.utils.clip_grad_norm_(netD.parameters(), 5.00)
            #errD = -errD
            #errD.backward()
            optimizerD.step()

        # ---------------------------
        #        Optimize over NetG
        # ---------------------------
        for p in netD.parameters():
            p.requires_grad = False

        for j in range(Giters):
            if i == len(dataloaders_dict['train']):
                break

            data = data_iter.next()
            i += 1
            count+=1
            netG.zero_grad()
            uinputs, inputs,_, labels, captions, lengths = data

            inp0 = uinputs[0]
            #inp0= inp0.to(device)
            inp0= inp0.cuda()
            N = inp0.size(0)
            noise_ = noise[:N]
            n1 = fixed_noise[:N]
            captions, lengths= adjust_padding(captions, lengths)
            #captions = captions.to(device)
            #lengths = lengths.to(device)
            captions = captions.cuda()
            lengths = lengths.cuda()
            ##########################################################
                # (1) Image embedding from pretrained model
            ##########################################################
            
            #img_embedding = enc(inp0)
            img_embedding, _, _ = enc(inp0)
            x = img_embedding
            
            ##########################################################
                # (2) Text embedding from pretrained model
            ##########################################################
            text_embedding = model.rnn(pass_type='encode',batch_positions=captions, text_length=lengths) #text_embedding
            

            #######################################################
            
            #######################################################
                # (1) Generate fake text_embedding
            ######################################################
            noise_.data.normal_(0, 1)
            y = netG(text_embedding[0].detach()) #img_embedding_fake

            f_enc_X, f_dec_X = netD(x.detach())

           

            f_enc_Y, f_dec_Y = netD(y)

            # compute biased MMD2 and use ReLU to prevent negative value
            mmd2_G = mix_rbf_mmd2(f_enc_X, f_enc_Y, sigma_list)
            mmd2_G = F.relu(mmd2_G)

            # compute rank hinge loss
            one_side_errG = one_sided(f_enc_X.mean(0) - f_enc_Y.mean(0))

            errG = torch.sqrt(mmd2_G) + lambda_rg * one_side_errG
            errG.backward(one)#positive grad
            torch.nn.utils.clip_grad_norm_(netG.parameters(), 5.00)
            optimizerG.step()

            gen_iterations += 1

        run_time = (timeit.default_timer() - time) / 60.0
        print('[%3d/%3d][%3d/%3d] [%5d] (%.2f m) MMD2_D %.6f hinge %.6f L2_AE_X %.6f L2_AE_Y %.6f loss_D %.6f Loss_G %.6f f_X %.6f f_Y %.6f |gD| %.4f |gG| %.4f'
              % ((t+1), args.max_iter, i, len(dataloaders_dict['train']), gen_iterations, run_time,
                 mmd2_D.item(), one_side_errD.item(),
                 L2_AE_X_D.item(), L2_AE_Y_D.item(),
                 errD.item(), errG.item(),
                 f_enc_X_D.mean().item(), f_enc_Y_D.mean().item(),
                 base_module.grad_norm(netD), base_module.grad_norm(netG)))
        if gen_iterations % 25==0:
            writer.add_scalar('D_loss', errD.item(), count)
            writer.add_scalar('G_loss', errG.item(), count)
            writer.add_scalar('mmd2_D',  mmd2_D.item(), count)
            writer.add_scalar('L2_AE_X_D', L2_AE_X_D.item(), count)
            writer.add_scalar('L2_AE_Y_D', L2_AE_Y_D.item(), count)

        if gen_iterations % 500 == 0:
            with torch.no_grad():
                y_fixed = netG(text_embedding[0].detach()) #img embedd
                #fake_imgs_o, _, _ = dec(n1, y_fixed.detach())
                fake_imgs_o = dec(n1, y_fixed.detach())
                texts_i = vocab.decode_positions(captions)
                save_results(fake_imgs_o[2], texts_i, count)
                save_model(netG, optimizerG, netD, optimizerD, count, model_dir)
            
            
    if (t+1) % 50 == 0:
        save_model(netG, optimizerG, netD, optimizerD, count, model_dir)
    
######################Evaluate###################################
        
ecount=0
        ######setting in eval mode##########################
enc.eval()
dec.eval()
model.eval()
netG.eval()
        ############################################################

#"""

caption_list=['this flower has one big purple petal that goes around and a green pistil.',
              'this flower has one big red petal that goes around and a green pistil.',
              'this flower has one big yellow petal that goes around and a green pistil.',
              'this flower has one big white petal that goes around and a green pistil.',
              'this flower has petals of different shapes and patterns accompanied by a big pistill.',
              'this flower has petals of different shapes and patterns accompanied by a small pistill',
              'this flower has petals of different shapes and patterns accompanied by a large pistill',
              'this flower has petals that are pink with white stamen',
              'this flower has petals that are pink with blue stamen',
              'this flower has petals that are yellow with white stamen',
              'this flower has petals that are blue with white stamen',
              'this flower has long white petals and a large yellow stigma in the middle',
              'this flower has long pink petals and a large yellow stigma in the middle',
              'this flower has long red petals and a large yellow stigma in the middle',
              'this flower has long red petals and a large green stigma in the middle',
              'this flower has long white petals and a small yellow stigma in the middle',
              'the petals of this flower are yellow with long stigma',
              'the petals of this flower are green with long stigma',
              'the petals of this flower are purple with long stigma',
              'the petals of this flower are purple with short stigma']
       
c_list=[]
l_list=[]
for caption in caption_list:
    #N = inp0.size(0)
    #n1 = fixed_noise[:N]
    caption = caption.strip()
    caption_encoded, caption_length = ds.vocab_builder.encode_sentences([caption], ds.max_sent_length)
    caption_encoded = caption_encoded.squeeze()
    caption_length = caption_length.squeeze()
    c_list.append(caption_encoded)
    l_list.append(caption_length)

c_tensor=torch.stack(c_list)
l_tensor=torch.stack(l_list)
#print('caption_shape:',c_tensor.shape)
#print('caption length:',l_tensor)
print('###############################################################')

captions, lengths= adjust_padding(c_tensor, l_tensor)
captions = captions.cuda()
lengths = lengths.cuda()
with torch.no_grad():
    N = captions.size(0)
    n1 = fixed_noise[:N]
    #img_embedding = enc(inp0)
    #img_embedding, _, _ = enc(inp0)
    text_embedding = model.rnn(pass_type='encode',batch_positions=captions, text_length=lengths)
    img_embedding_fake = netG(text_embedding[0].detach())
    #fake_imgs_o, _, _ = dec(n1, img_embedding_fake.detach())
    fake_imgs_o = dec(n1, img_embedding_fake.detach())
    texts_i = vocab.decode_positions(captions)
    if ecount % 1 == 0:
        save_results(fake_imgs_o[2], texts_i, -1)
    ecount = ecount +1
 