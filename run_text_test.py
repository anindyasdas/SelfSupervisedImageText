from __future__ import print_function
from __future__ import division
import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np
import pickle
import torchvision
import time
import sys
from data.datasets1 import ShapesDataset, BirdstextDataset, BillionDataset, FlowerstextDataset
from torchvision import transforms
import models.text_auto_models1 as text_models
from data.resultwriter import ResultWriter
from models.utils1 import EarlyStoppingWithOpt, Logger
import datetime
from pathlib import Path
from tempfile import mkdtemp
from tensorboardX import SummaryWriter

print("PyTorch Version: ",torch.__version__)
print("Torchvision Version: ",torchvision.__version__)



###########################################################################
###########################################################################
####################### variables#######################################
##########################################################################
########################################################################
# Top level data directory
data_dir = sys.argv[2]
# Flower dataset:1 Birds dataset 2
data_set_type = sys.argv[1]
# output folder name where results will be saved

# Models to choose from [resnet, alexnet, vgg, squeezenet, densenet, inception]
stored_model_dir='stored_model_dir/'
model_name = "AutoEncoderD"
additional ='glove100_flower'
ver='False150'#'True88' #initial version of the model to be loaded
glove_folder ='glove.6B_flowers'
gname = glove_folder + '/glove.6B.100d.txt' # for glove embedding
#gname = None

######################################################################
################step 1. Run with gname location, benchmark= True#for pretraining the model with glove embedding
################step 2. Run with gname location, benchmark=False# for finetuning the pretrained modelwith glove embedding
################ if gname location is not given it will be trained from scratch with embedding also trained
######################################################################
benchmark = False #when True run for benchmark dataset , when false Run for Actual dataset

#gname = None
if gname is None: # when gname is not provided no benchmark training is done
    benchmark= False #because we take first 50k words of glove during benchmark training
    
# Number of classes in the dataset
embedding_dim = 100
hidden_dim = 100


# Batch size for training (change depending on how much memory you have)
batch_size = 128

# Number of epochs to train for
num_epochs = 500
restart_epoch = 152 #1 #49#restartting from failed step

# Flag for feature extracting. When False, we finetune the whole model, else we only extract features
feature_extract = False
# Variational Autoencoder turn on
var_ae=False
#checkpoint_path = "checkpoints"
# mean and std calculated fort the dataset
mean_r= 0
mean_g= 0
mean_b= 0
std_r= 1
std_g= 1
std_b= 1
lr1 = 1e-3

##################################################
chkpt= stored_model_dir + model_name + additional + ver +'.pt'
early_stopping = EarlyStoppingWithOpt(patience=20, verbose=True, checkpoint= chkpt)
#########################Seting GPU###########################


print("Using GPU device", torch.cuda.current_device())
########################################################################
######################################################################
#########################################################################

runId = datetime.datetime.now().isoformat()
experiment_dir = Path('experiments/mylogg')
experiment_dir.mkdir(parents=True, exist_ok=True)
runPath = mkdtemp(prefix=runId, dir=str(experiment_dir))
sys.stdout = Logger('{}/run.log'.format(runPath))
print('Expt:', runPath)
print('RunID:', runId)
# output folder name where results will be saved
print('current directory:', os.getcwd())
results_writer_val = os.path.join(runPath,'val')
if not os.path.exists(results_writer_val):
    os.mkdir(results_writer_val)





#############################################################
##############store loss values#################################################
loss_dict ={'train_losses': [],'val_losses': []}
loss_dict_name= model_name + additional +'_loss_dict.p'
writer = SummaryWriter('data1/tensorboard/newmodel_' + model_name+ additional)
#######################################################################
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
    

def load_checkpoint(model, optimizer, filename):
    # Note: Input model & optimizer should be pre-defined.  This routine only updates their states.
    print("=> loading checkpoint '{}'".format(filename))
    checkpoint = torch.load(filename)
    model.load_state_dict(checkpoint['state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer'])
    print('model load successful')
    return model, optimizer


def initialize_model(model_name, config, embeddings_matrix):
    # Initialize these variables which will be set in this if statement. Each of these
    #   variables is model specific.
    
    autoencoder_model= text_models.AutoEncoderD(config, embeddings_matrix)
    #decoder = G_NET()


    return autoencoder_model

def train_model(model, dataloaders, criterion, optimizer, num_epochs=25, restart_epoch=1):
    since = time.time()

    val_loss_history = []

    for epoch in range(restart_epoch-1, num_epochs):
        start_t = time.time()
        print('Epoch {}/{}'.format(epoch + 1, num_epochs))
        print('-' * 10)

        # Each epoch has a training and validation phase
        if benchmark:
            phases = ['train']
            save_model_phase = 'train'
            print('##############training for benchmark dataset#################')
        else:
            phases = ['train', 'val']
            save_model_phase = 'val'
            print('##############training for birds dataset#################')
            
            
        for phase in phases:
            #if phase == 'train':
             #   model.train()  # Set model to training mode
            #else:
             #   model.eval()   # Set model to evaluate mode

            running_loss = 0.0
            running_perplexity = 0.0
            
                

            # Iterate over data.
            for _, _, captions, lengths in dataloaders[phase]:
                #print('cap:',captions)
                #print('len:',lengths)
                captions, lengths= adjust_padding(captions, lengths)
                #print('new cap:',captions)
                #print('new_len:', lengths)
                #print('length of dataset',len(dataloaders[phase].dataset))
                
                captions = captions.cuda()
                lengths = lengths.cuda()
                # zero the parameter gradients
                optimizer.zero_grad()

                # forward
                # track history if only in train
                with torch.set_grad_enabled(phase == 'train'):
                    # Get model outputs and calculate loss
                    out, index = model(captions, lengths)
                    #print(out.shape)
                    #print(captions.shape)
                    #print(lengths)
                    # Since we train an autoencoder we compare the output to the original input
                    loss = criterion(out[:, 1:, :].contiguous().view(-1, out.shape[2]), captions[:, 1:].flatten())
                    # backward + optimize only if in training phase
                    perplexity  = torch.exp(loss)
                    if phase == 'train':
                        loss.backward()
                        torch.nn.utils.clip_grad_norm_(model.parameters(), 5.00)
                        optimizer.step()

                # statistics
                
                running_loss += loss.item() * captions.size(0)
                running_perplexity += perplexity.item()

            epoch_loss = running_loss / len(dataloaders[phase].dataset)
            epoch_perplexity = running_perplexity / len(dataloaders[phase])
            end_t = time.time()
            # calculating perplexity

            print('{} Loss: {:.4f} Perplexity: {:.4f}'.format(phase, epoch_loss, epoch_perplexity))
            print('time taken:', end_t - start_t)

            # deep copy the model
            
            if phase == save_model_phase:
                ################checking intermediate results################
                f =open(os.path.join(results_writer_val, 'result_epoch_'+str(epoch)+'.txt'), 'w')
                texts_i = vocab.decode_positions(captions)
                texts_o = vocab.decode_positions(index)
                for l, o in zip(texts_i, texts_o):
                    print(l,'\t',o, file = f)
                f.close()
                ##########################################################
                ver = str(benchmark) + str(epoch)
                chkpt= stored_model_dir + model_name + additional + ver +'.pt'
                early_stopping(epoch_loss, model, optimizer, chkpt)
                val_loss_history.append(epoch_loss)
                
        if early_stopping.early_stop:
            print("Early stopping")
            break

        print()

    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))
    print('Best val loss: {:4f}'.format(min(val_loss_history)))
    # load best model weights
    ver = str(benchmark) + str(epoch- early_stopping.counter)
    chkpt= stored_model_dir + model_name + additional + ver +'.pt'
    model, optimizer = load_checkpoint(model, optimizer, chkpt)
    #model.load_state_dict(torch.load('checkpoint.pt'))
    return model, val_loss_history
    



##########################################################################
#########################################################################
#####################################################MAIN###############
########################################################################



data_transforms = transforms.Compose([transforms.ToPILImage(), transforms.Pad(0), transforms.ToTensor(), transforms.Normalize([mean_r, mean_g, mean_b], [std_r, std_g, std_b])])
inv_normalize = transforms.Normalize(mean=[-mean_r/std_r, -mean_g/std_g, -mean_b/std_b],std=[1/std_r, 1/std_g, 1/std_b])
# Data augmentation and normalization for training
#pretrain_dir = '1-billion-word-language-modeling-benchmark-r13output/training-monolingual.tokenized.shuffled/*'
print("Initializing Datasets and Dataloaders...")
pretrain_dir = '1-billion-word-language-modeling-benchmark-r13output/training-monolingual.tokenized.shuffled/*'
benchmark_datasets = {x: BillionDataset(pretrain_dir, split=x) for x in ['train']}



# Create training and validation data
if data_set_type == '1':
    text_datasets = {x: FlowerstextDataset(os.path.join(data_dir), transform=data_transforms, split=x) for x in ['train', 'val']}
elif data_set_type == '2':
    text_datasets = {x: BirdstextDataset(os.path.join(data_dir), transform=data_transforms, split=x) for x in ['train', 'val']}
# Create training and validation dataloaders

ds = text_datasets['train']

vocab = ds.get_vocab_builder()
#print('the max length is', ds.max_sent_length)
#print('the vocab size of birds dataset', vocab.vocab_size())


    

#########################################################################
#print(os.getcwd())
if gname is not None:
    top50k =[]
    t2i= {}
    ###########################Loading################################
    file_ematrix = open(os.path.join(glove_folder,'emtrix.obj'), 'rb') 
    embeddings_matrix = pickle.load(file_ematrix)
    
    file_vocab_i2t = open(os.path.join(glove_folder,'vocab_i2t.obj'), 'rb')
    vocab.i2t = pickle.load(file_vocab_i2t)
    
    file_vocab_t2i = open(os.path.join(glove_folder,'vocab_t2i.obj'), 'rb')
    vocab.t2i = pickle.load(file_vocab_t2i)
    
    print('embedding matrix, vocab.i2t, vocab.t2i are saved at ', file_ematrix.name, file_vocab_i2t.name, file_vocab_t2i.name)
    text_datasets['val'].vocab_builder.i2t =vocab.i2t
    #f1 =open('i2t', 'w+')
    #print('benchmark dataset previous vocabsize', benchmark_datasets['train'].vocab_builder.i2t, file=f1)
    #print('length of emtarix', len(embeddings_index))
    #for i, word in enumerate(benchmark_datasets['train'].vocab_builder.i2t):
     #   embedding_vector = embeddings_index.get(word)
      #  if embedding_vector is None:
       #     print(word)
        
        
    benchmark_datasets['train'].vocab_builder.i2t = vocab.i2t
    #print('benchmark dataset new vocabsize', len(benchmark_datasets['train'].vocab_builder.i2t))
    text_datasets['val'].vocab_builder.t2i = vocab.t2i
    benchmark_datasets['train'].vocab_builder.t2i = vocab.t2i
    #print('length of t2i', len(benchmark_datasets['train'].vocab_builder.t2i), len(t2i))
    
    vocab = ds.get_vocab_builder()
    #print('compare vocubbuilder birds', ds.vocab_builder.vocab_size(), text_datasets['train'].vocab_builder.vocab_size())

else:
    embeddings_matrix = None
###################################################################################
bench_dataloader_dict = {x: DataLoader(benchmark_datasets[x], batch_size=batch_size, shuffle=True, num_workers=0) for x in ['train']}  
dataloaders_dict = {x: DataLoader(text_datasets[x], batch_size=batch_size, shuffle=True, num_workers=0) for x in ['train', 'val']}

#print('the new vocab size of birds dataset', vocab.vocab_size())
    
#print(type(embeddings_matrix))
#print(vocab.i2t)
    
config = {  'emb_dim': embedding_dim,
                'hid_dim': hidden_dim//2, #birectional is used so hidden become double
                'n_layers': 1,
                'dropout': 0.0,
                'vocab_size': vocab.vocab_size(),
                'sos': vocab.sos_pos(),
                'eos': vocab.eos_pos(),
                'pad': vocab.pad_pos(),
             }
#print('pad position', vocab.pad_pos())
# Initialize the model for this run() second one if want to load weights from previous check point
model_ft = initialize_model(model_name, config, embeddings_matrix)
#model_ft, input_size = initialize_model(model_name, feature_vector_dim, feature_extract, use_pretrained=True, vae=var_ae, use_finetuned='checkpoint.pt')
#############################################################################
# Print the model we just instantiated
print(model_ft)
#model_ft = torch.nn.DataParallel(model_ft)
# Send the model to GPU
model_ft.cuda()

# Gather the parameters to be optimized/updated in this run. If we are
#  finetuning we will be updating all parameters. However, if we are
#  doing feature extract method, we will only update the parameters
#  that we have just initialized, i.e. the parameters with requires_grad
#  is True.
params_to_update = model_ft.parameters()
print("Params to learn:")
for name,param in model_ft.named_parameters():
        if param.requires_grad == True:
            print("\t",name)

# Observe that all parameters are being optimized
optimizer_ft = optim.Adam(params_to_update)

if (restart_epoch > 1) or ((not benchmark) and gname is not None):
        print('loading model from checkpoint............')
        model_ft, optimizer = load_checkpoint(model_ft, optimizer_ft, chkpt)
        params_to_update = model_ft.parameters()
        optimizer_ft = optim.Adam(params_to_update)
        if (restart_epoch > 1): # when restarting both optimizer and model is loaded
            print('loading optimizer from checkpoint.................')
            optimizer_ft = optimizer
if not benchmark:
    dataloaders_dict = dataloaders_dict
    output_loader = dataloaders_dict['val']
else:
    dataloaders_dict = bench_dataloader_dict
    output_loader = dataloaders_dict['train']
    

# Setup the loss fxn
#criterion = loss_function
criterion = torch.nn.CrossEntropyLoss(ignore_index=vocab.pad_pos())
if torch.cuda.is_available():
    criterion.cuda()
    
# Train and evaluate
model_ft, hist = train_model(model_ft, dataloaders_dict, criterion, optimizer_ft, num_epochs=num_epochs, restart_epoch=restart_epoch)
#torch.save(model_ft.state_dict(), os.path.join(checkpoint_path, "pytorch_model.bin"))

# show reconstruction for first batch
model_ft.eval() #set eval mode for testing
#print('#################Eval mode##################')
f =open(os.path.join(results_writer_val, sys.argv[3]), 'w')
print('actual','\t','generated', file = f)
for _, _, cap, len1 in output_loader:
    cap, len1= adjust_padding(cap, len1)
    cap = cap.cuda()
    len1 = len1.cuda()
    _, ind = model_ft(cap, len1)
    texts_i = vocab.decode_positions(cap)
    texts_o = vocab.decode_positions(ind)
    for l, o in zip(texts_i, texts_o):
        print(l,'\t',o, file = f)
f.close()
