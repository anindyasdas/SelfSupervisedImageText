import torch.nn.functional as f
import torch
from torch import nn
import torch.nn.functional as F
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
import numpy as np
import sys

def fix(params):
    if not isinstance(params, list):
        params = [params]
    for param in params:
        # for param in model.parameters():
        param.requires_grad = False


def tune(params):
    if not isinstance(params, list):
        params = [params]
    for param in params:
        # for param in model.parameters():
        param.requires_grad = True


def set_grad_zero(models):
    for model in models:
        for params in model.parameters():
            if params.grad is not None:
                params.grad.zero_()


def get_activation_function(activation_name):
    if activation_name == 'LeakyRelu':
        return nn.LeakyReLU(0.2)
    elif activation_name == 'Swish':
        return Swish()


class Swish(nn.Module):
    def __init__(self):
        super(Swish, self).__init__()

    def forward(self, x):
        return x * torch.sigmoid(x)


class DPP(nn.Module):

    def __init__(self):
        super(DPP, self).__init__()

    def forward(self, real, fake):
        S_b = torch.mm(real.transpose(0, 1), real)
        D_b = torch.mm(fake.transpose(0, 1), fake)

        lambda_real, v_real = torch.symeig(S_b, eigenvectors=True)
        lambda_fake, v_fake = torch.symeig(D_b, eigenvectors=True)

        # throw away the imaginary values RIP
        # lambda_real = lambda_real[:, 0]
        # lambda_fake = lambda_fake[:, 0]

        lambda_real_norm = lambda_real / lambda_real.sum(0).expand_as(lambda_real)

        L_m = torch.norm((lambda_real - lambda_fake), 2).sum()

        L_s = - (lambda_real_norm * F.cosine_similarity(v_real, v_fake)).sum()

        return L_m + L_s



# import tensorflow as tf

# Implementing the GDPP loss in both Tensorflow and Pytorch
# np.random.seed(1)

def kld_loss(mu, logvar):

    # KLD is Kullback–Leibler divergence -- how much does one learned
    # distribution deviate from another, in this specific case the
    # learned distribution from the unit Gaussian

    # see Appendix B from VAE paper:
    # Kingma and Welling. Auto-Encoding Variational Bayes. ICLR, 2014
    # https://arxiv.org/abs/1312.6114
    # - D_{KL} = 0.5 * sum(1 + log(sigma^2) - mu^2 - sigma^2)
    # note the negative D_{KL} in appendix B of the paper
    KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    # Normalise by same number of elements as in reconstruction
    KLD /= mu.size()[0] * mu.size()[1]

    # BCE tries to make our reconstruction as accurate as possible
    # KLD tries to push the distributions as close as possible to unit Gaussian
    return KLD

class Logger(object):
    def __init__(self, filename, mode="a"):
        self.terminal = sys.stdout
        self.log = open(filename, mode)

    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)

    def flush(self):
        # this flush method is needed for python 3 compatibility.
        # this handles the flush command by doing nothing.
        # you might want to specify some extra behavior here.
        pass
    
def sample_z(mu, log_var, device):
    # Using reparameterization trick to sample from a gaussian
    eps = torch.autograd.Variable(torch.randn(mu.size()[0], mu.size()[1])).to(device)
    #return mu + torch.exp(log_var / 2) * eps
    return mu + log_var* eps


def compute_gdpp(phi_real, phi_fake):
    def compute_diversity(phi):
        phi = f.normalize(phi, p=2, dim=1)
        S_B = torch.mm(phi, phi.t())
        eig_vals, eig_vecs = torch.symeig(S_B, eigenvectors=True)
        # eig_vals, eig_vecs = torch.eig(S_B, eigenvectors=True)
        return eig_vals, eig_vecs
        # return eig_vals[:, 0], eig_vecs

    def normalize_min_max(eig_vals):
        min_v, max_v = torch.min(eig_vals), torch.max(eig_vals)
        return (eig_vals - min_v) / (max_v - min_v)

    fake_eig_vals, fake_eig_vecs = compute_diversity(phi_fake)
    real_eig_vals, real_eig_vecs = compute_diversity(phi_real)
    # Scaling factor to make the two losses operating in comparable ranges.
    magnitude_loss = 0.0001 * f.mse_loss(target=real_eig_vals, input=fake_eig_vals)
    structure_loss = -torch.sum(torch.mul(fake_eig_vecs, real_eig_vecs), 0)
    normalized_real_eig_vals = normalize_min_max(real_eig_vals)
    weighted_structure_loss = torch.sum(torch.mul(normalized_real_eig_vals, structure_loss))
    return magnitude_loss + weighted_structure_loss


def GDPPLoss(phiFake, phiReal, backward=True):
    r"""
    Implementation of the GDPP loss. Can be used with any kind of GAN
    architecture.
    Args:
        phiFake (tensor) : last feature layer of the discriminator on real data
        phiReal (tensor) : last feature layer of the discriminator on fake data
        backward (bool)  : should we perform the backward operation ?
    Returns:
        Loss's value. The backward operation in performed within this operator
    """
    def compute_diversity(phi):
        phi = F.normalize(phi, p=2, dim=1)
        SB = torch.mm(phi, phi.t())
        eigVals, eigVecs = torch.symeig(SB, eigenvectors=True)
        return eigVals, eigVecs

    def normalize_min_max(eigVals):
        minV, maxV = torch.min(eigVals), torch.max(eigVals)
        return (eigVals - minV) / (maxV - minV)

    fakeEigVals, fakeEigVecs = compute_diversity(phiFake)
    realEigVals, realEigVecs = compute_diversity(phiReal)

    # Scaling factor to make the two losses operating in comparable ranges.
    magnitudeLoss = 0.0001 * F.mse_loss(target=realEigVals, input=fakeEigVals)
    structureLoss = -torch.sum(torch.mul(fakeEigVecs, realEigVecs), 0)
    normalizedRealEigVals = normalize_min_max(realEigVals)
    weightedStructureLoss = torch.sum(
        torch.mul(normalizedRealEigVals, structureLoss))
    gdppLoss = magnitudeLoss + weightedStructureLoss

    if backward:
        gdppLoss.backward(retain_graph=True)

    return gdppLoss.item()


def compute_bleu(candidate, reference, ngramm_weights=(0.25, 0.25, 0.25, 0.25),
                 smoothing=SmoothingFunction().method1):
    reference = [reference.split()]
    candidate = candidate.split()
    score = sentence_bleu(reference, candidate, smoothing_function=smoothing)
    return score


        
class EarlyStopping:
    """
    Early stops the training if validation loss doesn't improve after a given patience.
    From https://github.com/Bjarten/early-stopping-pytorch/blob/master/pytorchtools.py
    """
    def __init__(self, patience=7, verbose=False, delta=0, checkpoint='checkpoint.pt' ):
        """
        Args:
            patience (int): How long to wait after last time validation loss improved.
                            Default: 7
            verbose (bool): If True, prints a message for each validation loss improvement.
                            Default: False
            delta (float): Minimum change in the monitored quantity to qualify as an improvement.
                            Default: 0
        """
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.Inf
        self.delta = delta
        self.checkpoint = checkpoint

    def __call__(self, val_loss, model):

        score = -val_loss

        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
        elif score < self.best_score - self.delta:
            self.counter += 1
            print(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
            self.counter = 0

    def save_checkpoint(self, val_loss, model):
        '''Saves model when validation loss decrease.'''
        if self.verbose:
            print(f'Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}).  Saving model ...')
        torch.save(model.state_dict(), self.checkpoint)
        self.val_loss_min = val_loss


class EarlyStoppingWithOpt:
    """
    Early stops the training if validation loss doesn't improve after a given patience.
    From https://github.com/Bjarten/early-stopping-pytorch/blob/master/pytorchtools.py
    """
    def __init__(self, patience=7, verbose=False, delta=0, checkpoint='checkpoint.pt'):
        """
        Args:
            patience (int): How long to wait after last time validation loss improved.
                            Default: 7
            verbose (bool): If True, prints a message for each validation loss improvement.
                            Default: False
            delta (float): Minimum change in the monitored quantity to qualify as an improvement.
                            Default: 0
        """
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.Inf
        self.delta = delta
        self.checkpoint = checkpoint

    def __call__(self, val_loss, model, optimizer, ckpt=None):

        score = -val_loss

        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_loss, model, optimizer, ckpt)
        elif score < self.best_score - self.delta:
            self.counter += 1
            print(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(val_loss, model, optimizer, ckpt)
            self.counter = 0

    def save_checkpoint(self, val_loss, model, optimizer, ckpt):
        if ckpt != None:
            self.checkpoint = ckpt 
        '''Saves model when validation loss decrease.'''
        if self.verbose:
            print(f'Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}).  Saving model ...',self.checkpoint)
            state = {'state_dict': model.state_dict(),
             'optimizer': optimizer.state_dict()}
        torch.save(state, self.checkpoint)
        self.val_loss_min = val_loss


class EarlyStoppingWithOpt1:
    """
    Early stops the training if validation loss doesn't improve after a given patience.
    From https://github.com/Bjarten/early-stopping-pytorch/blob/master/pytorchtools.py
    """
    def __init__(self, patience=7, verbose=False, delta=0, checkpoint='checkpoint.pt'):
        """
        Args:
            patience (int): How long to wait after last time validation loss improved.
                            Default: 7
            verbose (bool): If True, prints a message for each validation loss improvement.
                            Default: False
            delta (float): Minimum change in the monitored quantity to qualify as an improvement.
                            Default: 0
        """
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.Inf
        self.delta = delta
        self.checkpoint = checkpoint

    def __call__(self, val_loss, enc, model, optimizer):

        score = -val_loss

        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_loss, enc, model, optimizer)
        elif score < self.best_score - self.delta:
            self.counter += 1
            print(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(val_loss, enc, model, optimizer)
            self.counter = 0

    def save_checkpoint(self, val_loss, enc, model, optimizer):
        '''Saves model when validation loss decrease.'''
        if self.verbose:
            print(f'Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}).  Saving model ...')
            state = {'state_dict': model.state_dict(),
                     'enc_dict': enc.state_dict(),
             'optimizer': optimizer.state_dict()}
        torch.save(state, self.checkpoint)
        self.val_loss_min = val_loss

if __name__ == '__main__':
    batch_size = 5
    dimension = 4

    a = torch.rand((batch_size, dimension))
    b = torch.rand((batch_size, dimension))

    dpp = DPP()

    dpp(a, b)

    reference = "A large green triangle at the bottom right"
    candidate = "A large green triangle at the bottom right"

    score = compute_bleu(candidate, reference)
    print(score)

    candidate = "A large red triangle at the bottom right"

    score = compute_bleu(candidate, reference)
    print(score)

    candidate = "A large blue triangle at the bottom left"
    score = compute_bleu(candidate, reference)
    print(score)
