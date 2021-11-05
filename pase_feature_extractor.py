# insert pase model as feature extraction and followed by NN. 
import os
import sys
import torch
import numpy as np
import pdb
# import torchaudio
import librosa
from pase.models.frontend import wf_builder
from progressbar import ProgressBar

import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

kernel_sizes = [4,3,3]
strides = [2,2,1]
paddings=[0,0,1]

latent_dim = 300

class Discriminator(nn.Module):
    def __init__(
            self, num_gpu
            ):

        super(Discriminator, self).__init__()
        self.num_gpu = num_gpu

        self.main = nn.Sequential(
                    nn.Conv2d(1, 32, 4, 2, 1, bias=False),
                    nn.LeakyReLU(0.2, inplace=True),
                    nn.Conv2d(32, 64, 4, 2, 1, bias=False),
                    nn.BatchNorm2d(64),
                    nn.LeakyReLU(0.2, inplace=True),

                    nn.Conv2d(64, 64 * 2, 4, 2, 1, bias=False),
                    nn.BatchNorm2d(64 * 2),
                    nn.LeakyReLU(0.2, inplace=True),

                    nn.Conv2d(64 * 2, 64 * 4, 4, 2, 1, bias=False),
                    nn.BatchNorm2d(64 * 4),
                    nn.LeakyReLU(0.2, inplace=True),

                    nn.Conv2d(64 * 4, 64 * 8, 4, 2, 1, bias=False),
                    nn.BatchNorm2d(64 * 8),
                    nn.LeakyReLU(0.2, inplace=True),

                    nn.AdaptiveMaxPool2d(1),
                    nn.Conv2d(512,64,kernel_size=(1,1)),
                    nn.Conv2d(64,2,kernel_size=(1,1))

                    # nn.Linear(1,64),
                    # nn.LeakyReLU(0.2, inplace=True),
                    # nn.Linear(64,2),
                    # nn.LeakyReLU(0.2, inplace=True),
                    # nn.Linear(32,8),
                    # nn.LeakyReLU(0.2, inplace=True),
                    # nn.Linear(8,2)
                    # nn.softmax()
                    )

    def forward(self, input):
        return self.main( input )


def test():
    pase = wf_builder('cfg/frontend/PASE+.cfg').eval()
    pase.load_pretrained('FE_e199.ckpt', load_last=True, verbose=True)
    
    pase.cuda()
    
    # Now we can forward waveforms as Torch tensors
    import torch
    x = torch.randn(1, 1, 100000) # example with random noise to check shape
    # y size will be (1, 256, 625), which are 625 frames of 256 dims each
    y = pase(x.cuda())

def load_model(path='FE_e199.ckpt'):
    pase = wf_builder('cfg/frontend/PASE+.cfg').eval()
    pase.load_pretrained(path, load_last=True, verbose=True)
    pase.cuda()
    return pase


def generate_npy(positive_file_path,negative_file_path,output_feature_path):
    # positive
    #output_feature_path = './all_audio/positive/'
    #txtname = 'positive.txt'
    if not os.path.exists(output_feature_path):
        os.makedirs(output_feature_path)
    
    #f = open(output_feature_path + txtname)
    f = open(positive_file_path)
    a = f.read()
    fl = a.split('\n')[:-1]

    positive_folder=os.path.dirname(positive_file_path)

    pase = load_model()
    pbar = ProgressBar()
    for i in pbar(fl):
        x , sr = librosa.load(os.path.join(positive_folder,i), sr=None)
        if  len(x)==0: 
            print(i)
            continue
        x = torch.Tensor(x)
        x = torch.unsqueeze(x, 0)
        x = torch.unsqueeze(x, 0)
        y = pase(x.cuda())
        #y = pase.apply(x.cuda())
        print(y.shape)

        np.save(output_feature_path + i[:-4] +'.npy', y.detach().cpu().numpy())
    
    pdb.set_trace()
    #output_feature_path = './all_audio/negative/'
    #txtname = 'negative.txt'

    #f = open(output_feature_path + txtname)
    f = open(negative_file_path)
    a = f.read()
    fl = a.split('\n')[:-1]

    negative_folder=os.path.dirname(negative_file_path)

    pase = load_model()
    pbar = ProgressBar()
    for i in pbar(fl):
        x , sr = librosa.load(os.path.join(negative_folder,i), sr=None)
        if len(x)==0:
            print(i)
            continue
        x = torch.Tensor(x)
        x = torch.unsqueeze(x, 0)
        x = torch.unsqueeze(x, 0)
        y = pase(x.cuda())

        np.save(output_feature_path + i[:-4] +'.npy', y.detach().cpu().numpy()) 


if __name__ == '__main__':
    #generate_npy(positive_file_path,negative_file_path,output_feature_path)
    generate_npy(sys.argv[1],sys.argv[2],sys.argv[3])
