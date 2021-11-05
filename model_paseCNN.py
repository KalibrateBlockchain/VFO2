import torch
from torch.autograd import Variable
from torch.utils.data import Dataset
import torch.utils.data.dataloader as dataloader
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from numpy.matlib import repmat
import math
import pdb

AUDIO_TYPE_ID = {'vowel-u': 0,'vowel-i': 1,'vowel-a': 2,'alphabet-a-z': 3, 'cough': 4, 'count-1-20': 5,}

def uttToSpkChile(fullname):
    f = fullname.split('/')[-1]#[:-1]
    spk_id = f.split('_')[1]
    return spk_id

class Dataset(Dataset):
    'Characterizes a dataset for PyTorch'
    def __init__(self, X, Y, all_files, UTT_TO_SPK):
        self.X = X
        self.Y = Y
        self.UTT_TO_SPK = UTT_TO_SPK
        
    def __len__(self):
        'Denotes the total number of samples'
        return len(self.X)
    
    
    def __getitem__(self, index):
        x = self.X[index] # filename 
        y = self.Y[index] # int (0,1)

        pase_feat = np.load(x)

        audio_type = x.split('/')[-1].split('_')[0]
        spk = self.UTT_TO_SPK[uttToSpkChile(x)]

        return pase_feat, int(y), AUDIO_TYPE_ID[audio_type], spk

    
class paseCNN_model(nn.Module):
    def __init__(self):

        super(paseCNN_model, self).__init__()

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
                    nn.Conv2d(64,2,kernel_size=(1,1)),

                    )

    def forward(self, input):
        # return self.main( input.unsqueeze(0) )
        return self.main( input)


class spectrogramCNN_model(nn.Module):
    def __init__(self):

        super(spectrogramCNN_model, self).__init__()

        self.main = nn.Sequential(
                    nn.Conv2d(1, 32, 4, 2, 1, bias=False),
                    nn.LeakyReLU(0.2, inplace=True),
                    nn.Conv2d(32, 64, 4, 2, 1, bias=False),
                    nn.BatchNorm2d(64),
                    nn.LeakyReLU(0.2, inplace=True),

                    nn.Conv2d(64, 64 * 2, 4, 2, 1, bias=False),
                    nn.BatchNorm2d(64 * 2),
                    nn.LeakyReLU(0.2, inplace=True),

                    nn.Conv2d(64 * 2, 64 * 4, 3, 2, 1, bias=False),
                    nn.BatchNorm2d(64 * 4),
                    nn.LeakyReLU(0.2, inplace=True),

                    nn.Conv2d(64 * 4, 64 * 8, 3, 2, 1, bias=False),
                    nn.BatchNorm2d(64 * 8),
                    nn.LeakyReLU(0.2, inplace=True),

                    nn.AdaptiveMaxPool2d(1),
                    nn.Conv2d(512,64,kernel_size=(1,1)),
                    nn.Conv2d(64,2,kernel_size=(1,1)),

                    )

    def forward(self, input):
        # return self.main( input.unsqueeze(0) )
        return self.main( input)



class paseLSTM_model(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, num_classes):
        super(paseLSTM_model, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        # self.lstm = nn.LSTM(input_size, self.hidden_size, self.num_layers, batch_first=True, dropout=0.2)
        self.gru = nn.GRU(input_size, self.hidden_size, self.num_layers, batch_first=True, dropout=0.2)
        self.fc1 = nn.Linear(hidden_size, int(hidden_size/2))
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(int(hidden_size/2), int(hidden_size/2))
        self.fc3 = nn.Linear(int(hidden_size/2), num_classes)
    
    def forward(self, x):
        x = x.float()
        h0 = Variable(torch.zeros(self.num_layers, x.size(0), self.hidden_size).float()).cuda() 
        # c0 = Variable(torch.zeros(self.num_layers, x.size(0), self.hidden_size).float()).cuda()
        # out, _ = self.lstm(x, (h0,c0)) 
        out, _ = self.gru(x, h0)
        out = self.relu(self.fc1(out[:, -1, :]))
        out = self.relu(self.fc2(out))
        out = self.fc3(out) 
        return out

