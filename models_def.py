import torch
from torch.utils.data import Dataset
import torch.utils.data.dataloader as dataloader
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from numpy.matlib import repmat
import math

MAX_LEN = 300
AUDIO_TYPE_ID = {'vowel-u': 0,'vowel-i': 1,'vowel-a': 2,'alphabet-a-z': 3, 'cough': 4, 'count-1-20': 5,}
# IMP_OVEC_FEAT = [16, 100, 709 ,  88,  612,  484, 1390,  591,   94,  716,  499,  463,  373,   95, 1407, 86 ]#,  \
#                  194,  401, 1389,  380,  381, 49,  495,  319,    1,   24,  685,  465,  711,  727, 1132,  695,  \
#                  356,  726,  352,   10,  815,  729, 1153,  421,  332, 1327,  395,  700, 1432,  583, 1202,  754, 1306,  291]
# IMP_OVEC_FEAT = np.arange(1409,1582)
# IMP_OVEC_FEAT = np.arange(1582)
IMP_OVEC_FEAT = np.arange(10)
def uttToSpkChile(fullname):
    f = fullname.split('/')[-1]#[:-1]
    spk_id = f.split('_')[1]
    return spk_id

class Dataset(Dataset):
    'Characterizes a dataset for PyTorch'
    def __init__(self, X, Y,all_files_ovec, UTT_TO_SPK):
        self.X = X
        self.Y = Y
        self.mfc_to_ovec = self.mfcToOvec(X, all_files_ovec)
        self.UTT_TO_SPK = UTT_TO_SPK
        
    def __len__(self):
        'Denotes the total number of samples'
        return len(self.X)
    
    def mfcToOvec(self, all_files_mfc, all_files_ovec):
        '''return {mfcfile: ovecfile, ... }'''
        
        mfc_file_base = list(map(lambda x: x.split('/')[-1].split('.')[0], all_files_mfc ))
        ovec_file_base = list(map(lambda x: ('_').join(x.split('/')[-1].split('.')[0].split('_')[:-1]), all_files_ovec ))
#         print(mfc_file_base)
#         print(ovec_file_base)
        res = {}
        i = 0
        for mfc_file in mfc_file_base:
            j = 0
            for ovec_file in ovec_file_base:
                if mfc_file != ovec_file: 
                    j += 1
                    continue
                res[all_files_mfc[i]] = all_files_ovec[j]
                break
            i += 1
        return res 
    
    def __getitem__(self, index):
        x = self.X[index] # filename 
        y = self.Y[index] # int (0,1)
        ovec_file = self.mfc_to_ovec[x]
        
        ovec_feat = np.load(ovec_file)[IMP_OVEC_FEAT]
        # print(ovec_file)
        # print(np.load(ovec_file).shape)

        # print(ovec_feat)
        # exit()
#         print(x, ovec_file)
        audio_type = x.split('/')[-1].split('_')[0]
        spk = self.UTT_TO_SPK[uttToSpkChile(x)]
#         print(x, spk)
        feat = np.load(x)
        # print("FEAT: ", feat.shape)
        ### FOR SPEC ###
        # need to do the transpose for spectrograms but not for mfccs 
        # feat = feat.transpose()
        ################
        orig_len = feat.shape[0]
        feat = repmat(feat, int(math.ceil(MAX_LEN/(feat.shape[0]))),1)
        feat = feat[:MAX_LEN,:]

        #### shuffling the cylinder ##
        # pivot = np.random.randint(MAX_LEN)
        # idx1 = np.arange(pivot, MAX_LEN)
        # idx2 = np.arange(0, pivot)
        # idx = np.concatenate((idx1, idx2))
        # feat = feat[idx]
        ###############################

        feat = feat.transpose()
        return feat, int(y), AUDIO_TYPE_ID[audio_type], spk, ovec_feat


class BasicBlock(nn.Module):
    def __init__(self, planes):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv1d(planes, planes, 3, 1, 1, bias=False)
        self.bn1 = nn.BatchNorm1d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.leakyrelu = nn.LeakyReLU(inplace=True)
        self.conv2 = nn.Conv1d(planes, planes, 3, 1, 1, bias=False)
        self.bn2 = nn.BatchNorm1d(planes)
   
    def forward(self, x):
        residual = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.leakyrelu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        out += residual
        out = self.leakyrelu(out)
   
        return out
    

class conv_model1(nn.Module):
    def __init__(self, TOTAL_NUM_SPKS):
        super(conv_model1, self).__init__()
        self.num_filter = 128
        self.encoder = nn.Conv1d(40, self.num_filter, 3, 1, bias=False, padding=1)
        self.encoder1 = nn.Conv1d(self.num_filter, self.num_filter, 3, 1, bias=False, padding=1)
        self.encoder2 = nn.Conv1d(self.num_filter, self.num_filter, 3, 1, bias=False, padding=1, groups=self.num_filter)
#         self.decoder = nn.Conv1d(self.num_filter, 40, 3, 1, bias=False, padding=1)
        self.decoder = nn.ConvTranspose1d(self.num_filter, self.num_filter, 3,1,padding=1)
        self.decoder1 = nn.ConvTranspose1d(self.num_filter, self.num_filter, 3,1,padding=1)
        self.decoder2 = nn.ConvTranspose1d(self.num_filter, 40, 3,1,padding=1)
        
        self.f2 = nn.Linear(self.num_filter, 6) # 6 classes
        self.f3 = nn.Linear(self.num_filter, TOTAL_NUM_SPKS) 
        self.f4 = nn.Linear(self.num_filter, len(IMP_OVEC_FEAT))
        
        self.basic1 = BasicBlock(self.num_filter)
        self.basic2 = BasicBlock(self.num_filter)
        
        self.bn = nn.BatchNorm1d(40)
        
    def forward(self,x):
#         x = self.bn(x)
        enc = self.encoder(x)
        enc = nn.LeakyReLU()(enc)
        enc = self.encoder1(enc)
        enc = self.basic1(enc)
        enc = nn.LeakyReLU()(enc)
        enc = self.encoder2(enc)
        enc = nn.LeakyReLU()(enc)
#         print("enc.shape: ", enc.shape)
        dec = nn.LeakyReLU()(self.decoder(enc))
        dec = nn.LeakyReLU()(self.decoder1(dec))
        dec = self.basic2(dec)
        dec = nn.LeakyReLU()(self.decoder2(dec))
#         print("dec.shape: ", dec.shape)
        enc_permute = enc.permute(0,2,1)
#         print("enc_permute.shape ", enc_permute.shape)
        
        enc_pooled = F.avg_pool1d(enc, kernel_size=(enc.shape[2])).squeeze()
        out2 = nn.LeakyReLU()(self.f2(enc_pooled))
        out3 = nn.LeakyReLU()(self.f3(enc_pooled))
        out4 = nn.LeakyReLU()(self.f4(enc_pooled))
        return dec, out2, out3, out4

class classification_model(nn.Module):
    def __init__(self):
        super(classification_model, self).__init__()
        self.num_filter = 128
        self.encoder = nn.Conv1d(40, self.num_filter, 3, 1, bias=False, padding=1)
        self.encoder1 = nn.Conv1d(self.num_filter, self.num_filter, 3, 1, bias=False, padding=1, groups=self.num_filter)
        self.encoder2 = nn.Conv1d(self.num_filter, self.num_filter, 3, 1, bias=False, padding=1, groups=self.num_filter)
        self.f1 = nn.Linear(128, 2) # 2 classes
        self.bn = nn.BatchNorm1d(40)
        
    def forward(self,x):
#         print(x.shape)
#         x = self.bn(x)
        enc = self.encoder(x)
        enc = nn.LeakyReLU()(enc)
        enc = self.encoder1(enc)
        enc = nn.LeakyReLU()(enc)
        enc = self.encoder2(enc)
        enc = nn.LeakyReLU()(enc)
#         enc = enc.permute(0,2,1) #b,t,f
#         print("enc.shape ", enc.shape)
        enc_permute = enc.permute(0,2,1)
#         print("enc_permute.shape ", enc_permute.shape)
        
        enc_pooled = F.avg_pool1d(enc, kernel_size=(enc.shape[2])).squeeze()
#         print("enc_pooled ", enc_pooled.shape)
        out = self.f1(enc_pooled) #b,t,2
        
#         print(out.shape)
        return enc_pooled, out

class OVEC_model(nn.Module):
    def __init__(self, mode):
        super(OVEC_model, self).__init__()
        self.num_filter = 256
        self.inp_channel = 40 #40
        self.ovec_length = len(IMP_OVEC_FEAT)
        self.cnn1 = nn.Conv1d(self.inp_channel, self.num_filter, kernel_size=3, stride=1, bias=False, padding=1)
        self.cnn2 = nn.Conv1d(self.num_filter, self.num_filter, 3, 1, bias=False)
        self.cnn3 = nn.Conv1d(self.num_filter, self.num_filter, 3, 1, bias=False)
        self.f1 = nn.Linear(self.num_filter, self.ovec_length)
        self.f2 = nn.Linear(self.num_filter, 2)
        self.bn = nn.BatchNorm1d(self.num_filter)
        self.bn1 = nn.BatchNorm1d(self.ovec_length)
        self.mode = mode 

    def forward(self, x):
        enc = self.cnn1(x)
        enc = nn.LeakyReLU()(enc)
        enc = self.cnn2(enc)
        enc = nn.LeakyReLU()(enc)
        enc = self.cnn3(enc)
        enc = nn.LeakyReLU()(enc)
        enc_permute = enc.permute(0,2,1)
        enc_pooled = F.avg_pool1d(enc, kernel_size=(enc.shape[2])).squeeze()
        # enc_pooled = self.bn(enc_pooled)
        if self.mode == "ovec":
            # print(enc_pooled)
            out = self.f1(enc_pooled)
            # print(out)
            # out = self.bn1(out)
            # print(out)
            # exit()
        if self.mode == "class":
            out = self.f2(enc_pooled)
        return out 