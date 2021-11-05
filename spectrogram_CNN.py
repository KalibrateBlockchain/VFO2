#!/usr/bin/env python
# coding: utf-8

# In[1]:

import pdb
import numpy as np
import torch
torch.multiprocessing.set_sharing_strategy('file_system')
import matplotlib.pyplot as plt
import librosa
from gen_plot import *
import sklearn
from sklearn.model_selection import StratifiedKFold
import pandas as pd
import sys 
from sklearn.model_selection import GridSearchCV
from model_paseCNN import * 
import argparse
import os
import pickle
from adabelief_pytorch import AdaBelief
    
def read_meta_csv(pos, neg, covid_key):
    # pos_lines_spk = dict(zip(pd.read_csv(pos)['UPID'], pd.read_csv(pos)[covid_key]))
    # neg_lines_spk = dict(zip(pd.read_csv(neg)['UPID'], pd.read_csv(neg)[covid_key]))
    ones_label = np.ones(len(pd.read_csv(pos)['UPID']))
    zeros_label = np.zeros(len(pd.read_csv(neg)['UPID']))

    pos_lines_spk = dict(zip(pd.read_csv(pos)['UPID'], ones_label)) # 1 is covid pos 
    neg_lines_spk = dict(zip(pd.read_csv(neg)['UPID'], zeros_label)) # 0 is covid neg
    # all spks
    neg_lines_spk.update(pos_lines_spk) #{spk: (1,0), ...}
    return neg_lines_spk 

def save_prediction_with_speakerid_and_audioname(prediction,labels,speaker,audio_name,pkl_file):
    d={}
    d['pred']=prediction
    d['labels']=labels
    d['speaker'] = speaker
    d['audio_name'] = audio_name
    f=open(pkl_file,'wb')
    pickle.dump(d,f)


def check_spk(all_files):
    '''
    return {spk: [u_1, u_2] , ....}
    '''
    res = {}
    
    for fullname in all_files:
        f = fullname.split('/')[-1]#[:-1]
        spk_id = f.split('_')[1]
        if spk_id == 'a-z' or spk_id == '1-20':
             print(f)
             cmd = 'mv '+ fullname + ' '+ fullname.replace('_'+spk_id, '-'+spk_id)
             os.system(cmd)
    

def spkToUtt(all_files):
    '''
    return {spk: [u_1, u_2] , ....}
    '''
    res = {}

    for fullname in all_files:
        f = fullname.split('/')[-1]#[:-1]
        spk_id = f.split('_')[1]
        res[spk_id] = res.get(spk_id, []) + [fullname]

    return res


def uttToSpk(all_files):
    '''
    {utt: spk, ...}
    '''
    res = {}
    unique_id = {}
    curr = 0
    for fullname in all_files:
        f = fullname.split('/')[-1]#[:-1]
        spk_id = f.split('_')[1]
        if spk_id not in unique_id: 
            unique_id[spk_id] = curr
            curr += 1
        res[fullname] = spk_id
    return res, unique_id

def tripleNegatives(Xtrain, Ytrain, Xtest, Ytest):
    Xtrain, Ytrain, Xtest, Ytest = np.array(Xtrain), np.array(Ytrain), np.array(Xtest), np.array(Ytest)
    # print(Ytrain)
    neg_idx = np.where(Ytrain == 0)
    # x_neg = XtrainAll[neg_idx]
    Xtrain = np.concatenate((Xtrain, Xtrain[neg_idx]))
    Xtrain = np.concatenate((Xtrain, Xtrain[neg_idx]))
    Ytrain = np.concatenate((Ytrain, Ytrain[neg_idx]))
    Ytrain = np.concatenate((Ytrain, Ytrain[neg_idx]))
    return Xtrain, Ytrain, Xtest, Ytest

def parsedata(feat_list, spk_covid):
    spks = list(spk_covid.keys())
    spks_val = list(spk_covid.values())
    all_files_feat = open(feat_list, 'r').readlines()
    all_files_feat = list(map(lambda x: x[:-1], all_files_feat))
    
    if AUDIO_TYPE != "All":
        all_files_feat =list(filter( lambda x: x.split('/')[-1].split('_')[0] == AUDIO_TYPE, all_files_feat))

    # check_spk(all_files_feat)
    spk_to_utt = spkToUtt(all_files_feat)
   
    ## There are some speakers in the csv but for dont have the audio so i have to determine the two variables below again. 
    spks = list(spk_to_utt.keys())
    spks_val = [spk_covid[s] for s in spks] # if 'UPID' not in s]
    # global UTT_TO_SPK
    utt_to_spk, UTT_TO_SPK = uttToSpk(all_files_feat)
    
    skf = StratifiedKFold(n_splits=NUM_K_FOLD)
    n_samples = len(spks_val)
    XtrainAll, YtrainAll, XtestAll, YtestAll = [],[],[],[]
    for train_spk_idx, test_spk_idx in skf.split(X=np.zeros(n_samples), y=spks_val):

        Xtrain = []
        for i in train_spk_idx:
            Xtrain.extend(spk_to_utt[spks[i]]) 

        Xtest = []
        for i in test_spk_idx:
            Xtest.extend(spk_to_utt[spks[i]])

        Ytrain = [int(spk_covid[utt_to_spk[x]]) for x in Xtrain ]
        Ytest = [int(spk_covid[utt_to_spk[x]]) for x in Xtest ]
        Xtrain, Ytrain, Xtest, Ytest = tripleNegatives(Xtrain, Ytrain, Xtest, Ytest)
#         print(len(Xtrain), len(Ytrain), len(Xtest), len(Ytest))
        XtrainAll.append(Xtrain)
        YtrainAll.append(Ytrain)
        XtestAll.append(Xtest)
        YtestAll.append(Ytest)
#         print(len(XtrainAll), len(YtrainAll), len(XtestAll), len(YtestAll))

        
    return XtrainAll, YtrainAll, XtestAll, YtestAll, UTT_TO_SPK


def normalize_data(data):
    mean = torch.mean(data, 0)
    std = torch.std(data, 0)
    data = (data - mean) / std
    return data 

def train_epoch(model, dataloader, optimizer, mode):
    losses = []

    for i, (data, labels, audio_types, spks) in enumerate(dataloader):
        if CUDA:
            data=data.cuda().float()
            labels=torch.LongTensor(labels).cuda()
            # labels = torch.eye(2)[labels,:]
            audio_types =  torch.LongTensor(audio_types).cuda()
            spks = torch.LongTensor(spks).cuda()
#         print(data.shape)
        loss = 0
        if args.normalize:
            data = normalize_data(data)

        # features, covid_class = model(data)
        data = data.unsqueeze(0)
        data = data.transpose(2,3)
        covid_class = model(data)
        loss1 = F.cross_entropy(covid_class, labels.unsqueeze(0).unsqueeze(0))
        loss = loss1

        losses.append(loss.item())
            
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

    losses = np.array(losses)
    print("Loss: ", np.mean(losses))
    return model, losses


def test_epoch(model, dataloader, optimizer, mode):
    pred, true, all_spks = [], [], []
    spk_wise = {}
    all_features = []
    with torch.no_grad():
        losses = []
        for i, (data, labels, audio_type, spks) in enumerate(dataloader):
            if CUDA:
                data=data.cuda().float()
                labels=torch.LongTensor(labels).cuda()
            if args.normalize:
                data = normalize_data(data)

            data = data.unsqueeze(0)
            data = data.transpose(2,3)
            logits = model(data)
            loss1 = F.cross_entropy(logits, labels.unsqueeze(0).unsqueeze(0))
            # pdb.set_trace()
            # logits = torch.argmax(F.softmax(logits, dim=1)) # should be argmax
            logits = F.softmax(logits, dim=1)
            # pay attention to the dimension
            curr_pred = logits[0][1][0].cpu().detach().numpy()
            curr_label = labels.cpu().numpy()
            losses.append(loss1.item())
            pred.extend(curr_pred)
            true.extend(curr_label)
            all_spks.extend(spks)

            for j, spk in enumerate(spks):
                spk = spk.item()
                spk_wise[spk] = spk_wise.get(spk, []) + [(curr_pred[j], curr_label[j])]
            #             print(logits.cpu().detach().numpy().shape, labels.cpu().numpy().shape)
    
    all_features = np.array(all_features)
    # all_features = np.concatenate(all_features, axis=0)
    all_features = None
    losses = np.array(losses)
    print("Loss: ", np.mean(losses))
    return pred, true, all_spks, spk_wise, all_features

def train_paseCNN_class_models(train_loader, test_loader, k_round):
    pCNN_model = spectrogramCNN_model()


    if CUDA:
        pCNN_model = pCNN_model.cuda()

    # optimizer = optim.Adam(filter(lambda p: p.requires_grad, pCNN_model.parameters()), lr=0.00001, weight_decay=0.0001)
    optimizer = AdaBelief(pCNN_model.parameters(), lr=1e-3, eps=1e-12, betas=(0.9,0.999))

    losses = []
    for i in range(EPOCHS): #n_epochs
        pCNN_model, loss = train_epoch(pCNN_model, train_loader, optimizer, mode='pase')
        losses.append(sum(loss))
    torch.save(pCNN_model.state_dict(), 'models/' + args.img_name + 'trained_pase_model_model'+str(k_round)+'.pth')

    print('Done paseCNN_model Training')

    '''
    pred, true, all_spks, spk_wise, _ = test_epoch(pCNN_model, test_loader, None, None)
    for k, v in spk_wise.items():
        lab = spk_wise[k][0][1]
        spk_wise[k] = (np.mean(list(map(lambda x: x[0] , spk_wise[k]))), lab)

    spk_wise_labels = list(map(lambda x: x[1] , list(spk_wise.values())))
    spk_wise_pred = list(map(lambda x: x[0] , list(spk_wise.values())))

    print('SPEAKER WISE ')
    # plotting(spk_wise_labels, spk_wise_pred, '')
    print('IND WISE ')
    # plotting(true, pred, '')
    # plt.close()
    '''
    return pCNN_model


def test_paseCNN_class_models(test_loader, k_round):
    pCNN_model = paseCNN_model()

    if CUDA:
        pCNN_model = pCNN_model.cuda()

    print('Load paseCNN_model Trained')

    path = 'models/unknowntrained_pase_model_model0.pth'

    pCNN_model.load_state_dict(torch.load(path))

    pred, true, all_spks, spk_wise, _ = test_epoch(pCNN_model, test_loader, None, None)
    for k, v in spk_wise.items():
        lab = spk_wise[k][0][1]
        spk_wise[k] = (np.mean(list(map(lambda x: x[0] , spk_wise[k]))), lab)

    spk_wise_labels = list(map(lambda x: x[1] , list(spk_wise.values())))
    spk_wise_pred = list(map(lambda x: x[0] , list(spk_wise.values())))

    print('SPEAKER WISE ')
    # plotting(spk_wise_labels, spk_wise_pred, '')
    print('IND WISE ')
    # plotting(true, pred, '')
    # plt.close()
    return pCNN_model


parser = argparse.ArgumentParser(description='Process some integers.')
parser.add_argument('--audio_type',  type=str, default = 'All')
parser.add_argument('--img_name', type=str, default = 'unknown')
parser.add_argument('--normalize', type=int, default = 0)
parser.add_argument('--data', type=str, default = 'All')
parser.add_argument('--train_mode', type=int, default = '1')
parser.add_argument('--init_model_path', type=str, default = '')
parser.add_argument('--feature_list', type=str, default = '../all_spectrogram_wOct.ctl')
parser.add_argument('--EPOCHS', type=int, default=20)
parser.add_argument('--K_FOLD', type=int, default=5)


args = parser.parse_args()

if args.data == 'All':
    TOTAL_NUM_SPKS = 250 + 105
    # pase_list= '../all_chile_pase.ctl'
    # pase_list= '../all_chile_pase_andAug22.ctl'
    # pase_list = '../all_spectrogram_wOct.ctl'
    # pos_covid_csv = '../chile_data/COVID-19_2020_CHILE/all_Aug_pos_speaker_list_u.csv' # 250 
    # neg_covid_csv ='../chile_data/COVID-19_2020_CHILE/all_Aug_neg_speaker_list_u.csv' # 106
    pos_covid_csv = '../chile_data/COVID-19_2020_CHILE/positive_correct.csv'
    neg_covid_csv ='../chile_data/COVID-19_2020_CHILE/negative_correct.csv'
    covid_key_in_csv = None
    

pase_list = args.feature_list
all_files = open(pase_list, 'r').readlines()
all_files = list(map(lambda x: x[:-1], all_files))


CUDA = True
BATCH_SIZE = 1
EPOCHS = args.EPOCHS
NUM_K_FOLD = args.K_FOLD
NUM_WORKERS = 8
# for AUDIO_TYPE in  {'All'}: # , 'vowel-i', 'vowel-a', 'alphabet-a-z', 'count-1-20', 'cough', 'vowel-u'}:
AUDIO_TYPE = args.audio_type # 'All'
spk_covid = read_meta_csv(pos=pos_covid_csv, neg=neg_covid_csv, covid_key=covid_key_in_csv)
print("Spks read from meta csv: ", len(spk_covid.keys()))
XtrainAll, YtrainAll, XtestAll, YtestAll, UTT_TO_SPK = parsedata(pase_list, spk_covid)
# pdb.set_trace()

if args.train_mode:

    trained_class_models = []

    for k_round, (Xtrain, Ytrain, Xtest, Ytest) in enumerate(zip(XtrainAll, YtrainAll, XtestAll, YtestAll)):
        print("******************************* ", k_round , "******************************* ")
        print('train: ', len(Xtrain), ' test: ', len(Xtest))
        print('Train: pos: ', sum(Ytrain), 'neg: ', len(Ytrain) - sum(Ytrain))
        print('Test: pos: ', sum(Ytest), 'neg: ', len(Ytest) - sum(Ytest))

        # feed in the right data etc
        train_dataset = Dataset(Xtrain, Ytrain, all_files, UTT_TO_SPK)
        train_loader = dataloader.DataLoader(train_dataset, shuffle=True, batch_size=BATCH_SIZE, num_workers=NUM_WORKERS)

        test_dataset = Dataset(Xtest, Ytest, all_files, UTT_TO_SPK)
        test_loader = dataloader.DataLoader(test_dataset, shuffle=False, batch_size=BATCH_SIZE, num_workers=NUM_WORKERS)

        models = []
        train_model = train_paseCNN_class_models(train_loader, test_loader, k_round)
        trained_class_models.append(train_model)
        # k_round += 1
        

    pred_all, true_all, spk_wise_all, all_spks_all = [], [], [], []
    for _, _, Xtest, Ytest, class_model in zip(XtrainAll, YtrainAll, XtestAll, YtestAll, trained_class_models):
        # class_model = trained_class_models[-1]
        # print("ONLY USING DELETEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEE ABOVE ")
        print('train: ', len(Xtrain), ' test: ', len(Xtest))
        print('Train: pos: ', sum(Ytrain), 'neg: ', len(Ytrain) - sum(Ytrain))
        print('Test: pos: ', sum(Ytest), 'neg: ', len(Ytest) - sum(Ytest))
        test_dataset = Dataset(Xtest, Ytest, all_files, UTT_TO_SPK)
        test_loader = dataloader.DataLoader(test_dataset, shuffle=False, batch_size=BATCH_SIZE, num_workers=NUM_WORKERS)
        pred, true, all_spks, spk_wise, _ = test_epoch(class_model, test_loader, None, None)
        pred_all.extend(pred)
        true_all.extend(true)
        spk_wise_all.append(spk_wise)
        all_spks_all.extend(all_spks)

if not args.train_mode:

    ## check tihs code hira plz 
    allXtest, allYtest = [], [] 
    for Xtrain, Ytrain, Xtest, Ytest in zip(XtrainAll, YtrainAll, XtestAll, YtestAll):
        allXtest.extend(Xtest)
        allYtest.extend(Ytest)
    pred_all, all_spks_all, spk_wise_all, all_features_test = [], [], [], []
    for k_round in range(NUM_K_FOLD):
        class_model = classification_model() 
        # MODEL_NAME = args.init_model_path
        MODEL_NAME = 'models/' + args.img_name +'trained_class_model'+str(k_round)+'.pth'
        class_model.load_state_dict(torch.load(MODEL_NAME)) 
        if CUDA:
            class_model.cuda()
        pred, true, all_spks, spk_wise, all_features = test_epoch(class_model, test_loader, None, None)
        pred_all.append(pred)
        all_features_test.append(all_features)
    pred_all = np.array(pred_all)
    pred_all = np.mean(pred_all, axis=0).reshape(-1,1)
    true_all = np.array(true).reshape(-1,1)
    spk_wise_all = [spk_wise]
    all_spks_all = all_spks
    
        
print("Final")
# print(spk_wise_all)
spk_wise_res = {}
for d in spk_wise_all:
    for k, v in d.items():
        if k in spk_wise_res: spk_wise_res[k].append(v)
        else: spk_wise_res[k] = v

#### VOTING SCORES 
#### AVEGRAING SCORES 
for k, v in spk_wise_res.items():
    lab = spk_wise_res[k][0][1]
    spk_wise_res[k] = (np.mean(list(map(lambda x: x[0] , spk_wise_res[k]))), lab)

# print(spk_wise_res[33])

spk_wise_labels = list(map(lambda x: x[1] , list(spk_wise_res.values())))
spk_wise_pred = list(map(lambda x: x[0] , list(spk_wise_res.values())))

# pdb.set_trace()

# affix = 'spec_wOct_epoch30_Kfold10_adabelief'
affix = pase_list[1:-4] + '_EPOCH_' + str(EPOCHS) + '_K_FOLD_' + str(NUM_K_FOLD)
np.save(affix + '_spk_wise_labels.npy', spk_wise_labels)
np.save(affix + '_spk_wise_pred.npy', spk_wise_pred)
np.save(affix + '_true_all.npy', true_all)
np.save(affix + '_pred_all.npy', pred_all)
np.save(affix + '_XtestAll.npy', XtestAll)

# save_prediction_with_speakerid_and_audioname(prediction,labels,speaker,audio_name,pkl_file)
'''
pdb.set_trace()

print('SPEAKER WISE ')
plotting(spk_wise_labels, spk_wise_pred, 'spk_wise_'+args.img_name)
print('IND WISE ')
plotting(true_all, pred_all, 'ind_wise_' + args.img_name)
'''


