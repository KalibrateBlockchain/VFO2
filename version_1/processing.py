#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import torch
import matplotlib.pyplot as plt
import librosa
from gen_plot import *
import sklearn
from sklearn.model_selection import StratifiedKFold
import pandas as pd
import sys 
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
# from jade import jadeR
from fobi import FOBI
from models_def import *
import argparse

    
def read_meta_csv(pos, neg, covid_key):
    # pos_lines_spk = dict(zip(pd.read_csv(pos)['UPID'], pd.read_csv(pos)[covid_key]))
    # neg_lines_spk = dict(zip(pd.read_csv(neg)['UPID'], pd.read_csv(neg)[covid_key]))
    ones_label = np.ones(len(pd.read_csv(pos)['UPID']))
    zeros_label = np.zeros(len(pd.read_csv(neg)['UPID']))

    pos_lines_spk = dict(zip(pd.read_csv(pos)['UPID'], ones_label)) # 1 is covid pos 
    neg_lines_spk = dict(zip(pd.read_csv(neg)['UPID'], zeros_label)) # 0 is covid neg
    neg_lines_spk.update(pos_lines_spk) #{spk: (1,0), ...}
    return neg_lines_spk 

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

def parsedata(mfc_list, spk_covid):
    spks = list(spk_covid.keys())
    spks_val = list(spk_covid.values())
    all_files_mfc = open(mfc_list, 'r').readlines()
    all_files_mfc = list(map(lambda x: x[:-1], all_files_mfc))
    
    if AUDIO_TYPE != "All":
        all_files_mfc =list(filter( lambda x: x.split('/')[-1].split('_')[0] == AUDIO_TYPE, all_files_mfc))

    spk_to_utt = spkToUtt(all_files_mfc)
   
    ## There are some speakers in the csv but for dont have the audio so i have to determine the two variables below again. 
    spks = list(spk_to_utt.keys())
    spks_val = [spk_covid[s] for s in spks]
    # global UTT_TO_SPK
    utt_to_spk, UTT_TO_SPK = uttToSpk(all_files_mfc)
    
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

    for i, (data, labels, audio_types, spks, ovec_feat) in enumerate(dataloader):
        if CUDA:
            data=data.cuda().float()
            labels=torch.LongTensor(labels).cuda()
            audio_types =  torch.LongTensor(audio_types).cuda()
            spks = torch.LongTensor(spks).cuda()
            ovec_feat = ovec_feat.cuda().float()
#         print(data.shape)
        loss = 0
        if args.normalize:
            data = normalize_data(data)
        if mode == 'auto':
            dec, audio_type_class, spk_id_class, ovec_pred = model(data)
            loss += F.mse_loss(data, dec)
            if args.audio_type_loss:
                loss += F.cross_entropy(audio_type_class, audio_types)
            if args.spk_loss:
                loss += F.cross_entropy(spk_id_class, spks)
            if args.ovec_loss:
                loss +=F.mse_loss(ovec_feat, ovec_pred)
            
        if mode == "ovec":
            ovec_pred = model(data)   
            # print("+++++++++++++")
            # print(torch.mean(ovec_feat, axis=1), torch.mean(ovec_pred, axis=1))
            # print("+++++++++++++")         
            loss +=F.mse_loss(ovec_feat, ovec_pred)
            # loss = torch.abs(ovec_feat - ovec_pred)
            # loss = torch.mean(loss)

        if mode == 'class':
            # features, covid_class = model(data)
            covid_class = model(data)
            loss1 = F.cross_entropy(covid_class, labels)
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
        for i, (data, labels, audio_type, spks, ovec_feat) in enumerate(dataloader):
            if CUDA:
                data=data.cuda().float()
                labels=torch.LongTensor(labels).cuda()
            if args.normalize:
                data = normalize_data(data)
            # features, logits = model(data)
            logits = model(data)
            # all_features.append(features.cpu().numpy())
            #             print(logits.shape)
            logits = F.softmax(logits, dim=-1)[:,1]
            #             print(logits.shape)
            curr_pred = logits.cpu().detach().numpy().reshape(-1)
            curr_label = labels.cpu().numpy().reshape(-1)
        
            pred.extend(logits.cpu().detach().numpy().reshape(-1))
            true.extend(labels.cpu().numpy().reshape(-1))
            all_spks.extend(spks)
            #             print(logits.reshape(-1))
            #             print(spks)
            #             print("****")
            #             print(audio_type, spks)
            for j, spk in enumerate(spks):
                spk = spk.item()
                spk_wise[spk] = spk_wise.get(spk, []) + [(curr_pred[j], curr_label[j])]
            #             print(logits.cpu().detach().numpy().shape, labels.cpu().numpy().shape)
    
    # all_features = np.array(all_features)
    # all_features = np.concatenate(all_features, axis=0)
    all_features = None
    return pred, true, all_spks, spk_wise, all_features

def train_RF(Xtrain , ytrain, Xtest, ytest):
    # param_grid = { 
    #     'n_estimators': [10, 20, 50, 100, 200],
    #     'max_features': ['auto', 'sqrt', 'log2'],
    #     'max_depth' : [10, 50, 100, 200],
    #     'criterion' :['gini', 'entropy']
    # }
    param_grid = {
        'bootstrap': [True],
        'max_depth': [80, 90, 100, 110],
        'max_features': [2, 5, 10, 50],
        'min_samples_leaf': [3, 4, 5],
        'min_samples_split': [8, 10, 12],
        'n_estimators': [100, 200, 300, 1000]
    }

    RFmodel = RandomForestClassifier()#, class_weight='balanced')
    CV_rfc = GridSearchCV(estimator=RFmodel, param_grid=param_grid, cv= 5)
    CV_rfc.fit(Xtrain, ytrain.reshape(-1))
    pred = CV_rfc.predict(Xtest)
    return pred 

def init_weight_fobi(data_point, layer=0):
    if layer == 0:
        X = data_point.reshape(40*3, -1)
        W = FOBI(X) #120x120
        #         W_r = W.reshape(40,3,120)
        #         random_part = np.random.rand(40,3,8) 
        #         conv_init = np.concatenate((W_r, random_part), axis=2) #40,3,128
        #         conv_init = torch.Tensor(conv_init).permute(2,0,1) #128, 40, 3
        W_r = W.reshape(120, 40, 3)
        random_part = np.random.rand(8,40,3) 
        conv_init = np.concatenate((W_r, random_part), axis=0) #128, 40, 3
        conv_init = torch.Tensor(conv_init) #.permute(2,0,1) #128, 40, 3
    if layer == 1:
        X = data_point #128 xT
        W = FOBI(X) # 128 x 128 
        W_r = np.repeat(W[:, :, np.newaxis], 3, axis=2) #128x128x3
        conv_init = torch.Tensor(W_r)
    return conv_init

def train_ae_class_models(Xtrain, Ytrain, k_round):
    trained_class_models = []
    ae_model = conv_model1(TOTAL_NUM_SPKS)
    if args.init_fobi:
        def combine_data(train_dataset):
            all_data_train = train_dataset[0][0]
            for i, data in enumerate(train_dataset):
                if i == 0: continue 
                all_data_train = np.concatenate((all_data_train, data[0]), axis=1)# torch.cat((, ), 1)
            return all_data_train
        
        # all_data_train = combine_data(train_dataset)
        # print("After combing data is: ", all_data_train.shape)
        # data, labels, audio_type, spks, ovec_feat
        audio_using = 0
        #             a_t = train_dataset[audio_using][2]
        #             while a_t == 5:
        #                 audio_using += 1
        #                 a_t = train_dataset[audio_using][2]
        #             print('1 Using datapoint :', audio_using)
        #             data_point = train_dataset[audio_using][0] 
        #             print(train_dataset[audio_using][0].shape)
        #             exit()
        data_point = train_dataset[audio_using][0]
        conv_init = init_weight_fobi(data_point, layer=0)
        ae_model.encoder.weight = torch.nn.Parameter(conv_init)
        
        def weights_init(m):
            if isinstance(m, nn.Conv1d):
                torch.nn.init.xavier_uniform_(m.weight)

        #             ae_model.apply(weights_init) ## FOR XAVIER WEIGHT 
        #             ae_model.encoder.weight.requires_grad = False
        #             audio_using += 1
        #             a_t = train_dataset[audio_using][2] 
        #             while a_t == 3 or a_t == 5:
        #                 audio_using += 1
        #                 a_t = train_dataset[audio_using][2]
        #             print('2 Using datapoint :', audio_using)
        #             data_point = train_dataset[audio_using][0] 
        #             enc1 = ae_model.encoder(torch.Tensor(data_point).unsqueeze(0)).squeeze() # 128x300
        #             conv_init = init_weight_fobi(enc1.detach().numpy(), layer=1)
        #             ae_model.encoder1.weight = torch.nn.Parameter(conv_init)
                
    if CUDA:
        ae_model = ae_model.cuda()

    optimizer = optim.Adam(filter(lambda p: p.requires_grad, ae_model.parameters()), lr=0.001, weight_decay=0.0001)
    losses = []
    for i in range(EPOCHS): #n_epochs
        ae_model, loss = train_epoch(ae_model, train_loader, optimizer, mode='auto')
        losses.append(sum(loss))
    torch.save(ae_model.state_dict(), 'models/' + args.img_name + 'trained_ae_model'+str(k_round)+'.pth')
    #     del model 
    #         models.append(trained_model)
    print('Done AE Training')
    #     plt.figure()
    #     print(np.arange(len(losses)), losses)
    #     plt.plot(np.arange(len(losses)), losses, label='AE Loss')
    #     plt.show()
    class_model = classification_model()
    #         if args.init_fobi:
    #             data_point = train_dataset[0][0] 
    #             conv_init = init_weight_fobi(data_point)
    #             class_model.encoder.weight = torch.nn.Parameter(conv_init)
        
    if CUDA:
        class_model = class_model.cuda()

    class_model.encoder = ae_model.encoder
    #         class_model.encoder1 = ae_model.encoder1
    #     class_model.encoder2 = ae_model.encoder2

    optimizer = optim.Adam(filter(lambda p: p.requires_grad, class_model.parameters()), lr=0.001, weight_decay=0.0001)
    losses = []
    for i in range(EPOCHS): #n_epochs
        class_model, loss = train_epoch(class_model, train_loader, optimizer, mode='class')
        losses.append(sum(loss))

    torch.save(class_model.state_dict(), 'models/' + args.img_name + 'trained_class_model'+str(k_round)+'.pth')
    trained_class_models.append(class_model)

    #     plt.figure()
    #     plt.plot(np.arange(len(losses)), losses ,label='Class loss')
    #     plt.show()
    # torch.save(model.state_dict(), 'trained_model'+str(i)+'.pth')
    # del model 
    #     models.append(trained_model)
    print('Done Class model Training')

    pred, true, all_spks, spk_wise, _ = test_epoch(class_model, test_loader, None, None)
    for k, v in spk_wise.items():
        lab = spk_wise[k][0][1]
        spk_wise[k] = (np.mean(list(map(lambda x: x[0] , spk_wise[k]))), lab)

    spk_wise_labels = list(map(lambda x: x[1] , list(spk_wise.values())))
    spk_wise_pred = list(map(lambda x: x[0] , list(spk_wise.values())))
    #     pred_fold.append(pred, true)
    # plotting(true, pred, 'nn_gabor')
    print('SPEAKER WISE ')
    plotting(spk_wise_labels, spk_wise_pred, 'ae_cnn')
    print('IND WISE ')
    plotting(true, pred, 'ae_cnn')
    plt.close()
    return trained_class_models
 
def train_ovec_class_models(Xtrain, Ytrain, k_round):
    ovec_model = OVEC_model(mode="ovec")
      
    if CUDA:
        ovec_model = ovec_model.cuda()

    optimizer = optim.Adam(filter(lambda p: p.requires_grad, ovec_model.parameters()), lr=0.0001, weight_decay=0.0001)
    losses = []
    for i in range(EPOCHS): #n_epochs
        ovec_model, loss = train_epoch(ovec_model, train_loader, optimizer, mode='ovec')
        losses.append(sum(loss))
    torch.save(ovec_model.state_dict(), 'models/' + args.img_name + 'trained_ovec_model_model'+str(k_round)+'.pth')

    print('Done ovec_model pre Training')

    ovec_model.mode = "class"
    optimizer = optim.Adam(filter(lambda p: p.requires_grad, ovec_model.parameters()), lr=0.01, weight_decay=0.0001)

    losses = []
    for i in range(EPOCHS): #n_epochs
        ovec_model, loss = train_epoch(ovec_model, train_loader, optimizer, mode='class')
        losses.append(sum(loss))

    torch.save(ovec_model.state_dict(), 'models/' + args.img_name + 'trained_class_model'+str(k_round)+'.pth')

    print('Done Class model Training')

    pred, true, all_spks, spk_wise, _ = test_epoch(ovec_model, test_loader, None, None)
    for k, v in spk_wise.items():
        lab = spk_wise[k][0][1]
        spk_wise[k] = (np.mean(list(map(lambda x: x[0] , spk_wise[k]))), lab)

    spk_wise_labels = list(map(lambda x: x[1] , list(spk_wise.values())))
    spk_wise_pred = list(map(lambda x: x[0] , list(spk_wise.values())))

    print('SPEAKER WISE ')
    plotting(spk_wise_labels, spk_wise_pred, '')
    print('IND WISE ')
    plotting(true, pred, '')
    plt.close()
    return ovec_model

parser = argparse.ArgumentParser(description='Process some integers.')
parser.add_argument('--audio_type',  type=str)
parser.add_argument('--spk_loss', type=int, default = 0)
parser.add_argument('--ovec_loss', type=int, default = 0)
parser.add_argument('--audio_type_loss', type=int, default = 0)
parser.add_argument('--img_name', type=str, default = 'unknown')
parser.add_argument('--normalize', type=int, default = 0)
parser.add_argument('--data', type=str, default = '0')
parser.add_argument('--train_mode', type=int, default = '0')
parser.add_argument('--init_fobi', type=int, default = '0')
parser.add_argument('--init_model_path', type=str, default = '')
parser.add_argument('--use_RF', type=int, default = 0)

args = parser.parse_args()



if args.data == '0':
    TOTAL_NUM_SPKS = 60
    ovec_list= '/mnt/cvd/chile_process/ovec_list.ctl'
    mfc_list = '/mnt/cvd/chile_process/mfc_list.ctl'
    pos_covid_csv = '/mnt/cvd/datasets/chile_data_2020_06_28/COVID_19_Metadata_Pos_Neg_UPID_Anonym_20200628_positive.csv'
    neg_covid_csv ='/mnt/cvd/datasets/chile_data_2020_06_28/COVID_19_Metadata_Pos_Neg_UPID_Anonym_20200628_neg.csv'
    covid_key_in_csv = 'POSITIVE'
if args.data == '1':
    TOTAL_NUM_SPKS = 39
    ovec_list= '/mnt/cvd/chile_process/ovec_list_chile_data_2020_06_30.ctl'  
    mfc_list = '/mnt/cvd/chile_process/mfc_list_chile_data_2020_06_30.ctl'
    pos_covid_csv = '/mnt/cvd/datasets/chile_data_2020_06_30/tested_positive.csv'
    neg_covid_csv = '/mnt/cvd/datasets/chile_data_2020_06_30/tested_negative.csv'
    covid_key_in_csv = 'COVID_19_POSITIVE'
if args.data == 'All':
    TOTAL_NUM_SPKS = 250 + 105
    ovec_list= '/mnt/cvd/chile-ovec/all_chile_ovec.ctl'
    mfc_list = '/mnt/cvd/chile_process/mfc_list_chile_all_data.ctl' #'/mnt/cvd/chile_process/mfc_list_chile_all_data.ctl'
    pos_covid_csv = '/mnt/cvd/datasets/chile_data/COVID-19_2020_CHILE/all_pos_speaker_list_u.csv' # 105 
    neg_covid_csv ='/mnt/cvd/datasets/chile_data/COVID-19_2020_CHILE/all_neg_speaker_list_u.csv' # 250 
    covid_key_in_csv = None
    
all_files_ovec = open(ovec_list, 'r').readlines()
all_files_ovec = list(map(lambda x: x[:-1], all_files_ovec))


CUDA = True
BATCH_SIZE = 16
EPOCHS = 50 
NUM_K_FOLD = 5
NUM_WORKERS = 8
# for AUDIO_TYPE in  {'All'}: # , 'vowel-i', 'vowel-a', 'alphabet-a-z', 'count-1-20', 'cough', 'vowel-u'}:
AUDIO_TYPE = args.audio_type # 'All'
spk_covid = read_meta_csv(pos=pos_covid_csv, neg=neg_covid_csv, covid_key=covid_key_in_csv)
print("Spks read from meta csv: ", len(spk_covid.keys()))
XtrainAll, YtrainAll, XtestAll, YtestAll, UTT_TO_SPK = parsedata(mfc_list, spk_covid)

# print(len(XtrainAll), len(YtrainAll))
# print(XtestAll, YtestAll)


if args.train_mode:

     # = 
    trained_class_models = []

    for k_round, (Xtrain, Ytrain, Xtest, Ytest) in enumerate(zip(XtrainAll, YtrainAll, XtestAll, YtestAll)):
        print("******************************* ", k_round , "******************************* ")
        print('train: ', len(Xtrain), ' test: ', len(Xtest))
        print('Train: pos: ', sum(Ytrain), 'neg: ', len(Ytrain) - sum(Ytrain))
        print('Test: pos: ', sum(Ytest), 'neg: ', len(Ytest) - sum(Ytest))

        train_dataset = Dataset(Xtrain, Ytrain, all_files_ovec, UTT_TO_SPK)
        train_loader = dataloader.DataLoader(train_dataset, shuffle=True, batch_size=BATCH_SIZE, num_workers=NUM_WORKERS)

        test_dataset = Dataset(Xtest, Ytest, all_files_ovec, UTT_TO_SPK)
        test_loader = dataloader.DataLoader(test_dataset, shuffle=False, batch_size=BATCH_SIZE, num_workers=NUM_WORKERS)

        models = []
        # trained_class_models = train_ae_class_models(Xtrain, Ytrain, k_round)
        train_model = train_ovec_class_models(Xtrain, Ytrain, k_round)
        trained_class_models.append(train_model)
        # k_round += 1
        

    pred_all, true_all, spk_wise_all, all_spks_all = [], [], [], []
    for _, _, Xtest, Ytest, class_model in zip(XtrainAll, YtrainAll, XtestAll, YtestAll, trained_class_models):
        # class_model = trained_class_models[-1]
        # print("ONLY USING DELETEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEE ABOVE ")
        print('train: ', len(Xtrain), ' test: ', len(Xtest))
        print('Train: pos: ', sum(Ytrain), 'neg: ', len(Ytrain) - sum(Ytrain))
        print('Test: pos: ', sum(Ytest), 'neg: ', len(Ytest) - sum(Ytest))
        test_dataset = Dataset(Xtest, Ytest, all_files_ovec, UTT_TO_SPK)
        test_loader = dataloader.DataLoader(test_dataset, shuffle=False, batch_size=BATCH_SIZE, num_workers=NUM_WORKERS)
        pred, true, all_spks, spk_wise, _ = test_epoch(class_model, test_loader, None, None)
        pred_all.extend(pred)
        true_all.extend(true)
        spk_wise_all.append(spk_wise)
        all_spks_all.extend(all_spks)

if not args.train_mode:

    if args.use_RF:
        print("TRAINING RANDOM FOREST FROM THE FEATURES ")
        true_all, pred_all, _, _, _ = [], [], [], [], []


        # for k_round in range(NUM_K_FOLD):
        for k_round, (Xtrain, Ytrain, Xtest, Ytest) in enumerate(zip(XtrainAll, YtrainAll, XtestAll, YtestAll)):

            test_dataset = Dataset(Xtest, Ytest, all_files_ovec, UTT_TO_SPK)
            test_loader = dataloader.DataLoader(test_dataset, shuffle=False, batch_size=BATCH_SIZE, num_workers=NUM_WORKERS)

            train_dataset = Dataset(Xtrain, Ytrain, all_files_ovec, UTT_TO_SPK)
            train_loader = dataloader.DataLoader(train_dataset, shuffle=False, batch_size=BATCH_SIZE, num_workers=NUM_WORKERS)

            class_model = classification_model() 
            # MODEL_NAME = args.init_model_path
            MODEL_NAME = 'models/' + args.img_name +'trained_class_model'+str(k_round)+'.pth'
            class_model.load_state_dict(torch.load(MODEL_NAME)) 
            if CUDA:
                class_model.cuda()
            _, true_train, _, _, all_features_train = test_epoch(class_model, train_loader, None, None)
            _, true_test, _, _, all_features_test = test_epoch(class_model, test_loader, None, None)
        

            pred = train_RF(Xtrain=all_features_train, ytrain=np.array(true_train).reshape(-1,1), \
                                Xtest=all_features_test, ytest=np.array(true_test).reshape(-1,1) )
            pred_all.extend(pred)
            true_all.extend(true_test)
                
        true_all = np.array(true_all)
        pred_all = np.array(pred_all)
        print(true_all)
        print(pred_all)
        plotting(true_all, pred_all, None)

        exit()

    if not args.use_RF:
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
print(spk_wise_all)
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

print('SPEAKER WISE ')
plotting(spk_wise_labels, spk_wise_pred, 'spk_wise_'+args.img_name)
print('IND WISE ')
plotting(true_all, pred_all, 'ind_wise_' + args.img_name)



