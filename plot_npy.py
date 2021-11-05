import argparse
import pdb
import numpy as np
import matplotlib.pyplot as plt
from gen_plot import *
import pickle

# testname = 'pase_wOct_epoch20_Kfold5'
# testname = 'pase_all_wOct_EPOCH_15_K_FOLD_5'
# testname = 'pase_normalized_all_wOct_EPOCH_15_K_FOLD_5'
# testname = 'pase_all_wOct_EPOCH_15_K_FOLD_8'
# testname = 'pase_all_wOct_EPOCH_25_K_FOLD_5'
# testname = 'pase_normalized_all_wOct_EPOCH_15_K_FOLD_10'
# testname = 'pase_normalized_removed_recovered_nosymptoms_wOct_EPOCH_12_K_FOLD_5' 
# testname = 'pase_normalized_removed_recovered_nosymptoms_wOct_EPOCH_15_K_FOLD_5'
# testname = 'pase_normalized_removed_recovered_nosymptoms_wOct_EPOCH_20_K_FOLD_5'
# testname = 'pase_normalized_all_wOct_EPOCH_25_K_FOLD_5' 
# testname = 'pase_normalized_all_wOct_EPOCH_12_K_FOLD_5'
# testname = 'pase_normalized_all_wOct_EPOCH_10_K_FOLD_5'
# testname = 'pase_normalized_all_wOct_EPOCH_20_K_FOLD_8'
# testname = 'pase_normalized_all_wOct_EPOCH_20_K_FOLD_10'
# testname = 'pase_normalized_all_wOct_correct20_EPOCH_20_K_FOLD_5' 
# testname = 'featurelist_tillAug16_correct20_EPOCH_15_K_FOLD_5'
# testname = 'vowel-a-all_wOct_correct20_EPOCH_10_K_FOLD_5'
# testname = 'vowel-i-all_wOct_correct20_EPOCH_10_K_FOLD_5'
# testname = 'vowel-u-all_wOct_correct20_EPOCH_10_K_FOLD_5'
# testname = 'vowel-a-all_wOct_correct20_EPOCH_20_K_FOLD_5'
# testname = 'vowel-aiu-all_wOct_correct20_EPOCH_20_K_FOLD_5' 
# testname = 'cough-all_wOct_correct20_EPOCH_10_K_FOLD_5'
# testname = 'cough-all_wOct_correct20_EPOCH_15_K_FOLD_5'
# testname = 'count-1-20-all_wOct_correct20_EPOCH_15_K_FOLD_5'
# testname = 'alphabet-a-z-all_wOct_correct20_EPOCH_15_K_FOLD_5'
# testname = 'featurelist_tillAug16_correct20_EPOCH_15_K_FOLD_5_rerun'
# testname = 'featurelist_diag7_correct20_EPOCH_15_K_FOLD_5'
# testname = 'cough_pase_normalized_all_wOct_correct20_EPOCH_15_K_FOLD_5'

parser = argparse.ArgumentParser(description='Process some integers.')
parser.add_argument('--FILE_NAME', type=str, default = '')
args = parser.parse_args()

testname = args.FILE_NAME

spk_wise_labels = np.load(testname + '_spk_wise_labels.npy')
spk_wise_pred = np.load(testname + '_spk_wise_pred.npy')

true_all = np.load(testname + '_true_all.npy')
pred_all = np.load(testname + '_pred_all.npy')

print('SPEAKER WISE ')
plotting(spk_wise_labels, spk_wise_pred, 'spk_wise')
print('IND WISE ')
plotting(true_all, pred_all, 'ind_wise')


def save_prediction_with_speakerid(prediction,labels,speaker,pkl_file):
    d={}
    d['pred']=prediction
    d['labels']=labels
    d['speaker'] = speaker
    f=open(pkl_file,'wb')
    pickle.dump(d,f)

def save_prediction_with_speakerid_and_audioname(prediction,labels,speaker,audio_name,pkl_file):
    d={}
    d['pred']=prediction
    d['labels']=labels
    d['speaker'] = speaker
    d['audio_name'] = audio_name
    f=open(pkl_file,'wb')
    pickle.dump(d,f)


Alltest = np.load(testname + '_XtestAll.npy', allow_pickle=True)
speaker = []
audio_name = []
for i in range(len(Alltest)):
    test = Alltest[i]
    for filen in test:
        # pdb.set_trace()
        audio = filen.split('/')[-1]
        speaker_name = filen.split('_')[-2]
        speaker.append(speaker_name)
        audio_name.append(audio)

pkl_file = testname + '.pkl'
save_prediction_with_speakerid_and_audioname(pred_all,true_all,speaker,audio_name,pkl_file)
