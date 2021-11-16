#!/bin/bash
epochs=1
#### NOTE -- CHange value of kfold to 5 before running -- this is set to 2 for the data in repo only! 
k_fold=2
feature_list=$1
lr=0.001
decay=0.00001
batch_size=1
has_scheduler=1
patience=2
factor=0.4
model='paseCNN_model'

python paseCNN.py --EPOCHS=$epochs --K_FOLD=$k_fold --feature_list=$feature_list --LR=$lr --W_DECAY=$decay --BATCH_SIZE=$batch_size --HAS_SCHEDULER=$has_scheduler --PATIENCE=$patience
feature_list_name=${feature_list%.ctl}
python plot_npy.py --FILE_NAME="${feature_list_name}_correct20_EPOCH_${epochs}_K_FOLD_${k_fold}"

