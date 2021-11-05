import numpy as np
import matplotlib.pyplot as plt 
import matplotlib
from sklearn.metrics import precision_recall_curve
from sklearn.metrics import plot_precision_recall_curve
from sklearn.metrics import roc_curve, auc
from sklearn.metrics import roc_auc_score
import pdb

def gen_roc_curve(fpr, tpr, roc_auc,classifier_type):
    plt.figure()
    lw = 2
    plt.plot(fpr, tpr, color='darkorange', \
        lw=lw, label='ROC curve (area = %0.2f)' % roc_auc)
    plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver operating characteristic example for {}'.format(classifier_type))
    plt.legend(loc="lower right")
    if classifier_type:
        plt.savefig(classifier_type+'roc')
    plt.show()
    
def gen_prec_recall_curve(recall, precision, classifier_type):
    plt.figure()
    lw = 2
    plt.plot(recall, precision, color='darkorange', \
        lw=lw)
    plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
    plt.xlim([-0.05, 1.0])
    plt.ylim([-0.05, 1.05])
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    i = 0
    prev_label = '1'    
    plt.title('Precision Recall curve for {}'.format(classifier_type))
    if classifier_type:
        plt.savefig(classifier_type+'pr')
    plt.show()
    
def computeDET(labels,scores):
    # Sort scores.  We will use the sorted scores themselves as thresholds.
    idx = [id[0] for id in sorted(enumerate(scores), key=lambda x:x[1])]
    idx.reverse()

    numtrue = sum(labels)
    numfalse = len(labels) - numtrue
    false_pos = []
    false_neg = []
    true_pos = []
    true_neg = []

    # Initially (threshold = -infinity) reject everything. There are no false positive,
    # but all numtrue positives are falsely rejected
    false_pos.append(0)
    false_neg.append(numtrue)
    true_pos.append(0)
    true_neg.append(numfalse)
    

    # At threshold = score[0], we accept the zeroth instance.
    # If the true label is 1, we get a true positive, so false rejection is reduced by 1.
    # Otherwise we get a false positive.
    if (labels[idx[0]] == 1):
        false_neg[0] = false_neg[0] - 1
        true_pos[0] = 1
#         false_pos[0] = false_pos[0] + 1
#         true_neg[0] -= 1
    else:
        false_pos[0] = 1
        
    auc = 0

    # At threshold = score[i], we accept the i-th instance. If the true label of the i-th
    # instance is 1, we get one lesser false rejection than when the threshold is score[i-1].
    # Otherwise, we get one more false positive than when the threshold is score[i-1].
    for i in range(1, len(idx)):
        if (labels[idx[i]] == 1):
            false_pos.append(false_pos[i-1])
            true_neg.append(true_neg[i-1])
            
            false_neg.append(false_neg[i-1] - 1)
            true_pos.append(true_pos[i-1] + 1)
            
            auc = auc + false_pos[i]
        else:
            false_neg.append(false_neg[i-1])
            true_pos.append(true_pos[i-1])
            
            false_pos.append(false_pos[i-1] + 1)
            true_neg.append(true_neg[i-1] - 1)
    
   # Compute the area under the curve using the wilcoxon-mann-whitney formula
    if (not numtrue <= 0):
        auc = auc / (numtrue * numfalse)
    else:
        auc = auc / (numfalse)
    # Normalize FP and FR to compute rates.  The constant added to the denominator avoids division by 0

    for i in range(0, len(idx)):
        false_pos[i] = false_pos[i] / (numfalse + 1e-30)
        true_neg[i] = true_neg[i] / (numfalse + 1e-30)
        
        false_neg[i] = false_neg[i] / (numtrue + 1e-30)
        true_pos[i] = true_pos[i] / (numtrue + 1e-30)
        
    return (false_neg,false_pos, true_neg, true_pos, auc,idx)

def calc_precision_recall(false_neg,false_pos, true_neg, true_pos):
    nums = len(false_neg)
    prec, rec = [], []
    for i in range(nums):
        p = (true_pos[i] / (true_pos[i] + false_pos[i]))
        r = (true_pos[i] / (true_pos[i] + false_neg[i]))
        prec.append(p)
        rec.append(r)
    return prec, rec

def plotting(y_true, y_pred, label=None):
    fpr, tpr, thresholds = roc_curve(y_true, y_pred)
    false_neg,false_pos, true_neg, true_pos, auc_val,idx = computeDET(y_true, y_pred)
    roc_auc = auc(fpr, tpr)
    gen_roc_curve(fpr, tpr, roc_auc, label)
    
    prec, rec, thresholds = precision_recall_curve(y_true, y_pred)
    gen_prec_recall_curve(rec, prec, label)
    prec, rec = calc_precision_recall(false_neg,false_pos, true_neg, true_pos)
    print('AUC = ', roc_auc)

