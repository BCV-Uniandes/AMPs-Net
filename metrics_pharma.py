import torch.nn as nn
import numpy as np
import sklearn.metrics as metrics
import pdb


import warnings

warnings.filterwarnings("ignore")
import numpy as np

from itertools import cycle
from sklearn.utils.extmath import stable_cumsum
from sklearn.preprocessing import label_binarize
from sklearn.metrics import (
    precision_recall_curve,
    average_precision_score,
    roc_curve,
    roc_auc_score,
    auc,
)

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pylab as plt
import matplotlib.pyplot as plt
import os

def norm_ap(output, target, num_classes=102):

    assert num_classes > 2, 'Binary classification not supported by this implementation'

    F1_T = []
    N_total = len(output) / num_classes
    area_t = []

    for clas in range(0, num_classes):
        area = 0
        R_total = []
        P_total = []
        F1_total = []
        for thr in np.arange(0, 1.0001, 0.0001):
            groundtruth = target[:,clas]
            #groundtruth = (target == clas).astype(int)
            predicted_thr = (output[:, clas] >= thr).astype(int)
            TP = np.sum((predicted_thr == 1) & (groundtruth == 1))
            FN = np.sum((predicted_thr == 0) & (groundtruth == 1))
            FP = np.sum((predicted_thr == 1) & (groundtruth == 0))

            # Recall
            Recall = TP / (TP + FN)

            # Normalized Precision
            if Recall == 0:
                Precision = 0
            else:
                Precision = (Recall * N_total) / ((Recall * N_total) + FP)
            # F-measure
            denom = Precision + Recall
            if denom == 0:
                denom = 1
            F1 = 2 * (Precision * Recall) / denom
            R_total.append(Recall)
            P_total.append(Precision)
            F1_total.append(F1)
        # Compute Area under Normalized Precision Recall curve
        area = metrics.auc(R_total,P_total)


        # F-measure
        F1_T.append(np.max(F1_total))
        area_t.append(area)
    # Compute Area under NAP curve
    area_under_curve = [0, 1]
    area_under_curve += area_t
    area_under_curve = sorted(area_under_curve)
    nap_area = 0.0
    scores = np.arange(num_classes, -1, -1)
    score = np.insert(scores, 0, num_classes)

    area = metrics.auc(score,area_under_curve)
    

    F1_T.append(np.mean(F1_T))

    # Final value for area under NAP curve divided by the number of classes.
    area_t.append(area / num_classes)
    return area_t[-1], F1_T[-1]
    #return area_t

def pltmap_bin(output, target, num_classes=2):

    assert num_classes == 2, 'Multiclass classification not supported by this implementation'

    """ Compute Average Precision """

    # A "micro-average": quantifying score on all classes jointly
    precision, recall, _ = precision_recall_curve(target, output[:,1])
    average_precision = average_precision_score(target, output[:,1], average="macro")
    denom = precision + recall 
    denom[denom == 0.] = 1
    fmeasure = np.max(2*precision*recall/ denom)

    return average_precision, fmeasure

def pltauc(output, target, num_classes):
    
    assert num_classes > 2, 'Binary classification not supported by this implementation'
    

    new_labels = label_binarize(target, classes=list(range(0, num_classes)))
    n_classes = new_labels.shape[1]
    fpr = dict()
    tpr = dict()
    roc_auc = dict()

    for i in range(n_classes):
        fpr[i], tpr[i], _ = roc_curve(new_labels[:, i], output[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])

    # A "micro-average": quantifying score on all classes jointly
    fpr["micro"], tpr["micro"], _ = roc_curve(new_labels.ravel(), output.ravel())
    roc_auc["micro"] = roc_auc_score(new_labels, output, average="macro")

    return roc_auc["micro"]

def plotbinauc(output, target, num_classes=2):
    
    assert num_classes == 2, 'Binary classification not supported by this implementation'
    

    """ Compute Area Under Curve """

    # A "micro-average": quantifying score on all classes jointly
    roc_auc = roc_auc_score(target, output[:,1], average="macro")
        
    return roc_auc 

def pltmap(output, target, num_classes):
    
    assert num_classes > 2, 'Binary classification not supported by this implementation'
    

    new_labels = label_binarize(target, classes=list(range(0, num_classes)))
    n_classes = new_labels.shape[1]
    precision = dict()
    recall = dict()
    average_precision = dict()
    fmeasure = dict()

    for i in range(n_classes):
        precision[i], recall[i], _ = precision_recall_curve(
            new_labels[:, i], output[:, i]
        )
        average_precision[i] = average_precision_score(new_labels[:, i], output[:, i])
        denom = precision[i] + recall[i]
        denom[denom == 0.0] = 1
        fmeasure[i] = np.max(2 * precision[i] * recall[i] / denom)

    # A "micro-average": quantifying score on all classes jointly
    precision["micro"], recall["micro"], _ = precision_recall_curve(
        new_labels.ravel(), output.ravel()
    )
    average_precision["micro"] = average_precision_score(
        new_labels, output, average="macro"
    )
    denom = precision["micro"] + recall["micro"]
    denom[denom == 0.0] = 1
    fmeasure["micro"] = np.max(2 * precision["micro"] * recall["micro"] / denom)

    return average_precision, fmeasure


def plot_nap(train_nap, val_nap, save_dir,
              num_epoch, save=True, show=False):
    fig, ax = plt.subplots()
    ax.set_xlim(0,num_epoch)
    ax.set_ylim(min(np.min(train_nap),np.min(val_nap)),max(np.max(train_nap),np.max(val_nap))*1.1)
    plt.xlabel('Epoch {0}'.format(num_epoch))
    plt.ylabel('NAP values')
    plt.plot(train_nap, label='Train')
    plt.plot(val_nap, label='Validation')
    plt.legend()
    
    # Save figure
    if save:
        if not os.path.exists(save_dir):
            os.mkdir(save_dir)
        save_fn = save_dir + '/NAP.png'
        plt.savefig(save_fn)
    if show:
        plt.show()
    else:
        plt.close()
        
        
def plot_loss(train_losses, val_losses, save_dir,
              num_epoch, save=True, show=False):
    fig, ax = plt.subplots()
    ax.set_xlim(0,num_epoch)
    ax.set_ylim(min(np.min(train_losses),np.min(val_losses)),max(np.max(val_losses),np.max(train_losses))*1.1)
    plt.xlabel('Epoch')
    plt.ylabel('Loss values')
    plt.plot(train_losses, label='Train')
    plt.plot(val_losses, label='Validation')
    plt.legend()
    
    # Save figure
    if save:
        if not os.path.exists(save_dir):
            os.mkdir(save_dir)
        save_fn = save_dir + '/Losses.png'
        plt.savefig(save_fn)
    if show:
        plt.show()
    else:
        plt.close()
        
