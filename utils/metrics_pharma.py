import numpy as np
import warnings
warnings.filterwarnings("ignore")

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
import sklearn.metrics as metrics


def norm_ap_optimized_binary(output,target, num_classes = 2):
    
    assert num_classes == 2, 'Multiclass classification not supported by this implementation'
    
    """ Compute Normalized Average Precision in an optimized manner """
    
    F1_T = []
    N_total = np.sum(target)
    area_t = []

    area = 0
    y_true = (target==1).astype(int)
    y_score = output[:,1]
    desc_score_indices = np.argsort(y_score, kind="mergesort")[::-1]
    y_score = y_score[desc_score_indices]
    y_true = y_true[desc_score_indices]
    distinct_value_indices = np.where(np.diff(y_score))[0]
    threshold_idxs = np.r_[distinct_value_indices, y_true.size - 1]
    TP = stable_cumsum(y_true)[threshold_idxs]  
    FP = stable_cumsum((1 - y_true))[threshold_idxs]
    TN = [FP[-1] for x in range(0,len(FP))]
    TN = np.subtract(TN,FP)
    FN = [TP[-1] for x in range(0,len(TP))]
    FN = np.subtract(FN,TP)     
    Fc = np.add(FP,FN) 
    Recall = np.divide(TP,np.add(TP,FN))
    Precision = np.divide(np.multiply(Recall,N_total),np.add(np.multiply(Recall,N_total),FP))
    denom = np.add(Precision,Recall)
    denom[denom == 0] = 1
    F1= np.divide(np.multiply(2,np.multiply(Precision,Recall)),denom)
    area = metrics.auc(Recall,Precision)
    F1_T = np.max(F1)
    
    return area, F1_T

def norm_ap_binary(output,groundtruth, num_classes = 2):

    assert num_classes == 2, 'Multiclass classification not supported by this implementation'

    """ Compute Normalized Average Precision """
    N_total = np.sum(groundtruth)
    #N_total = len(output)/num_classes
    area = 0
    R_total = [1.0001]
    #R_total = []
    #P_total = []
    P_total = [0]
    F1_total = []
    for thr in np.arange(0,1,0.0001):
    #thr= 0.5        
        predicted_thr = (output[:,1]>=thr).astype(int)
        TP = np.sum((predicted_thr == 1) & (groundtruth ==1))
        FN = np.sum((predicted_thr == 0) & (groundtruth ==1))
        FP = np.sum((predicted_thr == 1) & (groundtruth ==0))
        Recall = TP/(TP+FN)
        Fc = np.sum(predicted_thr != groundtruth)
        if (Recall*N_total)+FP !=0:
            Precision = (Recall * N_total)/((Recall*N_total)+FP)
        else:
            Precision = 0
        # To compute F measure
        denom = Precision + Recall
        if denom == 0:
            denom = 1
        F1= 2*(Precision*Recall)/denom
        R_total.append(Recall)
        P_total.append(Precision)
        F1_total.append(F1)
    R_total.append(0)
    P_total.append(1)

    area = metrics.auc(R_total,P_total)
    return area, np.max(F1_total)


def norm_ap_optimized(output, target, num_classes=102):

    assert num_classes > 2, 'Binary classification not supported by this implementation'

    F1_T = []
    N_total = len(output) / (max(target) + 1)
    area_t = []
    FP_Totales = 0
    FN_Totales = 0
    for clas in range(0, num_classes):
        area = 0
        y_true = (target == clas).astype(int)
        y_score = output[:, clas]
        desc_score_indices = np.argsort(y_score, kind="mergesort")[::-1]
        y_score = y_score[desc_score_indices]
        y_true = y_true[desc_score_indices]
        distinct_value_indices = np.where(np.diff(y_score))[0]
        threshold_idxs = np.r_[distinct_value_indices, y_true.size - 1]
        TP = stable_cumsum(y_true)[threshold_idxs]
        FP = stable_cumsum((1 - y_true))[threshold_idxs]
        TN = [FP[-1] for x in range(0, len(FP))]
        TN = np.subtract(TN, FP)
        FN = [TP[-1] for x in range(0, len(TP))]
        FN = np.subtract(FN, TP)
        FP_Totales+=sum(FP)
        FN_Totales+=sum(FN)
        # Recall
        Recall = np.divide(TP, np.add(TP, FN))

        # Normalized Precision
        Precision = np.divide(
            np.multiply(Recall, N_total), np.add(np.multiply(Recall, N_total), FP)
        )

        denom = np.add(Precision, Recall)
        denom[denom == 0] = 1

        # F-measure
        F1 = np.divide(np.multiply(2, np.multiply(Precision, Recall)), denom)

        # Compute Area under Normalized Precision Recall curve
        area = metrics.auc(Recall,Precision)
        F1_T.append(np.max(F1))
        area_t.append(area)

    # Compute Area under NAP curve
    area_under_curve = [0,1]
    area_under_curve += area_t
    area_under_curve = sorted(area_under_curve)
    nap_area = 0.0
    scores = np.arange(num_classes, -1, -1)
    score = np.insert(scores, 0, num_classes)
    nap_area = metrics.auc(score, area_under_curve)
    # For same size
    F1_T.append(np.mean(F1_T))

    # Final value for area under NAP curve divided by the number of classes.
    area_t.append(nap_area / num_classes)

    return area_t[-1], F1_T[-1]
    #return area_t, FP_Totales, FN_Totales

def norm_ap(output, target, num_classes=102):

    assert num_classes > 2, 'Binary classification not supported by this implementation'

    F1_T = []
    N_total = len(output) / (max(target) + 1)
    area_t = []

    for clas in range(0, num_classes):
        area = 0
        R_total = []
        P_total = []
        F1_total = []
        for thr in np.arange(0, 1.0001, 0.0001):
            groundtruth = (target == clas).astype(int)
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
    nap_area = metrics.auc(score, area_under_curve)

    F1_T.append(np.mean(F1_T))

    # Final value for area under NAP curve divided by the number of classes.
    area_t.append(nap_area / num_classes)
    return area_t[-1], F1_T[-1]
    #return area_t

#! IMPLEMENTAR MAP BINARIO
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
        
def plot_FP_FN(d_losses, g_losses, save_dir='/data/pruiz/DEEPER_GCN/Molecules-Graphs/deep_gcns_torch-master/examples/ogb/dude_dataset/log/Baseline_All_Epochs/Fold2',
              num_epoch=150, save=True, show=False):
    fig, ax = plt.subplots()
    ax.set_xlim(0,num_epoch)
    ax.set_ylim(min(np.min(d_losses),np.min(g_losses)), max(np.max(d_losses),np.max(g_losses)))
    plt.xlabel('Epoch {0}'.format(num_epoch))
    plt.ylabel('Number of FP/FN')
    plt.plot(d_losses, label='FP')
    plt.plot(g_losses, label='FN')
    plt.legend()
    
    # Save figure
    if save:
        if not os.path.exists(save_dir):
            os.mkdir(save_dir)
        save_fn = save_dir + 'losses_{:d}_FP_FN'.format(num_epoch) + '.png'
        plt.savefig(save_fn)
    if show:
        plt.show()
    else:
        plt.close()