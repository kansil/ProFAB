import numpy as np
import copy
from math import sqrt
from scipy import stats
from sklearn import preprocessing, metrics
from sklearn.metrics import confusion_matrix
import torch 

def get_cindex(Y, P):
    summ = 0
    pair = 0

    for i in range(1, len(Y)):
        for j in range(0, i):
            if i is not j:
                if (Y[i] > Y[j]):
                    pair += 1
                    summ += 1 * (P[i] > P[j]) + 0.5 * (P[i] == P[j])

    if pair is not 0:
        return summ / pair
    else:
        return 0


def r_squared_error(y_obs, y_pred):
    y_obs = np.array(y_obs)
    y_pred = np.array(y_pred)
    y_obs_mean = [np.mean(y_obs) for y in y_obs]
    y_pred_mean = [np.mean(y_pred) for y in y_pred]

    mult = sum((y_pred - y_pred_mean) * (y_obs - y_obs_mean))
    mult = mult * mult

    y_obs_sq = sum((y_obs - y_obs_mean) * (y_obs - y_obs_mean))
    y_pred_sq = sum((y_pred - y_pred_mean) * (y_pred - y_pred_mean))

    return mult / float(y_obs_sq * y_pred_sq)


def get_k(y_obs, y_pred):
    y_obs = np.array(y_obs)
    y_pred = np.array(y_pred)

    return sum(y_obs * y_pred) / float(sum(y_pred * y_pred))


def squared_error_zero(y_obs, y_pred):
    k = get_k(y_obs, y_pred)

    y_obs = np.array(y_obs)
    y_pred = np.array(y_pred)
    y_obs_mean = [np.mean(y_obs) for y in y_obs]
    upp = sum((y_obs - (k * y_pred)) * (y_obs - (k * y_pred)))
    down = sum((y_obs - y_obs_mean) * (y_obs - y_obs_mean))

    return 1 - (upp / float(down))


def get_rm2(ys_orig, ys_line):
    r2 = r_squared_error(ys_orig, ys_line)
    r02 = squared_error_zero(ys_orig, ys_line)

    return r2 * (1 - np.sqrt(np.absolute((r2 * r2) - (r02 * r02))))


"""
@author: Anna Cichonska
"""



def mse(y,f):
    """
    Task:    To compute root mean squared error (RMSE)

    Input:   y      Vector with original labels (pKd [M])
             f      Vector with predicted labels (pKd [M])

    Output:  mse   MSE
    """

    mse = ((y - f)**2).mean(axis=0)

    return mse

def rmse(y,f):
    """
    Task:    To compute root mean squared error (RMSE)

    Input:   y      Vector with original labels (pKd [M])
             f      Vector with predicted labels (pKd [M])

    Output:  rmse   RSME
    """

    rmse = sqrt(((y - f)**2).mean(axis=0))

    return rmse


def pearson(y,f):
    """
    Task:    To compute Pearson correlation coefficient

    Input:   y      Vector with original labels (pKd [M])
             f      Vector with predicted labels (pKd [M])

    Output:  rp     Pearson correlation coefficient
    """

    rp = np.corrcoef(y, f)[0,1]

    return rp


def spearman(y,f):
    """
    Task:    To compute Spearman's rank correlation coefficient

    Input:   y      Vector with original labels (pKd [M])
             f      Vector with predicted labels (pKd [M])

    Output:  rs     Spearman's rank correlation coefficient
    """

    rs = stats.spearmanr(y, f)[0]

    return rs

def calc_f(cm):#ok

    #maF = metrics.f1_score(y,f,labels = np.unique(y),average = 'macro')
    #miF = metrics.f1_score(y,f,labels = np.unique(y),average = 'micro')
    
    miF = (cm[:,1,1].sum())/((cm[:,1,1].sum()) + 0.5*((cm[:,0,1].sum())+(cm[:,1,0].sum())))
    
    
    #macro_f: avg of F
    maF = sum((cm[:,1,1])/((cm[:,1,1]) + 0.5*((cm[:,0,1])+(cm[:,1,0]))))/len(cm)

    return maF,miF

def calc_precision(cm):#ok
 
    # maP = metrics.precision_score(y,f,labels = np.unique(y),average = 'macro')
    # miP = metrics.precision_score(y,f,labels = np.unique(y),average = 'micro')
    
    miP = cm[:,1,1].sum()/(cm[:,1,1].sum() + cm[:,0,1].sum())
    
    maP = sum(cm[:,1,1]/(cm[:,1,1] + cm[:,0,1]))/len(cm)
    
    return maP,miP
def calc_recall(cm):#ok
    # maR = metrics.recall_score(y,f,labels = np.unique(y),average = 'macro')
    # miR = metrics.recall_score(y,f,labels = np.unique(y),average = 'micro')
    
    
    miR = cm[:,1,1].sum()/(cm[:,1,1].sum() + cm[:,1,0].sum())
    
    maR = sum(cm[:,1,1]/(cm[:,1,1] + cm[:,1,0]))/len(cm)
    
    return maR,miR

def calc_f05(cm):#ok

    beta = 0.5
    miF5 = (1+beta**2)*cm[:,1,1].sum()/(beta**2*(cm[:,1,1].sum()+cm[:,1,0].sum())+ cm[:,0,1].sum()+cm[:,1,1].sum())
    
    maF5 = sum((1+beta**2)*cm[:,1,1]/(beta**2*(cm[:,1,1]+cm[:,1,0])+ cm[:,0,1]+cm[:,1,1]))/len(cm)
    
    #macro_f: avg of F
    #maF = sum((1+beta**2)*cm[:,1,1]/(beta**2 * cm[:,1,1] + cm[:,0,1]+cm[:,1,0]))/len(cm)

    # maF5= metrics.fbeta_score(y,f,labels = np.unique(y),average = 'macro',beta = 0.5)
    # miF5 = metrics.fbeta_score(y,f,labels = np.unique(y),average = 'micro',beta = 0.5)
    
    return maF5,miF5

def calc_auc(y,f):#ok
    maRoc= metrics.roc_auc_score(y,f,labels = np.unique(y),average = 'macro')
    miRoc = metrics.roc_auc_score(y,f,labels = np.unique(y),average = 'micro')
    
    return maRoc,miRoc

def calc_auprc(y,f):#nok
    auprc = []
    for i in range(y.shape[-1]):
        precision, recall, thresholds = metrics.precision_recall_curve(y[:,i], f[:,i], pos_label=1)
        auprc.append(metrics.auc(recall, precision))
    maPRC = sum(auprc)/y.shape[-1]
    return maPRC
    
def calc_mcc(cm):#nok

    
    maMCC = sum((cm[:,1,1] * cm[:,0,0] - cm[:,0,1] * cm[:,1,0])/
            ((cm[:,1,1] + cm[:,0,1]) * (cm[:,1,1] + cm[:,1,0]) * (cm[:,0,0] + cm[:,0,1]) * (cm[:,0,0] + cm[:,1,0]))**0.5)/len(cm)
    
    
    miMCC = (cm[:,1,1].sum() * cm[:,0,0].sum() - cm[:,0,1].sum() * cm[:,1,0].sum())/(
        (cm[:,1,1].sum() + cm[:,0,1].sum()) * (cm[:,1,1].sum() + cm[:,1,0].sum()) * (cm[:,0,0].sum() + cm[:,0,1].sum()) * (cm[:,0,0].sum() + cm[:,1,0].sum()))**0.5
        
    return maMCC,miMCC
    
def calc_acc(cm):#nok
    
    maAcc = sum((cm[:,1,1] + cm[:,0,0])/(cm[:,1,0] + cm[:,0,0] + cm[:,1,1] + cm[:,0,1]))/len(cm)
    miAcc = (cm[:,1,1].sum() + cm[:,0,0].sum())/(cm[:,1,0].sum() + cm[:,0,0].sum() + cm[:,1,1].sum() + cm[:,0,1].sum())
    
    return maAcc,miAcc
    
def cl_prec_rec_f1_acc_mcc_multilabel(y_true, y_pred):
    cm = metrics.multilabel_confusion_matrix(y_true,y_pred)
    performance_threshold_dict = {}
    performance_threshold_dict['Macro_Precision'],performance_threshold_dict['Micro_Precision'] = calc_precision(cm)
    performance_threshold_dict['Macro_Recall'],performance_threshold_dict['Micro_Recall'] = calc_recall(cm)
    performance_threshold_dict['Macro_F1_Score'],performance_threshold_dict['Micro_F1_Score'] = calc_f(cm)
    performance_threshold_dict['Macro_F05_Score'],performance_threshold_dict['Micro_F05_Score'] = calc_f05(cm)#y_true,y_pred)
    performance_threshold_dict['Macro_AUPRC'] = calc_auprc(y_true,y_pred)
    performance_threshold_dict['Macro_AUC'],performance_threshold_dict['Micro_AUC'] = calc_auc(y_true,y_pred)
    

    print('confusuion matrix:\n',cm)
    performance_threshold_dict['Macro_Accuracy'],performance_threshold_dict['Micro_Accuracy'] = calc_acc(cm)
    #print('defrgdg')
    performance_threshold_dict['Macro_MCC'],performance_threshold_dict['Micro_MCC'] = calc_mcc(cm)
    #print('bbbbbbb neden neeeeeeeeddddeeeee') 
    
    return performance_threshold_dict


def cl_prec_rec_f1_acc_mcc(y_true, y_pred):
    """
    Task:    To compute F1 score using the threshold of 7 M
             to binarize pKd's into true class labels.

    Input:   y      Vector with original labels (pKd [M])
             f      Vector with predicted labels (pKd [M])

    Output:  f1     F1 score
    """
    performance_threshold_dict  = dict()
    precision = metrics.precision_score(y_true, y_pred)
    recall = metrics.recall_score(y_true, y_pred)
    f1_score = metrics.f1_score(y_true, y_pred)
    accuracy = metrics.accuracy_score(y_true, y_pred)
    mcc = metrics.matthews_corrcoef(y_true, y_pred)
    f05_score = 1.25*precision*recall/(0.25*precision+recall)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    auc = classif_AUC(y_true,y_pred)
    auprc = classif_AUPRC(y_true, y_pred)
    
    performance_threshold_dict["Precision"] = precision
    performance_threshold_dict["Recall"] = recall
    performance_threshold_dict["F1-Score"] = f1_score
    performance_threshold_dict["F05-Score"] = f05_score
    performance_threshold_dict["Accuracy"] = accuracy
    performance_threshold_dict["MCC"] = mcc
    performance_threshold_dict["AUC"] = auc
    performance_threshold_dict["AUPRC"] = auprc
    performance_threshold_dict["TP"] = tp
    performance_threshold_dict["FP"] = fp
    performance_threshold_dict["TN"] = tn
    performance_threshold_dict["FN"] = fn
    
    return performance_threshold_dict


def reg_prec_rec_f1_acc_mcc(y,f):
    """
    Task:    To compute F1 score using the threshold of 7 M
             to binarize pKd's into true class labels.

    Input:   y      Vector with original labels (pKd [M])
             f      Vector with predicted labels (pKd [M])

    Output:  f1     F1 score
    """
    # 10 uM, 1 uM, 100 nM
    str_threshold_lst = ["10uM", "1uM", "100nM", "30nM"]
    threshold_lst = [5.0, 6.0, 7.0, 7.522878745280337]
    dict_threshold = {str_threshold_lst[0]:threshold_lst[0] ,str_threshold_lst[1]:threshold_lst[1],
                      str_threshold_lst[2]:threshold_lst[2], str_threshold_lst[3]:threshold_lst[3]}

    performance_threshold_dict = dict()
    for str_thre, threshold in dict_threshold.items():
        y_binary = copy.deepcopy(y)
        y_binary = preprocessing.binarize(y_binary.reshape(1,-1), threshold, copy=False)[0]
        f_binary = copy.deepcopy(f)
        f_binary = preprocessing.binarize(f_binary.reshape(1,-1), threshold, copy=False)[0]
        precision = metrics.precision_score(y_binary, f_binary)
        recall = metrics.recall_score(y_binary, f_binary)
        f1_score = metrics.f1_score(y_binary, f_binary)
        accuracy = metrics.accuracy_score(y_binary, f_binary)
        mcc = metrics.matthews_corrcoef(y_binary, f_binary)
        performance_threshold_dict["Precision {}".format(str_thre)] = precision
        performance_threshold_dict["Recall {}".format(str_thre)] = recall
        performance_threshold_dict["F1-Score {}".format(str_thre)] = f1_score
        performance_threshold_dict["Accuracy {}".format(str_thre)] = accuracy
        performance_threshold_dict["MCC {}".format(str_thre)] = mcc

    return performance_threshold_dict

def classif_AUC(y,f):
    
    """
    Task:    To compute average area under the ROC curves (AUC)

    Input:   y      Vector with original labels (pKd [M])
             f      Vector with predicted labels (pKd [M])

    Output:  avAUC   average AUC

    """
    fpr, tpr, thresholds = metrics.roc_curve(y, f, pos_label=1)
    auc = metrics.auc(fpr, tpr)
    return auc
    
def classif_AUPRC(y,f):
    
    precision, recall, thresholds = metrics.precision_recall_curve(y, f, pos_label=1)
    auprc = metrics.auc(recall, precision)


    return auprc
 

def average_AUC(y,f):

    """
    Task:    To compute average area under the ROC curves (AUC) given ten
             interaction threshold values from the pKd interval [6 M, 8 M]
             to binarize pKd's into true class labels.

    Input:   y      Vector with original labels (pKd [M])
             f      Vector with predicted labels (pKd [M])

    Output:  avAUC   average AUC

    """

    thr = np.linspace(6,8,10)
    auc = np.empty(np.shape(thr)); auc[:] = np.nan

    for i in range(len(thr)):
        y_binary = copy.deepcopy(y)
        y_binary = preprocessing.binarize(y_binary.reshape(1,-1), threshold=thr[i], copy=False)[0]
        fpr, tpr, thresholds = metrics.roc_curve(y_binary, f, pos_label=1)
        auc[i] = metrics.auc(fpr, tpr)

    avAUC = np.mean(auc)

    return avAUC

def average_AUPRC(y,f):

    """
    Task:    To compute average area under the ROC curves (AUC) given ten
             interaction threshold values from the pKd interval [6 M, 8 M]
             to binarize pKd's into true class labels.

    Input:   y      Vector with original labels (pKd [M])
             f      Vector with predicted labels (pKd [M])

    Output:  avAUC   average AUC

    """

    thr = np.linspace(6,8,10)
    auc = np.empty(np.shape(thr)); auc[:] = np.nan

    for i in range(len(thr)):
        y_binary = copy.deepcopy(y)
        y_binary = preprocessing.binarize(y_binary.reshape(1,-1), threshold=thr[i], copy=False)[0]
        precision, recall, thresholds = metrics.precision_recall_curve(y_binary, f, pos_label=1)
        auc[i] = metrics.auc(recall, precision)

    avAUC = np.mean(auc)

    return avAUC



def get_list_of_scores():
    score_list = ["rm2", "CI", "MSE", "RMSE", "Pearson", "Spearman",
                  "Average AUC", "Average AUPRC",
                  "Precision 10uM", "Recall 10uM", "F1-Score 10uM", "Accuracy 10uM", "MCC 10uM",
                  "Precision 1uM", "Recall 1uM", "F1-Score 1uM", "Accuracy 1uM", "MCC 1uM",
                  "Precision 100nM", "Recall 100nM", "F1-Score 100nM", "Accuracy 100nM", "MCC 100nM",
                  "Precision 30nM", "Recall 30nM", "F1-Score 30nM", "Accuracy 30nM", "MCC 30nM",
                  ]
    return score_list


def get_validation_test_metric_list_of_scores():
    score_list =  get_list_of_scores()
    test_score_list = ["test {}".format(scr) for scr in score_list]
    validation_score_list = ["validation {}".format(scr) for scr in score_list]
    validation_test_metric_list = test_score_list + validation_score_list
    # print(validation_test_list)
    return validation_test_metric_list

def get_scores_generic(labels, predictions, validation_test, print_scores=False):

    score_dict = {"rm2": None, "CI": None, "RMSE": None, "MSE": None, "Pearson": None,
                  "Spearman": None, "Average AUC": None, "Average AUPRC":None,
                  "Precision 10uM": None, "Recall 10uM": None, "F1-Score 10uM": None, "Accuracy 10uM": None, "MCC 10uM": None,
                  "Precision 1uM": None, "Recall 1uM": None, "F1-Score 1uM": None, "Accuracy 1uM": None, "MCC 1uM": None,
                  "Precision 100nM": None, "Recall 100nM": None, "F1-Score 100nM": None, "Accuracy 100nM": None, "MCC 100nM": None,
                  "Precision 30nM": None, "Recall 30nM": None, "F1-Score 30nM": None, "Accuracy 30nM": None, "MCC 30nM": None,}
    score_list = get_list_of_scores()

    score_dict["rm2"] = get_rm2(np.asarray(labels), np.asarray(
        predictions))
    score_dict["CI"] = get_cindex(np.asarray(labels), np.asarray(
        predictions))
    score_dict["MSE"] = mse(np.asarray(labels), np.asarray(
        predictions))
    score_dict["RMSE"] = rmse(np.asarray(labels), np.asarray(
        predictions))
    score_dict["Pearson"] = pearson(np.asarray(labels), np.asarray(predictions))
    score_dict["Spearman"] = spearman(np.asarray(labels), np.asarray(predictions))
    score_dict["Average AUC"] = average_AUC(np.asarray(labels), np.asarray(predictions))
    score_dict["Average AUPRC"] = average_AUPRC(np.asarray(labels), np.asarray(predictions))

    prec_rec_f1_acc_mcc_threshold_dict = reg_prec_rec_f1_acc_mcc(np.asarray(labels), np.asarray(predictions))
    for key in prec_rec_f1_acc_mcc_threshold_dict.keys():
        score_dict[key] = prec_rec_f1_acc_mcc_threshold_dict[key]

    if print_scores:
        for scr in score_list:
            print("{} {}:\t{}".format(validation_test, scr, score_dict[scr]))
    return score_dict

def get_scores(labels, predictions, validation_test, total_training_loss, total_validation_test_loss, epoch, fold_epoch_results, print_scores=False, fold=None):

    score_dict = {"rm2": None, "CI": None, "MSE": None, "Pearson": None,
                  "Spearman": None, "Average AUC": None, "Average AUPRC":None,
                  "Precision 10uM": None, "Recall 10uM": None, "F1-Score 10uM": None, "Accuracy 10uM": None, "MCC 10uM": None,
                  "Precision 1uM": None, "Recall 1uM": None, "F1-Score 1uM": None, "Accuracy 1uM": None, "MCC 1uM": None,
                  "Precision 100nM": None, "Recall 100nM": None, "F1-Score 100nM": None, "Accuracy 100nM": None, "MCC 100nM": None,
                  "Precision 30nM": None, "Recall 30nM": None, "F1-Score 30nM": None, "Accuracy 30nM": None, "MCC 30nM": None,}
    score_list = get_list_of_scores()

    score_dict["rm2"] = get_rm2(np.asarray(labels), np.asarray(
        predictions))
    score_dict["CI"] = get_cindex(np.asarray(labels), np.asarray(
        predictions))
    score_dict["MSE"] = mse(np.asarray(labels), np.asarray(
        predictions))
    score_dict["RMSE"] = rmse(np.asarray(labels), np.asarray(
        predictions))
    score_dict["Pearson"] = pearson(np.asarray(labels), np.asarray(predictions))
    score_dict["Spearman"] = spearman(np.asarray(labels), np.asarray(predictions))
    score_dict["Average AUC"] = average_AUC(np.asarray(labels), np.asarray(predictions))
    score_dict["Average AUPRC"] = average_AUPRC(np.asarray(labels), np.asarray(predictions))

    prec_rec_f1_acc_mcc_threshold_dict = reg_prec_rec_f1_acc_mcc(np.asarray(labels), np.asarray(predictions))
    for key in prec_rec_f1_acc_mcc_threshold_dict.keys():
        score_dict[key] = prec_rec_f1_acc_mcc_threshold_dict[key]

    if print_scores:
        if fold!=None:
            fold_epoch_results[-1].append(score_dict)
            print("Fold:{}\tEpoch:{}\tTraining Loss:{}\t{} Loss:{}".format(fold + 1, epoch, total_training_loss,
                                                                           validation_test, total_validation_test_loss))
        else:
            fold_epoch_results.append(score_dict)
            print("Epoch:{}\tTraining Loss:{}\t{} Loss:{}".format(epoch, total_training_loss, validation_test,
                                                                  total_validation_test_loss))
        for scr in score_list:
            print("{} {}:\t{}".format(validation_test, scr, score_dict[scr]))
    return score_dict

def evaluate_score(model,X,y, preds = False, learning_method = 'classif', isDeep = False):
    
    """
    Description:
        Predict new labels and evaluate scoring metrics.
        
    Parameters
        model: model type used to predict new label
        X: feature matrix
        y: label matrix
        preds: {bool}, default = False, if True, function returns predicted labels, too
        learning_method: {string}, {'binary','multilabel','rgr'} default = 'binary', return scoring metric according
                        to learning method
        isDeep: {bool}, default = False, If True, model is evaluated with torch.no_grad()
    Returns
        Scores: {dict}, recall, precision, f1, acc, f 0.5, mcc scores
        f: {numpy array}, predicted label
    """
    if isDeep:
        if isinstance(y[0],int):#y.shape[-1] == 1:
            sgm = torch.nn.Sigmoid()
        
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        
            if isinstance(X,np.ndarray) or isinstance(X,list):
            
                X = torch.tensor(X).to(device)
            #y = torch.tensor(y).to(device)
        
            if len(X.size()) == 1:
                X = X.unsqueeze(0).unsqueeze(0).float()
            elif len(X.size()) == 2:
                X = X.unsqueeze(1).float() 
            model.eval()
            with torch.no_grad():
                pred = model(X)
                pred = sgm(pred)
            f = np.where(pred.cpu().detach().numpy()<0.5,0,1)
        else:
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

            if isinstance(X,np.ndarray) or isinstance(X,list):
                X = torch.tensor(X).to(device)
            if len(X.size()) == 1:
                X = X.unsqueeze(0).unsqueeze(0).float()
            elif len(X.size()) == 2:
                X = X.unsqueeze(1).float()
            model.eval()
            with torch.no_grad():

                #pred = model(X)
                f = []
                for x in X:
                    x = x.unsqueeze(0)
                    #with torch.no_grad():
                    pred = model(x)
                    sf = np.where(pred.cpu().detach().numpy()<0.5,0,1)
                    #print(sf)
                    f.append(sf[:])
                f = np.array(f)
                f = f.squeeze()  
            f = np.where(f<0.5,0,1)
            #for i in range(len(f)):
            #    print(pred[i],' ',f[i],' ',y[i])
        #print(type(f))
        #print(f)
    else:
        f = model.predict(X)
    
    if learning_method == 'rgr':
        a = mse(y,f)
        b = rmse(y,f)
        c = spearman(y,f)
        d = pearson(y,f)
        e = average_AUC(y,f)
        g = reg_prec_rec_f1_acc_mcc(y,f)
        
        Scores = {'MSE':a,'RMSE':b,'Spearman':c,'Pearson':d,'Average_AUC':e,'threshold based Metrics':g}
    
    elif learning_method == 'multilabel':
        Scores = cl_prec_rec_f1_acc_mcc_multilabel(y,f)
        
    else:
        Scores = cl_prec_rec_f1_acc_mcc(y,f)
    
    if preds:
        return Scores,f
    return Scores
