from exp.fastai_imports import *
from exp.nb_AmishDataLoaders import *
from sklearn.metrics import roc_auc_score,average_precision_score, roc_curve,precision_recall_curve
import matplotlib.pyplot as plt
import pandas  as pd
import numpy as np


def get_preds_bootstrap_fastai(learner, tasks, ds_type=None, rands=None,plot=False,prev_pred_bunch=None,use_softmax=False,iter=iter):
    
    ds_type = DatasetType.Valid if ds_type is None else ds_type
    
    # predict
    if(prev_pred_bunch is None):
        y_pred,y_true = learner.get_preds(ds_type=ds_type)
        if(use_softmax): 
            y_pred = torch.softmax(y_pred,dim=1)[:,1]
            y_true = y_true[:,1]
        else:
            y_pred =  torch.sigmoid(y_pred)
            
        y_pred_cpu = y_pred.cpu().data.numpy()
        y_true_cpu = y_true.cpu().data.numpy()
        pred_bunch = (y_pred_cpu,y_true_cpu)
    else:
        y_pred_cpu,y_true_cpu = pred_bunch

    res_dict,rands,pred_bunch = get_preds_bootstrap(y_pred_cpu,y_true_cpu,tasks, rands=rands,plot=plot,prev_pred_bunch=prev_pred_bunch,ITER=iter)
    return res_dict,rands,pred_bunch


def print_ci(tasks,y_pred=None ,y_true=None,mod=None,learner=None, ds_type=None, rands=None,
             prev_pred_bunch=None,plot=False,ax=None,figsize=(8,8),
             title='Receiver Operating Characteristic',use_softmax=False,ax2=None,
             title2='2-class Precision-Recall curve',t_test=False,iter=10000):
    
    ci_dict={}
    ttest_flag = False
    roc_pval=''
    pr_pval=''
    
    if(plot):
        _,ax = plt.subplots(figsize=figsize) if ax is None else (None,ax)
    
    if(learner is not None):
        res_dict, rands, pred_bunch = get_preds_bootstrap_fastai(learner, tasks, ds_type=ds_type, rands=rands,plot=plot, prev_pred_bunch=prev_pred_bunch,use_softmax=use_softmax,iter=iter)
    else:
        res_dict, rands, pred_bunch = get_preds_bootstrap(y_pred ,y_true, tasks, rands=rands,plot=plot, prev_pred_bunch=prev_pred_bunch,ITER=iter)
    
    print('Model','ROC','PR','ROC P-Value','PR P-Value',sep='\t')
    for tsk in tasks:
        roc_ci,pr_ci,fpr, tpr, precision, recall, rocs_us, prs_us = res_dict[tsk]
        
        ci_dict[tsk]=(roc_ci,pr_ci)
        if(plot):
            ax.plot(fpr,tpr,label=f'{tsk} AUC: {roc_ci[0]:.2f}[{roc_ci[1]:.2f},{roc_ci[2]:.2f}]')
            ax.legend()
            ax.set_title(title)
            ax.set_xlabel('False Positive Rate')
            ax.set_ylabel('True Positive Rate')
            
            if(ax2 is not None):
                ax2.step(recall, precision,where='post',label=f'{tsk} AP: {pr_ci[0]:.2f}[{pr_ci[1]:.2f},{pr_ci[2]:.2f}]',lw=2)
#                 ax2.fill_between(recall, precision, alpha=0.2,step='post')
                ax2.legend()
                ax2.set_title(title2)
                ax2.set_xlabel('Recall')
                ax2.set_ylabel('Precision')
        
        # P-Values 
        #----------------
        if(t_test):
            if(ttest_flag):
                diff_roc = np.array(prev_roc) - np.array(rocs_us)
                diff_pr  = np.array(prev_pr)  - np.array(prs_us)
                roc_pval = (diff_roc > 0).sum()/len(diff_roc); roc_pval = 1/iter if roc_pval==0 else roc_pval
                pr_pval =  (diff_pr > 0).sum()/len(diff_pr); pr_pval = 1/iter if pr_pval==0 else pr_pval
            else:
                roc_pval = 'Baseline'
                pr_pval = 'Baseline'
                prev_roc = rocs_us 
                prev_pr  = prs_us
                ttest_flag = True
       
        # PRINT RESULTS
        #-------------------
        print(f'{tsk} \t{roc_ci[0]:.2f}[{roc_ci[1]:.2f},{roc_ci[2]:.2f}] \t{pr_ci[0]:.2f}[{pr_ci[1]:.2f},{pr_ci[2]:.2f}] \t{roc_pval} \t{pr_pval}')
                
            
        
    
    return pred_bunch,ci_dict, res_dict
    

def get_preds_bootstrap(y_pred ,y_true ,tasks, rands=None,plot=False,prev_pred_bunch=None,ITER=10000):
    pred_bunch = (y_pred ,y_true)
    #random sampling
    n = y_pred.shape[0]
    rands =  np.random.choice(n,size=(ITER,n)) if rands is None else rands
    res_dict = {}
    
    #reshape in the case of one task
    if((y_pred.ndim==1) | (y_true.ndim==1)):
        y_pred = y_pred.reshape(-1,1)
        y_true = y_true.reshape(-1,1)
        
    for i,tsk in enumerate(tasks):
        # original prediction scores
        y_pred_p = y_pred[:,i]
        y_true_p = y_true[:,i]
        try:
            roc,pr = roc_auc_score(y_true_p,y_pred_p),average_precision_score(y_true_p,y_pred_p)
            fpr, tpr, _ = roc_curve(y_true_p,y_pred_p)
            precision, recall, _ = precision_recall_curve(y_true_p,y_pred_p)
        except:
            roc,pr = np.nan,np.nan
            fpr, tpr = np.nan,np.nan
            precision, recall = np.nan,np.nan
        
        
        # bootsraping
        preds = []
        ys = []
        rocs = []
        prs = []
        
        for randi in rands:
            predi,yi = y_pred_p[randi],y_true_p[randi]
            try:
                roc_i,pr_i = roc_auc_score(yi,predi),average_precision_score(yi,predi)
            except:
                roc_i,pr_i = np.nan,np.nan
                
            preds.append(predi)
            ys.append(yi)
            rocs.append(roc_i)
            prs.append(pr_i)
        
        # sort bootstrap
        rocs_us = rocs.copy()
        prs_us = prs.copy()
        rocs.sort()
        prs.sort()
        roc_ci = (roc,rocs[int(0.025*ITER)],rocs[int(0.975*ITER)])
        pr_ci = (pr,prs[int(0.025*ITER)],prs[int(0.975*ITER)])        
        res_dict[tsk]=(roc_ci,pr_ci,fpr,tpr,precision, recall,rocs_us,prs_us)
        
        
    return res_dict,rands,pred_bunch
    
def ci2errs(vals,ci):
    return np.abs(np.array(ci).T - np.array(vals))



def scores2medians(scores):        
    return np.median(np.array(scores),axis=0)  

def CI_str(score,lower,upper):
    return f'{score:.2f}[{lower:.2f},{upper:.2f}]'

def MedianCI(model_name,res_dicts,pathologies,verbose=False,bl_roc=None,bl_pr=None):
    
    bootstrap_rocs = []
    bootstrap_prs = []
    rocs = []
    prs = []
    
    for res_dict,p in zip(res_dicts,pathologies):
        roc_ci,pr_ci,fpr,tpr,precision, recall,rocs_us,prs_us = res_dict[model_name]

        bootstrap_rocs.append(rocs_us)
        bootstrap_prs.append(prs_us)
        rocs.append(roc_ci[0])
        prs.append(pr_ci[0])
        
    roc_med = scores2medians(rocs)
    pr_med = scores2medians(prs)

    bootstrap_roc_med = scores2medians(bootstrap_rocs)
    bootstrap_pr_med = scores2medians(bootstrap_prs)

    bootstrap_roc_med_sorted = np.sort(bootstrap_roc_med)
    bootstrap_pr_med_sorted = np.sort(bootstrap_pr_med)

    n_iter = len(bootstrap_roc_med)
    med_roc_ci = (roc_med,bootstrap_roc_med_sorted[int(0.025*n_iter)],bootstrap_roc_med_sorted[int(0.975*n_iter)])
    med_pr_ci = (pr_med,bootstrap_pr_med_sorted[int(0.025*n_iter)],bootstrap_pr_med_sorted[int(0.975*n_iter)]) 
    
    roc_p_val = f'{scores_pval(bl_roc,bootstrap_roc_med):.2f}' if bl_roc is not None else 'Baseline'
    pr_p_val = f'{scores_pval(bl_pr,bootstrap_pr_med):.2f}' if bl_pr is not None else 'Baseline'
    
    if(verbose):
        print(f'{model_name:<20s}\t{CI_str(*med_roc_ci)}\t{CI_str(*med_pr_ci)}\t{roc_p_val}\t{pr_p_val}')
        
    return med_roc_ci, med_pr_ci, bootstrap_roc_med, bootstrap_pr_med,rocs,prs


def scores_pval(bl_score,score):
    score_diff = np.array(bl_score) - np.array(score)
    score_pval = max((score_diff > 0).sum(),1)/len(score_diff)
    return score_pval