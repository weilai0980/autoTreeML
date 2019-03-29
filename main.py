#!/usr/bin/python

import sys
import os

# local packages
from utils_libs import *
from autoML_models import *

# --- parameter set-up from parameter-file ---

'''
# load the parameter file
para_dict = para_parser("para_file.txt")

para_order_minu = para_dict['para_order_minu']
para_order_hour = para_dict['para_order_hour']
bool_feature_selection = para_dict['bool_feature_selection']

# ONLY USED FOR ROLLING EVALUATION
interval_len = para_dict['interval_len']
roll_len = para_dict['roll_len']

para_step_ahead = para_dict['para_step_ahead']

bool_add_feature = True
'''

# ----


def train_eval_models(xtrain, 
                      ytrain, 
                      xval, 
                      yval, 
                      xtest, 
                      ytest):
    
    '''
    Argu: numpy array
    
    '''
    
    best_err_ts = []
   
    # log transformation of y
    log_ytrain = []
    #log(ytrain+1e-5)
    
    # note: remove the training error calculation
    # Gaussain process 
    if 'gp' in model_list:
        tmperr = gp_train_validate(xtrain, 
                                   ytrain, 
                                   xval,
                                   yval, 
                                   xtest, 
                                   ytest, 
                                   path_result, 
                                   path_model + '_gp.sav', 
                                   log_ytrain,
                                   path_pred)
        best_err_ts.append(tmperr)
    
    
    # Bayesian regression
    if 'bayes' in model_list:
        tmperr = bayesian_reg_train_validate(xtrain, 
                                             ytrain, 
                                             xval, 
                                             yval, 
                                             xtest, 
                                             ytest, 
                                             path_result, 
                                             path_model + '_bayes.sav',\
                                             log_ytrain, 
                                             path_pred)
        best_err_ts.append(tmperr)
    
    # ElasticNet
    if 'enet' in model_list:
        tmperr = elastic_net_train_validate(xtrain, 
                                            ytrain, 
                                            xval, 
                                            yval, 
                                            xtest, 
                                            ytest, 
                                            path_result,
                                            path_model + '_enet.sav',
                                            log_ytrain,
                                            path_pred)
        best_err_ts.append(tmperr)
    
    #Ridge regression
    if 'ridge' in model_list:
        tmperr = ridge_reg_train_validate(xtrain, 
                                          ytrain, 
                                          xval, 
                                          yval, 
                                          xtest, 
                                          ytest, 
                                          path_result, 
                                          path_model + '_ridge.sav',
                                          log_ytrain, 
                                          path_pred)
        best_err_ts.append(tmperr)
    
    # Lasso 
    if 'lasso' in model_list:
        tmperr = lasso_train_validate(xtrain, 
                                      ytrain, 
                                      xval, 
                                      yval, 
                                      xtest,
                                      ytest, 
                                      path_result,
                                      path_model + '_lasso.sav',
                                      log_ytrain, 
                                      path_pred)
        best_err_ts.append(tmperr)
    
    
    '''
    # EWMA
    if 'ewma' in model_list:
        if para_step_ahead != 0 or len(xtest) != 0: 
            
            tmperr = ewma_instance_validate(autotrain, ytrain, autoval, yval, autotest, ytest, path_result, path_pred)
        
        else:
            tmperr = ewma_validate(ytrain, yval, path_result, path_pred)
            
        best_err_ts.append(tmperr)
        
    '''
        
    
    # GBT gradient boosted tree
    if 'gbt' in model_list:
        tmperr = gbt_train_validate(xtrain, 
                                    ytrain, 
                                    xval, 
                                    yval, 
                                    xtest, 
                                    ytest, 
                                    0.0, 
                                    bool_clf, 
                                    path_result, \
                                    path_model +'_gbt.sav', 
                                    path_pred)
        best_err_ts.append(tmperr)
    
    # Random forest performance
    if 'rf' in model_list:
        tmperr = rf_train_validate(xtrain, 
                                   ytrain, 
                                   xval, 
                                   yval, 
                                   xtest, 
                                   ytest, 
                                   bool_clf, 
                                   path_result, 
                                   path_model + '_rf.sav',
                                   path_pred)
        best_err_ts.append(tmperr)
    
    # XGBoosted extreme gradient boosted
    if 'xgt' in model_list:
        tmperr = xgt_train_validate(xtrain, 
                                    ytrain, 
                                    xval, 
                                    yval, 
                                    xtest, 
                                    ytest, 
                                    bool_clf, 
                                    0, 
                                    path_result, 
                                    path_model + '_xgt.sav', 
                                    path_pred)
        best_err_ts.append(tmperr)
    
    return best_err_ts


def data_reshape(data):
    
    # data: [yi, ti, [xi_src1, xi_src2, ...]]
    src_num = len(data[0][2])

    tmpx = []
    for src_idx in range(src_num):
        tmpx.append(np.asarray([tmp[2][src_idx] for tmp in data]))
        print(np.shape(tmpx[-1]))
    
    tmpy = np.asarray([tmp[0] for tmp in data])
    
    # output shape: x [S N T D],  y [N 1]
    return tmpx, np.expand_dims(tmpy, -1)

def flatten_multi_source_x(x):
    # x [S N T D]
    
    # [N S T D]
    tmp_x = np.transpose(x, [1, 0, 2, 3])
    tmp_n = len(tmp_x)
    
    return np.reshape(tmp_x, [tmp_n, -1])
    

# ----- main process  

if __name__ == '__main__':
    
    
    # ----- log
    path_result = "../results/vol/log_error_ml.txt"
    path_model = "../results/vol/"
    path_pred = "../results/vol/"
    bool_clf = False

    model_list = ['gbt']
# 'gbt', 'rf', 'xgt', 'gp', 'bayes', 'enet', 'ridge', 'lasso', 'ewma'
    
    # fix random seed
    np.random.seed(1)
    
    '''
    # ----- log
    
    path_log_error = "../results/mixture/log_error_mix.txt"
    path_log_epoch  = "../results/mixture/log_epoch_mix.txt"
    path_data = "../dataset/bitcoin/double_trx/"
    '''
    path_data = "../dataset/bitcoin/double_trx/"
    
    # ----- data
    
    import pickle
    tr_dta = pickle.load(open(path_data + 'train.p', "rb"), encoding = 'latin1')
    val_dta = pickle.load(open(path_data + 'val.p', "rb"), encoding = 'latin1')
    ts_dta = pickle.load(open(path_data + 'test.p', "rb"), encoding = 'latin1')
    
    # output from the reshape 
    # y [N 1], x [S N T D]    
    
    tr_x, tr_y = data_reshape(tr_dta)
    val_x, val_y = data_reshape(val_dta)
    ts_x, ts_y = data_reshape(ts_dta)
    
    tr_x = flatten_multi_source_x(tr_x)
    val_x = flatten_multi_source_x(val_x)
    ts_x = flatten_multi_source_x(ts_x)
    
    print('\n shape of training, validation and testing data: \n')
    print(np.shape(tr_x), np.shape(tr_y))
    print(np.shape(val_x), np.shape(val_y))
    print(np.shape(ts_x), np.shape(ts_y))
        
         
    # train, validate and test models
    tmp_errors = train_eval_models(tr_x, 
                                   tr_y, 
                                   val_x, 
                                   val_y,
                                   ts_x, 
                                   ts_y)
        
    print(list(zip(model_list, tmp_errors)))
    
    