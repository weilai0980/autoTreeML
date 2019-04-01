#!/usr/bin/python

import sys
import os

# local packages
from utils_libs import *
from autoML_models import *

# ----- hyper parameter

# -- log
path_result = "../results/vol/log_error_ml.txt"
path_model = "../results/vol/"
path_pred = "../results/vol/"
bool_clf = False

path_data = "../dataset/bitcoin/single_trx/"

model_list = ['ridge',]
# 'rf', 'gbt'
# 'gbt', 'rf', 'xgt', 'gp', 'bayes', 'enet', 'ridge', 'lasso', 'ewma'


# -----

def train_eval_models(xtrain, 
                      ytrain, 
                      xval, 
                      yval, 
                      xtest, 
                      ytest):
    
    '''
    Argu: numpy array
    
    shape of x: [N, D]
    shape of y: [N, ]
    
    '''
    
    best_err_ts = []
   
    # log transformation of y
    log_ytrain = []
    #log(ytrain+1e-5)
    
    # -- Gaussain process 
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
    
    
    # -- Bayesian regression
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
        
    
    # -- ElasticNet
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
        
    
    # -- Ridge regression
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
        
    
    # -- Lasso 
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
        
    
    # -- GBT gradient boosted tree
    
    hyper_para_dict = {"n_steps": list(range(10, 50, 10)),
                       "n_depth": list(range(3, 5))}
    
    if 'gbt' in model_list:
        tmperr = gbt_train_validate(xtrain,
                                    ytrain,
                                    xval,
                                    yval,
                                    xtest,
                                    ytest,
                                    0.0,
                                    bool_clf,
                                    path_result,
                                    path_model + '_gbt.sav',
                                    path_pred,
                                    hyper_para_dict = hyper_para_dict)
        best_err_ts.append(tmperr)
        
    
    # -- Random forest
    
    hyper_para_dict = {"n_trees": list(range(10, 100, 10)),
                       "n_depth": list(range(3, 15))}
    
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
                                   path_pred,
                                   hyper_para_dict = hyper_para_dict)
        best_err_ts.append(tmperr)
        
    
    # -- XGBoosted extreme gradient boosted
    
    hyper_para_dict = {"n_trees": list(range(10, 100, 10)),
                       "n_depth": list(range(3, 15))}
    
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
        
        # test
        tmpx.append(np.asarray([tmp[2][src_idx] for tmp in data]))
        print(np.shape(tmpx[-1]))
    
    tmpy = np.asarray([tmp[0] for tmp in data])
    
    # output shape: x [S N T D], y [N 1]
    return tmpx, np.expand_dims(tmpy, -1)


def test_reshape(data):
    
    # data: [yi, ti, [xi_src1, xi_src2, ...]]
    src_num = len(data[0][2])
    
    step_num = len(data[0][2][0])
    
    print(step_num)

    tmpx = []
    for src_idx in range(src_num):
        
        tmp_src = []
        for tmp in data:
            tmp_src.append( [tmp[2][src_idx][j][0] for j in range(4, step_num)] )
        
        tmpx.append(np.asarray(tmp_src))
        
        # test
        #tmpx.append(np.asarray( [tmp[2][src_idx][j][0] for j in range(step_num) for tmp in data] ))
        print(np.shape(tmpx[-1]))
    
    tmpy = np.asarray([tmp[0] for tmp in data])
    
    # output shape: x [S N T D], y [N 1]
    return tmpx, np.expand_dims(tmpy, -1)


'''
def data_reshape_ml(data):
    
    # data: [yi, ti, [xi_src1, xi_src2, ...]]
    src_num = len(data[0][2])
1
    tmpx = []
    for src_idx in range(src_num):
        
        tmpx.append(np.asarray([tmp[2][src_idx] for tmp in data]))
        print(np.shape(tmpx[-1]))
    
    tmpy = np.asarray([tmp[0] for tmp in data])
    
    # output shape: x [S N T D], y [N 1]
    return tmpx, np.expand_dims(tmpy, -1)
'''

def flatten_multi_source_x(x):
    # x [S N T D]
    
    ins_num = len(x[0])
    src_num = len(x)
    
    tmp_x = []
    for i in range(ins_num):
        # [S T*D] - [ S*T*D]
        tmp_x.append(np.concatenate([x[j][i].flatten() for j in range(src_num)], -1))
        
        
    '''
    # [N S T D]
    tmp_x = np.transpose(x, [1, 0, 2, 3])
    tmp_n = len(tmp_x)
    '''
    
    return tmp_x
#np.reshape(tmp_x, [tmp_n, -1])
    

# ----- main process  

if __name__ == '__main__':
    
    # fix random seed
    np.random.seed(1)
    
    # ----- data
    
    import pickle
    tr_dta = pickle.load(open(path_data + 'train.p', "rb"), encoding = 'latin1')
    val_dta = pickle.load(open(path_data + 'val.p', "rb"), encoding = 'latin1')
    ts_dta = pickle.load(open(path_data + 'test.p', "rb"), encoding = 'latin1')
    
    # -- output from the reshape 
    # y [N 1], x [S N T D]  
    
    # N: number of data samples
    # S: number of sources
    # T: source-specific timesteps
    # D: source-specific feature dimensionality at each timestep
    
    '''
    tr_x, tr_y = data_reshape(tr_dta)
    val_x, val_y = data_reshape(val_dta)
    ts_x, ts_y = data_reshape(ts_dta)
    '''
    
    tr_x, tr_y = test_reshape(tr_dta)
    val_x, val_y = test_reshape(val_dta)
    ts_x, ts_y = test_reshape(ts_dta)
    
    # -- data shapes for machine learning models
    
    # x: [N S*T*D]
    # y: [N]
    
    tr_x = flatten_multi_source_x(tr_x)
    val_x = flatten_multi_source_x(val_x)
    ts_x = flatten_multi_source_x(ts_x)
    
    tr_y = tr_y.reshape((len(tr_y),))
    val_y = val_y.reshape((len(val_y),))
    ts_y = ts_y.reshape((len(ts_y),))
    
    print('\n shape of training, validation and testing data: \n')
    print(np.shape(tr_x), np.shape(tr_y))
    print(np.shape(val_x), np.shape(val_y))
    print(np.shape(ts_x), np.shape(ts_y))
        
    
    # test
    print(tr_x[100])
    
    
    # ----- train, validate and test models
    tmp_errors = train_eval_models(tr_x, 
                                   tr_y, 
                                   val_x, 
                                   val_y,
                                   ts_x, 
                                   ts_y)
        
    print('\n', list(zip(model_list, tmp_errors)))
    
    