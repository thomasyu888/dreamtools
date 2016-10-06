import numpy as np
import pandas
from sklearn.metrics import auc

import warnings
warnings.filterwarnings('ignore')

def __get_blockWise_stats(sub_stats):
    
    #group to calculate group wise stats for each block
    grouped = sub_stats.groupby(['predict'], sort=False)
    
    #instantiate a pandas dataframe to store the results for each group (tied values)
    result = pandas.DataFrame.from_dict({'block':xrange(len(grouped)),
                                         'block_numElements'  : np.nan,
                                         'block_truePos_density' : np.nan,
                                         'block_truePos'      : np.nan,
                                         'blockValue'   : np.nan
                                         })
    
    for block,grp in enumerate(grouped):
        name,grp = grp[0],grp[1]
        truePositive = sum(grp.truth == 1)
        grp_truePositive_density = truePositive / float(len(grp))
        idxs = result.block == block
        result.block_truePos_density[idxs] = grp_truePositive_density
        result.block_numElements[idxs] = len(grp)
        result.block_truePos[idxs] = truePositive
        result.blockValue[idxs] = grp.predict.unique()
    result.block = result.block + 1
    result['cum_numElements'] = result.block_numElements.cumsum()
    result['cum_truePos'] = result.block_truePos.cumsum()
    
    return(result)


def _get_precision_recall_fpr(truth, pred):
    
    sub_stats = pandas.DataFrame.from_dict({'predict':pred, 'truth':truth}, dtype='float64')
    sub_stats = sub_stats.sort_values(by=['predict'],ascending=False)

    blockWise_stats = __get_blockWise_stats(sub_stats)
    grouped = sub_stats.groupby(['predict'],sort=False)
    sub_stats = grouped.apply(__nonlinear_interpolated_evalStats,blockWise_stats)
#    sub_stats = sub_stats.sort(columns=['precision'])
    precision, recall,  fpr, threshold = sub_stats.precision.values, sub_stats.recall.values, sub_stats.fpr.values, sub_stats.predict.values 
    
    #YFG suggestion - for the case when Truth == Prediction
    # REF - https://github.com/Sage-Bionetworks/DARPA_Challenge/blob/master/challenge_config.py#L162
    #PR curve AUC (Fixes error when prediction == truth)
    #recall_new=list(recall)
    #precision_new=list(precision)
    #recall_new.reverse()
    #recall_new.append(0)
    #recall_new.reverse()
    #precision_new.reverse()
    #precision_new.append(precision_new[len(precision_new)-1])
    #precision_new.reverse()
    
    ### Implementing the change using numpy style  // Abhishek Pratap - 08/31/2016
    recall_mod = np.insert(recall,0,0)  ## adding 0 at the beginning
    precision_mod = np.insert(precision,0,precision[0]) ## adding corresponding value at the beginning 
    fpr_mod = np.insert(fpr,0,fpr[0]) ## adding corresponding value at the beginning 

    return(precision_mod, recall_mod, fpr_mod, threshold)

def __nonlinear_interpolated_evalStats(block_df, blockWise_stats):
    """
    //needs to be updated
    """
    
    blockValue = block_df.predict.unique()
    if len(blockValue) != 1:
        raise Exception("grouping by predict column doesnt yield unique predict vals per group..WIERD")
    blockValue = blockValue[0]
    blockStats = blockWise_stats[blockWise_stats.blockValue == blockValue].squeeze() #squeeze will convert one row df to series
    
    block_precision = []
    block_recall = []
    block_fpr = []
    test_FP = []
    test_TP = []
    total_elements = blockWise_stats.cum_numElements.max()
    total_truePos = blockWise_stats.cum_truePos.max()
    total_trueNeg = total_elements - total_truePos
    for block_depth,row in enumerate(block_df.iterrows()):
        block_depth += 1  #increase block depth by 1 
        #calculate the cumulative true positives seen till the last block from the current active block
        # and the total number of elements(cumulative) seen till the last block
        if blockStats.block == 1: #no previous obviously
            cum_truePos_till_lastBlock = 0
            cum_numElements_till_lastBlock = 0
            cum_trueNeg_till_lastBlock = 0
        elif blockStats.block > 1:
            last_blockStats = blockWise_stats[blockWise_stats.block == (blockStats.block-1)].squeeze()
            cum_truePos_till_lastBlock = last_blockStats['cum_truePos']
            cum_numElements_till_lastBlock = last_blockStats['cum_numElements']
            cum_trueNeg_till_lastBlock = cum_numElements_till_lastBlock - cum_truePos_till_lastBlock
            
        truePos = cum_truePos_till_lastBlock + (blockStats.block_truePos_density*block_depth)
        falsePos = cum_trueNeg_till_lastBlock + ((1 - blockStats.block_truePos_density ) * block_depth)
        test_FP.append(falsePos)
        test_TP.append(truePos)
        #precision
        interpolated_precision = truePos / float((cum_numElements_till_lastBlock+block_depth))
        block_precision.append(interpolated_precision)
        #recall == true positive rate
        interpolated_recall = truePos / float(total_truePos)
        block_recall.append(interpolated_recall)
        #fpr == false positive rate
        interpolated_fpr = falsePos / float(total_trueNeg)
        block_fpr.append(interpolated_fpr)
        
    block_df['precision'] = block_precision
    block_df['recall'] = block_recall
    block_df['fpr'] = block_fpr
    block_df['block_depth'] = np.arange(1,block_df.shape[0]+1)
    block_df['block'] = blockStats.block
    return(block_df)

def get_AUROC(truth, pred):
    """
    Obtain the nonlinear interpolated AUROC.  Make sure the values passed in are already matched
    
    :param truth: vector of truth values
    :param pred:  vector of prediction values

    :returns: AUROC
    """
    precision, recall, fpr, threshold= _get_precision_recall_fpr(truth, pred)

    AUROC = auc(fpr,recall,reorder=True)
    return(AUROC)