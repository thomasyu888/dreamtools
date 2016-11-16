import rpy2.robjects as robjects
import os
filePath = os.path.join(os.path.dirname(os.path.abspath(__file__)),'getAUC.R')
robjects.r("source('%s')" % filePath)

##Validate submissions
AUC_pAUC = robjects.r('GetScores')

def get_AUROC(truth, pred):
    """
    Obtain the nonlinear interpolated AUROC.  Make sure the values passed in are already matched
    
    :param truth: vector of truth values
    :param pred:  vector of prediction values

    :returns: (AUC, partial AUC)
    """
    pred = robjects.FloatVector(pred)
    truth = robjects.FloatVector(truth)
    results = AUC_pAUC(truth, pred)
    return(results[0], results[1])

