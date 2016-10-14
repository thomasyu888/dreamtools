import rpy2.robjects as robjects
robjects.r("source('getAUC.R')")

##Validate submissions
AUC_pAUC = robjects.r('GetScores')

def get_AUROC(truth, pred):
    """
    Obtain the nonlinear interpolated AUROC.  Make sure the values passed in are already matched
    
    :param truth: vector of truth values
    :param pred:  vector of prediction values

    :returns: (AUC, partial AUC)
    """
	pred = robjects.FloatVector([0.9,0.8,0.8,0.8,0.8,0.8,0.8,0.8,0.8,0.8,0.8,0.8,0.7])
	truth = robjects.FloatVector([1,1,1,1,1,1,0,0,0,0,0,0,1])
	results = AUC_pAUC(truth, pred)
    return(results[0], results[1])

