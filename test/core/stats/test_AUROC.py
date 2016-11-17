from dreamtools.core.stats import AUC
import numpy as np
import os 

def setup(module):
    print('\n')
    print('~' * 60)
    print(os.path.basename(__file__))
    print('~' * 60)

### Same pred and truth
def test_AUROC():
	pred = np.concatenate([np.repeat(1,25), np.repeat(0,30)])
	truth = pred

	auc, pAuc = AUC.get_AUROC(truth,pred)

	assert auc == 1, "Passing in the same prediction and truth should give AUROC of 1"

	pred = np.array([0.9,0.8,0.8,0.8,0.8,0.8,0.8,0.8,0.8,0.8,0.8,0.8,0.7])
	truth = np.array([1,1,1,1,1,1,0,0,0,0,0,0,1])

	first_auroc, first_pauc = AUC.get_AUROC(truth,pred)
	truth = np.array([1,0,0,0,0,0,0,1,1,1,1,1,1])
	second_auroc, second_pauc = AUC.get_AUROC(truth,pred)

	assert first_auroc == second_auroc, "Make sure that ties are dealt with correctly"



