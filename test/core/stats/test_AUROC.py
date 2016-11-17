from dreamtools.core.stats import AUC
from nose.tools import assert_raises, assert_equals
import numpy as np
import os 

def setup(module):
    print('\n')
    print('~' * 60)
    print(os.path.basename(__file__))
    print('~' * 60)

### Same pred and truth
def test_AUROC():

	#Correct order
	truth = np.array([1,1,1,1,0,0,0,0,0,0])
	pred = np.array([0.93, 0.78, 0.77, 0.63, 0.43, 0.32, 0.26, 0.07, 0.06, 0.04])
	auc, pAUCse, SpecAtSens = AUC.get_AUROC(truth,pred)

	assert auc == 1 and pAUCse ==  0.19999999999999996 and SpecAtSens == 1, "AUC, pAUCse and SpecAtSens should be 1, 0.2, 1 respectively"

	## Ex.2. Got the order of two labels (4 and 8) wrong.
	pred = np.array([0.93, 0.78, 0.77, 0.33, 0.43, 0.32, 0.26, 0.87, 0.06, 0.04])
	auc, pAUCse, SpecAtSens = AUC.get_AUROC(truth,pred)
	assert auc == 0.8333333333333334 and pAUCse == 0.1333333333333333 and SpecAtSens == 0.6666666666666666, "AUC, pAUCse and SpecAtSens should be 0.8334, 0.1333, 0.6667 respectively"

	## Ex.3. The order is the same as in Ex.2. 
	pred = np.array([0.93, 0.78, 0.77, 0.33, 0.43, 0.32, 0.26, 0.87, 0.06, 0.04])
	auc2, pAUCse2, SpecAtSens2 = AUC.get_AUROC(truth,pred)
	assert auc == auc2 and pAUCse == pAUCse2 and SpecAtSens == SpecAtSens2, "AUC, pAUCse and SpecAtSens should be the same as Ex.2."

	## Ex.4. Predictions outside the [0, 1] interval. 
	## (Added 1 to the predictions in Ex.2.) 
	## But the order is still the same as in Ex.2, and we 
	## should get the same results.
	pred = np.array([0.93, 0.78, 0.77, 0.33, 0.43, 0.32, 0.26, 0.87, 0.06, 0.04]) +1
	auc2, pAUCse2, SpecAtSens2 = AUC.get_AUROC(truth,pred)
	assert auc == auc2 and pAUCse == pAUCse2 and SpecAtSens == SpecAtSens2, "AUC, pAUCse and SpecAtSens should be the same as Ex.2. (1 is added to all the predictions)"

	## Ex.5. Predictions very close to zero but still the same order as in Ex.2. 
	## Should get the same results as in Ex.2.
	pred = np.array([0.0093, 0.0078, 0.0077, 0.0033, 0.0043, 0.0032, 0.0026, 0.0087, 0.0006, 0.0004])
	auc2, pAUCse2, SpecAtSens2 = AUC.get_AUROC(truth,pred)
	assert auc == auc2 and pAUCse == pAUCse2 and SpecAtSens == SpecAtSens2, "AUC, pAUCse and SpecAtSens should be the same as Ex.2. (Predictions are different but in the same order)"


	## Ex.6. Predictions very close to one but still the same order as in Ex.2. 
	## Should get the same results as in Ex.2.
	pred = np.array([0.9993, 0.9978, 0.9977, 0.9933, 0.9943, 0.9932, 0.9926, 0.9987, 0.9906, 0.9904])
	auc2, pAUCse2, SpecAtSens2 = AUC.get_AUROC(truth,pred)
	assert auc == auc2 and pAUCse == pAUCse2 and SpecAtSens == SpecAtSens2, "AUC, pAUCse and SpecAtSens should be the same as Ex.2. (Predictions are different but in the same order)"

	## Ex.7 constant prediction (all zeros). 
	## Should get AUC = 0.5, pAUC = 0.02, and SpecAtSensi = 0.2
	pred = np.array(np.repeat(0,10))
	auc, pAUCse, SpecAtSens = AUC.get_AUROC(truth,pred)
	assert auc == 0.5 and pAUCse == 0.01999999999999999 and SpecAtSens == 0.19999999999999996, "AUC, pAUCse and SpecAtSens should be 0.5, 0.02, 0.2 respectively"
	
	## Ex.8 constant prediction (all ones). 
	## Should get AUC = 0.5, pAUC = 0.02, and SpecAtSensi = 0.2
	pred = np.array(np.repeat(1,10))
	auc, pAUCse, SpecAtSens = AUC.get_AUROC(truth,pred)
	assert auc == 0.5 and pAUCse == 0.01999999999999999 and SpecAtSens == 0.19999999999999996, "AUC, pAUCse and SpecAtSens should be 0.5, 0.02, 0.2 respectively"

	## Ex.9 constant prediction. 
	## Should get AUC = 0.5, pAUC = 0.02, and SpecAtSensi = 0.2
	pred = np.array(np.repeat(13,10))
	auc, pAUCse, SpecAtSens = AUC.get_AUROC(truth,pred)
	assert auc == 0.5 and pAUCse == 0.01999999999999999 and SpecAtSens == 0.19999999999999996, "AUC, pAUCse and SpecAtSens should be 0.5, 0.02, 0.2 respectively"

	## Ex.10 pROC handle NAs by removing the pair if either 
	## the prediction, the label (or both) is NA. 
	## (This really doesn't matter if we are handling the
	## NAs in the validation phase.)
	##
	pred = np.array([0.93, 0.78, 0.77, 0.33,   np.nan, 0.32, 0.26, 0.87, 0.06, 0.04])
	auc, pAUCse, SpecAtSens = AUC.get_AUROC(truth,pred)
	assert auc == 0.8500000000000001 and pAUCse == 0.15999999999999998 and SpecAtSens == 0.8, "AUC, pAUCse and SpecAtSens should be 0.85, 0.16, 0.8 respectively"

	truth = np.array([1.00, 1.00, 1.00, 1.00, np.nan, 0.00, 0.00, 0.00, 0.00, 0.00])
	pred = np.array([0.93, 0.78, 0.77, 0.33, 0.43, 0.32, 0.26, 0.87, 0.06, 0.04])
	auc, pAUCse, SpecAtSens = AUC.get_AUROC(truth,pred)
	assert auc == 0.8500000000000001 and pAUCse == 0.15999999999999998 and SpecAtSens == 0.8, "AUC, pAUCse and SpecAtSens should be 0.85, 0.16, 0.8 respectively"

	truth = np.array([1.00, 1.00, 1.00, 1.00, np.nan, 0.00, 0.00, 0.00, 0.00, 0.00])
	pred = np.array([0.93, 0.78, 0.77, 0.33, np.nan, 0.32, 0.26, 0.87, 0.06, 0.04])
	auc, pAUCse, SpecAtSens = AUC.get_AUROC(truth,pred)
	assert auc == 0.8500000000000001 and pAUCse == 0.15999999999999998 and SpecAtSens == 0.8, "AUC, pAUCse and SpecAtSens should be 0.85, 0.16, 0.8 respectively"

	truth = np.array([1.00, 1.00, 1.00, 1.00, 0.00, 0.00, 0.00, 0.00, 0.00])
	pred = np.array([0.93, 0.78, 0.77, 0.33, 0.32, 0.26, 0.87, 0.06, 0.04])
	auc, pAUCse, SpecAtSens = AUC.get_AUROC(truth,pred)
	assert auc == 0.8500000000000001 and pAUCse == 0.15999999999999998 and SpecAtSens == 0.8, "AUC, pAUCse and SpecAtSens should be 0.85, 0.16, 0.8 respectively"

	# Ex.11 Ties should be dealt with correctly and they should give the same result even if its in a different order
	pred = np.array([0.9,0.8,0.8,0.8,0.8,0.8,0.8,0.8,0.8,0.8,0.8,0.8,0.7])
	truth = np.array([1,1,1,1,1,1,0,0,0,0,0,0,1])

	first_auroc, first_pauc, first_spec = AUC.get_AUROC(truth,pred)
	truth = np.array([1,0,0,0,0,0,0,1,1,1,1,1,1])
	second_auroc, second_pauc, second_spec = AUC.get_AUROC(truth,pred)

	assert first_auroc == second_auroc and first_pauc == second_pauc and first_spec == second_spec, "Make sure that ties are dealt with correctly"

