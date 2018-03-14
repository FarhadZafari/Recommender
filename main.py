import MF
import BMF
import ReadData as datareader
from scipy.stats import ttest_ind
import numpy as np

################################################
reader = datareader.Read()
Users_Jobs, Users_Jobs_Train, Users_Jobs_Test, Users, Jobs, User_train, Users_test, Jobs_train, Jobs_test, Users_Jobs_Train_Hash, Users_Jobs_Test_Hash = reader.readDataCSV()

num_repeats = 10
ttest = True
BMF_precision = []
MF_precision = []
BMF_recall = []
MF_recall = []
BMF_accuracy = []
MF_accuracy = []
BMF_UserHit = []
MF_UserHit = []

for i in range(num_repeats):
    bmf = BMF.BMF(Users_Jobs, Users_Jobs_Train, Users_Jobs_Test, Users, Jobs, User_train, Users_test, Jobs_train, Jobs_test, Users_Jobs_Train_Hash, Users_Jobs_Test_Hash)
    bmf.train()
    precision_test_BMF, recall_test_BMF, accuracy_test_BMF = bmf.ModelPrecisionRecallAccuracyTest()
    userHit_BMF = bmf.ModelUserHit()
    # BMF_precision.append(precision_test_BMF)
    # BMF_recall.append(recall_test_BMF)
    # BMF_accuracy.append(accuracy_test_BMF)
    BMF_UserHit.append(userHit_BMF)
    print("Training BMF finished!------------------------")

    mf = MF.MF(Users_Jobs, Users_Jobs_Train, Users_Jobs_Test, Users, Jobs, User_train, Users_test, Jobs_train, Jobs_test, Users_Jobs_Train_Hash, Users_Jobs_Test_Hash)
    mf.train()
    precision_test_MF, recall_test_MF, accuracy_test_MF = mf.ModelPrecisionRecallAccuracyTest()
    userHit_MF = mf.ModelUserHit()
    # MF_precision.append(precision_test_MF)
    # MF_recall.append(recall_test_MF)
    # MF_accuracy.append(accuracy_test_MF)
    MF_UserHit.append(userHit_MF)
    print("Training MF finished!------------------------")

#################################################################
if ttest is True:
    # precision_t, precision_p = ttest_ind(BMF_precision, MF_precision)
    # recall_t, recall_p = ttest_ind(BMF_recall, MF_recall)
    # accuracy_t, accuracy_p = ttest_ind(BMF_accuracy, MF_accuracy)
    userhit_t, userhit_p = ttest_ind(BMF_UserHit, MF_UserHit)

#################################################################
# temp = "Avg Precision BMF:" + str(np.mean(BMF_precision)) + "Avg Precision MF:" + str(np.mean(MF_precision))
# if ttest is True:
#     temp = temp + "t value:" + str(precision_t) + " p value:" + str(precision_p)
# print(temp)
# temp = "Avg Recall BMF:" + str(np.mean(BMF_recall)) + "Avg Recall MF:" + str(np.mean(MF_recall))
# if ttest is True:
#     temp = temp + "t value:" + str(recall_t) + " p value:" + str(recall_p)
# print(temp)
# temp = "Avg Accuracy BMF:" + str(np.mean(BMF_accuracy)) + "Avg Accuracy MF:" + str(np.mean(MF_accuracy))
# if ttest is True:
#     temp = temp + "t value:" + str(accuracy_t) + " p value:" + str(accuracy_p)
# print(temp)
temp = "Avg User Hit BMF:" + str(np.mean(BMF_UserHit)) + " Avg User Hit MF:" + str(np.mean(MF_UserHit))
if ttest is True:
    temp = temp + "t value:" + str(userhit_t) + " p value:" + str(userhit_p)
print(temp)
#################################################################
