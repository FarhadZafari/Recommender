import MF
import BMF
import ReadData as datareader
from scipy.stats import ttest_ind
import numpy as np

# Just testing the github.
#################################################################
reader = datareader.Read()
Users_Jobs, Users_Jobs_Train, Users_Jobs_Test, Users, Jobs = reader.readDataCSV()

BMF_precision = []
MF_precision = []
BMF_recall = []
MF_recall = []
BMF_accuracy = []
MF_accuracy = []

for i in range(5):
    bmf = BMF.BMF(Users_Jobs, Users_Jobs_Train, Users_Jobs_Test, Users, Jobs)
    bmf.train()
    precision_test_BMF, recall_test_BMF, accuracy_test_BMF = bmf.ModelPrecisionRecallAccuracyTest()
    BMF_precision.append(precision_test_BMF)
    BMF_recall.append(recall_test_BMF)
    BMF_accuracy.append(accuracy_test_BMF)
    print("Training BMF finished!------------------------")

    mf = MF.MF(Users_Jobs, Users_Jobs_Train, Users_Jobs_Test, Users, Jobs)
    mf.train()
    precision_test_MF, recall_test_MF, accuracy_test_MF = mf.ModelPrecisionRecallAccuracyTest()
    MF_precision.append(precision_test_MF)
    MF_recall.append(recall_test_MF)
    MF_accuracy.append(accuracy_test_MF)
    print("Training MF finished!------------------------")

#################################################################
precision_t, precision_p = ttest_ind(BMF_precision, MF_precision)
recall_t, recall_p = ttest_ind(BMF_recall, MF_recall)
accuracy_t, accuracy_p = ttest_ind(BMF_accuracy, MF_accuracy)

#################################################################
print("Avg Precision BMF:", np.mean(BMF_precision), "Avg Precision MF:", np.mean(MF_precision), "t value:", precision_t, "p value:", precision_p)
print("Avg Recall BMF:", np.mean(BMF_recall), "Avg Recall MF:", np.mean(MF_recall), "t value:", recall_t, "p value:", recall_p)
print("Avg Accuracy BMF:", np.mean(BMF_accuracy), "Avg Accuracy MF:", np.mean(MF_accuracy), "t value:", accuracy_t, "p value:", accuracy_p)
#################################################################
