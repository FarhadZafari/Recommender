import MF
import BMF
import ReadData as datareader
import click
import sys

model = ''

@click.option('--model', default = 'MF', help='Apply to date')

def main(model):
    reader = datareader.Read()
    Users_Jobs, Users_Jobs_Train, Users_Jobs_Test, Users, Jobs, User_train, Users_test, Jobs_train, Jobs_test, Users_Jobs_Train_Hash, Users_Jobs_Test_Hash = reader.readDataCSV()
    if model == 'BMF':
        m = BMF.BMF(Users_Jobs, Users_Jobs_Train, Users_Jobs_Test, Users, Jobs, User_train, Users_test, Jobs_train,
        Jobs_test, Users_Jobs_Train_Hash, Users_Jobs_Test_Hash)
    elif model == 'MF':
        m = MF.MF(Users_Jobs, Users_Jobs_Train, Users_Jobs_Test, Users, Jobs, User_train, Users_test, Jobs_train,
        Jobs_test, Users_Jobs_Train_Hash, Users_Jobs_Test_Hash)
    m.train()
    precision, recall, accuracy = m.ModelPrecisionRecallAccuracyTest()
    UserHit = m.ModelUserHit()
    print("Training model finished!------------------------")
    temp = "Avg Precision BMF:" + str(precision) + "Avg Precision MF:" + str(precision)
    print(temp)
    temp = "Avg Recall BMF:" + str(recall) + "Avg Recall MF:" + str(recall)
    print(temp)
    temp = "Avg Accuracy BMF:" + str(accuracy) + "Avg Accuracy MF:" + str(accuracy)
    print(temp)
    temp = "Avg User Hit BMF:" + str(UserHit) + " Avg User Hit MF:" + str(UserHit)
    print(temp)

if str(sys.argv[1] == ''):
    sys.argv[1] = 'MF'
main(sys.argv[1])
