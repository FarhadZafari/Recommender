import MF as mf
import random
import numpy as np
import ReadData as datareader

class BMF(mf.MF):
    UserBias = {}
    JobBias = {}
    avg = 0
    bias_reg = 0.01

    def __init__(self, users_Jobs, users_Jobs_Train, users_Jobs_Test, users, jobs):
        #print("this is the initializer!")
        super().__init__(users_Jobs, users_Jobs_Train, users_Jobs_Test, users, jobs)

        for user in self.Users:
            self.UserBias[user] = 0

        for job in self.Jobs:
            self.JobBias[job] = 0

        self.avg = self.Average()

    def predict(self, user_id, job_id):
        v = self.avg + self.UserBias[user_id] + self.JobBias[job_id]
        v = v + np.dot(self.P[user_id], self.Q[job_id])
        if v > 1:
            return 1
        elif v < 0:
            return 0
        else:
            return v

    def error(self, user_id, job_id):
        return self.real(user_id, job_id) - self.predict(user_id, job_id)

    def train(self):
        print("Training the model started...")
        for reps in range(self.max_learning_repeats):
            loss = 0
            for (user,job,value) in self.Users_Jobs_Train:
                error = self.error(user, job)
                loss = loss + error * error

                #Update user bias values:
                self.UserBias[user] = self.UserBias[user] + self.learningRate * (error - self.bias_reg * self.UserBias[user])
                loss = loss + self.bias_reg * self.UserBias[user] * self.UserBias[user]

                #Update job bias values:
                self.JobBias[job] = self.JobBias[job] + self.learningRate * (error - self.bias_reg * self.JobBias[job])
                loss = loss + self.bias_reg * self.JobBias[job] * self.JobBias[job]

                u_old = self.P[user].copy()
                v_old = self.Q[job].copy()

                self.P[user] = u_old + self.learningRate * (error * v_old - self.user_reg * u_old)
                self.Q[job] = v_old + self.learningRate * (error * u_old - self.job_reg * v_old)

                loss = loss + self.user_reg * np.dot(self.P[user], self.P[user]) + self.job_reg * np.dot(self.Q[job], self.Q[job])
            loss = loss / 2
            #print(reps, loss)

# reader = datareader.Read()
# Users_Jobs, Users_Jobs_Train, Users_Jobs_Test, Users, Jobs = reader.readDataCSV()
# bmf = BMF(Users_Jobs, Users_Jobs_Train, Users_Jobs_Test, Users, Jobs)
# bmf.train()
# precision_test_BMF, recall_test_BMF, accuracy_test_BMF = bmf.ModelPrecisionRecallAccuracyTest()
# print(precision_test_BMF, recall_test_BMF, accuracy_test_BMF)

