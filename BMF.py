import MF as mf
import random


class BMF(mf.MF):
    UserBias = {}
    JobBias = {}
    avg = 0
    bias_reg = 0.1

    def __init__(self, users_Jobs, users_Jobs_Train, users_Jobs_Test, users, jobs):
        #print("this is the initializer!")
        super().__init__(users_Jobs, users_Jobs_Train, users_Jobs_Test, users, jobs)

        for user in self.Users:
            self.UserBias[user] = 0

        for job in self.Jobs:
            self.JobBias[job] = 0

        self.avg = self.Average()

    def predict(self, user_id, job_id):
        sum = self.avg + self.UserBias[user_id] + self.JobBias[job_id]
        for factor in range(self.num_factors):
            sum = sum + self.P[(user_id,factor)] * self.Q[(job_id,factor)]
        if sum > 1:
            return 1
        elif sum < 0:
            return 0
        else:
            return sum

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

                for factor in range(self.num_factors):
                    self.P[(user, factor)] = self.P[(user, factor)] + self.learningRate * (error * self.Q[(job, factor)] - self.user_reg * self.P[(user, factor)])
                    self.Q[(job, factor)] = self.Q[(job, factor)] + self.learningRate * (error * self.P[(user, factor)] - self.job_reg * self.Q[(job, factor)])
                    loss = loss + (self.user_reg * self.P[(user, factor)] * self.P[(user, factor)]) + (self.job_reg * self.Q[(job, factor)] * self.Q[(job, factor)])
            loss = loss / 2
            #print(reps, loss)
