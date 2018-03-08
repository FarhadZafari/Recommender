import ReadData as datareader
import IterativeRec as ir
import random


class MF(ir.IterativeRec):

    P = {}
    Q = {}
    num_factors = 10
    learningRate = 0.1
    user_reg = 0.1
    job_reg = 0.1

    def __init__(self, users_Jobs, users_Jobs_Train, users_Jobs_Test, users, jobs):
        #print("this is the initializer!")
        super().__init__(users_Jobs, users_Jobs_Train, users_Jobs_Test, users, jobs)

        #print("training: ", self.Users_Jobs_Train)
        #print("size", len(self.Users_Jobs_Train))

        #print("testing: ", self.Users_Jobs_Test)
        #print("size", len(self.Users_Jobs_Test))

        #Initializing matrix P.
        for factor1 in range(self.num_factors):
            for user in self.Users:
                self.P[(user, factor1)] = random.uniform(0.45,0.55)

        #print(self.P[('id_11610047', 0)])

        # Initializing matrix Q.
        for factor2 in range(self.num_factors):
            for job in self.Jobs:
                self.Q[(job, factor2)] = random.uniform(0.45,0.55)

        #print(self.Q[('id_50417156', 0)])

    def predict(self, user_id, job_id):
        sum = 0
        for factor in range(self.num_factors):
            sum = sum + self.P[(user_id,factor)] * self.Q[(job_id,factor)]
        if sum > 1:
            return 1
        elif sum < 0:
            return 0
        else:
            return sum


    def error(self, user_id, job_id):
        return self.real(user_id, job_id) - self.predict(user_id, job_id)

    def train(self):
        for reps in range(self.max_learning_repeats):
            loss = 0
            for (user,job, value) in self.Users_Jobs_Train:
                error = self.error(user, job)
                loss = loss + error * error
                for factor in range(self.num_factors):
                    self.P[(user, factor)] = self.P[(user, factor)] + self.learningRate * (error * self.Q[(job, factor)] - self.user_reg * self.P[(user, factor)])
                    self.Q[(job, factor)] = self.Q[(job, factor)] + self.learningRate * (error * self.P[(user, factor)] - self.job_reg * self.Q[(job, factor)])
                    loss = loss + (self.user_reg * self.P[(user, factor)] * self.P[(user, factor)]) + (self.job_reg * self.Q[(job, factor)] * self.Q[(job, factor)])
            loss = loss / 2
            #print(reps, loss)
