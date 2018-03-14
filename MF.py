import ReadData as datareader
import IterativeRec as it
import random
import numpy as np


class MF(it.IterativeRec):

    P = {}
    Q = {}
    num_factors = 10
    learningRate = 0.1
    user_reg = 0.01
    job_reg = 0.01

    def __init__(self, users_Jobs, users_jobs_train, users_jobs_test, users, jobs, users_train, users_test, jobs_train, jobs_test, users_jobs_train_hash, users_jobs_test_hash):
        print("this is the initializer!")

        super().__init__(users_Jobs, users_jobs_train, users_jobs_test, users, jobs, users_train, users_test, jobs_train, jobs_test, users_jobs_train_hash, users_jobs_test_hash)

        #print("training: ", self.Users_Jobs_Train)
        #print("size", len(self.Users_Jobs_Train))

        #print("testing: ", self.Users_Jobs_Test)
        #print("size", len(self.Users_Jobs_Test))

        #Initializing matrix P.
        for user in self.Users:
            l = []
            for factor1 in range(self.num_factors):
                l.append(random.uniform(0, 1))
                self.P[user] = np.array(l)
                #self.P[user] = 0.1 * (np.random.rand(self.num_factors) - 0.5)

        #print(self.P[('id_11610047', 0)])

        # Initializing matrix Q.
        for job in self.Jobs:
            l = []
            for factor2 in range(self.num_factors):
                l.append(random.uniform(0, 1))
                self.Q[job] = np.array(l)
                #self.Q[job] = 0.1 * (np.random.rand(self.num_factors) - 0.5)

        #print(self.Q[('id_50417156', 0)])

    def predict(self, user_id, job_id):
        v = np.dot(self.P[user_id], self.Q[job_id])
        if v > 1:
            return 1
        elif v < 0:
            return 0
        else:
            return v

    def error(self, user_id, job_id):
        return self.real(user_id, job_id) - self.predict(user_id, job_id)

    def train(self):
        print("Training MF started...")
        for reps in range(self.max_learning_repeats):
            loss = 0
            for (user,job, value) in self.Users_Jobs_Train:
                error = self.error(user, job)
                loss = loss + error * error

                u_old = self.P[user].copy()
                v_old = self.Q[job].copy()

                self.P[user] = u_old + self.learningRate * (error * v_old - self.user_reg * u_old)
                self.Q[job] = v_old + self.learningRate * (error * u_old - self.job_reg * v_old)

                loss = loss + self.user_reg * np.dot(self.P[user], self.P[user]) + self.job_reg * np.dot(self.Q[job], self.Q[job])
            loss = loss / 2
            #print(reps, loss)

# reader = datareader.Read()
# Users_Jobs, Users_Jobs_Train, Users_Jobs_Test, Users, Jobs, Users_train, Users_test, Jobs_train, Jobs_test, Users_Jobs_Train_Hash, Users_Jobs_Test_Hash = reader.readDataCSV()
# mf = MF(Users_Jobs, Users_Jobs_Train, Users_Jobs_Test, Users, Jobs, Users_train, Users_test, Jobs_train, Jobs_test, Users_Jobs_Train_Hash, Users_Jobs_Test_Hash)
# mf.train()
# mf.ModelUserHit()


