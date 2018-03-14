import numpy as np
import ReadData as datareader
import operator


class IterativeRec:
    max_learning_repeats = 100
    Users_Jobs = []
    Users_Jobs_Train = []
    Users_Jobs_Test = []
    Users_Jobs_Train_Hash = {}
    Users_Jobs_Test_Hash = {}
    Users = set()
    Users_train = set()
    Users_test = set()
    Jobs_train = set()
    Jobs_test = set()
    Jobs = set()

    def __init__(self, users_Jobs, users_Jobs_Train, users_Jobs_Test, users, jobs, users_train, users_test, jobs_train, jobs_test, users_jobs_train_hash, users_jobs_test_hash):
        #print("this is the initializer!")
        self.Users_Jobs, self.Users_Jobs_Train, self.Users_Jobs_Test, self.Users, self.Jobs, self.Users_train, self.Users_test, self.Jobs_train, self.Jobs_test, self.Users_Jobs_Train_Hash, self.Users_Jobs_Test_Hash = users_Jobs, users_Jobs_Train, users_Jobs_Test, users, jobs, users_train, users_test, jobs_train, jobs_test, users_jobs_train_hash, users_jobs_test_hash

    def real(self, user_id, job_id):
        if (user_id,job_id, 0) in self.Users_Jobs:
            return 0
        elif (user_id,job_id, 1) in self.Users_Jobs:
            return 1

    def ModelErrorTrain(self):
        mae_train = 0
        rmse_train = 0
        for (user, job, value) in self.Users_Jobs_Train:
            mae_train = mae_train + abs(self.real(user, job) - self.predict(user, job))
            rmse_train = rmse_train + pow((self.real(user, job) - self.predict(user, job)), 2)
        mae_train = mae_train / len(self.Users_Jobs_Train)
        rmse_train = rmse_train / len(self.Users_Jobs_Train)
        rmse_train = rmse_train ** (0.5)
        return mae_train, rmse_train

    def ModelErrorTest(self):
        mae_test = 0
        rmse_test = 0
        for (user, job, value) in self.Users_Jobs_Test:
            mae_test = mae_test + abs(self.real(user, job) - self.predict(user, job))
            rmse_test = rmse_test + pow((self.real(user, job) - self.predict(user, job)), 2)
        mae_test = mae_test / len(self.Users_Jobs_Test)
        rmse_test = rmse_test / len(self.Users_Jobs_Test)
        rmse_test = rmse_test ** (0.5)
        return mae_test, rmse_test

    def ModelPrecisionRecallAccuracyTrain(self):
        # First one real, second one prediction.
        TT = 0
        TF = 0
        FT = 0
        FF = 0
        for (user, job, value) in self.Users_Jobs_Train:
            #print(self.real(user,job), self.predict(user,job))
            if self.real(user,job) >= 0.5 and self.predict(user,job) >= 0.5:
                TT = TT + 1
            elif self.real(user,job) < 0.5 and self.predict(user,job) >= 0.5:
                FT = FT + 1
            elif self.real(user,job) >= 0.5 and self.predict(user,job) < 0.5:
                TF = TF + 1
            elif self.real(user,job) >= 0.5 and self.predict(user,job) < 0.5:
                FF = FF + 1
        recall = TT / (TT + TF)
        precision = TT / (TT + FT)
        accuracy = (TT + FF) / (TT + TF + FT + FF)
        return precision, recall, accuracy

    def Average(self):
        avg = 0
        for (user,job, value) in self.Users_Jobs_Train:
            avg = avg + value
        avg = avg / len(self.Users_Jobs_Train)
        return avg

    def ModelPrecisionRecallAccuracyTest(self):
        # First one real, second one prediction.
        TT = 0
        TF = 0
        FT = 0
        FF = 0
        for (user, job, value) in self.Users_Jobs_Test:
            #print(self.real(user,job), self.predict(user,job))
            if self.real(user,job) >= 0.5 and self.predict(user,job) >= 0.5:
                TT = TT + 1
            elif self.real(user,job) < 0.5 and self.predict(user,job) >= 0.5:
                FT = FT + 1
            elif self.real(user,job) >= 0.5 and self.predict(user,job) < 0.5:
                TF = TF + 1
            elif self.real(user,job) >= 0.5 and self.predict(user,job) < 0.5:
                FF = FF + 1
        recall = TT / (TT + TF)
        precision = TT / (TT + FT)
        accuracy = (TT + FF) / (TT + TF + FT + FF)
        return precision, recall, accuracy

    def ModelUserHit(self, num_top_recommendations = 5000):
        #print("Number of users being tested in User Hit: " + str(self.Users_test))
        #print("Number of jobs being tested in Jobs Hit: " + str(self.Jobs_test))

        TopJobsPerUserPred = {}
        TopJobsPerUserReal = {}
        for user in self.Users_test:
            pred = {}
            topjobsreal = set()
            for job in self.Jobs_test:
                pred[job] = self.predict(user,job)
                if (user, job) in self.Users_Jobs_Test_Hash.keys():
                    if self.Users_Jobs_Test_Hash[(user, job)] == 1:
                        topjobsreal.add(job)
            TopJobsPerUserReal[user] = topjobsreal
            sorted_pred = sorted(pred.items(), key=operator.itemgetter(1))
            topjobspred = set()
            for (job,apply) in sorted_pred[len(sorted_pred) - num_top_recommendations: len(sorted_pred)]:
                topjobspred.add(job)
            TopJobsPerUserPred[user] = topjobspred

        UserHits = {}
        for user in TopJobsPerUserPred:
            UserHits[user] = TopJobsPerUserPred[user].intersection(TopJobsPerUserReal[user])

        hits = []
        for user in UserHits.keys():
            hits.append(len(UserHits.get(user)) / len(TopJobsPerUserPred[user]))
        hit = np.mean(hits)
        #print(hit)
        return hit

# r = datareader.Read()
# users_jobs, users_jobs_train, users_jobs_test, users, jobs, user_train, users_test, jobs_train, jobs_test  = r.readDataCSV()
# print("done")
# bmf = BMF(users_jobs, users_jobs_train, users_jobs_test, users, jobs, user_train, users_test, jobs_train, jobs_test)
# bmf.train()
# t = IterativeRec(users_jobs, users_jobs_train, users_jobs_test, users, jobs, user_train, users_test, jobs_train, jobs_test)
# t.ModelUserHit()