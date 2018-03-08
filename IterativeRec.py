class IterativeRec:
    max_learning_repeats = 20
    Users_Jobs = []
    Users_Jobs_Train = []
    Users_Jobs_Test = []
    Users = set()
    Jobs = set()

    def __init__(self, users_Jobs, users_Jobs_Train, users_Jobs_Test, users, jobs):
        #print("this is the initializer!")
        self.Users_Jobs, self.Users_Jobs_Train, self.Users_Jobs_Test, self.Users, self.Jobs = users_Jobs, users_Jobs_Train, users_Jobs_Test, users, jobs

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