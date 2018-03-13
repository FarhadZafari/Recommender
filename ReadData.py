import json
import random
from pprint import pprint
import itertools

class Read:
    # Maximum number of records to use for training.
    max_records = 2000
    users_jobs = []
    users_jobs_train = []  # A list that includes users as keys and the jobs they have applied for as values (train set).
    users_jobs_test = []  # A list that includes users as keys and the jobs they have applied for as values (Recommender set).
    users_jobs_train_hash = {}
    users_jobs_test_hash = {}
    users = set()
    jobs = set()
    split = 0.6

    def notinusersjobslist(self, users_jobs, user_id, job_id):
        if users_jobs.get((user_id, job_id)) == 1:
            return False
        else:
            return True

    #---------------------------------------------------------------------------------------------
    def readDataJson(self, path = '/Users/fzafari/all.json'):
        data = json.load(open(path)) #Opening the dataset.

        #Reading user-job pairs from the dataset and putting them in users_jobs and jobs_users dictionaries.
        #num_applies = len(data["response"]["docs"])

        x = 0
        for d in data["response"]["docs"]:
            if x <= self.max_records:
                self.users_jobs.append((d["applyCVId"],d["applyJobId"],1))
                self.users.add(d["applyCVId"])
                self.jobs.add(d["applyJobId"])
                x = x + 1
            else:
                print("Breaking...")
                break

        num_applies = len(self.users_jobs)
        i = 0
        while i <= num_applies:
            user = random.choice(list(self.users))
            job = random.choice(list(self.jobs))
            if self.notinusersjobslist(users_jobs, user, job) == 0:
                self.users_jobs.append((user, job, 0))
                i = i + 1
            else:
                continue
            

        #Getting the number of users and jobs from the dictionaries.
        num_users = len(self.users)
        num_jobs = len(self.jobs)

        random.shuffle(self.users_jobs)

        train_data_size = round(self.split * len(self.users_jobs))
        self.users_jobs_train, self.users_jobs_test = self.random_split(self.users_jobs, train_data_size)

        #print(self.users_jobs)
        total_num_applies_notapplies = len(self.users_jobs)

        #printing number of users and jobs.
        print("Total number of users:", num_users)
        print("Total number of jobs:", num_jobs)
        print("Total number of applies plus not applies:", total_num_applies_notapplies)
        print("Total number of applies in train set:" + str(len(self.users_jobs_train)))
        print("Total number of applies in test set:" + str(len(self.users_jobs_test)))

        #print(users_jobs_train)
        #print(users_jobs_test)

        #Just trying to see if I can create a boolean matrix that shows whether
        #a user has applied for a job or not. Of course I am not gonna use this
        #matrix since the ratings matrix is very sparse.
        #for u in users_jobs.keys():
        #    for j in jobs_users.keys():
        #        if (u,j) in users_jobs.items():
        #            print(u,"has applied for", j)
        #        else:
        #            print(u,"has not applied for", j)

        #users = {'user_ids':list(users_jobs.keys())}
        #jobs = {'job_ids':list(jobs_users.keys())}

        #users_json = json.dumps(users, indent=4)
        #jobs_json = json.dumps(jobs, indent=4)

        #print(users_json)
        #print(jobs_json)

        #f1 = open("/Users/fzafari/user_ids.json","w")
        #f1.write(users_json)

        #f2 = open("/Users/fzafari/job_ids.json","w")
        #f2.write(jobs_json)

        return self.users_jobs, self.users_jobs_train,self.users_jobs_test, self.users,self.jobs
    #---------------------------------------------------------------------------------------------

    # ---------------------------------------------------------------------------------------------
    def readDataCSV(self, negativeSampling = True, trainset = 'sgtrain.txt', testset = 'sgtest.txt'):
        path='/Users/fzafari/'+ trainset
        data = open(path)  # Opening the train dataset.
        i = 0
        for d in data:
            if i == 1000:
                break
            line = str(d)
            l = line.split(",")
            self.users.add(l[0])
            self.jobs.add(l[1])
            self.users_jobs_train.append((l[0], l[1], 1))
            self.users_jobs_train_hash[(l[0], l[1])] = 1
            i = i + 1
        print("train dataset loaded!")

        i = 0
        path='/Users/fzafari/' + testset
        data = open(path)  # Opening the test datasel[0], l[1], 1t.
        for d in data:
            if i == 1000:
                break
            line = str(d)
            l = line.split(",")
            self.users.add(l[0])
            self.jobs.add(l[1])
            self.users_jobs_test.append((l[0], l[1], 1))
            self.users_jobs_test_hash[(l[0], l[1])] = 1
            i = i + 1
        print("test dataset loaded!")

        num_users = len(self.users)
        num_jobs = len(self.jobs)
        num_applies_train = len(self.users_jobs_train)
        num_applies_test = len(self.users_jobs_test)

        print("Total number of users:", num_users)
        print("Total number of jobs:", num_jobs)
        print("Total number of applies in train set:" + str(len(self.users_jobs_train)))
        print("Total number of applies in test set:" + str(len(self.users_jobs_test)))

        if negativeSampling is True:
            i = 0
            while i <= 1 * num_applies_train:
                user = random.choice(list(self.users))
                job = random.choice(list(self.jobs))
                if self.notinusersjobslist(self.users_jobs_train_hash, user, job) is True:
                    self.users_jobs_train.append((user, job, 0))
                    i = i + 1
                else:
                    continue
                    print(i)
            print("train dataset completed!")

            i = 0
            while i <= 1 * num_applies_test:
                user = random.choice(list(self.users))
                job = random.choice(list(self.jobs))
                if self.notinusersjobslist(self.users_jobs_test_hash, user, job) is True:
                    self.users_jobs_test.append((user, job, 0))
                    i = i + 1
                else:
                    continue
                    print(i)
            print("test dataset completed!")

            random.shuffle(self.users_jobs_train)
            train_file = open('/Users/fzafari/train.txt', 'w')
            for (user,job,value) in self.users_jobs_train:
                st = user + ',' + job + ',' + str(value) + '\n'
                #print(st)
                train_file.write(st)

            print("writing train dataset completed!")

            random.shuffle(self.users_jobs_test)
            test_file = open('/Users/fzafari/test.txt', 'w')
            for (user,job,value) in self.users_jobs_test:
                st = user + ',' + job + ',' + str(value) + '\n'
                #print(st)
                test_file.write(st)

            print("writing test dataset completed!")

        self.users_jobs = self.users_jobs_train + self.users_jobs_test

        #printing number of users and jobs.
        print("Total number of users:", num_users)
        print("Total number of jobs:", num_jobs)
        print("Total number of applies in train set:" + str(len(self.users_jobs_train)))
        print("Total number of applies in test set:" + str(len(self.users_jobs_test)))

        return self.users_jobs, self.users_jobs_train,self.users_jobs_test, self.users,self.jobs
    # ---------------------------------------------------------------------------------------------

    def random_split(self, l, a_size):
        a, b = [], []
        m = len(l)
        which = ([a] * a_size) + ([b] * (m - a_size))
        random.shuffle(which)

        for array, sample in zip(which, l):
            array.append(sample)

        return a, b

#r = Read()
#users_jobs, users_jobs_train, users_jobs_test, users, jobs  = r.readDataCSV()
#print("done")
