import json
import random
from pprint import pprint

class Read:
    # Maximum number of records to use for training.
    max_records = 2000
    users_jobs = []
    users_jobs_train = []  # A list that includes users as keys and the jobs they have applied for as values (train set).
    users_jobs_test = []  # A list that includes users as keys and the jobs they have applied for as values (Recommender set).
    users = set()
    jobs = set()
    split = 0.6

    def notinusersjobslist(self, user_id, job_id):
        if (user_id, job_id) in self.users_jobs:
            return 1
        return 0

    #---------------------------------------------------------------------------------------------
    def readData(self, path = '/Users/fzafari/all.json'):
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
            if self.notinusersjobslist(user, job) == 0:
                self.users_jobs.append((user, job,0))
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
        print("Total number of applies in Recommender set:" + str(len(self.users_jobs_test)))

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

    def random_split(self, l, a_size):
        a, b = [], []
        m = len(l)
        which = ([a] * a_size) + ([b] * (m - a_size))
        random.shuffle(which)

        for array, sample in zip(which, l):
            array.append(sample)

        return a, b

#r = Read()
#users_jobs, users_jobs_train, users_jobs_test, users, jobs  = r.readData()
#print("done")