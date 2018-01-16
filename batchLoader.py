import random
import numpy as np
import dataLoader

class batch:
    def __init__(self):
        #Load the data
        self.dataSet, self.labels = dataLoader.loadData('.\\data\\trainingData.mat')
        self.dataSet_valid, self.labels_valid = dataLoader.loadData('.\\data\\testData.mat')
    	
        #Initialize variables
        self.epoch = 0
        self.current_sample=0
        self.train_size = self.dataSet.shape[0]

        #Create a list od indices and shuffle them
        self.train_rand_idx = list(range(0,self.train_size))
        random.shuffle(self.train_rand_idx)

    def getTrain(self,batch_size):
        #If number of training samples taken from the pool is larger than the pool
        #Then the epoch is over and reshuffle
        if (self.current_sample + batch_size) >= self.train_size:
            self.epoch+=1
            self.current_sample=0
            random.shuffle(self.train_rand_idx)

        #Initialize the batch variables
        batch_input = np.zeros((batch_size, self.dataSet.shape[1], 3))
        batch_labels = np.zeros(batch_size)

        #Get batch_size of random samples
        for i in range(batch_size):
            idx = self.train_rand_idx[self.current_sample+i]
            batch_input[i] = self.dataSet[idx]
            batch_labels[i] = self.labels[idx]

        #Increment the current sample
        self.current_sample += batch_size

        batch_labels = np.resize(batch_labels, [batch_size, 1])
        return batch_input, batch_labels

    def getValidation(self):
        return self.dataSet_valid, self.labels_valid

    def getEpoch(self):
        return self.epoch