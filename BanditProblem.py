import numpy as np

class BanditProblem(object):
    # the number of arms is equal to the number of entries in the trueActionValues
    # epsilon - epsilon probability value for selecting non-greedy actions
    # ucPpolicy - uncertainity term sumemd with mean reward for earch arm
    
    def __init__(self, trueActionValues, epsilon, totalSteps, c=1):
        
        # number of arms
        self.armNumber=np.size(trueActionValues)
        self.epsilon=epsilon  
        self.currentStep=0
        self.N=np.zeros(self.armNumber)
        self.totalSteps=totalSteps
        self.trueActionValues=trueActionValues #means of each arm, we use normal distribution later
        self.armMeanRewards=np.zeros(self.armNumber) #mean rewards of every arm
        self.currentReward=0 #Q for each step
        self.Q=np.zeros(totalSteps+1)
        self.c=c #UCB constant
        self.ucbPolicy=np.zeros(self.armNumber) #UCB policy values

    def selectActionsEgreedy(self):
        probabilityDraw=np.random.rand()
        
        if (self.currentStep==0) or (probabilityDraw<=self.epsilon): # first step is always random
            selectedArmIndex=np.random.choice(self.armNumber)
            
        if (probabilityDraw>self.epsilon):
            selectedArmIndex=np.argmax(self.armMeanRewards)
            
        self.currentStep += 1
        self.N[selectedArmIndex] += 1
        self.currentReward = np.random.normal(self.trueActionValues[selectedArmIndex], 1)

        self.Q[self.currentStep] = self.Q[self.currentStep-1] + (1/(self.currentStep)) * (self.currentReward - self.Q[self.currentStep-1])
        
        self.armMeanRewards[selectedArmIndex] += (1/(self.N[selectedArmIndex])) * (self.currentReward - self.armMeanRewards[selectedArmIndex])
    
    def selectActionsUCB(self):
        # calculate UCB values for each arm
        for arm in range(self.armNumber):
            if self.N[arm] > 0:
                self.ucbPolicy[arm] = self.armMeanRewards[arm] + self.c*(np.sqrt(np.log(self.currentStep + 1) / self.N[arm]))
            else:
                # if arm has not been selected, set UCB to a high value to ensure it gets selected
                self.ucbPolicy[arm] = float('inf')
        
        selectedArmIndex = np.argmax(self.ucbPolicy)
        self.currentStep += 1
        self.N[selectedArmIndex] += 1
        self.currentReward = np.random.normal(self.trueActionValues[selectedArmIndex], 1)
  
        self.Q[self.currentStep] = self.Q[self.currentStep-1] + (1/(self.currentStep)) * (self.currentReward - self.Q[self.currentStep-1])

        self.armMeanRewards[selectedArmIndex] += (1/(self.N[selectedArmIndex])) * (self.currentReward - self.armMeanRewards[selectedArmIndex])
    
    def playGame(self, action):
        if action == "Egreedy":
            for i in range(self.totalSteps):
                if self.currentStep >= self.totalSteps:
                    break
                self.selectActionsEgreedy()
        elif action == "UCB":
            for i in range(self.totalSteps):
                if self.currentStep >= self.totalSteps:
                    break
                self.selectActionsUCB()
 
    def reset(self):
        self.currentStep = 0
        self.N = np.zeros(self.armNumber)
        self.armMeanRewards = np.zeros(self.armNumber)
        self.currentReward = 0
        self.Q = np.zeros(self.totalSteps + 1)
        self.ucbPolicy = np.zeros(self.armNumber)
