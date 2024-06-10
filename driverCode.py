import numpy as np
import matplotlib.pyplot as plt
from BanditProblem import BanditProblem

# these are the means of the action values that are used to simulate the multi-armed bandit problem
actionValues=np.array([1,0.7,2,0.2,-1])

# epsilon values to investigate the performance of the method
epsilon1=0
epsilon2=0.1
epsilon3=0.01
c=0
# total number of simulation steps 
totalSteps=10000

# create four different bandit problems and simulate the method performance
Bandit1=BanditProblem(actionValues, epsilon1, totalSteps)
Bandit1.playGame("Egreedy")
epsilon1MeanReward=Bandit1.Q

Bandit2=BanditProblem(actionValues, epsilon2, totalSteps)
Bandit2.playGame("Egreedy")
epsilon2MeanReward=Bandit2.Q

Bandit3=BanditProblem(actionValues, epsilon3, totalSteps)
Bandit3.playGame("Egreedy")
epsilon3MeanReward=Bandit3.Q

Bandit4 = BanditProblem(actionValues, c, totalSteps)  # epsilon value is irrelevant for UCB
Bandit4.playGame("UCB")
ucbMeanReward = Bandit4.Q

#plot the results
plt.plot(np.arange(totalSteps+1),epsilon1MeanReward,linewidth=2, color='r', label='epsilon =0')
plt.plot(np.arange(totalSteps+1),epsilon2MeanReward,linewidth=2, color='k', label='epsilon =0.1')
plt.plot(np.arange(totalSteps+1),epsilon3MeanReward,linewidth=2, color='m', label='epsilon =0.01')
plt.plot(np.arange(totalSteps+1),ucbMeanReward,linewidth=2, color='b', label='UCB')
plt.xscale("log")
plt.xlabel('Steps')
plt.ylabel('Average reward')
plt.legend()
plt.savefig('results.png', dpi = 500)
plt.show()


