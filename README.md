# Reinforcement Learning
# Precision Pest Management for Blueberries

# Abstract

The spotted wing drosophila (SWD) is an invasive pest of blueberries. It has spread throughout many of the primary fruit production regions of the United States since 2008. Controlling the number of SWD becomes an important task on farming blueberries. In this project, I designed a Markov decision process (MDP) model on this task and use machine learning and reinforcement learning to solve this problem. Finally, I gave strategies to control the pest.

# Data

The data set I have includes the number of both male and female SWD in June, July and August in 2016 in a farm. It also has the spray records. I firstly build an MDP model onto this data set and then use K-means, value iteration and policy iteration to solve the problem and finally give policy to control spotted wing drosophila. In order to the confidentiality agreement, I uploaded a dummy data set here. 

# Model

a. States: decided by K-means

b. Actions: 0/1 to do spray/not do spray

c. Transition Probabilities: computed by iterating the data set, stored in transition matrices

d. Rewards: +10/-10 to smallest number/largest number of pest 

# Methods

1. K-means

K-means is used to make clusters for data points which are two dimensions vectors whose x is the mean number of males of SWD and y is the mean number of females of SWD. Then, I use the clusters to decide states. The stop criterion is a maximum number of iterations, which is 100. To decide the K, I used all the vectors to make a plot showing the change of mean distance between data points and their clusters’ centroids over the value of K. I choose the first “elbow point” where the rate of decrease sharply shifts as the number of K, which is 4, from the original data set.

2. Value Iteration & Polic Iteration & Q-learning

Value Iteration, Policy Iteration, and Q-learning are used to generate policy. The policies are doing different kind of action (do spray/not do spray) based on different kind of states (0, 1, 2, 3) in different month (June or July).

# Usage

python main.py filename method_name(v/p/q)



