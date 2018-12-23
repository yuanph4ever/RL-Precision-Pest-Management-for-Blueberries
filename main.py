import sys
from doKmeans import doKmeans
import numpy as np
from valueIteration import valueIteration
from policyIteration import policyIteration
from qlearning import qlearning
from transMatrix import transMatrix

# because labels are assigned randomly so they cannot be used as states directly
# this function creates a dictionary which gives a map of label to state
def sortCentroids(km_centroids):
    dict = []
    for cenList in km_centroids:
        sub_dict = {}
        for i in range(len(cenList)):
            x = np.array(cenList[i])
            val = np.linalg.norm(x)
            state = 0
            others = []
            for j in range(len(cenList)):
                if j != i:
                    others.append(j)
            for k in others:
                y = np.array(cenList[k])
                vy = np.linalg.norm(y)
                if val >= vy:
                    state += 1
            sub_dict[i] = state

        dict.append(sub_dict)
    return dict

# this function converts label list to state list using the dictionary of label to state
def labelList2stateList(labelList, label_state_list):
    stateList = []
    for i in range(len(labelList)):
        temp = []
        labels = labelList[i]
        label_state_dict = label_state_list[i]
        for label in labels:
            temp.append(label_state_dict[label])
        stateList.append(temp)

    stateListJun = stateList[0]
    stateListJul = stateList[1]
    stateListAug = stateList[2]

    finalStateList = []
    for j in range(len(stateListJun)):
        tripleList = [stateListJun[j], stateListJul[j], stateListAug[j]]
        finalStateList.append(tripleList)

    return finalStateList

def getActionNames(actions, actions_names):
    policy = []
    for a in actions:
        policy.append(actions_names[int(a)])
    return policy

if __name__ == '__main__':

    #filename = '/Users/paul.yuan/Desktop/MasterProject/Blueberry/SWDdata/DataSWD2016.xls'
    if len(sys.argv) < 3:
        print("usage: python main.py filename method_name(v/p/q)")
        exit(0)

    filename = sys.argv[1]
    method_name = sys.argv[2]

    if method_name != 'v' and method_name != 'p' and method_name != 'q':
        print("Invalid method name: use either v, p or q")
        print("usage: python main.py filename method_name(v/p/q)")
        exit(-1)

    show_transition_matrices = 0
    k = 4 # this is the number of clusters and also the number of states

    print("modeling data ...")
    kmeans = doKmeans(filename, k, 100, 16, 100)
    #kmeans.generatePlot(0)
    #kmeans.generatePlot(1)
    #kmeans.generatePlot(2)
    #kmeans.makeMeanDisPlot(kmeans.dataList[0])
    label_state_list = sortCentroids(kmeans.centroids)
    stateList = labelList2stateList(kmeans.labels, label_state_list)
    print(len(stateList))

    # then we have the state list, which contains multiple triple lists having state of Jun, Jul, and Aug
    # and we also have the spray list, which contains multiple triple lists having times of spray of Jun, Jul and Aug
    # we use those two lists to make the transition matrices
    # we'll have four matrices: not do spray Jun->Jul Jul->Aug and do spray Jun->Jul Jul->Aug
    shape = [k, k]
    transM = transMatrix(stateList, kmeans.sprayList, shape)

    if show_transition_matrices:
        print("\nDisplay transition matrices\n")
        print("--- not do spray: June to July ---")
        print(transM.tmJun2Jul)
        print("--- do spray: June to July ---")
        print(transM.stmJun2Jul)
        print("--- not do spray: July to August ---")
        print(transM.tmJul2Aug)
        print("--- do spray: July to August ---")
        print(transM.stmJul2Aug)

    states = [0, 1, 2, 3]
    actions = [0, 1]
    actions_names = ['not do spray', 'do spray']
    rewards = {0:10, 1:0, 2:0, 3:-10}
    DISCOUNT_FACTOR = 0.9
    print("\nmodeling done ...")

    show_round = 1
    for month in ['June', 'July']:
        if month == 'June':
            action_tm = {0: transM.tmJun2Jul, 1: transM.stmJun2Jul} # transition matrices
        else:
            action_tm = {0: transM.tmJul2Aug, 1: transM.stmJul2Aug} # transition matrices

        policy = []
        if method_name == 'v':
            print("\nValue Iteration")
            value_iteration = valueIteration(states, rewards, actions, action_tm, DISCOUNT_FACTOR, display_process=show_round)
            policy = getActionNames(value_iteration.generate_policy(), actions_names)
        elif method_name == 'p':
            print("\nPolicy Iteration")
            policy_iteration = policyIteration(states, rewards, actions, action_tm, DISCOUNT_FACTOR, display_process=show_round)
            policy = getActionNames(policy_iteration.generate_policy(), actions_names)
        else:
            print("\nQ-learning")
            q_learning = qlearning(states, rewards, actions, action_tm, DISCOUNT_FACTOR, alpha=0.1, display_process=show_round)
            policy = getActionNames(q_learning.generate_policy(), actions_names)

        print("\nIn " + month + " policy is " + str(policy))
        












