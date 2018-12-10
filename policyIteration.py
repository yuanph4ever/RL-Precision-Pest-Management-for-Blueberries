import numpy as np

class policyIteration:

    def __init__(self, states, rewards, actions, action_tm, discount_factor = 0.9, theta = 0.1, display_process = 0):
        self.states = states
        self.rewards = rewards
        self.actions = actions
        self.action_tm = action_tm
        self.dis = discount_factor
        self.theta = theta
        self.statesValue = np.zeros(len(states))
        self.display = display_process

    def next_best_action(self, now_state):
        action_values = np.zeros(len(self.actions))
        for a in self.actions:
            transM = self.action_tm[a]
            probs = transM[now_state]
            actionV = 0.0
            for new_state in range(len(probs)):
                actionV += probs[new_state] * (self.rewards[now_state] + self.dis * self.statesValue[new_state])
            action_values[a] = actionV
        return np.argmax(action_values), np.max(action_values)

    def evaluate_policy(self, policy):
        delta = float("inf")
        while delta > self.theta:
            delta = 0
            for state in self.states:
                action = policy[state]
                transition_mat = self.action_tm[action]
                probs = transition_mat[state]
                stateV = 0
                for next_state in self.states:
                    stateV += probs[next_state] * (self.rewards[state] + self.dis * self.statesValue[next_state])
                delta = max(delta, np.abs(self.statesValue[state] - stateV))
                self.statesValue[state] = stateV

    def generate_policy(self):
        stable = 0
        round_num = 0
        policy = [0, 0, 0, 0]
        while not stable:
            stable = 1
            self.evaluate_policy(policy)
            if self.display:
                print("\nRound " + str(round_num))
                print("current policy: " + str(policy))
                print("states values: " + str(self.statesValue))
            round_num += 1
            for state in self.states:
                best_action, best_action_value = self.next_best_action(state)
                policy_action = policy[state]
                policy[state] = best_action
                if best_action != policy_action:
                    stable = 0
        return policy





