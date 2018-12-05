import numpy as np

class valueIteration:

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

    def generate_policy(self):
        delta = float("inf")
        round_num = 0

        while delta > self.theta:

            if self.display:
                print("\nRound " + str(round_num))
                print(self.statesValue)  # TODO can be reshaped to a better look

            delta = 0
            for s in self.states:
                best_action, best_action_value = self.next_best_action(s)
                delta = max(delta, np.abs(best_action_value - self.statesValue[s]))
                self.statesValue[s] = best_action_value
            round_num += 1

        policy = np.zeros(len(self.states))
        for s in self.states:
            best_action, best_action_value = self.next_best_action(s)
            policy[s] = best_action

        return policy




