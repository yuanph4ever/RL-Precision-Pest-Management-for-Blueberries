import numpy as np
import itertools

class qlearning:

    def __init__(self, states, rewards, actions, action_tm, discount_factor = 0.9, num_episodes=1000, alpha=0.1, display_process = 0):
        self.qtable = np.zeros((len(states), len(actions)))
        self.states = states
        self.actions = actions
        self.action_transition_matrices = action_tm
        self.rewards = rewards
        self.df = discount_factor
        self.num_episodes = num_episodes
        self.alpha = alpha
        self.display = display_process

    def generate_policy(self):

        for ith in range(self.num_episodes + 1):
            if self.display:
                if ith % 100 == 0:
                    print("Episode " + str(ith) + ":\n" + str(self.qtable))

            state = np.random.choice(self.states)

            for it in itertools.count():
                reward = self.rewards[state]
                action = np.random.choice(self.actions)
                probs = self.action_transition_matrices[action][state]
                next_state = np.random.choice(self.states, p = probs if sum(probs) == 1 else None)
                self.qtable[state][action] = (1-self.alpha)*self.qtable[state][action] + self.alpha*(reward+self.df*np.max(self.qtable[next_state]))

                if next_state == 0 or it == 1000:
                    break

                state = next_state

        policy = []
        for q in self.qtable:
            policy.append(np.argmax(q))

        return policy








