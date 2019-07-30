import numpy as np

"""
Contains the definition of the agent that will run in an
environment.
"""

class RandomAgent:
    def __init__(self):
        """Init a new agent.
        """

    def choose(self):
        """Acts given an observation of the environment.

        Takes as argument an observation of the current state, and
        returns the chosen action.
        """
        return np.random.randint(0, 10)

    def update(self, action, reward):
        """Receive a reward for performing given action on
        given observation.

        This is where your agent can learn.
        """
        pass


##############################################
# IMPLEMENTATIONS OF THE MANDATORY ALGORITHMS

class epsGreedyAgent:
    def __init__(self):
        self.A          = [0,1,2,3,4,5,6,7,8,9]
        self.rewards    = np.zeros(len(self.A))
        self.play_count = np.zeros(len(self.A))
        self.epsilon    = 1

    def choose(self):
        if np.random.random() < self.epsilon:
            self.epsilon *= 0.92
            return np.random.randint(0, 10)
        else:
            return np.argmax(self.rewards / self.play_count)

    def update(self, action, reward):
        self.rewards[action] += reward
        self.play_count[action] += 1


class BesaAgent():
    # https://hal.archives-ouvertes.fr/hal-01025651v1/document
    def __init__(self):
        self.A = [0,1,2,3,4,5,6,7,8,9]
        self.rewards    = [[] for _ in range(len(self.A))]
        self.play_count = [0] * len(self.A)
        self.n = 0

    def besa_2_arms(self, index1, index2):
        if self.play_count[index1] < self.play_count[index2]:
            index_min = index1
            index_max = index2
        else:
            index_min = index2
            index_max = index1

        N = self.play_count[index_min]
        mean1 = np.mean(self.rewards[index_min])
        mean2 = np.mean(np.random.choice(self.rewards[index_max], N, replace=False))

        if mean1 < mean2:
            return index_max
        else:
            return index_min

    def besa_k_actions(self, indices):
        if len(indices) == 1:
            return indices[0]

        index = len(indices)//2
        return self.besa_2_arms(self.besa_k_actions(indices[:index]), self.besa_k_actions(indices[index:]))

    def choose(self):
        # Play each machine once at the beginning
        self.n += 1
        if self.n < len(self.A) + 1:
            return self.n - 1

        return self.besa_k_actions(self.A)

    def update(self, action, reward):
        self.rewards[action].append(reward)
        self.play_count[action] += 1


class UCBAgent:
    # https://homes.di.unimi.it/~cesabian/Pubblicazioni/ml-02.pdf
    def __init__(self):
        self.A = [0,1,2,3,4,5,6,7,8,9]
        self.rewards    = np.zeros(len(self.A))
        self.play_count = np.zeros(len(self.A))
        self.n = 0

    def choose(self):
        # Play each machine once at the beginning
        self.n += 1
        if self.n < (len(self.A) + 1):
            return self.n - 1

        avg_reward = self.rewards / self.play_count

        return np.argmax(avg_reward + np.sqrt(2 * np.log(self.n) / self.play_count))

    def update(self, action, reward):
        self.rewards[action] += reward
        self.play_count[action] += 1


class SoftmaxAgent:
    # https://www.cs.mcgill.ca/~vkules/bandits.pdf
    def __init__(self):
        self.A          = [0,1,2,3,4,5,6,7,8,9]
        self.rewards    = np.zeros(len(self.A))
        self.play_count = np.zeros(len(self.A))
        self.n = 0
        self.tau = 0.29

    def choose(self):
        # Play each machine once at the beginning
        self.n += 1
        if self.n < len(self.A) + 1:
            return self.n - 1

        if self.n > 180:
            return np.argmax(self.rewards/self.play_count)

        exp_rewards = np.exp(self.rewards / self.play_count / self.tau)
        exp_sum = np.sum(exp_rewards)
        prob_softmax = exp_rewards / exp_sum

        # Roulette wheel selection
        cumulative_prob = [np.sum(prob_softmax[:k]) for k in range(1, len(prob_softmax) + 1)]
        random_number = np.random.random()
        index = 0
        while random_number > cumulative_prob[index]:
            index += 1

        return index

    def update(self, action, reward):
        self.rewards[action] += reward
        self.play_count[action] += 1

##############################################
# IMPLEMENTATION OF THE BONUS ALGORITHM (KL-UCB)

from scipy.optimize import minimize

class KLUCBAgent:
    # See: https://hal.archives-ouvertes.fr/hal-00738209v2
    def __init__(self):
        self.A = [0,1,2,3,4,5,6,7,8,9]
        self.rewards    = np.zeros(len(self.A))
        self.play_count = np.zeros(len(self.A))
        self.n = 0
        self.c = 3

    def kl(self, p, q):
        if p == 0 or p == 1:
            return 0
        if q == 0:
            return 10
        eps = 1e-15
        p = min(max(p, eps), 1 - eps)
        q = min(max(q, eps), 1 - eps)

        return p * np.log(p / q) + (1 - p) * np.log((1 - p) / (1 - q))

    def choose(self):
        self.n = self.n + 1
        nb_actions = len(self.A)
        if self.n < ( nb_actions + 1):
            return self.n - 1

        list_q = np.zeros(nb_actions)
        membre_equation_droit = np.log(self.n) + self.c * np.log(np.log(self.n))
        x0 = 0.5

        for k in range(nb_actions):
            cons = ({'type': 'ineq', 'fun': lambda x: membre_equation_droit - (self.play_count[k] * self.kl(self.rewards[k]/self.play_count[k], x))})
            bounds = [(0, 1)]
            res = minimize(lambda x: -x, x0, bounds=bounds, constraints=cons, method='SLSQP', tol=0.1)
            list_q[k] = res['x'][0]
        return np.argmax(list_q)


    def update(self, action, reward):
        self.rewards[action] += reward
        self.play_count[action] += 1


##############################################
# OTHER ALGORITHM (JUST FOR TESTING)

class epsSoftmaxAgent:
    # https://www.cs.mcgill.ca/~vkules/bandits.pdf
    def __init__(self):
        self.A = [0,1,2,3,4,5,6,7,8,9]
        self.rewards = np.zeros(len(self.A))
        self.play_count = np.zeros(len(self.A))
        self.epsilon = 1
        self.tau = 0.31

    def choose(self):
        if np.random.random() > self.epsilon:
            exp_rewards = np.exp(self.rewards / self.play_count / self.tau)
            exp_sum = np.sum(exp_rewards)
            prob_softmax = exp_rewards / exp_sum
            cumulative_prob = [np.sum(prob_softmax[:k]) for k in range(1, len(prob_softmax) + 1)]
            random_number = np.random.random()
            i = 0
            while random_number > cumulative_prob[i]:
                i += 1

            return i
        else:
            self.epsilon *= 0.95
            return np.random.randint(0, 10)

    def update(self, action, reward):
        self.rewards[action] += reward
        self.play_count[action] += 1


# Choose which Agent is run for scoring
Agent = UCBAgent
