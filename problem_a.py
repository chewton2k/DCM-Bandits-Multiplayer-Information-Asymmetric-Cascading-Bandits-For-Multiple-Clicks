import numpy as np
import random
import math

class CascadingBandit:
    def __init__(self, num_arms, probabilities, num_positions, num_players):
        self.num_arms = num_arms
        self.probabilities = probabilities
        self.num_positions = num_positions
        self.num_players = num_players
        self.reset()

    def reset(self):
        self.history = []

    def recommend(self, selected_arms):
        assert len(selected_arms) == self.num_positions
        
        for i, arm in enumerate(selected_arms):
            if np.random.rand() < self.probabilities[arm]:
                click = i
                break
        else:
            click = self.num_positions
        
        self.history.append((selected_arms, click))
        return click

def compute_ucb(empirical_means, counts, t):
    ucb_values = np.zeros(len(empirical_means))
    for e in range(len(empirical_means)):
        if counts[e] == 0:
            ucb_values[e] = np.inf
        else:
            ucb_values[e] = empirical_means[e] + math.sqrt((1.5 * math.log(t + 1)) / counts[e])
    return ucb_values

def simulate_cascading_bandit(total_rounds, indiv_arms, num_arms, num_positions, num_players, cl_prob):
    click_probabilities = cl_prob
    bandit = CascadingBandit(num_arms, click_probabilities, num_positions, num_players)
    
    empirical_means = np.zeros(num_arms)
    counts = np.zeros(num_arms)
    regret = []
    optimal_score = 1 - np.prod([1 - p for p in sorted(click_probabilities, reverse=True)[:num_positions]])
    current_regret = 0
    
    for t in range(1, total_rounds + 1):
        ucb_values = compute_ucb(empirical_means, counts, t)
        selected_arms = np.argsort(ucb_values)[-num_positions:][::-1]
        click = bandit.recommend(selected_arms)
        
        for i, arm in enumerate(selected_arms[:click + 1]):
            reward = 1 if i == click else 0
            counts[arm] += 1
            empirical_means[arm] = ((empirical_means[arm] * (counts[arm] - 1)) + reward) / counts[arm]
        
        score = 1 - np.prod([1 - click_probabilities[a] for a in selected_arms])
        current_regret += optimal_score - score
        regret.append(current_regret)
    
    return regret
