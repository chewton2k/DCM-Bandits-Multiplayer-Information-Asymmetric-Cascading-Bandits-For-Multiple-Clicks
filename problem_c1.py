import numpy as np
import random
import math

def calc_score(recommendation, probs, num_positions):
    """Expected reward under cascade (termination) model."""
    reward = 0.0
    cont_prob = 1.0
    for i in range(min(len(recommendation), num_positions)):
        arm = recommendation[i]
        reward += cont_prob * probs[arm]
        cont_prob *= (1 - probs[arm])
    return reward

def simulate_cascading_bandit(T, indiv_arms, num_arms, num_positions, num_players, click_probabilities):
    # Each player maintains their own empirical means and counts
    w_hat = np.zeros((num_players, num_arms))  # per-player empirical means
    n = np.zeros((num_players, num_arms))      # per-player observation counts

    cumulative_regret = 0
    regret = np.zeros(T)
    lam = 1                     # Exploration multiplier
    t = 0                       # Current round
    
    # Optimal expected reward (global)
    sorted_arms = sorted(range(num_arms), key=lambda i: -click_probabilities[i])
    optimal_arms = sorted_arms[:num_positions]
    optimal_score = calc_score(optimal_arms, click_probabilities, num_positions)

    while t < T:
        # === Exploration Phase ===
        exploration_rounds = lam * num_arms
        for _ in range(exploration_rounds):
            if t >= T:
                break

            # Round-robin selection of arms
            j = (t // lam) % num_arms
            selected_arms = [(j + k) % num_arms for k in range(num_positions)]
            scores = []

            for p in range(num_players):
                # Simulate clicks for all K arms
                rewards = [1 if random.random() < click_probabilities[selected_arms[k]] else 0
                           for k in range(num_positions)]

                # Update all K arms with their observed rewards
                for k, arm_idx in enumerate(selected_arms):
                    n[p][arm_idx] += 1
                    w_hat[p][arm_idx] = ((n[p][arm_idx] - 1) * w_hat[p][arm_idx] + rewards[k]) / n[p][arm_idx]

                scores.append(calc_score(selected_arms, click_probabilities, num_positions))

            avg_score = np.mean(scores)
            regret[t] = regret[t-1] + (optimal_score - avg_score) if t > 0 else (optimal_score - avg_score)
            t += 1

        if t >= T:
            break

        # === Exploitation Phase ===
        # Each player picks top-K arms based on its local w_hat
        Rt_list = []
        for p in range(num_players):
            Rt_p = sorted(range(num_arms), key=lambda i: (-w_hat[p][i], i))[:num_positions]
            Rt_list.append(Rt_p)

        # Run exploitation until next power-of-two boundary
        r = math.floor(math.log2(t)) + 1
        Texploit = min(2**r - t, T - t)

        for _ in range(Texploit):
            if t >= T:
                break
            scores = []

            for p in range(num_players):
                Rt_p = Rt_list[p]
                rewards = [1 if random.random() < click_probabilities[Rt_p[k]] else 0
                           for k in range(num_positions)]

                # Update all K arms with their observed rewards
                for k, arm_idx in enumerate(Rt_p):
                    n[p][arm_idx] += 1
                    w_hat[p][arm_idx] = ((n[p][arm_idx] - 1) * w_hat[p][arm_idx] + rewards[k]) / n[p][arm_idx]

                scores.append(calc_score(Rt_p, click_probabilities, num_positions))

            avg_score = np.mean(scores)
            regret[t] = regret[t-1] + (optimal_score - avg_score) if t > 0 else (optimal_score - avg_score)
            t += 1

        lam += 1

    return regret
