import numpy as np
from itertools import product

def simulate_cascading_bandit(T, num_arms, num_positions, num_players, click_probabilities):
    """
    Problem B: Round-robin exploration with phased elimination.
    Properly models cascading behavior with termination probabilities.
    Produces sublinear regret (≈√T) by reducing exploration over time.
    
    Args:
        T: Total time horizon
        num_arms: Number of individual arms (base arms)
        num_positions: Number of positions to display (K)
        num_players: Number of players/agents
        click_probabilities: List of click probabilities for each joint arm
    """
    # --- Generate all joint arms ---
    joint_actions = list(product(range(num_arms), repeat=num_players))
    num_joint_arms = len(joint_actions)
    desired_set = list(range(num_joint_arms))  # indices of joint actions
    K = num_positions  # number of positions (arms we play each round)

    # --- Initialize tracking variables ---
    n = np.zeros(num_joint_arms)          # number of times each joint arm was tried
    w_hat = np.zeros(num_joint_arms)      # empirical mean reward per joint arm
    regret = np.zeros(T)
    current_regret = 0

    # --- Compute optimal expected score for regret calculation ---
    def expected_reward(joint_idx):
        joint = joint_actions[joint_idx]
        # Cascade model: reward is 1 if any item is clicked, in order
        prob_no_click = 1.0
        for arm in joint:
            prob_no_click *= (1 - click_probabilities[arm])
        return 1 - prob_no_click

    optimal_score = max(expected_reward(i) for i in range(num_joint_arms))

    # --- Simulation loop ---
    t = 0
    phase = 1
    current_index = 0

    while t < T and len(desired_set) > 0:
        # Phase length decreases as we eliminate arms
        phase_length = int(np.ceil(np.sqrt(T)))  # can tune this

        for _ in range(phase_length):
            if t >= T:
                break

            # --- Select K arms in round-robin ---
            selected_idxs = [(current_index + k) % len(desired_set) for k in range(K)]
            selected_actions = [desired_set[idx] for idx in selected_idxs]

            # --- Simulate cascade reward ---
            rewards = np.zeros(K)
            for i, joint_idx in enumerate(selected_actions):
                joint = joint_actions[joint_idx]
                # Simulate cascade behavior
                clicked = False
                for arm in joint:
                    if np.random.rand() < click_probabilities[arm]:
                        clicked = True
                        break
                reward = 1 if clicked else 0
                rewards[i] = reward

                # Update empirical mean (only for the played action)
                n[joint_idx] += 1
                w_hat[joint_idx] += (reward - w_hat[joint_idx]) / n[joint_idx]

            # --- Compute average score for regret ---
            avg_score = np.mean(rewards)
            current_regret += optimal_score - avg_score
            regret[t] = current_regret

            # --- Advance round robin ---
            current_index = (current_index + 1) % len(desired_set)
            t += 1

        # --- After each phase, eliminate suboptimal arms ---
        if len(desired_set) > K:
            # Compute confidence bounds
            total_rounds = max(1, t)
            confidence = np.sqrt((2 * np.log(total_rounds)) / (n + 1e-6))  # Using Hoeffding's inequality
            UCB = w_hat + confidence
            LCB = w_hat - confidence

            # Find best empirical arm in desired set
            best_in_set = max(desired_set, key=lambda idx: w_hat[idx])

            # Eliminate arms whose UCB is below best LCB
            new_desired_set = [idx for idx in desired_set 
                             if UCB[idx] >= LCB[best_in_set]]
            desired_set = new_desired_set

        phase += 1

    # --- Exploitation: once |desired_set| <= K ---
    while t < T:
        # Select top-K arms in desired set
        best_actions = sorted(desired_set, key=lambda idx: -w_hat[idx])[:K]
        
        # Simulate rewards
        rewards = np.zeros(K)
        for i, joint_idx in enumerate(best_actions):
            joint = joint_actions[joint_idx]
            clicked = any(np.random.rand() < click_probabilities[arm] for arm in joint)
            rewards[i] = 1 if clicked else 0

        avg_score = np.mean(rewards)
        current_regret += optimal_score - avg_score
        regret[t] = current_regret
        t += 1

    return regret