import numpy as np

class CascadingBanditPB:
    def __init__(self, num_arms, probabilities, num_positions):
        assert len(probabilities) == num_arms, "Probabilities must match the number of arms."
        assert num_positions <= num_arms, "Number of positions cannot exceed number of arms."

        self.num_arms = num_arms
        self.probabilities = probabilities
        self.num_positions = num_positions
        self.reset()

    def reset(self):
        self.history = []  # Stores history of arm selections and clicks

    def recommend(self, selected_arms):
        assert len(selected_arms) == self.num_positions, "Number of selected arms must match num_positions."

        clicks = [int(np.random.rand() < self.probabilities[arm]) for arm in selected_arms]
        self.history.append((selected_arms, clicks))
        return clicks

def optimize(click_probabilities, num_positions):
    click_probabilities.sort(reverse=True)
    return calc_score(list(range(num_positions)), click_probabilities)

def calc_score(positions, click_probabilities):
    prob = 1
    for arm in positions:
        arm = int(arm)
        prob *= (1 - click_probabilities[arm])
    return 1 - prob

def simulate_cascading_bandit(total_rounds, num_arms, num_positions, num_players, click_probabilities):
    bandit = CascadingBanditPB(num_arms=num_arms, probabilities=click_probabilities, num_positions=num_positions)

    empirical_means = np.zeros((num_players, num_arms))
    observations = np.zeros(num_arms)
    desired_set = list(range(num_arms))
    current_order = np.arange(num_positions)
    UCB = np.zeros((num_players, num_arms))
    LCB = np.zeros((num_players, num_arms))

    optimal_score = optimize(click_probabilities.copy(), num_positions)
    current_regret = 0
    regret = []

    for t in range(total_rounds):
        num_popped = 0

        # Update UCB/LCB intervals
        for p in range(num_players):
            for arm in range(num_arms):
                if observations[arm] == 0:
                    UCB[p][arm] = np.inf
                    LCB[p][arm] = -np.inf
                else:
                    confidence = ((1.5) * np.log(total_rounds) / observations[arm]) ** 0.5
                    UCB[p][arm] = empirical_means[p][arm] + confidence
                    LCB[p][arm] = empirical_means[p][arm] - confidence

        recommendations = [desired_set[i] for i in current_order]

        # Remove disjoint arms from recommendation set
        popped = False
        for p in range(num_players):
            if len(desired_set) > num_positions:
                for i, rec in enumerate(recommendations):
                    counter = sum(UCB[p][rec] < LCB[p][arm] for arm in desired_set)
                    if counter >= num_positions:
                        recommendations[i] = desired_set[(current_order[-1] + (i + 1)) % num_positions]
                        desired_set.pop((current_order[0] + i) % len(desired_set))
                        num_popped += 1
                        popped = True
                        break
            if popped:
                break

        # Collect full feedback
        clicks_all_players = [bandit.recommend(recommendations) for _ in range(num_players)]
        score = calc_score(recommendations, click_probabilities)

        # Update estimates using all k arms
        for p in range(num_players):
            for i, arm in enumerate(recommendations):
                inc = clicks_all_players[p][i]
                empirical_means[p][arm] = (
                    empirical_means[p][arm] * observations[arm] + inc
                ) / (observations[arm] + 1)
                observations[arm] += (1 / num_players)

        # Update recommendation ordering
        if num_popped == 0:
            current_order = (current_order + 1) % len(desired_set)
        else:
            current_order = np.arange(num_positions) % len(desired_set)

        current_regret += optimal_score - score
        regret.append(current_regret)

    return regret
