import os
import sys
import json
import numpy as np
import pandas as pd

sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from rl.seir_simulator import run_simulation


class LinUCBAgent:

    def __init__(self, context_dim, alpha=0.5):
        self.alpha = alpha
        self.A = np.identity(context_dim)
        self.b = np.zeros(context_dim)

    def get_theta(self):
        A_inv = np.linalg.inv(self.A)
        return A_inv @ self.b

    def score(self, context):
        A_inv = np.linalg.inv(self.A)
        theta = A_inv @ self.b

        exploitation = theta @ context

        exploration = self.alpha * np.sqrt(
            context @ A_inv @ context
        )

        return exploitation + exploration

    def update(self, context, reward):
        self.A += np.outer(context, context)
        self.b += reward * context


def load_data():
    project_root = os.path.dirname(
        os.path.dirname(__file__)
    )

    with open(
        os.path.join(
            project_root,
            "region_risk_output.json"
        ),
        "r"
    ) as f:
        risk_data = json.load(f)

    demo = pd.read_csv(
        os.path.join(
            project_root,
            "data",
            "processed",
            "tn_demographics.csv"
        )
    )

    population_map = dict(
        zip(
            demo["region_id"],
            demo["population"]
        )
    )

    return risk_data["regions"], population_map


def build_context(region):
    context = [
        region["risk_score"],
        min(region["predicted_cases"] / 1000, 1.0)
    ]

    context.extend(region["embedding"])

    return np.array(context, dtype=float)


def calculate_reward(
    population,
    initial_infected,
    vaccines,
    testing_kits,
    staff
):
    no_intervention = run_simulation(
        population=population,
        initial_infected=initial_infected,
        days=30
    )

    intervention = run_simulation(
        population=population,
        initial_infected=initial_infected,
        days=30,
        vaccine_doses=vaccines,
        testing_kits=testing_kits,
        staff=staff
    )

    no_cases = sum(
        day["infected"]
        for day in no_intervention
    )

    intervention_cases = sum(
        day["infected"]
        for day in intervention
    )

    if no_cases == 0:
        return 0

    reward = (
        no_cases - intervention_cases
    ) / no_cases

    return reward


def train_bandit(episodes=20):

    regions, population_map = load_data()

    sample_context = build_context(regions[0])

    agent = LinUCBAgent(
        context_dim=len(sample_context),
        alpha=0.5
    )

    total_vaccines = 10000
    total_testing = 5000
    total_staff = 100

    print("\nTRAINING LINUCB AGENT")
    print("---------------------")

    for episode in range(episodes):

        scores = []

        for region in regions:
            context = build_context(region)

            bandit_score = agent.score(context)

            scores.append(
                max(bandit_score, 0.001)
            )

        total_score = sum(scores)

        episode_reward = 0

        for region, score in zip(
            regions,
            scores
        ):

            proportion = score / total_score

            vaccines = int(
                total_vaccines * proportion
            )

            testing_kits = int(
                total_testing * proportion
            )

            staff = int(
                total_staff * proportion
            )

            population = int(
                population_map[
                    region["region_id"]
                ]
            )

            initial_infected = max(
                int(region["predicted_cases"]),
                1
            )

            reward = calculate_reward(
                population=population,
                initial_infected=initial_infected,
                vaccines=vaccines,
                testing_kits=testing_kits,
                staff=staff
            )

            context = build_context(region)

            agent.update(
                context,
                reward
            )

            episode_reward += reward

        average_reward = (
            episode_reward / len(regions)
        )

        print(
            f"Episode {episode + 1}: "
            f"Average Reward = "
            f"{average_reward:.4f}"
        )

    print("\nTraining complete.")

    return agent


if __name__ == "__main__":
    train_bandit()