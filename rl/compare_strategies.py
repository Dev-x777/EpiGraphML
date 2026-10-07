import os
import sys
import json
import random

import numpy as np
import pandas as pd

sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from rl.seir_simulator import run_simulation


def load_data():
    project_root = os.path.dirname(os.path.dirname(__file__))

    with open(
        os.path.join(project_root, "region_risk_output.json"),
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
        zip(demo["region_id"], demo["population"])
    )

    return risk_data["regions"], population_map


def simulate_region(
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

    no_burden = sum(
        day["infected"]
        for day in no_intervention
    )

    intervention_burden = sum(
        day["infected"]
        for day in intervention
    )

    prevented = (
        no_burden - intervention_burden
    )

    if no_burden > 0:
        reward = prevented / no_burden
    else:
        reward = 0

    return reward, prevented, no_burden


def create_allocations(
    regions,
    weights,
    total_vaccines=10000,
    total_testing=5000,
    total_staff=100
):
    weights = np.array(
        weights,
        dtype=float
    )

    weights = np.maximum(
        weights,
        0.0001
    )

    total_weight = np.sum(weights)

    allocations = []

    for region, weight in zip(
        regions,
        weights
    ):
        proportion = (
            weight / total_weight
        )

        allocations.append({
            "region_id": region["region_id"],
            "vaccines": int(
                total_vaccines * proportion
            ),
            "testing_kits": int(
                total_testing * proportion
            ),
            "staff": int(
                total_staff * proportion
            )
        })

    return allocations


def random_strategy(regions):
    weights = [
        random.random()
        for _ in regions
    ]

    return create_allocations(
        regions,
        weights
    )


def population_strategy(
    regions,
    population_map
):
    weights = [
        population_map[
            region["region_id"]
        ]
        for region in regions
    ]

    return create_allocations(
        regions,
        weights
    )


def risk_strategy(regions):
    weights = [
        region["risk_score"]
        for region in regions
    ]

    return create_allocations(
        regions,
        weights
    )


class LinUCBAgent:

    def __init__(
        self,
        context_dimension,
        alpha=0.5
    ):
        self.alpha = alpha

        self.A = np.identity(
            context_dimension
        )

        self.b = np.zeros(
            context_dimension
        )

    def score(self, context):
        A_inv = np.linalg.inv(self.A)

        theta = (
            A_inv @ self.b
        )

        exploitation = (
            theta @ context
        )

        exploration = (
            self.alpha
            * np.sqrt(
                context
                @ A_inv
                @ context
            )
        )

        return (
            exploitation
            + exploration
        )

    def update(
        self,
        context,
        reward
    ):
        self.A += np.outer(
            context,
            context
        )

        self.b += (
            reward * context
        )


def build_context(region):
    context = [
        region["risk_score"],
        min(
            region["predicted_cases"]
            / 1000,
            1.0
        )
    ]

    context.extend(
        region["embedding"]
    )

    return np.array(
        context,
        dtype=float
    )


def linucb_strategy(
    regions,
    population_map,
    episodes=20
):
    context_dimension = len(
        build_context(
            regions[0]
        )
    )

    agent = LinUCBAgent(
        context_dimension,
        alpha=0.5
    )

    for _ in range(episodes):

        scores = []

        for region in regions:
            context = build_context(
                region
            )

            score = agent.score(
                context
            )

            scores.append(
                max(score, 0.001)
            )

        allocations = create_allocations(
            regions,
            scores
        )

        for region, allocation in zip(
            regions,
            allocations
        ):
            region_id = region[
                "region_id"
            ]

            population = int(
                population_map[
                    region_id
                ]
            )

            initial_infected = max(
                int(
                    region[
                        "predicted_cases"
                    ]
                ),
                1
            )

            reward, _, _ = simulate_region(
                population=population,
                initial_infected=initial_infected,
                vaccines=allocation[
                    "vaccines"
                ],
                testing_kits=allocation[
                    "testing_kits"
                ],
                staff=allocation[
                    "staff"
                ]
            )

            context = build_context(
                region
            )

            agent.update(
                context,
                reward
            )

    final_scores = []

    for region in regions:
        context = build_context(
            region
        )

        score = agent.score(
            context
        )

        final_scores.append(
            max(score, 0.001)
        )

    return create_allocations(
        regions,
        final_scores
    )


def evaluate_strategy(
    name,
    regions,
    population_map,
    allocations
):
    total_reward = 0
    total_prevented = 0
    total_no_burden = 0

    for region, allocation in zip(
        regions,
        allocations
    ):
        region_id = region[
            "region_id"
        ]

        population = int(
            population_map[
                region_id
            ]
        )

        initial_infected = max(
            int(
                region[
                    "predicted_cases"
                ]
            ),
            1
        )

        reward, prevented, no_burden = (
            simulate_region(
                population=population,
                initial_infected=initial_infected,
                vaccines=allocation[
                    "vaccines"
                ],
                testing_kits=allocation[
                    "testing_kits"
                ],
                staff=allocation[
                    "staff"
                ]
            )
        )

        total_reward += reward
        total_prevented += prevented
        total_no_burden += no_burden

    average_reward = (
        total_reward
        / len(regions)
    )

    overall_reduction = (
        total_prevented
        / total_no_burden
        if total_no_burden > 0
        else 0
    )

    return {
        "strategy": name,
        "average_reward":
            average_reward,
        "infection_burden_prevented":
            total_prevented,
        "overall_reduction":
            overall_reduction
    }


def evaluate_random_multiple_times(
    regions,
    population_map,
    trials=30
):
    rewards = []
    prevented_values = []
    reduction_values = []

    for trial in range(trials):

        random.seed(trial)

        allocations = (
            random_strategy(regions)
        )

        result = evaluate_strategy(
            "Random",
            regions,
            population_map,
            allocations
        )

        rewards.append(
            result[
                "average_reward"
            ]
        )

        prevented_values.append(
            result[
                "infection_burden_prevented"
            ]
        )

        reduction_values.append(
            result[
                "overall_reduction"
            ]
        )

    return {
        "strategy": "Random",
        "average_reward":
            np.mean(rewards),
        "infection_burden_prevented":
            np.mean(prevented_values),
        "overall_reduction":
            np.mean(reduction_values)
    }


def compare():
    regions, population_map = (
        load_data()
    )

    print(
        "\nCOMPARING RESOURCE "
        "ALLOCATION STRATEGIES"
    )

    print(
        "--------------------------------"
    )

    print(
        "\nEvaluating Random baseline..."
    )

    random_result = (
        evaluate_random_multiple_times(
            regions,
            population_map,
            trials=30
        )
    )

    print(
        "Evaluating Population strategy..."
    )

    population_result = (
        evaluate_strategy(
            "Population",
            regions,
            population_map,
            population_strategy(
                regions,
                population_map
            )
        )
    )

    print(
        "Evaluating Risk-Based strategy..."
    )

    risk_result = (
        evaluate_strategy(
            "Risk-Based",
            regions,
            population_map,
            risk_strategy(regions)
        )
    )

    print(
        "Training and evaluating LinUCB..."
    )

    linucb_allocations = (
        linucb_strategy(
            regions,
            population_map
        )
    )

    linucb_result = (
        evaluate_strategy(
            "LinUCB",
            regions,
            population_map,
            linucb_allocations
        )
    )

    results = [
        random_result,
        population_result,
        risk_result,
        linucb_result
    ]

    print("\nRESULTS")
    print("-------------------------------")

    for result in results:

        print(
            f"\nStrategy: "
            f"{result['strategy']}"
        )

        print(
            f"Average Reward: "
            f"{result['average_reward']:.4f}"
        )

        print(
            f"Infection Burden Prevented: "
            f"{result['infection_burden_prevented']:.0f}"
        )

        print(
            f"Overall Reduction: "
            f"{result['overall_reduction'] * 100:.2f}%"
        )

    best = max(
        results,
        key=lambda x:
        x[
            "infection_burden_prevented"
        ]
    )

    print(
        "\n==============================="
    )

    print(
        f"Best Strategy by "
        f"Infection Burden Prevented: "
        f"{best['strategy']}"
    )


if __name__ == "__main__":
    compare()