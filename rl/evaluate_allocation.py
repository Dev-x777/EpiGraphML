import os
import sys
import pandas as pd

sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from rl.bandit_agent import allocate_resources
from rl.seir_simulator import run_simulation


def load_populations():

    project_root = os.path.dirname(
        os.path.dirname(__file__)
    )

    demo_path = os.path.join(
        project_root,
        "data",
        "processed",
        "tn_demographics.csv"
    )

    df = pd.read_csv(demo_path)

    population_map = {}

    for _, row in df.iterrows():
        population_map[row["region_id"]] = int(
            row["population"]
        )

    return population_map


def evaluate():

    allocations = allocate_resources()

    populations = load_populations()

    print("\nRESOURCE ALLOCATION + SEIR EVALUATION")
    print("-------------------------------------")

    total_reward = 0
    evaluated_regions = 0

    for allocation in allocations:

        region = allocation["region_id"]

        if region not in populations:
            print(
                f"\nSkipping {region}: population not found"
            )
            continue

        population = populations[region]

        predicted_cases = allocation["predicted_cases"]

        initial_infected = max(
            int(predicted_cases),
            1
        )

        # Without intervention
        no_intervention = run_simulation(
            population=population,
            initial_infected=initial_infected,
            days=30
        )

        # With intervention
        intervention = run_simulation(
            population=population,
            initial_infected=initial_infected,
            days=30,
            vaccine_doses=allocation["vaccine_doses"],
            testing_kits=allocation["testing_kits"],
            staff=allocation["staff"]
        )

        no_intervention_cases = sum(
            day["infected"]
            for day in no_intervention
        )

        intervention_cases = sum(
            day["infected"]
            for day in intervention
        )

        cases_prevented = (
            no_intervention_cases
            - intervention_cases
        )

        if no_intervention_cases > 0:
            reward = (
                cases_prevented
                / no_intervention_cases
            )
        else:
            reward = 0

        total_reward += reward
        evaluated_regions += 1

        print(f"\nRegion: {region}")

        print(
            f"Population: {population:,}"
        )

        print(
            f"Predicted Cases: "
            f"{predicted_cases}"
        )

        print(
            f"Vaccines: "
            f"{allocation['vaccine_doses']}"
        )

        print(
            f"Testing Kits: "
            f"{allocation['testing_kits']}"
        )

        print(
            f"Staff: "
            f"{allocation['staff']}"
        )

        print(
            f"Cases Without Intervention: "
            f"{no_intervention_cases:.0f}"
        )

        print(
            f"Cases With Intervention: "
            f"{intervention_cases:.0f}"
        )

        print(
            f"Cases Prevented: "
            f"{cases_prevented:.0f}"
        )

        print(
            f"Reward: "
            f"{reward:.4f}"
        )

    if evaluated_regions > 0:

        average_reward = (
            total_reward / evaluated_regions
        )

        print("\n=====================================")

        print(
            f"Average Allocation Reward: "
            f"{average_reward:.4f}"
        )


if __name__ == "__main__":
    evaluate()