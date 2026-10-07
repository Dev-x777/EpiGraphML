import os
import sys

# Allow imports from the project
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from rl.bandit_agent import allocate_resources
from rl.seir_simulator import run_simulation


def evaluate():

    allocations = allocate_resources()

    print("\nRESOURCE ALLOCATION + SEIR EVALUATION")
    print("-------------------------------------")

    total_reward = 0

    for allocation in allocations:

        region = allocation["region_id"]
        predicted_cases = allocation["predicted_cases"]

        # Temporary population assumption.
        # Later we will replace this with real district population.
        population = 1_000_000

        # Avoid starting simulation with 0 infections
        initial_infected = max(predicted_cases, 1)

        # --------------------------
        # Without intervention
        # --------------------------

        no_intervention = run_simulation(
            population=population,
            initial_infected=initial_infected,
            days=30
        )

        # --------------------------
        # With intervention
        # --------------------------

        intervention = run_simulation(
            population=population,
            initial_infected=initial_infected,
            days=30,
            vaccine_doses=allocation["vaccine_doses"],
            testing_kits=allocation["testing_kits"],
            staff=allocation["staff"]
        )

        # Infection burden over simulation period
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
            reward = cases_prevented / no_intervention_cases
        else:
            reward = 0

        total_reward += reward

        print(f"\nRegion: {region}")

        print(
            f"Cases without intervention: "
            f"{no_intervention_cases:.0f}"
        )

        print(
            f"Cases with intervention: "
            f"{intervention_cases:.0f}"
        )

        print(
            f"Cases prevented: "
            f"{cases_prevented:.0f}"
        )

        print(
            f"Reward: "
            f"{reward:.4f}"
        )

    average_reward = total_reward / len(allocations)

    print("\n=====================================")

    print(
        f"Average Allocation Reward: "
        f"{average_reward:.4f}"
    )


if __name__ == "__main__":
    evaluate()