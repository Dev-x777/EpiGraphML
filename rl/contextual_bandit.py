import json
import os
import numpy as np


class LinUCBAllocator:

    def __init__(self, alpha=1.0):
        self.alpha = alpha

    def load_risk_data(self):

        project_root = os.path.dirname(
            os.path.dirname(__file__)
        )

        file_path = os.path.join(
            project_root,
            "region_risk_output.json"
        )

        with open(file_path, "r") as f:
            data = json.load(f)

        return data


    def build_context(self, region):

        risk_score = region["risk_score"]
        predicted_cases = region["predicted_cases"]

        embedding = region["embedding"]

        # Normalize predicted cases roughly
        normalized_cases = min(
            predicted_cases / 1000,
            1.0
        )

        context = [
            risk_score,
            normalized_cases
        ]

        # Add graph embedding
        context.extend(embedding)

        return np.array(context, dtype=float)


    def calculate_score(self, context):

        # Simple LinUCB-style scoring
        # Since we do not yet have learned historical parameters,
        # risk-related information acts as initial exploitation signal.

        theta = np.ones(len(context))

        exploitation = np.dot(
            theta,
            context
        )

        exploration = (
            self.alpha
            * np.sqrt(
                np.dot(context, context)
            )
        )

        score = exploitation + exploration

        return score


    def allocate(
        self,
        total_vaccines=10000,
        total_testing_kits=5000,
        total_staff=100
    ):

        data = self.load_risk_data()

        regions = data["regions"]

        scored_regions = []

        for region in regions:

            context = self.build_context(region)

            score = self.calculate_score(context)

            scored_regions.append({
                "region_id": region["region_id"],
                "risk_score": region["risk_score"],
                "predicted_cases": region["predicted_cases"],
                "score": max(score, 0)
            })

        total_score = sum(
            region["score"]
            for region in scored_regions
        )

        allocations = []

        for region in scored_regions:

            if total_score == 0:
                proportion = 1 / len(scored_regions)
            else:
                proportion = (
                    region["score"]
                    / total_score
                )

            vaccines = int(
                total_vaccines
                * proportion
            )

            testing_kits = int(
                total_testing_kits
                * proportion
            )

            staff = int(
                total_staff
                * proportion
            )

            allocations.append({
                "region_id": region["region_id"],
                "risk_score": region["risk_score"],
                "predicted_cases": region["predicted_cases"],
                "bandit_score": round(
                    region["score"],
                    4
                ),
                "vaccine_doses": vaccines,
                "testing_kits": testing_kits,
                "staff": staff
            })

        return allocations


if __name__ == "__main__":

    agent = LinUCBAllocator(
        alpha=0.5
    )

    allocations = agent.allocate()

    print("\nLINUCB RESOURCE ALLOCATION")
    print("--------------------------")

    for allocation in allocations:

        print(
            f"\nRegion: "
            f"{allocation['region_id']}"
        )

        print(
            f"Risk Score: "
            f"{allocation['risk_score']}"
        )

        print(
            f"Predicted Cases: "
            f"{allocation['predicted_cases']}"
        )

        print(
            f"Bandit Score: "
            f"{allocation['bandit_score']}"
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