import json
import os


def load_risk_data():
    file_path = os.path.join(
        os.path.dirname(os.path.dirname(__file__)),
        "region_risk_output.json"
    )

    with open(file_path, "r") as f:
        data = json.load(f)

    return data


def allocate_resources(
    total_vaccines=10000,
    total_testing_kits=5000,
    total_staff=100
):
    data = load_risk_data()

    regions = data["regions"]

    # Total risk is used to divide resources proportionally
    total_risk = sum(region["risk_score"] for region in regions)

    allocations = []

    for region in regions:
        risk = region["risk_score"]

        if total_risk == 0:
            proportion = 1 / len(regions)
        else:
            proportion = risk / total_risk

        vaccines = int(total_vaccines * proportion)
        testing_kits = int(total_testing_kits * proportion)
        staff = int(total_staff * proportion)

        allocations.append({
            "region_id": region["region_id"],
            "risk_score": risk,
            "predicted_cases": region["predicted_cases"],
            "vaccine_doses": vaccines,
            "testing_kits": testing_kits,
            "staff": staff
        })

    return allocations


if __name__ == "__main__":

    allocations = allocate_resources()

    print("\nRESOURCE ALLOCATION PLAN")
    print("------------------------")

    for allocation in allocations:

        print(
            f"\nRegion: {allocation['region_id']}"
        )

        print(
            f"Risk Score: {allocation['risk_score']}"
        )

        print(
            f"Predicted Cases: {allocation['predicted_cases']}"
        )

        print(
            f"Vaccines: {allocation['vaccine_doses']}"
        )

        print(
            f"Testing Kits: {allocation['testing_kits']}"
        )

        print(
            f"Staff: {allocation['staff']}"
        )