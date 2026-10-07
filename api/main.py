import os
import sys
import json

import pandas as pd
from fastapi import FastAPI, HTTPException

PROJECT_ROOT = os.path.dirname(
    os.path.dirname(__file__)
)

sys.path.append(PROJECT_ROOT)

from rl.bandit_agent import allocate_resources
from rl.seir_simulator import run_simulation


app = FastAPI(
    title="EpiGraphML API",
    description=(
        "Disease outbreak risk prediction, "
        "resource allocation and SEIR simulation API"
    ),
    version="1.0"
)


def load_risk_data():

    risk_path = os.path.join(
        PROJECT_ROOT,
        "region_risk_output.json"
    )

    with open(risk_path, "r") as f:
        data = json.load(f)

    return data


def load_demographics():

    demo_path = os.path.join(
        PROJECT_ROOT,
        "data",
        "processed",
        "tn_demographics.csv"
    )

    return pd.read_csv(demo_path)


def get_region_data(region_id):

    data = load_risk_data()

    for region in data["regions"]:

        if region["region_id"] == region_id:
            return region

    return None


def get_population(region_id):

    df_demo = load_demographics()

    row = df_demo[
        df_demo["region_id"] == region_id
    ]

    if row.empty:
        return None

    return int(
        row.iloc[0]["population"]
    )


def get_district_name(region_id):

    df_demo = load_demographics()

    row = df_demo[
        df_demo["region_id"] == region_id
    ]

    if row.empty:
        return region_id

    return str(
        row.iloc[0]["district"]
    )


def get_region_allocation(region_id):

    allocations = allocate_resources()

    for allocation in allocations:

        if allocation["region_id"] == region_id:
            return allocation

    return None


@app.get("/")
def home():

    return {
        "message": "EpiGraphML API is running",
        "status": "success"
    }


@app.get("/risk")
def get_risk():

    return load_risk_data()


@app.get("/risk/{region_id}")
def get_region_risk(region_id: str):

    region = get_region_data(
        region_id
    )

    if region is None:

        raise HTTPException(
            status_code=404,
            detail="Region not found"
        )

    return region


@app.get("/allocation")
def get_allocation():

    allocations = allocate_resources()

    return {
        "total_regions": len(
            allocations
        ),
        "allocations": allocations
    }


@app.get("/allocation/{region_id}")
def get_single_allocation(
    region_id: str
):

    allocation = get_region_allocation(
        region_id
    )

    if allocation is None:

        raise HTTPException(
            status_code=404,
            detail="Allocation not found"
        )

    return allocation


@app.get("/simulation/{region_id}")
def simulate_region(
    region_id: str
):

    region = get_region_data(
        region_id
    )

    if region is None:

        raise HTTPException(
            status_code=404,
            detail="Region not found"
        )

    population = get_population(
        region_id
    )

    if population is None:

        raise HTTPException(
            status_code=404,
            detail="Population data not found"
        )

    allocation = get_region_allocation(
        region_id
    )

    if allocation is None:

        raise HTTPException(
            status_code=404,
            detail="Allocation not found"
        )

    predicted_cases = int(
        region["predicted_cases"]
    )

    initial_infected = max(
        predicted_cases,
        1
    )

    without_intervention = run_simulation(
        population=population,
        initial_infected=initial_infected,
        days=30
    )

    with_intervention = run_simulation(
        population=population,
        initial_infected=initial_infected,
        days=30,
        vaccine_doses=allocation[
            "vaccine_doses"
        ],
        testing_kits=allocation[
            "testing_kits"
        ],
        staff=allocation[
            "staff"
        ]
    )

    no_intervention_burden = sum(
        day["infected"]
        for day in without_intervention
    )

    intervention_burden = sum(
        day["infected"]
        for day in with_intervention
    )

    burden_prevented = (
        no_intervention_burden
        - intervention_burden
    )

    if no_intervention_burden > 0:

        reduction_percentage = (
            burden_prevented
            / no_intervention_burden
        ) * 100

    else:

        reduction_percentage = 0

    return {
        "region_id": region_id,
        "district": get_district_name(region_id),
        "population": population,
        "risk_score": region["risk_score"],
        "predicted_cases": predicted_cases,
        "contributing_factors":
            region["contributing_factors"],
        "allocation": allocation,

        "simulation_summary": {

            "infection_burden_without_intervention":
                round(
                    no_intervention_burden,
                    2
                ),

            "infection_burden_with_intervention":
                round(
                    intervention_burden,
                    2
                ),

            "infection_burden_prevented":
                round(
                    burden_prevented,
                    2
                ),

            "reduction_percentage":
                round(
                    reduction_percentage,
                    2
                )
        },

        "without_intervention":
            without_intervention,

        "with_intervention":
            with_intervention
    }


@app.get("/regions")
def get_regions():

    data = load_risk_data()

    df_demo = load_demographics()

    regions = []

    for region in data["regions"]:

        region_id = region[
            "region_id"
        ]

        row = df_demo[
            df_demo["region_id"]
            == region_id
        ]

        if not row.empty:
            district = row.iloc[0][
                "district"
            ]
        else:
            district = region_id

        regions.append({
            "region_id":
                region_id,

            "district":
                district,

            "risk_score":
                region["risk_score"],

            "predicted_cases":
                region[
                    "predicted_cases"
                ]
        })

    return {
        "total_regions":
            len(regions),

        "regions":
            regions
    }