import os
import sys
import json

import pandas as pd
from fastapi import FastAPI, HTTPException

# --------------------------------------------------
# PROJECT IMPORT SETUP
# --------------------------------------------------

PROJECT_ROOT = os.path.dirname(
    os.path.dirname(__file__)
)

sys.path.append(PROJECT_ROOT)

from rl.bandit_agent import allocate_resources
from rl.seir_simulator import run_simulation


# --------------------------------------------------
# FASTAPI APP
# --------------------------------------------------

app = FastAPI(
    title="EpiGraphML API",
    description=(
        "Disease outbreak risk prediction, "
        "resource allocation and SEIR simulation API"
    ),
    version="1.0"
)


# --------------------------------------------------
# HELPER FUNCTIONS
# --------------------------------------------------

def load_risk_data():

    risk_path = os.path.join(
        PROJECT_ROOT,
        "region_risk_output.json"
    )

    with open(
        risk_path,
        "r"
    ) as f:
        data = json.load(f)

    return data


def load_demographics():

    demo_path = os.path.join(
        PROJECT_ROOT,
        "data",
        "processed",
        "tn_demographics.csv"
    )

    df = pd.read_csv(
        demo_path
    )

    return df


def get_region_data(region_id):

    data = load_risk_data()

    for region in data["regions"]:

        if region["region_id"] == region_id:
            return region

    return None


def get_population(region_id):

    df_demo = load_demographics()

    region_row = df_demo[
        df_demo["region_id"] == region_id
    ]

    if region_row.empty:
        return None

    population = int(
        region_row.iloc[0]["population"]
    )

    return population


def get_region_allocation(region_id):

    allocations = allocate_resources()

    for allocation in allocations:

        if allocation["region_id"] == region_id:
            return allocation

    return None


# --------------------------------------------------
# HOME ENDPOINT
# --------------------------------------------------

@app.get("/")
def home():

    return {
        "message": "EpiGraphML API is running",
        "status": "success"
    }


# --------------------------------------------------
# GET ALL RISK DATA
# --------------------------------------------------

@app.get("/risk")
def get_risk():

    data = load_risk_data()

    return data


# --------------------------------------------------
# GET SINGLE REGION RISK
# --------------------------------------------------

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


# --------------------------------------------------
# GET RESOURCE ALLOCATION
# --------------------------------------------------

@app.get("/allocation")
def get_allocation():

    allocations = allocate_resources()

    return {
        "total_regions": len(
            allocations
        ),
        "allocations": allocations
    }


# --------------------------------------------------
# GET ALLOCATION FOR ONE REGION
# --------------------------------------------------

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


# --------------------------------------------------
# SIMULATE A REGION
# --------------------------------------------------

@app.get("/simulation/{region_id}")
def simulate_region(
    region_id: str
):

    # ------------------------------
    # Get region prediction
    # ------------------------------

    region = get_region_data(
        region_id
    )

    if region is None:

        raise HTTPException(
            status_code=404,
            detail="Region not found"
        )

    # ------------------------------
    # Get real population
    # ------------------------------

    population = get_population(
        region_id
    )

    if population is None:

        raise HTTPException(
            status_code=404,
            detail="Population data not found"
        )

    # ------------------------------
    # Get resource allocation
    # ------------------------------

    allocation = get_region_allocation(
        region_id
    )

    if allocation is None:

        raise HTTPException(
            status_code=404,
            detail="Allocation not found"
        )

    # ------------------------------
    # Initial infection value
    # ------------------------------

    predicted_cases = int(
        region["predicted_cases"]
    )

    initial_infected = max(
        predicted_cases,
        1
    )

    # ------------------------------
    # Simulation without intervention
    # ------------------------------

    without_intervention = (
        run_simulation(
            population=population,
            initial_infected=initial_infected,
            days=30
        )
    )

    # ------------------------------
    # Simulation with intervention
    # ------------------------------

    with_intervention = (
        run_simulation(
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
    )

    # ------------------------------
    # Calculate summary metrics
    # ------------------------------

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

    # ------------------------------
    # API RESPONSE
    # ------------------------------

    return {
        "region_id": region_id,

        "population": population,

        "risk_score":
            region["risk_score"],

        "predicted_cases":
            predicted_cases,

        "contributing_factors":
            region[
                "contributing_factors"
            ],

        "allocation":
            allocation,

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


# --------------------------------------------------
# LIST AVAILABLE REGIONS
# --------------------------------------------------

@app.get("/regions")
def get_regions():

    data = load_risk_data()

    regions = []

    for region in data["regions"]:

        regions.append({
            "region_id":
                region["region_id"],

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