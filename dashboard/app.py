import requests
import pandas as pd
import streamlit as st
import plotly.graph_objects as go


API_BASE = "http://127.0.0.1:8000"


# --------------------------------------------------
# PAGE CONFIG
# --------------------------------------------------

st.set_page_config(
    page_title="EpiGraphML Dashboard",
    page_icon="🦠",
    layout="wide"
)


# --------------------------------------------------
# CUSTOM DARK UI
# --------------------------------------------------

st.markdown(
    """
    <style>

    .block-container {
        padding-top: 1.8rem;
        padding-bottom: 3rem;
    }

    .hero {
        background: linear-gradient(
            135deg,
            #111827,
            #1f2937,
            #0f172a
        );
        border: 1px solid #334155;
        padding: 30px;
        border-radius: 18px;
        margin-bottom: 26px;
        box-shadow: 0 8px 25px rgba(0,0,0,0.25);
    }

    .hero-title {
        color: #f8fafc;
        font-size: 38px;
        font-weight: 800;
        margin-bottom: 7px;
    }

    .hero-subtitle {
        color: #cbd5e1;
        font-size: 16px;
        line-height: 1.6;
    }

    .section-title {
        color: #f8fafc;
        font-size: 25px;
        font-weight: 750;
        margin-top: 28px;
        margin-bottom: 15px;
    }

    .metric-card {
        background: #111827;
        border: 1px solid #334155;
        padding: 20px;
        border-radius: 16px;
        min-height: 120px;
        box-shadow: 0 4px 16px rgba(0,0,0,0.18);
    }

    .metric-label {
        color: #94a3b8;
        font-size: 14px;
        margin-bottom: 7px;
    }

    .metric-value {
        color: #f8fafc;
        font-size: 28px;
        font-weight: 750;
    }

    .info-card {
        background: #111827;
        border: 1px solid #334155;
        padding: 18px;
        border-radius: 14px;
        margin-bottom: 16px;
    }

    .risk-high {
        display: inline-block;
        background-color: #7f1d1d;
        color: #fecaca;
        padding: 8px 15px;
        border-radius: 999px;
        font-weight: 700;
    }

    .risk-medium {
        display: inline-block;
        background-color: #78350f;
        color: #fde68a;
        padding: 8px 15px;
        border-radius: 999px;
        font-weight: 700;
    }

    .risk-low {
        display: inline-block;
        background-color: #14532d;
        color: #bbf7d0;
        padding: 8px 15px;
        border-radius: 999px;
        font-weight: 700;
    }

    .factor-chip {
        display: inline-block;
        background-color: #312e81;
        color: #c7d2fe;
        padding: 7px 12px;
        border-radius: 999px;
        margin-right: 8px;
        margin-bottom: 8px;
        font-size: 13px;
        font-weight: 600;
    }

    .strategy-box {
        background: #172554;
        color: #dbeafe;
        border: 1px solid #1d4ed8;
        border-left: 5px solid #3b82f6;
        padding: 17px;
        border-radius: 12px;
        line-height: 1.6;
        margin-bottom: 20px;
    }

    .footer-note {
        color: #64748b;
        text-align: center;
        font-size: 13px;
        margin-top: 35px;
        padding-top: 18px;
        border-top: 1px solid #334155;
    }

    div[data-testid="stMetric"] {
        background: #111827;
        border: 1px solid #334155;
        padding: 17px;
        border-radius: 14px;
        box-shadow: 0 4px 15px rgba(0,0,0,0.18);
    }

    div[data-testid="stMetricLabel"] {
        color: #94a3b8;
    }

    div[data-testid="stMetricValue"] {
        color: #f8fafc;
    }

    section[data-testid="stSidebar"] {
        border-right: 1px solid #334155;
    }

    </style>
    """,
    unsafe_allow_html=True
)


# --------------------------------------------------
# API FUNCTIONS
# --------------------------------------------------

def get_regions():

    response = requests.get(
        API_BASE + "/regions"
    )

    response.raise_for_status()

    return response.json()


def get_region_risk(region_id):

    response = requests.get(
        API_BASE + "/risk/" + region_id
    )

    response.raise_for_status()

    return response.json()


def get_region_allocation(region_id):

    response = requests.get(
        API_BASE + "/allocation/" + region_id
    )

    response.raise_for_status()

    return response.json()


def get_simulation(region_id):

    response = requests.get(
        API_BASE + "/simulation/" + region_id
    )

    response.raise_for_status()

    return response.json()


# --------------------------------------------------
# LOAD DATA
# --------------------------------------------------

try:

    region_data = get_regions()

except Exception as error:

    st.error(
        "Could not connect to the FastAPI backend."
    )

    st.code(
        "uvicorn api.main:app --reload"
    )

    st.code(
        str(error)
    )

    st.stop()


regions = region_data["regions"]


region_map = {}

for region in regions:

    region_map[
        region["region_id"]
    ] = region["district"]


region_ids = list(
    region_map.keys()
)


# --------------------------------------------------
# SIDEBAR
# --------------------------------------------------

st.sidebar.title(
    "EpiGraphML"
)

st.sidebar.caption(
    "Disease intelligence and resource allocation controls"
)


def format_region(region_id):

    return (
        region_map[region_id]
        + " ("
        + region_id
        + ")"
    )


selected_region = st.sidebar.selectbox(
    "Select District",
    region_ids,
    format_func=format_region
)


st.sidebar.markdown("---")

st.sidebar.markdown(
    "### System Pipeline"
)

st.sidebar.markdown(
    """
    **1.** Graph-based regional modelling

    **2.** XGBoost case forecasting

    **3.** Risk scoring

    **4.** Resource allocation

    **5.** SEIR intervention simulation
    """
)


st.sidebar.markdown("---")

st.sidebar.caption(
    "Simulation-validated academic prototype"
)


# --------------------------------------------------
# SELECTED REGION DATA
# --------------------------------------------------

try:

    risk = get_region_risk(
        selected_region
    )

    allocation = get_region_allocation(
        selected_region
    )

    simulation = get_simulation(
        selected_region
    )

except Exception as error:

    st.error(
        "Could not load data for this region."
    )

    st.code(
        str(error)
    )

    st.stop()


district_name = simulation["district"]

population = simulation["population"]

summary = simulation[
    "simulation_summary"
]


risk_score = risk["risk_score"]

predicted_cases = risk[
    "predicted_cases"
]

reduction = summary[
    "reduction_percentage"
]


# --------------------------------------------------
# HERO
# --------------------------------------------------

st.markdown(
    """
    <div class="hero">
        <div class="hero-title">
            EpiGraphML
        </div>
        <div class="hero-subtitle">
            Disease outbreak forecasting,
            intelligent healthcare resource allocation,
            and SEIR-based intervention simulation.
        </div>
    </div>
    """,
    unsafe_allow_html=True
)


# --------------------------------------------------
# REGION OVERVIEW
# --------------------------------------------------

st.markdown(
    '<div class="section-title">'
    + district_name
    + ' Overview</div>',
    unsafe_allow_html=True
)


col1, col2, col3, col4 = st.columns(4)


with col1:

    st.markdown(
        """
        <div class="metric-card">
            <div class="metric-label">
                Risk Score
            </div>
            <div class="metric-value">
        """
        + format(risk_score, ".4f")
        +
        """
            </div>
        </div>
        """,
        unsafe_allow_html=True
    )


with col2:

    st.markdown(
        """
        <div class="metric-card">
            <div class="metric-label">
                Predicted Cases
            </div>
            <div class="metric-value">
        """
        + str(predicted_cases)
        +
        """
            </div>
        </div>
        """,
        unsafe_allow_html=True
    )


with col3:

    st.markdown(
        """
        <div class="metric-card">
            <div class="metric-label">
                Population
            </div>
            <div class="metric-value">
        """
        + format(population, ",")
        +
        """
            </div>
        </div>
        """,
        unsafe_allow_html=True
    )


with col4:

    st.markdown(
        """
        <div class="metric-card">
            <div class="metric-label">
                Outbreak Reduction
            </div>
            <div class="metric-value">
        """
        + format(reduction, ".2f")
        + "%"
        +
        """
            </div>
        </div>
        """,
        unsafe_allow_html=True
    )


# --------------------------------------------------
# RISK ASSESSMENT
# --------------------------------------------------

st.markdown(
    '<div class="section-title">'
    'Risk Assessment'
    '</div>',
    unsafe_allow_html=True
)


if risk_score >= 0.7:

    risk_label = "High Risk"
    risk_class = "risk-high"

elif risk_score >= 0.4:

    risk_label = "Medium Risk"
    risk_class = "risk-medium"

else:

    risk_label = "Low Risk"
    risk_class = "risk-low"


st.markdown(
    '<div class="info-card">'
    '<span class="'
    + risk_class
    + '">'
    + risk_label
    + '</span>'
    '</div>',
    unsafe_allow_html=True
)


# --------------------------------------------------
# CONTRIBUTING FACTORS
# --------------------------------------------------

st.markdown(
    '<div class="section-title">'
    'Contributing Factors'
    '</div>',
    unsafe_allow_html=True
)


factors = risk.get(
    "contributing_factors",
    []
)


if factors:

    factor_html = ""

    for factor in factors:

        readable = (
            factor
            .replace("_", " ")
            .title()
        )

        factor_html += (
            '<span class="factor-chip">'
            + readable
            + '</span>'
        )

    st.markdown(
        '<div class="info-card">'
        + factor_html
        + '</div>',
        unsafe_allow_html=True
    )

else:

    st.info(
        "No major contributing factors identified."
    )


# --------------------------------------------------
# RESOURCE ALLOCATION
# --------------------------------------------------

st.markdown(
    '<div class="section-title">'
    'Recommended Resource Allocation'
    '</div>',
    unsafe_allow_html=True
)


st.markdown(
    """
    <div class="strategy-box">
        The current recommendation uses the
        <b>risk-based allocation strategy</b>.
        In the simulation comparison, this strategy
        produced the highest total outbreak reduction.
    </div>
    """,
    unsafe_allow_html=True
)


r1, r2, r3 = st.columns(3)


with r1:

    st.metric(
        "Vaccine Doses",
        allocation[
            "vaccine_doses"
        ]
    )


with r2:

    st.metric(
        "Testing Kits",
        allocation[
            "testing_kits"
        ]
    )


with r3:

    st.metric(
        "Medical Staff",
        allocation[
            "staff"
        ]
    )


# --------------------------------------------------
# SEIR RESULTS
# --------------------------------------------------

st.markdown(
    '<div class="section-title">'
    'SEIR Intervention Results'
    '</div>',
    unsafe_allow_html=True
)


s1, s2, s3, s4 = st.columns(4)


with s1:

    st.metric(
        "Without Intervention",
        round(
            summary[
                "infection_burden_without_intervention"
            ]
        )
    )


with s2:

    st.metric(
        "With Intervention",
        round(
            summary[
                "infection_burden_with_intervention"
            ]
        )
    )


with s3:

    st.metric(
        "Burden Prevented",
        round(
            summary[
                "infection_burden_prevented"
            ]
        )
    )


with s4:

    st.metric(
        "Reduction",
        format(
            summary[
                "reduction_percentage"
            ],
            ".2f"
        ) + "%"
    )


# --------------------------------------------------
# DATA
# --------------------------------------------------

without_data = pd.DataFrame(
    simulation[
        "without_intervention"
    ]
)


with_data = pd.DataFrame(
    simulation[
        "with_intervention"
    ]
)


# --------------------------------------------------
# INFECTION CURVE
# --------------------------------------------------

st.markdown(
    '<div class="section-title">'
    'Projected Infection Curve'
    '</div>',
    unsafe_allow_html=True
)


infection_fig = go.Figure()


infection_fig.add_trace(
    go.Scatter(
        x=without_data["day"],
        y=without_data["infected"],
        mode="lines",
        name="Without Intervention",
        line=dict(
            width=3
        )
    )
)


infection_fig.add_trace(
    go.Scatter(
        x=with_data["day"],
        y=with_data["infected"],
        mode="lines",
        name="With Intervention",
        line=dict(
            width=3
        )
    )
)


infection_fig.update_layout(
    title=(
        "Projected Infection Burden — "
        + district_name
    ),
    xaxis_title="Day",
    yaxis_title="Infected Population",
    hovermode="x unified",
    height=440,
    template="plotly_dark",
    margin=dict(
        l=20,
        r=20,
        t=60,
        b=20
    )
)


st.plotly_chart(
    infection_fig,
    use_container_width=True
)


# --------------------------------------------------
# SEIR COMPARTMENTS
# --------------------------------------------------

st.markdown(
    '<div class="section-title">'
    'SEIR Compartment Dynamics'
    '</div>',
    unsafe_allow_html=True
)


seir_fig = go.Figure()


for column, label in [
    ("susceptible", "Susceptible"),
    ("exposed", "Exposed"),
    ("infected", "Infected"),
    ("recovered", "Recovered")
]:

    seir_fig.add_trace(
        go.Scatter(
            x=with_data["day"],
            y=with_data[column],
            mode="lines",
            name=label
        )
    )


seir_fig.update_layout(
    title=(
        "SEIR Dynamics With Intervention — "
        + district_name
    ),
    xaxis_title="Day",
    yaxis_title="Population",
    hovermode="x unified",
    height=480,
    template="plotly_dark",
    margin=dict(
        l=20,
        r=20,
        t=60,
        b=20
    )
)


st.plotly_chart(
    seir_fig,
    use_container_width=True
)


# --------------------------------------------------
# TECHNICAL DETAILS
# --------------------------------------------------

st.markdown(
    '<div class="section-title">'
    'Technical Details'
    '</div>',
    unsafe_allow_html=True
)


with st.expander(
    "Risk Prediction Data"
):

    st.json(risk)


with st.expander(
    "Resource Allocation Data"
):

    st.json(allocation)


with st.expander(
    "Simulation Summary"
):

    st.json(summary)


# --------------------------------------------------
# FOOTER
# --------------------------------------------------

st.markdown(
    """
    <div class="footer-note">
        EpiGraphML • Disease Outbreak Prediction
        & Resource Allocation System
    </div>
    """,
    unsafe_allow_html=True
)