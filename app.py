from datetime import datetime, timedelta
from pathlib import Path
import pickle

import numpy as np
import streamlit as st

# ============================================================
# PAGE CONFIG
# ============================================================
st.set_page_config(
    page_title="Flight Fare Predictor",
    page_icon="✈️",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ============================================================
# FILE PATHS
# ============================================================
BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "model.pkl"
COLUMNS_PATH = BASE_DIR / "columns.pkl"

# ============================================================
# ORIGINAL MODEL CATEGORIES
# Keep these exactly aligned with your trained model.
# ============================================================
AIRLINES = [
    "Air Asia",
    "Air India",
    "GoAir",
    "IndiGo",
    "Jet Airways",
    "Jet Airways Business",
    "Multiple carriers",
    "Multiple carriers Premium economy",
    "SpiceJet",
    "Trujet",
    "Vistara",
    "Vistara Premium economy",
]

SOURCES = [
    "Banglore",
    "Chennai",
    "Delhi",
    "Kolkata",
    "Mumbai",
]

DESTINATIONS = [
    "Banglore",
    "Cochin",
    "Delhi",
    "Hyderabad",
    "Kolkata",
    "New Delhi",
]

STOPS = {
    "Non-stop": 0,
    "1 Stop": 1,
    "2 Stops": 2,
    "3 Stops": 3,
    "4 Stops": 4,
}

# ============================================================
# PROFESSIONAL FLIGHT-BOOKING STYLE
# ============================================================
st.markdown(
    """
    <style>

    /* ---------- Main background ---------- */
    .stApp {
        background:
            linear-gradient(
                90deg,
                rgba(3, 20, 45, 0.92) 0%,
                rgba(7, 45, 85, 0.78) 45%,
                rgba(3, 20, 45, 0.82) 100%
            ),
            url("https://images.unsplash.com/photo-1436491865332-7a61a109cc05?auto=format&fit=crop&w=2200&q=85");
        background-size: cover;
        background-position: center;
        background-attachment: fixed;
    }

    /* ---------- Remove default top padding ---------- */
    .block-container {
        max-width: 1180px;
        padding-top: 2rem;
        padding-bottom: 3rem;
    }

    /* ---------- Hide Streamlit menu/footer ---------- */
    #MainMenu {
        visibility: hidden;
    }

    footer {
        visibility: hidden;
    }

    header {
        background: transparent !important;
    }

    /* ---------- Header ---------- */
    .hero {
        text-align: center;
        color: white;
        padding: 18px 20px 28px 20px;
    }

    .hero-badge {
        display: inline-block;
        padding: 8px 16px;
        border-radius: 999px;
        background: rgba(255, 255, 255, 0.15);
        border: 1px solid rgba(255, 255, 255, 0.25);
        backdrop-filter: blur(10px);
        font-size: 14px;
        font-weight: 600;
        letter-spacing: 0.5px;
        margin-bottom: 12px;
    }

    .hero h1 {
        margin: 0;
        font-size: 46px;
        font-weight: 800;
        letter-spacing: -1px;
    }

    .hero p {
        margin-top: 10px;
        font-size: 17px;
        color: rgba(255, 255, 255, 0.85);
    }

    /* ---------- Main glass card ---------- */
    .booking-card {
        background: rgba(255, 255, 255, 0.96);
        border-radius: 26px;
        padding: 28px 32px 30px 32px;
        box-shadow: 0 24px 70px rgba(0, 0, 0, 0.30);
        border: 1px solid rgba(255, 255, 255, 0.75);
    }

    .section-title {
        font-size: 21px;
        font-weight: 800;
        color: #102a43;
        margin-bottom: 4px;
    }

    .section-subtitle {
        font-size: 13px;
        color: #627d98;
        margin-bottom: 20px;
    }

    /* ---------- Input labels ---------- */
    label {
        font-weight: 650 !important;
        color: #243b53 !important;
    }

    /* ---------- Input widgets ---------- */
    div[data-baseweb="select"] > div,
    div[data-baseweb="input"] > div,
    div[data-testid="stDateInput"] > div,
    div[data-testid="stTimeInput"] > div {
        border-radius: 12px !important;
        border: 1px solid #d9e2ec !important;
        background: #ffffff !important;
        min-height: 44px;
    }

    /* ---------- Predict button ---------- */
    .stButton > button {
        width: 100%;
        min-height: 52px;
        border-radius: 14px;
        border: none;
        background: linear-gradient(135deg, #0b63ce, #1683ff);
        color: white;
        font-size: 17px;
        font-weight: 750;
        box-shadow: 0 10px 25px rgba(11, 99, 206, 0.30);
        transition: all 0.2s ease;
    }

    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 14px 30px rgba(11, 99, 206, 0.40);
    }

    /* ---------- Result card ---------- */
    .fare-result {
        margin-top: 24px;
        padding: 24px;
        border-radius: 20px;
        background: linear-gradient(135deg, #e9f7ef, #f4fff8);
        border: 1px solid #b7e4c7;
        text-align: center;
    }

    .fare-label {
        color: #386641;
        font-size: 14px;
        font-weight: 650;
        text-transform: uppercase;
        letter-spacing: 1px;
    }

    .fare-price {
        color: #1b4332;
        font-size: 42px;
        font-weight: 850;
        margin: 5px 0;
    }

    .fare-note {
        color: #52796f;
        font-size: 13px;
    }

    /* ---------- Feature cards ---------- */
    .mini-card {
        background: rgba(255, 255, 255, 0.14);
        border: 1px solid rgba(255, 255, 255, 0.20);
        border-radius: 16px;
        padding: 15px;
        text-align: center;
        color: white;
        backdrop-filter: blur(8px);
    }

    .mini-card-title {
        font-size: 12px;
        opacity: 0.78;
        margin-bottom: 4px;
    }

    .mini-card-value {
        font-size: 16px;
        font-weight: 750;
    }

    /* ---------- Responsive ---------- */
    @media (max-width: 768px) {
        .hero h1 {
            font-size: 32px;
        }

        .booking-card {
            padding: 22px 18px;
            border-radius: 20px;
        }
    }

    </style>
    """,
    unsafe_allow_html=True,
)

# ============================================================
# HERO
# ============================================================
st.markdown(
    """
    <div class="hero">
        <div class="hero-badge">✈️ SMART FLIGHT FARE PREDICTOR</div>
        <h1>Find Your Flight Fare</h1>
        <p>Enter your journey details and get an estimated ticket price instantly.</p>
    </div>
    """,
    unsafe_allow_html=True,
)

# ============================================================
# LOAD MODEL
# ============================================================
@st.cache_resource
def load_model():
    with open(MODEL_PATH, "rb") as file:
        return pickle.load(file)


@st.cache_data
def load_columns():
    with open(COLUMNS_PATH, "rb") as file:
        return list(pickle.load(file))


try:
    model = load_model()
    columns = load_columns()
except Exception as error:
    st.error(f"Could not load model files: {error}")
    st.stop()

# ============================================================
# HELPER FUNCTIONS
# ============================================================
def duration_parts(departure: datetime, arrival: datetime) -> tuple[int, int]:
    """
    Calculate flight duration.

    If arrival time is earlier than departure time,
    assume arrival is on the next day.
    """
    if arrival < departure:
        arrival += timedelta(days=1)

    duration = arrival - departure
    total_minutes = int(duration.total_seconds() // 60)

    return divmod(total_minutes, 60)


def build_feature_row(
    airline: str,
    source: str,
    destination: str,
    total_stops: int,
    departure: datetime,
    arrival: datetime,
) -> list[int]:

    duration_hours, duration_mins = duration_parts(
        departure,
        arrival,
    )

    # IMPORTANT:
    # This order must stay exactly the same as the original model.
    return [
        total_stops,
        departure.day,
        departure.month,
        departure.hour,
        departure.minute,
        arrival.hour,
        arrival.minute,
        duration_hours,
        duration_mins,

        # Airline one-hot encoding
        *(1 if airline == option else 0 for option in AIRLINES),

        # Source one-hot encoding
        1 if source == "Chennai" else 0,
        1 if source == "Delhi" else 0,
        1 if source == "Kolkata" else 0,
        1 if source == "Mumbai" else 0,

        # Destination one-hot encoding
        1 if destination == "Cochin" else 0,
        1 if destination == "Delhi" else 0,
        1 if destination == "Hyderabad" else 0,
        1 if destination == "Kolkata" else 0,
        1 if destination == "New Delhi" else 0,
    ]


# ============================================================
# BOOKING CARD
# ============================================================
st.markdown('<div class="booking-card">', unsafe_allow_html=True)

st.markdown(
    """
    <div class="section-title">🧳 Enter Your Flight Details</div>
    <div class="section-subtitle">
        Add your journey, airline and route information below.
    </div>
    """,
    unsafe_allow_html=True,
)

# ------------------------------------------------------------
# Date
# ------------------------------------------------------------
journey_date = st.date_input(
    "Journey Date",
    value=datetime.now().date(),
)

# ------------------------------------------------------------
# Departure / Arrival
# ------------------------------------------------------------
col1, col2 = st.columns(2)

with col1:
    departure_time = st.time_input(
        "Departure Time",
        value=datetime.strptime("10:00", "%H:%M").time(),
    )

with col2:
    arrival_time = st.time_input(
        "Arrival Time",
        value=datetime.strptime("12:00", "%H:%M").time(),
    )

# ------------------------------------------------------------
# Route
# ------------------------------------------------------------
col3, col4 = st.columns(2)

with col3:
    source = st.selectbox(
        "From",
        SOURCES,
    )

with col4:
    destination = st.selectbox(
        "To",
        DESTINATIONS,
    )

# ------------------------------------------------------------
# Airline / Stops
# ------------------------------------------------------------
col5, col6 = st.columns(2)

with col5:
    airline = st.selectbox(
        "Airline",
        AIRLINES,
    )

with col6:
    stop_label = st.selectbox(
        "Total Stops",
        list(STOPS.keys()),
    )

st.markdown("<br>", unsafe_allow_html=True)

# ------------------------------------------------------------
# Predict
# ------------------------------------------------------------
predict_clicked = st.button(
    "✈️  Predict Flight Fare",
    type="primary",
)

st.markdown("</div>", unsafe_allow_html=True)

# ============================================================
# PREDICTION
# ============================================================
if predict_clicked:

    departure = datetime.combine(
        journey_date,
        departure_time,
    )

    arrival = datetime.combine(
        journey_date,
        arrival_time,
    )

    feature_row = build_feature_row(
        airline=airline,
        source=source,
        destination=destination,
        total_stops=STOPS[stop_label],
        departure=departure,
        arrival=arrival,
    )

    # Your trained model expects the same number of
    # features stored in columns.pkl.
    if len(feature_row) != len(columns):
        st.error(
            f"Feature mismatch: app created {len(feature_row)} "
            f"features, but the model expects {len(columns)}."
        )
        st.stop()

    # IMPORTANT:
    # columns.pkl contains duplicate names such as Delhi/Kolkata
    # because Source and Destination were one-hot encoded separately.
    # Therefore, use a NumPy array instead of a DataFrame.
    input_data = np.array(
        feature_row,
        dtype=float,
    ).reshape(1, -1)

    try:
        prediction = model.predict(input_data)[0]
        prediction = float(prediction)

        # Result
        st.markdown(
            f"""
            <div class="fare-result">
                <div class="fare-label">Estimated Flight Fare</div>
                <div class="fare-price">₹{prediction:,.2f}</div>
                <div class="fare-note">
                    Estimated price based on the selected flight details.
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        # Summary cards
        st.markdown("<br>", unsafe_allow_html=True)

        summary1, summary2, summary3, summary4 = st.columns(4)

        with summary1:
            st.markdown(
                f"""
                <div class="mini-card">
                    <div class="mini-card-title">ROUTE</div>
                    <div class="mini-card-value">
                        {source} → {destination}
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )

        with summary2:
            st.markdown(
                f"""
                <div class="mini-card">
                    <div class="mini-card-title">AIRLINE</div>
                    <div class="mini-card-value">{airline}</div>
                </div>
                """,
                unsafe_allow_html=True,
            )

        with summary3:
            st.markdown(
                f"""
                <div class="mini-card">
                    <div class="mini-card-title">STOPS</div>
                    <div class="mini-card-value">{stop_label}</div>
                </div>
                """,
                unsafe_allow_html=True,
            )

        duration_hours, duration_mins = duration_parts(
            departure,
            arrival,
        )

        with summary4:
            st.markdown(
                f"""
                <div class="mini-card">
                    <div class="mini-card-title">DURATION</div>
                    <div class="mini-card-value">
                        {duration_hours}h {duration_mins}m
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )

    except Exception as error:
        st.error(f"Prediction failed: {error}")

# ============================================================
# FOOTER
# ============================================================
st.markdown(
    """
    <div style="
        text-align:center;
        color:rgba(255,255,255,0.70);
        font-size:12px;
        margin-top:30px;
    ">
        ✈️ Flight Fare Predictor &nbsp;•&nbsp; Machine Learning Project
    </div>
    """,
    unsafe_allow_html=True,
)
