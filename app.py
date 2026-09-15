from datetime import datetime, timedelta
from pathlib import Path
import pickle

import pandas as pd
import streamlit as st


st.set_page_config(page_title="Flight Fare Prediction")

BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "model.pkl"
COLUMNS_PATH = BASE_DIR / "columns.pkl"

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

SOURCES = ["Banglore", "Chennai", "Delhi", "Kolkata", "Mumbai"]
DESTINATIONS = ["Banglore", "Cochin", "Delhi", "Hyderabad", "Kolkata", "New Delhi"]
STOPS = {
    "non-stop": 0,
    "1 stop": 1,
    "2 stops": 2,
    "3 stops": 3,
    "4 stops": 4,
}


@st.cache_resource
def load_model():
    with open(MODEL_PATH, "rb") as file:
        return pickle.load(file)


@st.cache_data
def load_columns():
    with open(COLUMNS_PATH, "rb") as file:
        return list(pickle.load(file))


def duration_parts(departure: datetime, arrival: datetime) -> tuple[int, int]:
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
    duration_hours, duration_mins = duration_parts(departure, arrival)

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
        *(1 if airline == option else 0 for option in AIRLINES),
        1 if source == "Chennai" else 0,
        1 if source == "Delhi" else 0,
        1 if source == "Kolkata" else 0,
        1 if source == "Mumbai" else 0,
        1 if destination == "Cochin" else 0,
        1 if destination == "Delhi" else 0,
        1 if destination == "Hyderabad" else 0,
        1 if destination == "Kolkata" else 0,
        1 if destination == "New Delhi" else 0,
    ]


st.title("Flight Fare Prediction")

try:
    model = load_model()
    columns = load_columns()
except Exception as error:
    st.error(f"Could not load model files: {error}")
    st.stop()

journey_date = st.date_input("Journey date")

left, right = st.columns(2)
with left:
    departure_time = st.time_input("Departure time")
with right:
    arrival_time = st.time_input("Arrival time")

airline = st.selectbox("Airline", AIRLINES)
source = st.selectbox("Source", SOURCES)
destination = st.selectbox("Destination", DESTINATIONS)
stop_label = st.selectbox("Total stops", list(STOPS.keys()))

if st.button("Predict fare"):
    departure = datetime.combine(journey_date, departure_time)
    arrival = datetime.combine(journey_date, arrival_time)
    feature_row = build_feature_row(
        airline=airline,
        source=source,
        destination=destination,
        total_stops=STOPS[stop_label],
        departure=departure,
        arrival=arrival,
    )

    if len(feature_row) != len(columns):
        st.error(
            f"Feature mismatch: app created {len(feature_row)} columns, "
            f"but the model expects {len(columns)}."
        )
        st.stop()

    input_data = pd.DataFrame([feature_row], columns=columns)
    prediction = model.predict(input_data)[0]
    st.success(f"Estimated flight fare: Rs. {round(float(prediction), 2):,.2f}")
