import pandas as pd
import streamlit as st

from ecmwf_client import AVAILABLE_THRESHOLDS_MM, fetch_exceedance_probabilities

st.set_page_config(page_title="Rainfall Exceedance Forecast", page_icon="🌧️")

st.title("🌧️ Rainfall Exceedance Forecast")
st.caption("ECMWF ENS open data — probability of 24h rainfall exceeding a threshold")

# Stage 1: single fixed location. Add more here as you expand the app.
LOCATIONS = {
    "Guiguinto, Bulacan, Philippines": (14.842279, 120.859681),
}


@st.cache_data(ttl=3 * 60 * 60, show_spinner=False)  # ECMWF ENS updates 4x/day
def get_probabilities(lat: float, lon: float, threshold_mm: int, lead_days: int):
    return fetch_exceedance_probabilities(lat, lon, threshold_mm, max_lead_days=lead_days)


col1, col2 = st.columns(2)
with col1:
    location_name = st.selectbox("Location", list(LOCATIONS.keys()))
    lat, lon = LOCATIONS[location_name]
with col2:
    threshold = st.selectbox(
        "Threshold (24h accumulated)",
        AVAILABLE_THRESHOLDS_MM,
        index=3,  # defaults to 20 mm
        format_func=lambda x: f"{x} mm",
    )

lead_days = st.slider("Forecast range (days)", min_value=1, max_value=15, value=10)

if st.button("Get forecast", type="primary"):
    with st.spinner("Fetching latest ECMWF ENS forecast..."):
        try:
            probs = get_probabilities(lat, lon, threshold, lead_days)
        except Exception as e:
            st.error(f"Failed to fetch forecast: {e}")
            st.stop()

    if not probs:
        st.warning("No data returned for this request.")
        st.stop()

    first_window, first_val = next(iter(probs.items()))
    end_hour = first_window.split("-")[1]
    st.metric(f"P(24h rainfall ≥ {threshold} mm) — next {end_hour}h", f"{first_val:.0f}%")

    st.subheader("Full forecast timeline")
    df = pd.DataFrame(
        {"Window (h)": list(probs.keys()), "Probability (%)": list(probs.values())}
    )
    st.bar_chart(df.set_index("Window (h)"))
    st.dataframe(df, use_container_width=True, hide_index=True)

    st.caption(
        f"Location: {location_name} ({lat}, {lon}) · "
        "Source: ECMWF ENS Open Data (CC BY 4.0)"
    )
else:
    st.info("Choose a threshold and click **Get forecast**.")
