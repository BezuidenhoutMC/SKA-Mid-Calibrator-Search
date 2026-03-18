import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from astropy.coordinates import SkyCoord, Angle
import astropy.units as u

# --- Page config ---
st.set_page_config(page_title="ATCA Calibrator Viewer", layout="wide")

st.title("ATCA Calibrator Sky Distribution")

# --- File upload ---
# Load data
ATCAdb = pd.read_csv('ATCA Calibrators Database.csv')

st.write("Preview of data:")
st.dataframe(ATCAdb.head())

# --- Coordinate conversion ---
try:
    ra_vals = Angle(ATCAdb["R.A."], unit=u.hourangle)
    dec_vals = Angle(ATCAdb["Dec."], unit=u.degree)

    c = SkyCoord(ra=ra_vals, dec=dec_vals, frame="icrs")

    ra_rad = c.ra.wrap_at(180 * u.deg).radian
    dec_rad = c.dec.radian

    # --- Plot ---
    fig, ax = plt.subplots(
        figsize=(8, 4.2),
        subplot_kw=dict(projection="mollweide")
    )

    ax.grid(True)
    ax.scatter(ra_rad, dec_rad, marker="o", s=2, alpha=0.3)

    ax.set_title("ATCA calibrators (before cut)")

    # Show in Streamlit
    st.pyplot(fig)

    # --- Info ---
    st.success(f"{len(c)} sources before cut")

except Exception as e:
    st.error(f"Error processing data: {e}")
