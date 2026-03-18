import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from astropy.coordinates import SkyCoord, Angle
import astropy.units as u

# --- Page config ---
st.set_page_config(page_title="SKAO Calibrator Search", layout="wide")

st.title("SKAO Calibrator Search")

# --- File upload ---
# Load data
ATCAdb = pd.read_csv('ATCA Calibrators Database.csv')

# st.write("Preview of data:")
# st.dataframe(ATCAdb.head())

# --- Coordinate conversion ---
try:
    ra_vals = Angle(ATCAdb["R.A."], unit=u.hourangle)
    dec_vals = Angle(ATCAdb["Dec."], unit=u.degree)

    c = SkyCoord(ra=ra_vals, dec=dec_vals, frame="icrs")

    ra_rad = c.ra.wrap_at(180 * u.deg).radian
    dec_rad = c.dec.radian

    # # --- Plot ---
    # fig, ax = plt.subplots(
    #     figsize=(8, 4.2),
    #     subplot_kw=dict(projection="mollweide")
    # )

    # ax.grid(True)
    # ax.scatter(ra_rad, dec_rad, marker="o", s=2, alpha=0.3)

    # ax.set_title("ATCA calibrators (before cut)")

    # # Show in Streamlit
    # st.pyplot(fig)

    # --- Info ---
    st.success(f"{len(c)} sources before cut")

except Exception as e:
    st.error(f"Error processing ATACA data: {e}")

## Apply Declination & Flux density cuts at 4cm
try: 
    Dec_lim = 40 #deg
    Flux_lim = 5 #Jy

    ATCA_CutDec = ATCAdb[Angle(ATCAdb["Dec."],unit=u.deg).deg< Dec_lim]
    ATCA_CutF4cm = ATCA_CutDec[
        (ATCA_CutDec["4cm"] > Flux_lim) | (ATCA_CutDec["15mm"] > Flux_lim)]

except Exception as e:
    st.error(f"Error making ATCA cuts {e}")

## Plot ATCA catalog after cuts
try:
    ra_vals = Angle(ATCA_CutF4cm["R.A."], unit=u.hourangle)
    dec_vals = Angle(ATCA_CutF4cm["Dec."], unit=u.degree)

    c = SkyCoord(ra=ra_vals, dec=dec_vals, frame='icrs')
    ra_rad = c.ra.wrap_at(180 * u.deg).radian
    dec_rad = c.dec.radian

    fig, ax = plt.subplots(figsize=(8, 4.2), subplot_kw=dict(projection="mollweide"))
    ax.grid(True)
    ax.scatter(ra_rad, dec_rad, marker="o", s=2, alpha=0.3)
    fig.subplots_adjust(top=0.95, bottom=0.0)
    ax.set_title('ATCA Calibrators (after cut)')
    st.pyplot(fig)
    print(len(ATCA_CutF4cm), 'sources after 4cm & 16cm Flux & Dec cuts')
    print(ATCA_CutF4cm)
except Exception as e:
    st.error(f"Error plotting ATCA after cuts: {e}")