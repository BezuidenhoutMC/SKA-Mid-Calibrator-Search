import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from astropy.coordinates import SkyCoord, Angle
import astropy.units as u

# --- Page config ---
st.set_page_config(page_title="SKAO Calibrator Search", layout="wide")

st.title("SKAO Calibrator Search")


def ParseATCA(fname):
    try:
        ATCAdb = pd.read_csv(fname)
    except Exception as e:
        st.error(f"Error reading ATCA Database file: {e}")     

    # --- Coordinate conversion ---
    try:
        ra_vals = Angle(ATCAdb["R.A."], unit=u.hourangle)
        dec_vals = Angle(ATCAdb["Dec."], unit=u.degree)

        c = SkyCoord(ra=ra_vals, dec=dec_vals, frame="icrs")

        ra_rad = c.ra.wrap_at(180 * u.deg).radian
        dec_rad = c.dec.radian

        return ATCAdb, c,ra_rad,dec_rad
        # st.success(f"{len(c)} sources before cut")

    except Exception as e:
        st.error(f"Error processing ATCA data: {e}")

def ATCA_cuts(Dec_lim,Flux_lim,ATCAdb):
        st.write(ATCAdb)
        ATCA_CutDec = ATCAdb[Angle(ATCAdb["Dec."],unit=u.deg).deg< Dec_lim]
        ATCA_CutFlux = ATCA_CutDec[
            (ATCA_CutDec["4cm"] > Flux_lim) | (ATCA_CutDec["15mm"] > Flux_lim)]
        return ATCA_CutFlux

    # except Exception as e:
    #     st.error(f"Error making ATCA cuts {e}")

def plotSkyCoords(ra_rad,dec_rad):
    ## Plot ATCA catalog after cuts
    try:
        fig, ax = plt.subplots(figsize=(8, 4.2), subplot_kw=dict(projection="mollweide"))
        ax.grid(True)
        ax.scatter(ra_rad, dec_rad, marker="o", s=2, alpha=0.3)
        fig.subplots_adjust(top=0.95, bottom=0.0)
        ax.set_title('ATCA Calibrators (after cut)')
        st.pyplot(fig)
        st.success(f"{len(ATCA_CutFlux)} sources after Flux & Dec cuts")
    except Exception as e:
        st.error(f"Error plotting ATCA after cuts: {e}")

def main():
    st.sidebar.header("Filter settings")

    Dec_lim = st.sidebar.number_input(
    "Minimum Declination (deg)",
    min_value=-90.0,
    max_value=90.0,
    value=30.0,
    step=1.0
    )

    Flux_lim = st.sidebar.slider(
    "Minimum Flux (Jy)",
    min_value=0.0,
    max_value = 100.0,
    value=5.0,
    step=0.1
    )

    ATCAdb, c, ra_rad, dec_rad = ParseATCA('ATCA Calibrators Database.csv')
    ATCA_after_cuts = ATCA_cuts(Dec_lim,Flux_lim,ATCAdb)

    ra_vals = Angle(ATCA_after_cuts["R.A."], unit=u.hourangle)
    dec_vals = Angle(ATCA_after_cuts["Dec."], unit=u.degree)

    c = SkyCoord(ra=ra_vals, dec=dec_vals, frame='icrs')
    ra_rad = c.ra.wrap_at(180 * u.deg).radian
    dec_rad = c.dec.radian

    plotSkyCoords(ra_rad,dec_rad)

main()