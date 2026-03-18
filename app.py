import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import re

from astropy.coordinates import SkyCoord, Angle
import astropy.units as u

# --- Page config ---
st.set_page_config(page_title="SKAO Calibrator Search", layout="wide")

st.title("SKAO Calibrator Search")

# -------------------------
# Helper numeric parsers
# -------------------------
def ra_to_deg(ra_str):
    nums = re.findall(r'\d+\.?\d*', ra_str)
    if len(nums) < 3:
        raise ValueError(f"Cannot parse RA string: {ra_str!r}")
    h, m, s = map(float, nums[:3])
    return (h + m/60.0 + s/3600.0) * 15.0

def dec_to_deg(dec_str):
    s = dec_str.strip()
    # remove stray backslashes / double quotes etc we saw in HTML-copy
    s = s.replace("\\'", "'").replace('\\"', '"').replace('""', '"')
    sign = -1 if s.startswith('-') else 1
    nums = re.findall(r'\d+\.?\d*', s)
    if len(nums) < 3:
        raise ValueError(f"Cannot parse DEC string: {dec_str!r}")
    d, m, sec = map(float, nums[:3])
    return sign * (d + m/60.0 + sec/3600.0)

def setupSidebar():
    st.sidebar.header("Filter settings")

    Dec_lim = st.sidebar.number_input(
    "Minimum Declination (deg)",
    min_value=-90.0,
    max_value=90.0,
    value=30.0,
    step=1.0
    )

    with st.sidebar.expander("ATCA", expanded=True): 
        atca_bands = ["16cm", "4cm", "15mm", "7mm", "3mm"]

        atca_selected_bands = []
        atca_flux_limits = {}

        for band in atca_bands:
            # st.markdown(f"**{band}**")
            use_band = st.checkbox(f"{band}", value=(band in ["15mm", "4cm"]))

            if use_band:
                atca_selected_bands.append(band)

                atca_flux_limits[band] = st.number_input(
                    f"{band} flux limit (Jy)",
                    min_value=0.0,
                    value=1.0,
                    step=0.1,
                    key=f"{band}_flux"
                )

    with st.sidebar.expander("VLA", expanded=True): 
        vla_bands = ["P", "L", "C", "X", "U", "K", "Q"]

        vla_selected_bands = []
        vla_flux_limits = {}

        for band in vla_bands:
            # st.markdown(f"**{band}**")
            use_band = st.checkbox(f"{band}", value=(band in ["C", "X", "U"]))

            if use_band:
                vla_selected_bands.append(band)

                vla_flux_limits[band] = st.number_input(
                    f"{band} flux limit (Jy)",
                    min_value=0.0,
                    value=1.0,
                    step=0.1,
                    key=f"{band}_flux"
                )

        pos_quality_option = st.selectbox(
        "Positional accuracy",
            [
                "A (<0.002 arcsec)",
                "B (<0.01 arcsec)",
                "C (<0.015 arcsec)",
                "T (>0.015 arcsec)"
            ]
        )
        posq_map = {
            "A (<0.002 arcsec)": ["A"],
            "B (<0.01 arcsec)": ["A", "B"],
            "C (<0.015 arcsec)": ["A", "B", "C"],
            "T (>0.015 arcsec)": ["A", "B", "C", "T"],
            }

        vla_pos_quality = posq_map[pos_quality_option]

        amp_quality_option = st.selectbox(
        "Amplitude closure quality",
            [
                "P (<3%)",
                "S (<10%)",
                "W (>10%)"
            ]
        )
        ampq_map = {
            "P (<3%)": ["P"],
            "S (<10%)": ["P", "S"],
            "W (>10%)": ["P", "S", "W"],
        }

        vla_ampq = ampq_map[amp_quality_option]

        quality_mode = st.selectbox(
            "Amplitude closure quality configuration",
            [
                "Any",
                "A",
                "B",
                "C",
                "D"
            ]
        )

    return Dec_lim, atca_selected_bands, atca_flux_limits, vla_selected_bands, vla_flux_limits, vla_pos_quality, vla_ampq, quality_mode

def ParseATCA(fname):
    try:
        ATCAdb = pd.read_csv(fname)
        return ATCAdb

    except Exception as e:
        st.error(f"Error reading ATCA Database file: {e}")     

def ParseVLA(fn):
    with open(fn, 'r', encoding='utf-8') as fh:
        lines = [L.rstrip('\n') for L in fh]

    # Patterns
    # header lines (either J2000 or B1950)
    hdr_re = re.compile(
        r'^(?P<name>\S+)\s+'                          # source name
        r'(?P<frame>J2000|B1950)\s+'
        r'(?P<header_posq>\S)\s+'
        r'(?P<ra>\d{2}h\d{2}m\d{2}(?:\.\d+)?s)\s+'   # RA in hhmmss.sss format
        r'(?P<dec>[+-]?\d{2}d\d{2}\'\d{2}(?:\.\d+)?"?)' # Dec in ddmmss.ss"
        r'(?:\s+(?P<extra>.*))?'
    )

    # band lines: band, receiver, A B C D codes, optional flux as next numeric token
    band_re = re.compile(
        r'^\s*(?P<band>\S+)\s+'            # e.g. "0.7cm" or " 20cm"
        r'(?P<receiver>\S)\s+'            # single-letter receiver label (e.g. Q,L,U,...)
        r'(?P<A>\S)\s+(?P<B>\S)\s+(?P<C>\S)\s+(?P<D>\S)'  # the four A/B/C/D codes
        r'(?:\s+(?P<flux>[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?))?'  # optional float flux (Jy)
        r'.*$'
    )

    records = []           # each will be a dict: name, frame, ra, dec, band, receiver, A,B,C,D,flux,header_posq,extra
    cur_headers = []       # accumulate header lines until separator
    i = 0
    n_lines = len(lines)

    while i < n_lines:
        L = lines[i].rstrip()
        # collect header lines until we hit a dashed separator
        m = hdr_re.match(L)
        if m:
            cur_headers.append(m.groupdict())
            i += 1
            # continue collecting possible second header (B1950 or J2000)
            # do not finalize until we see the dashed separator line
            continue

        # dashed separator indicates end of header block and that band table follows
        if L.startswith('----') or L.startswith('==='):
            # select J2000 header if present, else B1950, else skip
            j2000_hdr = None
            b1950_hdr = None
            for h in cur_headers:
                if h['frame'] == 'J2000':
                    j2000_hdr = h
                elif h['frame'] == 'B1950':
                    b1950_hdr = h
            chosen_hdr = j2000_hdr or b1950_hdr

            # clear headers buffer
            cur_headers = []

            # move to the next lines which will likely include BAND heading and = lines; skip those
            i += 1
            while i < n_lines and (lines[i].strip().upper().startswith('BAND') or set(lines[i].strip()) <= set('= ')):
                i += 1

            # now parse band lines until blank or until next header/separator
            while i < n_lines:
                line2 = lines[i].rstrip()
                if line2.strip() == "":
                    i += 1
                    break
                # if we hit a new header line, break (it will be consumed in main loop)
                if hdr_re.match(line2):
                    break
                # if we hit a dashed separator for next source, break to outer loop to handle it
                if line2.startswith('----') or line2.startswith('==='):
                    break
                bm = band_re.match(line2)
                if bm and chosen_hdr:
                    bd = bm.groupdict()
                    rec = {}
                    # Record header-sourced info (prefer J2000 name; but we keep both name/frame if you want)
                    rec['name_j2000'] = chosen_hdr['name'] if chosen_hdr['frame']=='J2000' else None
                    rec['name_b1950'] = chosen_hdr['name'] if chosen_hdr['frame']=='B1950' else None
                    # For convenience keep a canonical name: if J2000 name exists prefer that, else B1950
                    rec['name'] = rec['name_j2000'] or rec['name_b1950'] or chosen_hdr.get('name')
                    rec['frame'] = chosen_hdr['frame']
                    rec['header_posq'] = chosen_hdr.get('header_posq')
                    rec['ra'] = chosen_hdr['ra']
                    rec['dec'] = chosen_hdr['dec']
                    rec['extra'] = chosen_hdr.get('extra')

                    # band-level fields
                    rec['band'] = bd.get('band')
                    rec['receiver'] = bd.get('receiver')
                    rec['A'] = bd.get('A')
                    rec['B'] = bd.get('B')
                    rec['C'] = bd.get('C')
                    rec['D'] = bd.get('D')
                    # flux might be missing -> keep as NaN
                    rec['flux'] = float(bd['flux']) if bd.get('flux') not in (None, '') else np.nan
                    # print(rec)
                    records.append(rec)

                # move to next band line
                i += 1
            # continue outer loop
            continue

        # otherwise nothing matched: advance
        i += 1

    VLAdb = pd.DataFrame.from_records(records)
    return VLAdb

def ATCA_cuts(Dec_lim,selected_bands,flux_limits, ATCAdb):
    # --- Dec cut ---
    ATCA_CutDec = ATCAdb[
        Angle(ATCAdb["Dec."], unit=u.deg).deg < Dec_lim
    ]

    # --- Flux cut (ALL selected bands must pass) ---
    if len(selected_bands) == 0:
        ATCA_CutFlux = ATCA_CutDec
    else:
        flux_mask = np.ones(len(ATCA_CutDec), dtype=bool)

        for band in selected_bands:
            if band in ATCA_CutDec.columns:
                limit = flux_limits[band]
                flux_mask &= (ATCA_CutDec[band] > limit)
            else:
                # If a selected band is missing, fail the condition
                flux_mask &= False

        ATCA_CutFlux = ATCA_CutDec[flux_mask]

    return ATCA_CutFlux

def VLA_cuts(VLAdb, Dec_lim, vla_selected_bands, vla_flux_limits, vla_pos_quality, vla_ampq, quality_mode):
    # -------------------------
    # Positional certainty filter
    # -------------------------
    VLAdb['header_posq'] = VLAdb['header_posq'].astype(str).str.strip().str.upper()
    VLAdb = VLAdb[VLAdb['header_posq'].isin(vla_pos_quality)]

    # -------------------------
    # Convert RA/Dec to decimals
    # -------------------------
    VLAdb = VLAdb.copy()
    VLAdb['ra_deg'] = VLAdb['ra'].apply(ra_to_deg)
    VLAdb['dec_deg'] = VLAdb['dec'].apply(dec_to_deg)

    # -------------------------
    # Build flux pivot (one row per canonical source)
    # -------------------------
    VLAdb['receiver'] = VLAdb['receiver'].str.strip().str.upper()
    for col in ['A','B','C','D']:
        VLAdb[col] = VLAdb[col].astype(str).str.strip().str.upper().replace({'NAN': np.nan})

    flux_pivot = VLAdb.pivot_table(
        index='name', columns='receiver', values='flux', aggfunc='max'
    )

    # -------------------------
    # Build quality pivot
    # -------------------------
    def amplitude_closer_quality(row):
        if quality_mode.lower() == 'any':
            return any((row.get(col) in vla_ampq) for col in ['A','B','C','D'])
        else:
            col = quality_mode.upper()
            return row.get(col) in vla_ampq

    VLAdb['quality_PS'] = VLAdb.apply(amplitude_closer_quality, axis=1)
    quality_pivot = VLAdb.pivot_table(
        index='name', columns='receiver', values='quality_PS', aggfunc='max'
    ).fillna(False)

    # -------------------------
    # Apply flux & quality cuts dynamically on selected bands
    # -------------------------
    if len(vla_selected_bands) == 0:
        valid_names = flux_pivot.index  # no flux restriction
    else:
        flux_mask = pd.Series(True, index=flux_pivot.index)
        quality_mask = pd.Series(True, index=flux_pivot.index)

        for band in vla_selected_bands:
            col = band.upper()
            # Treat missing columns as zeros / False
            band_flux = flux_pivot.get(col, pd.Series(0.0, index=flux_pivot.index)).fillna(0.0)
            band_quality = quality_pivot.get(col, pd.Series(False, index=flux_pivot.index)).fillna(False)

            flux_limit = vla_flux_limits.get(band, 0.0)

            flux_mask &= (band_flux > flux_limit)
            quality_mask &= band_quality

        # Only keep sources passing **all selected bands** AND quality
        valid_names = flux_pivot.index[flux_mask & quality_mask]

    # -------------------------
    # Build per-source coordinate table
    # -------------------------
    per_name = VLAdb.groupby('name').agg({
        'ra_deg': 'median',
        'dec_deg': 'median'
    })

    # Add flux columns for selected bands (optional, for diagnostics)
    for band in vla_selected_bands:
        per_name[f'flux_{band}'] = flux_pivot.get(band.upper())

    # Final selection with Dec cut
    sel = per_name.loc[per_name.index.intersection(valid_names)].copy()
    sel = sel[sel['dec_deg'] < Dec_lim]

    return sel

def plotSkyCoords(ra_rad, dec_rad):
    try:
        fig, ax = plt.subplots(
            figsize=(8, 4.2),
            subplot_kw=dict(projection="mollweide")
        )

        # --- Transparent background ---
        fig.patch.set_alpha(0)
        ax.set_facecolor("none")

        # --- Scatter ---
        ax.scatter(
            ra_rad,
            dec_rad,
            marker="o",
            s=4,                
            alpha=0.6,
            color="white"       
        )

        # --- Grid styling ---
        ax.grid(True, color="white", alpha=0.2, linestyle="--")

        # --- Tick + label colors ---
        ax.tick_params(colors="white")

        # Mollweide axes labels
        ax.set_xticklabels(ax.get_xticklabels(), color="white")
        ax.set_yticklabels(ax.get_yticklabels(), color="white")
        # ax.invert_xaxis()

        # --- Title ---
        ax.set_title("Calibrators After Cuts", color="white", pad=20)

        # --- Layout ---
        fig.subplots_adjust(top=0.9, bottom=0.05)

        col1, col2, col3 = st.columns([1, 3, 1])
        with col2:
            st.pyplot(fig, use_container_width=True)

    except Exception as e:
        st.error(f"Error plotting: {e}")

def joinTables(atca_table, vla_table):
    atca = pd.DataFrame.copy(atca_table)
    vla = pd.DataFrame.copy(vla_table)

    # -------------------------
    # Clean VLA table
    # -------------------------
    vla = vla.reset_index()
    vla['name'] = vla['name'].str.replace('"','').str.strip()

    # -------------------------
    # Clean ATCA table
    # -------------------------
    atca = atca.rename(columns={
        'Name': 'name',
        'R.A.': 'ra',
        'Dec.': 'dec'
    })

    coords = SkyCoord(atca['ra'], atca['dec'],
                      unit=(u.hourangle, u.deg), frame='icrs')

    atca['ra_deg'] = coords.ra.deg
    atca['dec_deg'] = coords.dec.deg

    atca['name'] = atca['name'].str.strip()

    # Rename ATCA flux columns dynamically
    for col in atca.columns:
        if col in ["16cm", "4cm", "15mm", "7mm", "3mm"]:
            atca = atca.rename(columns={col: f"flux_{col}"})

    # -------------------------
    # Identify columns
    # -------------------------
    base_cols = ['name', 'ra_deg', 'dec_deg']

    atca_flux_cols = [c for c in atca.columns if c.startswith("flux_")]
    vla_flux_cols  = [c for c in vla.columns  if c.startswith("flux_")]

    all_flux_cols = sorted(set(atca_flux_cols + vla_flux_cols))

    # -------------------------
    # Ensure both tables have same columns
    # -------------------------
    for col in all_flux_cols:
        if col not in atca:
            atca[col] = np.nan
        if col not in vla:
            vla[col] = np.nan

    # -------------------------
    # Keep only relevant columns
    # -------------------------
    final_cols = base_cols + all_flux_cols

    atca = atca[final_cols]
    vla  = vla[final_cols]

    # -------------------------
    # Merge catalogs
    # -------------------------
    calibrators = pd.concat([atca, vla], ignore_index=True)

    # remove duplicates by source name
    calibrators = calibrators.drop_duplicates(subset='name')

    return calibrators

def main():
    
    Dec_lim, atca_selected_bands, atca_flux_limits, vla_selected_bands, vla_flux_limits, vla_pos_quality, vla_ampq, quality_mode = setupSidebar()

    ATCAdb= ParseATCA('ATCA Calibrators Database.csv')
    ATCA_after_cuts = ATCA_cuts(Dec_lim, atca_selected_bands, atca_flux_limits, ATCAdb)

    VLAdb = ParseVLA('VLA Calibrator List 2.csv')

    VLA_after_cuts = VLA_cuts(VLAdb, Dec_lim, vla_selected_bands, vla_flux_limits, vla_pos_quality, vla_ampq, quality_mode)
    
    joint_cal_list = joinTables(ATCA_after_cuts,VLA_after_cuts)
    ra_joint_rad = np.radians(joint_cal_list['ra_deg'].values)
    dec_joint_rad = np.radians(joint_cal_list['dec_deg'].values)
    ra_joint_rad = np.remainder(ra_joint_rad + np.pi, 2*np.pi) - np.pi

    plotSkyCoords(ra_joint_rad,dec_joint_rad)
    st.success(f"{len(ATCA_after_cuts)} ATCA sources")
    st.success(f"{len(VLA_after_cuts)} VLA sources")
    st.success(f"{len(joint_cal_list)} combined sources")

    st.write(joint_cal_list)
main()