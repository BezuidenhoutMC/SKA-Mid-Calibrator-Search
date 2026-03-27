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

    records = []
    cur_headers = []
    i = 0
    n_lines = len(lines)

    while i < n_lines:
        L = lines[i].rstrip()

        # -------------------------
        # Collect header lines
        # -------------------------
        m = hdr_re.match(L)
        if m:
            cur_headers.append(m.groupdict())
            i += 1
            continue

        # -------------------------
        # Separator → process block
        # -------------------------
        if L.startswith('----') or L.startswith('==='):

            # --- ONLY use B1950 ---
            b1950_hdr = None
            for h in cur_headers:
                if h['frame'] == 'B1950':
                    b1950_hdr = h
                    break

            # clear headers buffer
            cur_headers = []

            # skip if no B1950 entry
            if b1950_hdr is None:
                i += 1
                continue

            # skip BAND header lines
            i += 1
            while i < n_lines and (
                lines[i].strip().upper().startswith('BAND') or
                set(lines[i].strip()) <= set('= ')
            ):
                i += 1

            # -------------------------
            # Parse band rows
            # -------------------------
            while i < n_lines:
                line2 = lines[i].rstrip()

                if line2.strip() == "":
                    i += 1
                    break

                if hdr_re.match(line2):
                    break

                if line2.startswith('----') or line2.startswith('==='):
                    break

                bm = band_re.match(line2)
                if bm:
                    bd = bm.groupdict()

                    rec = {}

                    # --- B1950 ONLY ---
                    rec['name'] = b1950_hdr['name']
                    rec['frame'] = 'B1950'
                    rec['header_posq'] = b1950_hdr.get('header_posq')
                    rec['ra'] = b1950_hdr['ra']
                    rec['dec'] = b1950_hdr['dec']
                    rec['extra'] = b1950_hdr.get('extra')

                    # band info
                    rec['band'] = bd.get('band')
                    rec['receiver'] = bd.get('receiver')
                    rec['A'] = bd.get('A')
                    rec['B'] = bd.get('B')
                    rec['C'] = bd.get('C')
                    rec['D'] = bd.get('D')

                    rec['flux'] = float(bd['flux']) if bd.get('flux') not in (None, '') else np.nan

                    records.append(rec)

                i += 1

            continue

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
            figsize=(15, 9),
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
        ax.set_title("Calibrators after all cuts", color="white", pad=20)

        # --- Layout ---
        fig.subplots_adjust(top=0.9, bottom=0.05)

        col1, col2, col3 = st.columns([1, 3, 1])
        with col2:
            st.pyplot(fig, use_container_width=True)

    except Exception as e:
        st.error(f"Error plotting: {e}")

def joinTables(atca, vla, atca_selected_bands,vla_selected_bands):
    # -------------------------
    # Clean VLA table
    # -------------------------
    vla = vla.reset_index()
    vla['name'] = vla['name'].str.replace('"','').str.strip()

    # Rename VLA flux columns to consistent naming if needed
    vla = vla.rename(columns={band: f"flux_{band}" for band in vla_selected_bands})

    # Keep only name, coords, selected fluxes
    vla = vla[['name','ra_deg','dec_deg'] + [f"flux_{b}" for b in vla_selected_bands]]
    vla['origin_VLA'] = True


    # -------------------------
    # Clean ATCA table
    # -------------------------
    atca = atca.rename(columns={
        'Name':'name',
        'R.A.':'ra',
        'Dec.':'dec'
    })

    coords = SkyCoord(atca['ra'], atca['dec'], unit=(u.hourangle, u.deg), frame='icrs')
    atca['ra_deg'] = coords.ra.deg
    atca['dec_deg'] = coords.dec.deg
    atca['name'] = atca['name'].str.strip()

    # Rename flux columns dynamically
    atca = atca.rename(columns={band: f"flux_{band}" for band in atca_selected_bands})

    # Keep only name, coords, selected fluxes
    atca = atca[['name','ra_deg','dec_deg'] + [f"flux_{b}" for b in atca_selected_bands]]
    atca['origin_ATCA'] = True

    # Optional: drop rows where required flux columns are missing
    atca = atca.dropna(subset=[f"flux_{b}" for b in atca_selected_bands])


    # -------------------------
    # Merge catalogs dynamically
    # -------------------------
    calibrators = pd.merge(
        atca,
        vla,
        on='name',
        how='outer',
        suffixes=('_atca','_vla')
    )


    # -------------------------
    # Combine coordinates (prefer ATCA if available)
    # -------------------------
    calibrators['ra_deg'] = calibrators['ra_deg_atca'].combine_first(calibrators['ra_deg_vla'])
    calibrators['dec_deg'] = calibrators['dec_deg_atca'].combine_first(calibrators['dec_deg_vla'])
    calibrators = calibrators.drop(columns=['ra_deg_atca','ra_deg_vla','dec_deg_atca','dec_deg_vla'])


    # -------------------------
    # Fill origin flags
    # -------------------------
    calibrators['origin_ATCA'] = calibrators['origin_ATCA'].fillna(False)
    calibrators['origin_VLA'] = calibrators['origin_VLA'].fillna(False)


    # -------------------------
    # Final column order (dynamic)
    # -------------------------
    final_cols = ['name','ra_deg','dec_deg'] + \
                [f"flux_{b}" for b in atca_selected_bands] + \
                [f"flux_{b}" for b in vla_selected_bands] + \
                ['origin_ATCA','origin_VLA']

    calibrators = calibrators[final_cols]

    return calibrators

def conesearch_internal(search_radius, pb_radius, self_radius, calibrators, VLAdb, ATCAdb):
    st.text(VLAdb)

    ra_calibrators = Angle(calibrators["ra_deg"], unit=u.degree)
    dec_calibrators = Angle(calibrators["dec_deg"], unit=u.degree)
    coords_calibrators = SkyCoord(ra=ra_calibrators, dec=dec_calibrators)

    ra_atca_full = Angle(ATCAdb["R.A."], unit=u.hourangle)
    dec_atca_full = Angle(ATCAdb["Dec."], unit=u.degree)
    coords_atca_full = SkyCoord(ra=ra_atca_full, dec=dec_atca_full)
    ra_vla_full = Angle(VLAdb["ra_deg"], unit=u.degree)
    dec_vla_full = Angle(VLAdb["dec_deg"], unit=u.degree)
    coords_vla_full = SkyCoord(ra=ra_vla_full, dec=dec_vla_full)
    all_coords = np.concatenate((coords_atca_full, coords_vla_full))

    # Iterate through each calibrator in the filtered calibrator list
    drop_list = []
    atca_old = ""
    vla_old = ""
    src_old = ""
    for i in range(len(coords_calibrators)):
        # Calculate separation from the current source to all sources in the full ATCAdb+VLAdb
        separations = coords_calibrators[i].separation(all_coords).to(u.deg)

        # Find indices of sources within the search radius
        mask_nearby = separations < search_radius
        self_sep = separations < self_radius
        mask_nearby_not_self = mask_nearby & ~self_sep
        # Get the original DataFrame index labels for these nearby sources in ATCAdb
        nearby_original = all_coords[mask_nearby_not_self]
        seps_nearby = separations[mask_nearby_not_self].to(u.deg)

        if np.any(nearby_original):
            for j in range(len(nearby_original)):
                src = nearby_original[j]
                # Try to identify if the source is from ATCAdb
                atca_source_mask = (coords_atca_full == src)
                myname_atca_series = ATCAdb[atca_source_mask]["Name"]

                vla_source_mask = (coords_vla_full == src)
                myname_vla_series = VLAdb[vla_source_mask]['name']

                if not myname_atca_series.empty:
                    confusing_atca_name = myname_atca_series.iloc[0] # Get the name string
                    if (confusing_atca_name in atca_old) or (calibrators["name"].iloc[i] in confusing_atca_name):
                        continue
                    atca_old = confusing_atca_name
                    flux_4cm_con = ATCAdb[atca_source_mask]["4cm"].iloc[0] # Ensure single value
                    flux_15mm_con = ATCAdb[atca_source_mask]["15mm"].iloc[0] # Ensure single value|

                    flux_4cm_cal = calibrators["flux_4cm"].iloc[i]
                    flux_15mm_cal = calibrators["flux_15mm"].iloc[i]

                    if (np.isnan(flux_4cm_cal) or np.isnan(flux_15mm_cal)):
                        flux_C_cal = calibrators["flux_C"].iloc[i]
                        flux_U_cal = calibrators["flux_U"].iloc[i]

                        frac_C = flux_4cm_con / flux_C_cal
                        frac_U = flux_15mm_con / flux_U_cal

                        print(f"Calibrator {calibrators["name"].iloc[i]} has confusing source "
                            f"{confusing_atca_name} with separation {seps_nearby[j]:.4f} and flux ratios"
                            f"(confusing/calibrator) at C band: {frac_C:.4f} and U band: {frac_U:.4f}.\n")
                        if seps_nearby[j] <= pb_radius:
                            if (frac_C > 0.1 or frac_U > 0.1):
                                drop_list.append(i)
                        elif seps_nearby[j] <= pb_radius * 2:
                            if (frac_C > 0.2 or frac_U > 0.2):
                                drop_list.append(i)
                        else:
                            if (frac_C > 1 or frac_U > 1):
                                drop_list.append(i)
                        continue

                    else:
                        frac_4cm = flux_4cm_con / flux_4cm_cal
                        frac_15mm = flux_15mm_con / flux_15mm_cal

                        print(f"Calibrator {calibrators["name"].iloc[i]} has confusing source "
                            f"{confusing_atca_name} with separation {seps_nearby[j]:.4f} and flux ratios "
                            f"(confusing/calibrator) at 4cm: {frac_4cm:.4f} and 15mm: {frac_15mm:.4f}.\n")

                    if seps_nearby[j] <= pb_radius:
                        if (frac_4cm > 0.1 or frac_15mm > 0.1):
                            drop_list.append(i)
                    elif seps_nearby[j] <= pb_radius * 2:
                        if (frac_4cm > 0.2 or frac_15mm > 0.2):
                            drop_list.append(i)
                    else:
                        if (frac_4cm > 1 or frac_15mm > 1):
                            drop_list.append(i)

                else: # If not an ATCA source, it must be a VLA source

                    if not myname_vla_series.empty:
                        confusing_vla_name = myname_vla_series.iloc[0]
                        if (confusing_vla_name in vla_old) or (calibrators["name"].iloc[i] in confusing_vla_name):
                            continue
                        vla_old = confusing_vla_name
                        # Now use flux_pivot to get the fluxes for this confusing VLA source
                        flux_C_con = flux_pivot.loc[confusing_vla_name, 'C'] if 'C' in flux_pivot.columns and confusing_vla_name in flux_pivot.index else np.nan
                        flux_U_con = flux_pivot.loc[confusing_vla_name, 'U'] if 'U' in flux_pivot.columns and confusing_vla_name in flux_pivot.index else np.nan

                        flux_C_cal = calibrators["flux_C"].iloc[i]
                        flux_U_cal = calibrators["flux_U"].iloc[i]

                    if (np.isnan(flux_C_cal) or np.isnan(flux_U_cal)):
                        flux_4cm_cal = calibrators["flux_4cm"].iloc[i]
                        flux_15mm_cal = calibrators["flux_15mm"].iloc[i]

                        frac_4cm = flux_C_con / flux_4cm_cal
                        frac_15mm = flux_U_con / flux_15mm_cal

                        print(f"Calibrator {calibrators["name"].iloc[i]} has confusing source "
                            "{confusing_vla_name} with separation {seps_nearby[j]:.4f} and flux ratios"
                            f"(confusing/calibrator) at 4cm: {frac_4cm:.4f} and 15mm: {frac_15mm:.4f}.\n")

                        if seps_nearby[j] <= pb_radius:
                            if (frac_4cm > 0.1 or frac_15mm > 0.1):
                                drop_list.append(i)
                        elif seps_nearby[j] <= pb_radius * 2:
                            if (frac_4cm > 0.2 or frac_15mm > 0.2):
                                drop_list.append(i)
                        else:
                            if (frac_4cm > 1 or frac_15mm > 1):
                                drop_list.append(i)
                        continue

                    frac_C = flux_C_con / flux_C_cal
                    frac_U = flux_U_con / flux_U_cal

                    print(f"Calibrator {calibrators["name"].iloc[i]} has confusing source "
                        f"{confusing_vla_name} with separation {seps_nearby[j]:.4f} and flux ratios "
                        f"(confusing/calibrator) at C band: {frac_C:.4f} and U band: {frac_U:.4f}.\n")

                    if seps_nearby[j] <= pb_radius:
                        if (frac_C > 0.1 or frac_U > 0.1):
                            drop_list.append(i)
                    elif seps_nearby[j] <= pb_radius * 2:
                        if (frac_C > 0.2 or frac_U > 0.2):
                            drop_list.append(i)
                    else:
                        if (frac_C > 1 or frac_U > 1):
                            drop_list.append(i)

        else:
            continue

    if drop_list:
        print(f"Number of calibrators in 'calbirators' with at least one other calibrator in ATCAdb+VLAdb within {search_radius}: {len(drop_list)}")
        print(f"Names of these calibrators to drop: {[calibrators["name"].iloc[i] for i in drop_list]}")
    else:
        print("No calibrators to drop")

    calibrators = calibrators.drop(calibrators.index[i] for i in drop_list)
    return calibrators

def main():
    
    Dec_lim, atca_selected_bands, atca_flux_limits, vla_selected_bands, vla_flux_limits, vla_pos_quality, vla_ampq, quality_mode = setupSidebar()

    ATCAdb= ParseATCA('ATCA Calibrators Database.csv')
    ATCA_after_cuts = ATCA_cuts(Dec_lim, atca_selected_bands, atca_flux_limits, ATCAdb)

    VLAdb = ParseVLA('VLA Calibrator List 2.csv')

    VLA_after_cuts = VLA_cuts(VLAdb, Dec_lim, vla_selected_bands, vla_flux_limits, vla_pos_quality, vla_ampq, quality_mode)
    
    joint_cal_list = joinTables(ATCA_after_cuts,VLA_after_cuts,atca_selected_bands,vla_selected_bands)
    
    search_radius = 2 * u.degree
    pb_radius = 0.4 / 2 * u.degree
    self_radius = 10/3600 * u.degree
    
    st.text(VLAdb)
    if search_radius != 0:
        joint_cal_list = conesearch_internal(search_radius, pb_radius, self_radius, joint_cal_list, VLAdb, ATCAdb)
    
    ra_joint_rad = np.radians(joint_cal_list['ra_deg'].values)
    dec_joint_rad = np.radians(joint_cal_list['dec_deg'].values)
    ra_joint_rad = np.remainder(ra_joint_rad + np.pi, 2*np.pi) - np.pi

    st.success(f"{len(joint_cal_list)} combined sources")
    st.success(f"{len(ATCA_after_cuts)} ATCA sources")
    st.success(f"{len(VLA_after_cuts)} VLA sources")

    
    

    plotSkyCoords(ra_joint_rad,dec_joint_rad)
    st.dataframe(joint_cal_list, height=600)    
    csv = joint_cal_list.to_csv(index=False)

    st.download_button(
        label="Download calibrator list as CSV",
        data=csv,
        file_name="calibrators.csv",
        mime="text/csv",
    )
main()