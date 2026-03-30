import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import re

from astropy.coordinates import SkyCoord, Angle, search_around_sky
import astropy.units as u
from astroquery.utils.tap.core import TapPlus

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

    with st.sidebar.expander("ATCA", expanded=False): 
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

    with st.sidebar.expander("VLA", expanded=False): 
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

    with st.sidebar.expander("Cone Search", expanded=False): 
        use_cone = st.checkbox('ATCA/VLA Search', value=False)
        st.text('Performs a cone search on all candidate sources against the full ATCA+VLA calibrator catalogues.')
        if use_cone:
            search_radius = st.number_input(
                    f"Search Radius (deg)",
                    min_value=0.0,
                    value=2.0,
                    step=0.1,
                    key="cone search radius"
                )
            search_radius = search_radius * u.degree

            pb_radius = st.number_input(
                    f"Primary Beam Radius (deg)",
                    min_value=0.0,
                    value=0.2,
                    step=0.05,
                    key="primary beam radius"
                )
            pb_radius = pb_radius * u.degree

            self_radius = st.number_input(
                    f"Self Radius (arcsec)",
                    min_value=0.0,
                    value=10.0,
                    step=1.0,
                    key="self radius"
                )
            self_radius = self_radius/3600 * u.degree
        else:
            search_radius = 0
            pb_radius = 0
            self_radius = 0

    return Dec_lim, atca_selected_bands, atca_flux_limits, vla_selected_bands, vla_flux_limits, vla_pos_quality, vla_ampq, quality_mode, search_radius, pb_radius, self_radius

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

def conesearch_internal(search_radius, pb_radius, self_radius,
                    calibrators, VLAdb, ATCAdb):

    # -------------------------
    # Precompute coordinates
    # -------------------------
    coords_cal = SkyCoord(ra=calibrators["ra_deg"].values * u.deg,
                          dec=calibrators["dec_deg"].values * u.deg)

    coords_atca = SkyCoord(ra=ATCAdb["R.A."].values,
                           dec=ATCAdb["Dec."].values,
                           unit=(u.hourangle, u.deg))

    VLAdb = VLAdb.copy()
    VLAdb['ra_deg'] = VLAdb['ra'].apply(ra_to_deg)
    VLAdb['dec_deg'] = VLAdb['dec'].apply(dec_to_deg)

    coords_vla = SkyCoord(ra=VLAdb["ra_deg"].values * u.deg,
                          dec=VLAdb["dec_deg"].values * u.deg)

    # Combine catalogs
    all_coords = SkyCoord(
        ra=np.concatenate([coords_atca.ra, coords_vla.ra]),
        dec=np.concatenate([coords_atca.dec, coords_vla.dec])
    )

    # Track origin
    n_atca = len(coords_atca)

    # -------------------------
    # Precompute flux tables
    # -------------------------
    flux_pivot = VLAdb.pivot_table(
        index='name', columns='receiver', values='flux', aggfunc='max'
    )

    atca_names = ATCAdb["Name"].values
    vla_names = VLAdb["name"].values

    # -------------------------
    # FAST cone search
    # -------------------------
    idx_cal, idx_all, sep2d, _ = search_around_sky(
        coords_cal, all_coords, search_radius
    )

    drop_mask = np.zeros(len(calibrators), dtype=bool)

    # -------------------------
    # Loop ONLY over matches
    # -------------------------
    for i_cal, i_all, sep in zip(idx_cal, idx_all, sep2d):

        # Skip self-match
        if sep < self_radius:
            continue

        name_cal = calibrators["name"].iloc[i_cal]

        # -------------------------
        # Determine catalog
        # -------------------------
        if i_all < n_atca:
            # ATCA source
            name_con = atca_names[i_all]

            if name_con == name_cal:
                continue

            flux_4cm_con = ATCAdb["4cm"].iloc[i_all]
            flux_15mm_con = ATCAdb["15mm"].iloc[i_all]

            flux_4cm_cal = calibrators["flux_4cm"].iloc[i_cal]
            flux_15mm_cal = calibrators["flux_15mm"].iloc[i_cal]

            if np.isnan(flux_4cm_cal) or np.isnan(flux_15mm_cal):
                flux_C_cal = calibrators["flux_C"].iloc[i_cal]
                flux_U_cal = calibrators["flux_U"].iloc[i_cal]

                frac1 = flux_4cm_con / flux_C_cal if flux_C_cal else 0
                frac2 = flux_15mm_con / flux_U_cal if flux_U_cal else 0
            else:
                frac1 = flux_4cm_con / flux_4cm_cal
                frac2 = flux_15mm_con / flux_15mm_cal

        else:
            # VLA source
            i_vla = i_all - n_atca
            name_con = vla_names[i_vla]

            if name_con == name_cal:
                continue

            flux_C_con = flux_pivot.loc[name_con, 'C'] if name_con in flux_pivot.index else np.nan
            flux_U_con = flux_pivot.loc[name_con, 'U'] if name_con in flux_pivot.index else np.nan

            flux_C_cal = calibrators["flux_C"].iloc[i_cal]
            flux_U_cal = calibrators["flux_U"].iloc[i_cal]

            if np.isnan(flux_C_cal) or np.isnan(flux_U_cal):
                flux_4cm_cal = calibrators["flux_4cm"].iloc[i_cal]
                flux_15mm_cal = calibrators["flux_15mm"].iloc[i_cal]

                frac1 = flux_C_con / flux_4cm_cal if flux_4cm_cal else 0
                frac2 = flux_U_con / flux_15mm_cal if flux_15mm_cal else 0
            else:
                frac1 = flux_C_con / flux_C_cal
                frac2 = flux_U_con / flux_U_cal

        # -------------------------
        # Apply thresholds
        # -------------------------
        if sep <= pb_radius:
            if frac1 > 0.1 or frac2 > 0.1:
                drop_mask[i_cal] = True
        elif sep <= 2 * pb_radius:
            if frac1 > 0.2 or frac2 > 0.2:
                drop_mask[i_cal] = True
        else:
            if frac1 > 1 or frac2 > 1:
                drop_mask[i_cal] = True

    # -------------------------
    # Drop sources
    # -------------------------
    drop_indices = np.where(drop_mask)[0]

    print(f"Dropping {len(drop_indices)} calibrators")

    return calibrators.drop(calibrators.index[drop_indices])

def conesearch_PMN(calibrators):
    # Query the PMN catalogues on vizier
    tap = TapPlus(url="http://tapvizier.u-strasbg.fr/TAPVizieR/tap")

    job = tap.launch_job("""
        SELECT *
        FROM "VIII/38/pmns"
        WHERE Flux > 100
        ORDER BY RAJ2000 ASC
    """)
    table = job.get_results()
    PMNS = table.to_pandas()

    job = tap.launch_job("""
        SELECT *
        FROM "VIII/38/pmnz"
        WHERE Flux > 100
        ORDER BY RAJ2000 ASC
    """)
    table = job.get_results()
    PMNZ = table.to_pandas()

    job = tap.launch_job("""
        SELECT *
        FROM "VIII/38/pmnt"
        WHERE Flux > 100
        ORDER BY RAJ2000 ASC
    """)
    table = job.get_results()
    PMNT = table.to_pandas()

    job = tap.launch_job("""
        SELECT *
        FROM "VIII/38/pmne"
        WHERE Flux > 100
        ORDER BY RAJ2000 ASC
    """)
    table = job.get_results()
    PMNE = table.to_pandas()

    PMN = pd.concat([PMNS, PMNZ, PMNT, PMNE], ignore_index=True)

    ra_PMN = Angle(PMN['RAJ2000'], unit=u.degree)
    dec_PMN = Angle(PMN['DEJ2000'], unit=u.degree)
    coords_PMN = SkyCoord(ra=ra_PMN, dec=dec_PMN, frame='icrs')

    PMN_cat_flux = PMN["Flux"] * 1e-3 #Jy
    PMN_cat_gflux = PMN["GFlux"] * 1e-3 #Jy

    atca_freq = 5.5e9
    PMN_freq = 4.850e9
    PMN_cat_flux_scaled = PMN_cat_flux * (atca_freq / PMN_freq) ** -0.7
    PMN_cat_gflux_scaled = PMN_cat_gflux * (atca_freq / PMN_freq) ** -0.7

    ra_cals = Angle(calibrators["ra_deg"], unit=u.degree)
    dec_cals = Angle(calibrators["dec_deg"], unit=u.degree)
    coords_cals = SkyCoord(ra=ra_cals, dec=dec_cals, frame='icrs')

    thresh_val_pb = 0.05
    thresh_val_out_pb = 0.5
    pb_radius = 0.4 / 2
    PMN_res = 5 / 60 / 2 # degree
    radius = 2 # degree

    remove_idx = []
    for i in range(len(ra_cals)):
        col = 'flux_4cm'
        ra = ra_cals[i].deg
        dec = dec_cals[i].deg
        flux = calibrators[col].iloc[i]
        if np.isnan(flux):
            col = 'flux_C'
            flux = calibrators[col].iloc[i]

        thresh = thresh_val_pb * flux

        mask = (PMN_cat_flux_scaled > thresh) | (PMN_cat_gflux_scaled > thresh if isinstance(PMN_cat_gflux_scaled, float) else False)
        if np.any(mask):
            remove = False
            idxs = np.where(mask)[0]
            # compute separations (degrees) for all masked indices
            input_coord = SkyCoord(ra=ra*u.deg, dec=dec*u.deg)
            seps_deg = np.array([
                input_coord.separation(coords_PMN[j]).to(u.degree).value
                for j in idxs
                ])
            order = np.argsort(seps_deg)
            sorted_idxs = idxs[order]
            sorted_seps_deg = seps_deg[order]

            for k, sep_deg in zip(sorted_idxs, sorted_seps_deg):
                PMN_flux = PMN_cat_gflux_scaled[k] if PMN_cat_gflux_scaled[k] else PMN_cat_flux_scaled[k]

                if np.isclose(sep_deg, 0.0, atol=1e-2):
                    continue
                else:
                    if PMN_res < sep_deg < radius:
                        print(f"Calibrator: {calibrators['name'].iloc[i]}, {col}: {calibrators[col].iloc[i]} Jy")
                        if PMN_res < sep_deg <= pb_radius:
                            print(f"Source {PMN['PMNJ'].iloc[k]} = {PMN_flux:.6g} Jy, separation = {sep_deg:.4f} deg. Drop calibrator.")
                            remove = True
                        else:
                            if PMN_flux/flux > thresh_val_out_pb:
                                print(f"Source {PMN['PMNJ'].iloc[k]} = {PMN_flux:.6g} Jy, separation = {sep_deg:.4f} deg. Drop calibrator.")
                                remove = True
                            else:
                                print(f"Source {PMN['PMNJ'].iloc[k]} = {PMN_flux:.6g} Jy, separation = {sep_deg:.4f} deg. No need to drop calibrator.")

        if remove:
            remove_idx.append(i)

    return calibrators.drop(calibrators.index[i] for i in remove_idx)

def main():
    
    Dec_lim, atca_selected_bands, atca_flux_limits, vla_selected_bands, vla_flux_limits, vla_pos_quality, vla_ampq, quality_mode, search_radius, pb_radius, self_radius = setupSidebar()

    ATCAdb= ParseATCA('ATCA Calibrators Database.csv')
    ATCA_after_cuts = ATCA_cuts(Dec_lim, atca_selected_bands, atca_flux_limits, ATCAdb)

    VLAdb = ParseVLA('VLA Calibrator List 2.csv')

    VLA_after_cuts = VLA_cuts(VLAdb, Dec_lim, vla_selected_bands, vla_flux_limits, vla_pos_quality, vla_ampq, quality_mode)
    
    joint_cal_list = joinTables(ATCA_after_cuts,VLA_after_cuts,atca_selected_bands,vla_selected_bands)
    

    if search_radius != 0:
        joint_cal_list = conesearch_internal(search_radius, pb_radius, self_radius, joint_cal_list, VLAdb, ATCAdb)
    
        joint_cal_list = conesearch_PMN(joint_cal_list)

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