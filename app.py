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

    Flux_lim = st.sidebar.number_input(
    "Minimum Flux (Jy)",
    min_value=0.0,
    max_value = 100.0,
    value=5.0,
    step=1.0
    )

    st.sidebar.subheader("Flux band selection")

    use_16cm = st.sidebar.checkbox("16 cm", value=False)
    use_4cm  = st.sidebar.checkbox("4 cm", value=True)
    use_15mm = st.sidebar.checkbox("15 mm", value=True)
    use_7mm  = st.sidebar.checkbox("7 mm", value=False)
    use_3mm  = st.sidebar.checkbox("3 mm", value=False)
    selected_bands = []
    if use_16cm:
        selected_bands.append("16cm")
    if use_4cm:
        selected_bands.append("4cm")
    if use_15mm:
        selected_bands.append("15mm")
    if use_7mm:
        selected_bands.append("7mm")
    if use_3mm:
        selected_bands.append("3mm")

    return Flux_lim, Dec_lim, selected_bands

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

def ATCA_cuts(Dec_lim,Flux_lim,selected_bands,ATCAdb):
    # --- Dec cut ---
    ATCA_CutDec = ATCAdb[
        Angle(ATCAdb["Dec."], unit=u.deg).deg < Dec_lim
    ]

    # --- Flux cut ---
    if len(selected_bands) == 0:
        # No bands selected → don't apply flux cut
        ATCA_CutFlux = ATCA_CutDec
    else:
        flux_mask = np.zeros(len(ATCA_CutDec), dtype=bool)

        for band in selected_bands:
            if band in ATCA_CutDec.columns:
                flux_mask |= (ATCA_CutDec[band] > Flux_lim)

        ATCA_CutFlux = ATCA_CutDec[flux_mask]

    st.success(f"{len(ATCA_CutFlux)} sources after Flux & Dec cuts")
    return ATCA_CutFlux

def VLA_cuts(VLAdb,Flux_lim, Dec_lim, quality_mode):
        # -------------------------
    # Positional certainty filter
    # -------------------------
    VLAdb['header_posq'] = VLAdb['header_posq'].astype(str).str.strip().str.upper()
    print("Unique canonical source names:",VLAdb['name'].nunique())
    VLAdb = VLAdb[VLAdb['header_posq'] == 'A']

    print("Unique canonical source names with positional certainty A:",
        VLAdb['name'].nunique())

    # -------------------------
    # Convert RA/Dec to decimals (use canonical header RA/Dec)
    # -------------------------
    VLAdb = VLAdb.copy()
    VLAdb['ra_deg'] = VLAdb['ra'].apply(ra_to_deg)
    VLAdb['dec_deg'] = VLAdb['dec'].apply(dec_to_deg)

    # -------------------------
    # Build flux pivot (one row per canonical source)
    # -------------------------
    # Normalize receiver and quality character columns
    VLAdb['receiver'] = VLAdb['receiver'].str.strip().str.upper()
    for col in ['A','B','C','D']:
        VLAdb[col] = VLAdb[col].astype(str).str.strip().str.upper().replace({'NAN': np.nan})

    # pivot flux table
    flux_pivot = VLAdb.pivot_table(index='name', columns='receiver', values='flux', aggfunc='max')

    # Build a per-source-per-band "P-quality" boolean using A-D columns
    # Two possible modes: 'any' => True if any of A/B/C/D == 'P' or 'S';
    # or a specific column: 'A','B','C' or 'D'
    def band_has_PS(row):
        good = ['P', 'S']   # acceptable amplitude closure quality

        if quality_mode.lower() == 'any':
            return any((row.get(col) in good) for col in ['A','B','C','D'])
        else:
            col = quality_mode.upper()
            return row.get(col) in good

    VLAdb['quality_PS'] = VLAdb.apply(band_has_PS, axis=1)

    # Now pivot quality into per-source receiver columns too (True/False)
    quality_pivot = VLAdb.pivot_table(index='name', columns='receiver', values='quality_PS', aggfunc='max')  # max on booleans = any True

    # -------------------------
    # Diagnostics: show how many sources have P in C & X
    # -------------------------
    # print("Receivers found (columns in flux_pivot):", list(flux_pivot.columns))
    # print("Number of sources with any flux entry (unique names):", VLAdb['name'].nunique())
    # Count sources with P in C and X (existence)
    has_C_P = 'C' in quality_pivot.columns and quality_pivot['C'].sum()
    has_X_P = 'X' in quality_pivot.columns and quality_pivot['X'].sum()
    # print("Sources with P in C (count):", int(has_C_P or 0))
    # print("Sources with P in X (count):", int(has_X_P or 0))

    # -------------------------
    # Example selection: require flux > Flux_lim in BOTH C and X AND P-quality in both (per chosen quality_mode)
    # -------------------------
    # treat missing values as 0 / False
    c_flux = flux_pivot.get('C', pd.Series(dtype=float)).fillna(0.0)
    x_flux = flux_pivot.get('X', pd.Series(dtype=float)).fillna(0.0)
    u_flux = flux_pivot.get('U', pd.Series(dtype=float)).fillna(0.0)

    c_quality = quality_pivot.get('C', pd.Series(dtype=bool)).fillna(False)
    x_quality = quality_pivot.get('X', pd.Series(dtype=bool)).fillna(False)
    u_quality = quality_pivot.get('U', pd.Series(dtype=bool)).fillna(False)

    valid_names = c_flux.index[
        ((c_flux > Flux_lim) |
        (x_flux > Flux_lim) |
        (u_flux > Flux_lim)) &
        ((c_quality) |
        (x_quality) |
        (u_quality))
    ]
    # print("Valid names satisfying Flux >1 Jy and P/S-quality in C, X or U:", len(valid_names))

    # apply declination cut on canonical header dec (we have dec_deg per band row, so pick unique header per name)
    # get unique canonical header (since we used .drop_duplicates earlier approach elsewhere, here we compute per-name median)
    # Build per-source coordinate table
    per_name = VLAdb.groupby('name').agg({
        'ra_deg': 'median',
        'dec_deg': 'median'
    })

    # Add flux columns for C, X, U bands
    per_name['flux_C'] = flux_pivot.get('C')
    per_name['flux_X'] = flux_pivot.get('X')
    per_name['flux_U'] = flux_pivot.get('U')

    # final selection dataframe (unique names)
    sel = per_name.loc[per_name.index.intersection(valid_names)].copy()

    # apply declination cut
    sel = sel[sel['dec_deg'] < Dec_lim]

    return sel

def plotSkyCoords(ra_rad,dec_rad):
    ## Plot ATCA catalog after cuts
    try:
        fig, ax = plt.subplots(figsize=(8, 4.2), subplot_kw=dict(projection="mollweide"))
        ax.grid(True)
        ax.scatter(ra_rad, dec_rad, marker="o", s=2, alpha=0.3)
        fig.subplots_adjust(top=0.95, bottom=0.0)
        ax.set_title('ATCA Calibrators (after cut)')
        st.pyplot(fig)
    except Exception as e:
        st.error(f"Error plotting ATCA after cuts: {e}")

def joinTables(atca_table,vla_table):
    atca = pd.DataFrame.copy(atca_table)
    vla = pd.DataFrame.copy(vla_table)

    # -------------------------
    # Clean VLA table
    # -------------------------
    vla = vla.reset_index()          # move name from index to column
    vla['name'] = vla['name'].str.replace('"','').str.strip()

    # keep flux columns
    vla = vla[['name','ra_deg','dec_deg','flux_C','flux_X','flux_U']]


    # -------------------------
    # Clean ATCA table
    # -------------------------
    # rename columns
    atca = atca.rename(columns={
        'Name':'name',
        'R.A.':'ra',
        'Dec.':'dec'
    })

    # convert RA/Dec to degrees
    coords = SkyCoord(atca['ra'], atca['dec'], unit=(u.hourangle, u.deg), frame='icrs')

    atca['ra_deg'] = coords.ra.deg
    atca['dec_deg'] = coords.dec.deg

    atca['name'] = atca['name'].str.strip()

    # rename ATCA flux columns to consistent names
    atca = atca.rename(columns={
        '4cm':'flux_4cm',
        '15mm':'flux_15mm'
    })

    # keep flux columns
    atca = atca[['name','ra_deg','dec_deg','flux_4cm','flux_15mm']]


    # -------------------------
    # Align column sets
    # -------------------------
    # ensure both tables contain the same columns
    for col in ['flux_C','flux_X','flux_U']:
        if col not in atca:
            atca[col] = np.nan

    for col in ['flux_4cm','flux_15mm']:
        if col not in vla:
            vla[col] = np.nan

    # reorder columns consistently
    cols = ['name','ra_deg','dec_deg','flux_4cm','flux_15mm','flux_C','flux_X','flux_U']

    atca = atca[cols]
    vla = vla[cols]


    # -------------------------
    # Merge catalogs
    # -------------------------
    calibrators = pd.concat([atca, vla], ignore_index=True)

    # remove duplicates by source name
    calibrators = calibrators.drop_duplicates(subset='name')

    return calibrators

def main():
    
    Flux_lim, Dec_lim, selected_bands = setupSidebar()

    ATCAdb= ParseATCA('ATCA Calibrators Database.csv')
    ATCA_after_cuts = ATCA_cuts(Dec_lim,Flux_lim, selected_bands, ATCAdb)

    ra_vals = Angle(ATCA_after_cuts["R.A."], unit=u.hourangle)
    dec_vals = Angle(ATCA_after_cuts["Dec."], unit=u.degree)

    c = SkyCoord(ra=ra_vals, dec=dec_vals, frame='icrs')
    ra_rad = c.ra.wrap_at(180 * u.deg).radian
    dec_rad = c.dec.radian

    # plotSkyCoords(ra_rad,dec_rad)

    VLAdb = ParseVLA('VLA Calibrator List 2.csv')

    VLA_after_cuts = VLA_cuts(VLAdb,Flux_lim, Dec_lim, quality_mode='any')
    
    c_vla = SkyCoord(ra=VLA_after_cuts['ra_deg'].values * u.deg,
                   dec=VLA_after_cuts['dec_deg'].values * u.deg,
                   frame='icrs')
    ra_vla_rad = c_vla.ra.wrap_at(180 * u.deg).radian
    dec_vla_rad = c_vla.dec.radian
    # plotSkyCoords(ra_vla_rad,dec_vla_rad)
    st.success(f"{len(VLA_after_cuts)} VLA sources after Flux & Dec cuts")

    joint_cal_list = joinTables(ATCA_after_cuts,VLA_after_cuts)
    ra_joint_rad = np.radians(joint_cal_list['ra_deg'].values)
    dec_joint_rad = np.radians(joint_cal_list['dec_deg'].values)
    ra_joint_rad = np.remainder(ra_joint_rad + np.pi, 2*np.pi) - np.pi

    plotSkyCoords(ra_joint_rad,dec_joint_rad)
    st.success(f"{len(joint_cal_list)} joint sources after Flux & Dec cuts")

main()