import numpy as np
import pandas as pd
import glob
import os
import re
import h5py as h5
from importlib import resources


def _bundled_csv() -> str:
    """Return the path to the station CSV bundled with this package."""
    ref = resources.files(__package__).joinpath("station_scintpi_codes_fsr.csv")
    # as_posix works for both traversable and Path objects
    return str(ref)


def load_targets(csv_path=None):
    """Loads and sorts station targets strictly from CSV."""
    if csv_path is None:
        csv_path = _bundled_csv()
    plot_targets = []
    try:
        scintpi_codes = pd.read_csv(csv_path, usecols=[0, 1, 2, 3, 4], encoding='latin1', header=None, skiprows=1)
        scintpi_codes.columns = ['Station Location', 'Latitude', 'Longitude', 'Code', 'Type']
        
        scintpi_codes['Station Location'] = scintpi_codes['Station Location'].astype(str).str.strip()
        scintpi_codes['Code'] = scintpi_codes['Code'].astype(str).str.strip()
        scintpi_codes['Latitude'] = pd.to_numeric(scintpi_codes['Latitude'], errors='coerce')
        scintpi_codes['Longitude'] = pd.to_numeric(scintpi_codes['Longitude'], errors='coerce')
        
        scintpi_codes = scintpi_codes.dropna(subset=['Latitude', 'Longitude'])
        
        for idx, row in scintpi_codes.iterrows():
            lat, lon = float(row['Latitude']), float(row['Longitude'])
            if abs(lat) < 0.001 and abs(lon) < 0.001:
                continue
                
            code_val = row['Code'] if row['Code'].lower() != 'nan' and row['Code'] else ""
            v_type = str(row['Type']).strip().upper() if pd.notnull(row['Type']) and str(row['Type']).strip() else "UNK"
            
            plot_targets.append({
                'csv_index': idx,
                'name': row['Station Location'],
                'lat': lat,
                'lon': lon,
                'code': code_val,
                'version': v_type,
                'valid_times': set()
            })

        # Sort stations by descending latitude
        plot_targets = sorted(plot_targets, key=lambda x: x['lat'], reverse=True)
        print(f"Loaded exactly {len(plot_targets)} target rows from CSV.")
        return plot_targets
    except Exception as e:
        raise RuntimeError(f"CRITICAL ERROR: CSV not found or invalid ({e}).")

def scan_legacy_files(plot_targets, cutoff, base_dir='/mfs/io/groups/uars/scintpi'):
    """Scans ScintPi 2/3 files by distance and version tag."""
    print("Scanning Legacy ScintPi 2/3 files...")
    
    search_path = os.path.join(base_dir, "*", "*")
    data = glob.glob(f"{search_path}.bin.zip")
    data.extend(glob.glob(f"{search_path}.dat.zip"))

    pattern = re.compile(r'(\w+?)_(\d{8})_(\d{4})_([\d.]+[WE]_[\d.]+[NS]).*\.zip$')

    for full_path in data:
        s = full_path if isinstance(full_path, str) else full_path.decode('utf-8', 'ignore')
        fname = os.path.basename(s).replace('\\', '/')
        v_tag = "SC2" if "scintpi2" in full_path.lower() else "SC3"
        
        m = pattern.search(fname)
        if not m: continue
            
        time_val = pd.to_datetime(m.group(2) + m.group(3), format='%Y%m%d%H%M', errors='coerce')
        if pd.notnull(time_val) and time_val >= cutoff.normalize():
            coord_m = re.search(r'_([0-9.]+)([EW])_([0-9.]+)([NS])', fname)
            if coord_m:
                ln, lnh, lt, lth = float(coord_m.group(1)), coord_m.group(2), float(coord_m.group(3)), coord_m.group(4)
                if ln > 180.0: ln /= 1e4
                if lt > 90.0: lt /= 1e4
                lt = -lt if lth == 'S' else lt
                ln = -ln if lnh == 'W' else ln
                
                if abs(lt) < 0.001 and abs(ln) < 0.001: continue
                
                best_target, min_dist = None, 9999
                for t in plot_targets:
                    dist = np.sqrt((lt - t['lat'])**2 + (ln - t['lon'])**2)
                    if dist <= 0.02 and dist < min_dist and t['version'] == v_tag:
                        min_dist, best_target = dist, t
                        
                if best_target:
                    best_target['valid_times'].add(time_val.normalize())

def scan_sc4_files(plot_targets, cutoff, sc4_dict, base_dir='/mfs/io/groups/uars/scintpi'):
    """Scans ScintPi 4.0 files mapped via explicit dictionary."""
    print("Scanning ScintPi 4 files...")
    sc4paths = []

    for prefix in sc4_dict.keys():
        search_pattern = os.path.join(base_dir, "*", "*", f"{prefix}*_")
        sc4paths.extend(glob.glob(search_pattern))

    for p in sc4paths:
        p_str = p.replace('\\', '/')
        fname = os.path.basename(p_str)
        if len(fname) < 10: continue 
        
        stat_prefix = fname[:4].lower()
        date_str = p_str.split('/')[-2]
        time_val = pd.to_datetime(date_str, format='%Y%m%d', errors='coerce')
        
        if pd.notnull(time_val) and time_val >= cutoff.normalize():
            mapped_code = sc4_dict.get(stat_prefix)
            if mapped_code:
                for t in plot_targets:
                    if t['code'] == mapped_code:
                        t['valid_times'].add(time_val.normalize())
                        break

def checklvl3datamissing(lvl3file, thres=900):
    """Helper: checks percent missing from Level-3 HDF5 file."""
    try:
        with h5.File(lvl3file, 'r') as f:
            rows = [pd.DataFrame({'group': g, 'sat': sat, 'NOS1': np.array(f[g][sat]['NOS1'][0])})
                    for g in f.keys() for sat in f[g].keys()]
            df = pd.concat(rows, ignore_index=False).reset_index()
            maxnumsamples = df.groupby('index')['NOS1'].max()
            enoughpoints = maxnumsamples[maxnumsamples > thres]
            return 1 - (len(enoughpoints) / len(maxnumsamples))
    except Exception:
        return np.nan