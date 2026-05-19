#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import pandas as pd
from emailer import (
    load_targets, 
    scan_legacy_files, 
    scan_sc4_files, 
    generate_availability_plot, 
    send_status_email
)

def main():
    # 1. Initialization
    now = pd.Timestamp.now()
    cutoff = now - pd.DateOffset(months=3)
    out_png = f"stations_availability_lvl3_{now:%Y_%m_%d}.png"
    
    sc4_stations_dict = {
        "mx01": "ME-MO1", "mx02": "ME-MO2",
        "cg01": "BR-CG1", "cg02": "BR-CG2",
        "cr01": "CR-CA1", "ho01": "TE-HO1",
    }
    email_recipients = ["tarunlsankar@gmail.com", "TLS220003@utdallas.edu", "IGW180000@utdallas.edu"]

    # 2. Pipeline Execution
    print("Starting ScintPi Pipeline...")
    try:
        targets = load_targets('station_scintpi_codes_fsr.csv')
        scan_legacy_files(targets, cutoff)
        scan_sc4_files(targets, cutoff, sc4_stations_dict)
        
        generate_availability_plot(targets, cutoff, now, out_png)
        send_status_email(image_path=out_png, now_date=now, to_list=email_recipients)
        
        print("Pipeline finished successfully.")
    except Exception as e:
        print(f"Pipeline failed: {e}")

if __name__ == "__main__":
    main()