# author : Lesley
# date: 09/10/2025
# ENCI 493 Research : Elevator in the friction building


import pandas as pd
import matplotlib.pyplot as plt
import glob
import os

# Read the data
# df = pd.read_csv(r'C:\Users\jzh204\OneDrive - University of Canterbury\4th_year2025\493\Data_Analysis')
# dict_dir = r'C:\Users\jzh204\OneDrive - University of Canterbury\4th_year2025\493\Data_Analysis'
dict_dir = r'E:\onedrive\OneDrive - University of Canterbury\4th_year2025\493\Data_Analysis' # with my own laptop
# define the file pattern to search for
file_pattern = os.path.join(dict_dir, 'trial_*')
# Use glob to find all files matching the pattern
file_list = glob.glob(file_pattern)


time = 'Elapsed (s)'
laser1_orig = 'Laser 1 (mm)'
laser2_orig = 'Laser 2 (mm)'
laser1_new = 'New Laser 1 (mm)'
laser2_new = 'New Laser 2 (mm)'

# Define the time range for filtering
time_start = 3
time_end = 20.0


if not file_list:
    print(f"ERROR: No files found matching the pattern: {file_pattern}")
    print("Please check the directory path and the file extension.")
else:
    print(f"Found {len(file_list)} files to process: {', '.join([os.path.basename(f) for f in file_list])}")


for file_path in file_list:
    # Extract a clean name for the trial
    file_name = os.path.basename(file_path)
    print(f"\n--- Starting Plot for: {file_name} ---")

    try:
        # Read the data using pandas
        df = pd.read_csv(file_path)
    except Exception as e:
        print(f"Error reading {file_name}. Skipping file. Error: {e}")
        continue



    """ Step 1. Normalize laser data """

    L1_offset = df[laser1_orig].iloc[0]
    L2_offset = df[laser2_orig].iloc[0]

    # Apply the normalization
    df[laser1_new] = df[laser1_orig] - L1_offset
    df[laser2_new] = df[laser2_orig] - L2_offset

    """ Step 2. Apply time filter """

    # Create a boolean mask: True for the time range, false for out of the range
    # Ture is invalid, false is valid
    mask_keep = (df[time] >= time_start) & (df[time] <= time_end)

    # Cut the time period
    time_cut =df.loc[mask_keep, time]
    
    # for excel data
    # laser1_cut = df.loc[mask_keep, laser1_new]
    # laser2_cut = df.loc[mask_keep, laser2_new]

    # Apply the filter
    df.loc[~mask_keep, laser1_new] = 0.0
    df.loc[~mask_keep, laser2_new] = 0.0

    """Export results"""   
    # df_filtered = pd.DataFrame({
    # time: time_cut,
    # laser1_new: laser1_cut,
    # laser2_new: laser2_cut
    # })

    # output_filename = file_name + '_filtered.csv'
    # output_path = os.path.join(os.path.dirname(file_path), output_filename)
    # df_filtered.to_csv(output_path, index=False)
    # print(f"--> Successfully exported to: {output_filename}")
    

    """Plotting"""

    plt.figure(figsize=(10, 6))
    plt.plot(time_cut, df.loc[mask_keep,laser1_new],label='Laser 1 Displacement', linewidth=2)
    plt.plot(time_cut, df.loc[mask_keep,laser2_new], label='Laser 2 Displacement', linewidth=2)
    plt.title(f'Time vs. Displacement for {file_name}')
    plt.xlabel(time)
    plt.ylabel('Displacement (mm)')
    plt.legend()
    plt.grid(True)
    #filter boundaries
    plt.axvline(x=4.0, color='r', linestyle='--', linewidth=2, label='Filter Start (4s)')
    plt.axvline(x=20.0, color='r', linestyle='--', linewidth=2, label='Filter End (20s)')
    plt.show()