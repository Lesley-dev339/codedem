import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

# Define the uploaded files
# files = {
#     "trial_1": Path(r"E:\onedrive\OneDrive - University of Canterbury\4th_year2025\493\Data_Analysis\trial_1"),
#     "trial_2": Path(r"E:\onedrive\OneDrive - University of Canterbury\4th_year2025\493\Data_Analysis\trial_2"),
#     "trial_3": Path(r"E:\onedrive\OneDrive - University of Canterbury\4th_year2025\493\Data_Analysis\trial_3"),
#     }

files = {
    # Floor 1 testing ( CWT at 1st floor)
    "trial_1": Path(r"C:\Users\jzh204\OneDrive - University of Canterbury\4th_year2025\493\Data_Analysis\trial_1"),
    "trial_2": Path(r"C:\Users\jzh204\OneDrive - University of Canterbury\4th_year2025\493\Data_Analysis\trial_2"),
    "trial_3": Path(r"C:\Users\jzh204\OneDrive - University of Canterbury\4th_year2025\493\Data_Analysis\trial_3"),
    "trial_4": Path(r"C:\Users\jzh204\OneDrive - University of Canterbury\4th_year2025\493\Data_Analysis\trial_4"),
    # Floor 2 tesing (CWT at 2nd floor)
    "trial_5": Path(r"C:\Users\jzh204\OneDrive - University of Canterbury\4th_year2025\493\Data_Analysis\trial_5"),
    "trial_6": Path(r"C:\Users\jzh204\OneDrive - University of Canterbury\4th_year2025\493\Data_Analysis\trial_6"),
    "trial_7": Path(r"C:\Users\jzh204\OneDrive - University of Canterbury\4th_year2025\493\Data_Analysis\trial_7"),
    }

# Define column names
time_col = "Elapsed (s)"
d1_col_orig = "Laser D1 (mm)"
d2_col_orig = "Laser D2 (mm)"
d3_col_orig = "Laser D3 (mm)"

# Define column names for normalized data
d1_col_norm = "Laser D1 Normalized (mm)"
d2_col_norm = "Laser D2 Normalized (mm)"
d3_col_norm = "Laser D3 Normalized (mm)"

# Define start times
start_times = {"trial_1": 3.43, "trial_2": 3.60, "trial_3": 3.90, "trial_4": 3.50, "trial_5": 3.00, "trial_6": 3.00, "trial_7": 3.00}

# Output directory
outdir = Path(r"C:\Users\jzh204\OneDrive - University of Canterbury\4th_year2025\493\Data_Analysis")
outdir.mkdir(exist_ok=True)

summary = []

for trial_name, file_path in files.items():

    # Read CSV
    df = pd.read_csv(file_path)

    # Convert to numeric and clean
    for col in [time_col, d1_col_orig, d2_col_orig, d3_col_orig]:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df = df.dropna(subset=[time_col, d1_col_orig, d2_col_orig, d3_col_orig])

    # Apply time filter
    start_t = start_times[trial_name]
    df_filtered = df[df[time_col] >= start_t].copy()

    # Normalise data   
    D1_offset = df_filtered[d1_col_orig].iloc[0]
    D2_offset = df_filtered[d2_col_orig].iloc[0]
    D3_offset = df_filtered[d3_col_orig].iloc[0]

    # Step 2: Apply the normalization to the filtered displacement data
    df_filtered[d1_col_norm] = df_filtered[d1_col_orig] - D1_offset
    df_filtered[d2_col_norm] = df_filtered[d2_col_orig] - D2_offset
    df_filtered[d3_col_norm] = df_filtered[d3_col_orig] - D3_offset


    # Compute drifts
    df_filtered["Drift_21 (mm)"] = df_filtered[d2_col_norm] - df_filtered[d1_col_norm]
    df_filtered["Drift_32 (mm)"] = df_filtered[d3_col_norm] - df_filtered[d2_col_norm]

    # Select final columns
    df_out = df_filtered[[time_col, "Drift_21 (mm)", "Drift_32 (mm)"]].copy()

    # Save filtered CSV
    out_csv = outdir / f"{trial_name}_inter-storey.csv"
    df_out.to_csv(out_csv, index=False)

    # Plot
    plt.figure()
    plt.plot(df_out[time_col], df_out["Drift_21 (mm)"], label="Drift between 1st floor and 2nd floor (mm)")
    plt.plot(df_out[time_col], df_out["Drift_32 (mm)"], label="Drift between 2nd floor and 3rd floor (mm)")
    plt.xlabel("Time (s)")
    plt.ylabel("Drift (mm)")
    plt.title(f"Inter-storey Drift vs Time - {trial_name}")
    plt.legend()
    out_fig = outdir / f"{trial_name}_drift_plot.png"
    # plt.savefig(out_fig, dpi=150, bbox_inches="tight")
    plt.show()

    # Add summary
    summary.append(
        {
            "Trial": trial_name,
            "Drift_21 min": df_out["Drift_21 (mm)"].min(),
            "Drift_21 max": df_out["Drift_21 (mm)"].max(),
            "Drift_21 mean": df_out["Drift_21 (mm)"].mean(),
            "Drift_32 min": df_out["Drift_32 (mm)"].min(),
            "Drift_32 max": df_out["Drift_32 (mm)"].max(),
            "Drift_32 mean": df_out["Drift_32 (mm)"].mean(),
        }
    )

# Create summary table
summary_df = pd.DataFrame(summary)
summary_csv = outdir / "1and2_floors_inter-storey_summary_table.csv"
summary_df.to_csv(summary_csv, index=False)

summary_df.head()
