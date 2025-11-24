# author : Lesley 
# date: 02/10/2025
# ENCI 493 Research : Elevator in the friction building 

""" Represent Elevator horizontal sliding from lasers' record """

import numpy as np 
import  matplotlib.pyplot as plt 
import pandas as pd

# import data

data = pd.read_csv(
    r"C:\Users\jzh204\OneDrive - University of Canterbury\4th_year2025\493\Data_Analysis\trial_1",
    delimiter="," , usecols=(0,1,2)
)

time = data.iloc[:, 0]      # Time column (A)
col1 = data.iloc[:, 1]      # 2nd Floor laser (B)
col2 = data.iloc[:, 2]      # 3rd Floor laser (F)

def main(): 
    
    # Correction data 
    col1_corr = col1 - col1[0]
    col2_corr = col2 - col2[0]

    # Apply boundary conditions (manually choose)
    output1 = np.where((time >= 3.5) & (time < 23), col1_corr, 0)
    output2 = np.where((time >= 3.5) & (time < 23), col2_corr, 0)

    return time, output1, output2


def plotting(): 
    plt.figure(figsize=(10, 6))    
    plt.plot(time, output1, label="laser 1 ", color="red", linewidth=2)
    plt.plot(time, output2, label="laser 2", color="blue", linewidth=2)
    plt.xlabel("Time (s)")
    plt.ylabel("Displacement (corrected)")
    plt.title("Conditioned Data vs Time (2nd & 3rd Floors)")
    plt.legend()
    plt.grid(True)
    plt.show()


if __name__ == "__main__":
    time, output1, output2 = main()
    plotting()