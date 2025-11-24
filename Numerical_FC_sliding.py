#author : Lesley 
# date: 21/10/2025
# ENCI 493 Research : Elevator in the friction building 

""" Represent Numericial firction connection slidings """

import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import savgol_filter

# import data 

""" CWT at the bottom/Middle/Top floor """

Time  = np.loadtxt(r'C:\Users\jzh204\OneDrive - University of Canterbury\4th_year2025\493\Data_Analysis\FC\node17.txt', usecols= 0)

# 3rd floor  TOP   
node17 = np.loadtxt(r'C:\Users\jzh204\OneDrive - University of Canterbury\4th_year2025\493\Data_Analysis\FC\node17.txt', usecols= 1)    
node8921 = np.loadtxt(r'C:\Users\jzh204\OneDrive - University of Canterbury\4th_year2025\493\Data_Analysis\FC\node8921.txt', usecols=1)

# 2nd floor   
node18 = np.loadtxt(r'C:\Users\jzh204\OneDrive - University of Canterbury\4th_year2025\493\Data_Analysis\FC\node18.txt', usecols= 1)
node8922 = np.loadtxt(r'C:\Users\jzh204\OneDrive - University of Canterbury\4th_year2025\493\Data_Analysis\FC\node8922.txt', usecols= 1)

# 1st floor 
node19 = np.loadtxt(r'C:\Users\jzh204\OneDrive - University of Canterbury\4th_year2025\493\Data_Analysis\FC\node19.txt', usecols= 1)
node8923 = np.loadtxt(r'C:\Users\jzh204\OneDrive - University of Canterbury\4th_year2025\493\Data_Analysis\FC\node8923.txt', usecols= 1) 

# bottom (assume same as 1st floor)
# node20 = np.loadtxt(r'C:\Users\jzh204\OneDrive - University of Canterbury\4th_year2025\493\Data_Analysis\FC\node20.txt', usecols= 1)
# node8924 = np.loadtxt(r'C:\Users\jzh204\OneDrive - University of Canterbury\4th_year2025\493\Data_Analysis\FC\node8924.txt', usecols= 1) 

"""for cwt at middle case """
all_nodes = [Time, node17, node8921, node18, node8922, node19, node8923]

# --- Function to Synchronize Array Lengths (More Concise) ---
def synchronize_lengths(node_list):
    """
    Finds the minimum length among all arrays in the list and truncates 
    all arrays to that minimum length. Returns the synchronized list.
    """
    min_len = min(len(node) for node in node_list)
    print(f"Minimum length found for all arrays: {min_len} data points.")
    
    # Use list comprehension for efficient slicing
    return [node[:min_len] for node in node_list]

# Synchronize all loaded node arrays 
Time, node17, node8921, node18, node8922, node19, node8923 = synchronize_lengths(all_nodes)

def FC_sliding():

    raw_sliding3 = node8921 - node17
    sliding3 = raw_sliding3 - raw_sliding3[0]


    raw_sliding2 = node8922 - node18
    sliding2 = raw_sliding2 - raw_sliding2[0]


    raw_sliding1 = node8923 - node19
    sliding1 = raw_sliding1 - raw_sliding1[0]

    return sliding3, sliding2, sliding1


def plotting():

    mask = Time <= 10
    Time_cut =Time[mask]
    sliding1, sliding2, sliding3= FC_sliding()
    sliding3_cut = sliding3[mask]
    sliding2_cut = sliding2[mask]
    sliding1_cut = sliding1[mask]   # buttom 
    

    plt.figure(figsize=(10, 6))    
    # plt.plot(Time_cut, sliding3_cut, label=" FC sliping at top floor ", color="red", linewidth=2)
    # plt.plot(Time_cut, sliding2_cut, label=" FC sliping at middle floor ", color="blue", linewidth=2)
    plt.plot(Time_cut, sliding1_cut, color="red", linewidth=2)
    plt.xlabel("Time (s)")
    plt.ylabel("Displacement (mm)")
    plt.title("CWT at the 1st Floor")
    plt.show()


""" Find MAX sliding on each floor """
def main():

    sliding3, sliding2, sliding1 = FC_sliding()

    # Max Absolute value
    max_abs_sliding3 = np.max(np.abs(sliding3))
    max_abs_sliding2 = np.max(np.abs(sliding2))
    max_abs_sliding1 = np.max(np.abs(sliding1))
    # max_sliding4 = np.max(sliding4)

    print(f"Maximum Friction Connection Sliding on the 3rd floor (TOP): {max_abs_sliding3:.2f} mm")
    print(f"Maximum Friction Connection Sliding on the 2nd floor: {max_abs_sliding2:.2f} mm")
    print(f"Maximum Friction Connection Sliding on the 1st floor: {max_abs_sliding1:.2f} mm")
    # print(f"Maximum Friction Connection Sliding on the bottom floor: {max_sliding4:.2f} mm")


if __name__ == "__main__": 
    
    plotting()
    main()

