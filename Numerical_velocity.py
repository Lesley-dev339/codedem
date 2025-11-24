# author : Lesley
# date   : October, 2025
# scope : Calculate the peak absolute velocity of each floor 

import numpy as np 

# Read the data
Time  = np.loadtxt('node444442_v.txt', usecols= 0,comments='#')    # unit: mm/s
V1 = np.loadtxt('node444442_v.txt', usecols= 1,skiprows=1)    
V2 = np.loadtxt('node444443_v.txt', usecols= 1,skiprows=1)    
V3 = np.loadtxt('node444444_v.txt', usecols= 1,skiprows=1)    


""" Return peak relative velocity of each floor in m/s """

def peak_velocity():
    peak_V1 = np.max(np.abs(V1)/1000)
    peak_V2 = np.max(np.abs(V2)/1000)
    peak_V3 = np.max(np.abs(V3)/1000)
    
    return peak_V1, peak_V2, peak_V3

def main():
    peak_V1, peak_V2, peak_V3 = peak_velocity()
    print(f"Maximum velocity on the 3rd floor (TOP): {peak_V3:.2f} m/s")
    print(f"Maximum velocity on the 2nd floor: {peak_V2:.2f} m/s")
    print(f"Maximum velocity on the 1st floor: {peak_V1:.2f} m/s")


if __name__ == "__main__":
    main()