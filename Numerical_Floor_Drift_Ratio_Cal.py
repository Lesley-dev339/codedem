# author : Lesley 
# date: 17/09/2025
# ENCI 493 Research : Elevator in the friction building 

""" This script calculated the intermediate drift ratio of the main frame, and the displacement of the elevator shafts and floor slabs. """
import numpy as np

# import data
""" Counterweight at the bottom/middle floor """ 
Time  = np.loadtxt('node444442.txt', usecols= 0,comments='#')    # read the time history
Disp1 = np.loadtxt('node444442.txt', usecols= 1,skiprows=1)    # read the horizontal displacement of node 444442 (bottom centre mass point)
Disp2 = np.loadtxt('node444443.txt', usecols= 1,skiprows=1)    # read the horizontal displacement of node 444443 (middle centre mass point)
Disp3 = np.loadtxt('node444444.txt', usecols= 1,skiprows=1)    # read the horizontal displacement of node 444444 (top centre mass point)

# define values 
H = 400   # inter-storey height in mm

""" Return time histories of inter-storey drift ratios in % """
def drift_ratio(): 
    
    drift_21 = np.abs(Disp2 - Disp1) / H * 100  # inter-storey drift at the first storey 
    drift_32 = np.abs(Disp3 - Disp2) / H * 100  # inter-storey drift at the second storey 
    
    return drift_21, drift_32

""" Display maximum drift ratios """ 

def main():
    drift_21, drift_32 = drift_ratio()

    max_drift_21 = np.max(drift_21)
    max_drift_32 = np.max(drift_32)

    print(f"Maximum Drift Ratio (Storey 2): {max_drift_21:.2f} %")
    print(f"Maximum Drift Ratio (Storey 3): {max_drift_32:.2f} %")



if __name__ == "__main__": 
    main()
    
    
    