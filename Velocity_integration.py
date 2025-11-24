""" integration velocity from accleration data  """
# data : 21/10/2025 
# author : lesley 


import pandas as pd
import numpy as np 
from scipy.integrate import simpson 

# df = pd.read_excel(r'C:\Users\jzh204\OneDrive - University of Canterbury\4th_year2025\493\1st_floor_acc.xlsx')
# df = pd.read_excel(r'C:\Users\jzh204\OneDrive - University of Canterbury\4th_year2025\493\2nd_floor_acc.xlsx')
df = pd.read_excel(r'E:\onedrive\OneDrive - University of Canterbury\4th_year2025\493\2nd_floor_acc.xlsx')

time = 'Elapsed (s)'
dt = 0.004
# all normalised acc data of Floor 1
# floor_cols = [
#     'F1_normalised_trial1 (m/s^2)',
#     'F1_normalised_trial2 (m/s^2)',
#     'F1_normalised_trial3 (m/s^2)',
#     'F1_normalised_trial4 (m/s^2)'
# ]

# all normalised acc data of Floor 2
floor_cols = [
    'F2_normalised_trial1 (m/s^2)',
    'F2_normalised_check3 (m/s^2)'
]

# df[floor_cols] = df[floor_cols].apply(pd.to_numeric, errors='coerce').fillna(0)  # clean the texts
df[floor_cols] = df[floor_cols].apply(pd.to_numeric, errors='coerce').fillna(0)  # clean the texts

t = df[time].to_numpy()


""" Simpson Rule """
def integrate_simpson(a, dt):
    v = np.zeros_like(a)
    for i in range(2, len(a), 2):
        v[i] = v[i-2] + (dt/3)*(a[i-2] + 4*a[i-1] + a[i])
        if i+1 < len(a):
            v[i+1] = v[i] + 0.5*dt*(a[i] + a[i+1])
    return v


for col in floor_cols:
    a = df[col].to_numpy()
    a = a - np.mean(a[:80])          # offset the tiem      
    v = integrate_simpson(a, dt)
    v -= np.linspace(0, v[-1], len(v))     # drift correction, keep the velocity value = 0 . 
    vmax = np.max(np.abs(v))
    print(f"{col}: Peak |v| = {vmax:.8f} m/s")

