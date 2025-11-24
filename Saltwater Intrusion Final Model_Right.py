"""
ENCN404 Modern Modelling Practices for Civil Engineering
Seawater Intrusion Model
Group Members: Leon Thomas (81644603), Tim Wilson (782781823), and Lesley Zhuo (78811607).
Last Updated: 11:30am 07/05/2025.
"""

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
from scipy.optimize import curve_fit
from functools import partial
from scipy.interpolate import interp1d

#Importing Existing Data
tq,q = np.genfromtxt('si_mass.txt', delimiter = ',', skip_header = 1).T
q = -q #kg/year
tp,p = np.genfromtxt('si_pres.txt', delimiter = ',', skip_header = 1).T
tsal,s = np.genfromtxt('si_salt.txt', delimiter = ',', skip_header = 1).T

v = 5725800063333.452 # Volume of the aquifer in [m^3] gloabally defined
c_ocean = 35    # Concentration of salt in the ocean [kg/m^3]

# This is the main function for executing all tasks.
def main():

    # This segment of code defines common model and numerical parameters to be used in the calculations
    q0 = -5E14      # Constant pumping rate in [kg/yr]
    a = 2.160e-15           # Lumped paramerter
    b_alps = 1.726e-01      # Lumped parameter
    b_ocean = 9.269e-01     # Lumped parameter
    p_alps = 5.346    # Alps pressure  [MPa]
    p_ocean = p_alps - 3  #MPa as total pressure drop is about 3 MPa # Ocean pressure [MPa] 
    p0 =  2.346   # Initial Pressure in the aquifer [MPa]
    x0 = 0          # Initial concentration of salt in the aquifer [kg]
    rho = 997                         # Density of water [kg/m^3]

    dt = 0.01       # Timestep [yr], a numerical parameter
    n = 600         # Number of timesteps, a numerical parameter

    # This function numerically solves the pressure ODE using backward Euler method
    ts, ps = p_backward_euler(a, b_alps, b_ocean, p_alps, p_ocean, q0, dt, n, p0, show_plot=True)

    # Benchmarking for aquifer pressure
    pa = p_analytical(ts, a, b_alps, b_ocean, p_alps, p_ocean, q0, p0)

    # This function numerically solves the salinity ODE using backward Euler method
    global v
    ts, xs = x_backward_euler(a, ps, b_ocean, p_ocean, dt, n, q0, v, x0, c_ocean, rho, show_plot=True)

    # This function performs a unit test for the Euler numerical solution for one step 
    unit_tests()

    # # Benchmarking for aquifer salt concentration
    xa = x_analytical(ts, xs, a, b_alps, b_ocean, p_alps, p_ocean, c_ocean, v, q0, x0, p0, rho)
    
    # #This segment of code generates plots for both pressure and salinity
    benchmark_plots(ts, xa, v, xs, pa, ps)

    a, b_alps, b_ocean, p0, p_alps, p_ocean, v = main_cali()

    get_familiar_with_model(q0, a, b_alps, b_ocean, p_alps, p_ocean, p0, c_ocean, x0)

    a_list,b_alps_list, b_ocean_list,posterior = grid_search(a, b_alps, b_ocean, p0, p_alps, p_ocean)
    N_samples = 20
    samples = construct_samples(a_list, b_alps_list, b_ocean_list, p0, p_alps, p_ocean, posterior, N_samples)

    t = np.linspace(2010, 2020, num=10)
    model_ensemble(samples, p0, p_alps, p_ocean)
    model_ensemble_salt(samples, a, b_alps, b_ocean, p0, p_alps, v, p_ocean, t)
    model_forecast(samples, a, b_alps, b_ocean, p_alps, p_ocean, p0, v)

"""------------------------------------------------------------------------------------------------------------------------------------"""
"""BENCHMARKING"""

# This function will solve the given ODE using backward Euler method, assuming a constant pumping rate, to find the pressure in the aquifer.
def p_backward_euler(a, b_alps, b_ocean, p_alps, p_ocean, q0, dt, n, p0, show_plot=True):
    """
    Numerically solves the pressure ODE using the Backward Euler method.
    General solution for the backward Euler method is as follows:
    P_i+1 = P_i + dt * dP_(i+1)/dt
    """

    # This segment of code initializes lists to store time steps (ts) and pressure values (ps).
    ts = [0]
    ps = [p0] # Pressure values [Pa] or [N/m^2]

    # This segment of code iterates through all time steps to solve for pressure at each time step.
    for i in range(n):
        t_current = ts[-1]
        p_current = ps[-1]

        denominator = 1 + dt * (b_alps + b_ocean)
        numerator = p_current + dt * (a * q0 + b_alps * p_alps + b_ocean * p_ocean)

        p_next = numerator / denominator
        t_next = t_current + dt

        ps.append(p_next)       # Appends the computed pressure value to the list.
        ts.append(t_next)       # Appends the next time step to the list.

    ts = [round(t, 2) for t in ts]

    return ts, ps

# This function computes the analytical solution for pressure in the aquifer.
def p_analytical(ts, a, b_alps, b_ocean, p_alps, p_ocean, q0, p0):
    """
    The analytical solution is given by:

        P(t) = c/(b_alps + b_ocean) + (p0 - c/(b_alps + b_ocean)) * exp(-(b_alps + b_ocean) * t)
       
    Where:

        c = a * q0 + b_alps * p_alps + b_ocean * p_ocean
    """

    pa = []
    c_value = a * q0 + b_alps * p_alps + b_ocean * p_ocean   # A lumped constant in the analytical solution

    # This segment of code loops over the time steps and calculates the analytical pressure at each time step.
    for t in ts:
        pa_value = c_value/(b_alps + b_ocean) + (p0 - c_value/(b_alps + b_ocean)) * np.exp(-(b_alps + b_ocean) * t)
        pa.append(pa_value)

    return pa

# This function will solve the given ODE using backward Euler method to find the mass of salt in the aquifer, thus salinity.
def x_backward_euler(a, ps, b_ocean, p_ocean, dt, n, q0, v, x0, c_ocean, rho, show_plot=True):
    """
    Numerically solves the salt ODE using the backward Euler method.
    General solution for the backward Euler method is as follows:
    X_i+1 = X_i + dt * dX_(i+1)/dt
    """

    # This segment of code initializes lists to store time steps (ts) and salt mass values (xs).
    ts = [0]
    xs = [x0]   #[kg/m^3]

    # This segment of code iterates through all time steps to solve for salt mass at each time step.
    for i in range(n):
        t_current = ts[-1]
        x_current = xs[-1]
        p_plus_one = ps[i + 1]

        denominator = 1 - dt * ((q0) / (v * rho))
        numerator = x_current - dt * (b_ocean / (a * rho)) * (p_plus_one - p_ocean) * c_ocean

        x_next = numerator / denominator
        t_next = t_current + dt

        xs.append(x_next)       # Appends the computed salt mass value to the list above.
        ts.append(t_next)       # Appends the next time step to the list above.

    ts = [round(t, 2) for t in ts]


    return ts, xs


# This function performs a unit test that checks the correct implementation of another function and verifys it.
#def unit_test_x_backward_euler(a, ps, b_ocean, p_ocean, dt, n, q0, v, x0, c_ocean, show_plot=False):
    ps = [100, 105]     

    numerator = 100 - 0.01 * (2 / 3) * (105 - 100) * 35
    x1_expected = numerator / (1 - 0.01 * ((3 * -5e14) / 5.19e11))

    xs = x_backward_euler(a, ps, b_ocean, p_ocean, dt, 1, q0, v, x0, c_ocean, show_plot=False)

    error = abs(xs[1] - x1_expected)
    assert error < 1e-8, f"UNIT TEST FAILED: expected {x1_expected}, but got {xs[1]} (error is {error:.2e}, where tolerance is 1e-8)"

    print(f"UNIT TEST PASSED: one step of x_backward_euler {xs[1]} ≈ x_expected {x1_expected} (error is {error:.2e}, where tolerance is 1e-8)")


#def unit_test_x_backward_euler(a, ps, b_ocean, p_ocean, dt, n, q0, v, x0, c_ocean, show_plot=False):
    """Unit test for one-step backward Euler method for salinity."""
    # Using the same fixed parameters as in the actual model
    ps = [p_ocean, p_ocean + 5e3]  

    # Manual calculation of expected x1 
    numerator = x0 - dt * (b_ocean / a) * (ps[1] - p_ocean) * c_ocean
    denominator = 1 - dt * ((a * q0) / v)
    x1_expected = numerator / denominator

    # Run the x_backward_euler function for one step (n=1)
    ts, xs_list = x_backward_euler(a, ps, b_ocean, p_ocean, dt, 1, q0, v, x0, c_ocean, show_plot=False)
    x1_actual = xs_list[1]

    # Calculate the error and assert
    error = abs(x1_actual - x1_expected)
    assert error < 1e-8, f"UNIT TEST FAILED: expected {x1_expected}, but got {x1_actual} (error is {error:.2e}, where tolerance is 1e-8)"

    print(f"UNIT TEST PASSED: one step of x_backward_euler {x1_actual} ≈ x_expected {x1_expected} (error is {error:.2e}, where tolerance is 1e-8)")

# This function finds the exact mass of salt in the aquifer, thus finds the analytical values for salinity.
def x_analytical(ts, xs, a, b_alps, b_ocean, p_alps, p_ocean, c_ocean, v, q0, x0, p0, rho):
    """
    The analytical solution for salt concentration should consider the initial amount of salt (x0) and devide it by the aquifers volume.
    It assumes that C_t = C_ocean such that water is flowing in from the ocean.
    """
    
    # This segment of code initializes the list `xa` to store the analytical salt concentration values
    xa = []  

    # This segment of code defines parameters and lumped paramters
    const = a * q0 + b_alps * p_alps + b_ocean * p_ocean 
    k = const / (b_alps + b_ocean)
    d = p0 - ((const)/(b_alps + b_ocean))
    b = b_alps + b_ocean
    omega = (b_ocean * c_ocean * d) / (a * rho)
    gamma = (b_ocean * p_ocean * c_ocean) / (a * rho)
    phi = gamma - ((b_ocean * k * c_ocean) / (a * rho))
    
    # This segment of code loops over the time steps to compute the analytical salt concentration at each time step
    for t in ts:
        
        x_value = -((rho * v * phi) / (q0)) + (omega / (b + ((q0) / (rho * v)))) * np.exp(-b * t) + \
            (x0 + ((rho * v * phi) / (q0)) - (omega / (b + ((q0) / (rho * v))))) * np.exp(((q0) / (rho * v)) * t)

        xa.append(x_value)

    # This segment of code plots the analytical solution for salinity concentration in the aquifer for the case where P < P_ocean.
    # plt.figure()
    # plt.plot(ts, np.array(xa)/v, 'r--', label="Analytical Solution (x)")    
    # plt.plot(ts, np.array(xs)/v, 'b.', markersize=3, label="Numerical Solution (x)")
    # plt.xlabel("Time [yr]")
    # plt.ylabel("Salinity in the aquifer [kg/m3]")
    # plt.title("Benchmarking Salinity Concentrations with P < P_ocean")
    # plt.legend()
    # plt.show()

    return xa

# This function performs a unit test that checks the correct implementation of another function and verifys it.
def unit_tests():
    """
    Unit tests for one-step backward Euler method for both pressure and salinity.
    """

    # Constants
    a = 1e-10
    b_alps = 0.5
    b_ocean = 1e-5
    p_alps = 4
    p_ocean = 1

    q0 = -5e10
    p0 = 320     
    x0 = 0        
    v = 5000   
    c_ocean = 35  
    rho = 997     

    dt = 0.01     # One timestep
    n = 1         # Only one step for unit test

    # Pressure Unit Test
    ts, ps_modelled = p_backward_euler(a, b_alps, b_ocean, p_alps, p_ocean, q0, dt, n, p0, show_plot=False)
    
    # Expected pressure using manual backward Euler calculation
    numerator = p0 + dt * (a * q0 + b_alps * p_alps + b_ocean * p_ocean)
    denominator = 1 + dt * (b_alps + b_ocean)
    p_expected = numerator / denominator #=318.378 from hand calculations

    error_p = abs(ps_modelled[1] - p_expected)
    assert error_p < 1e-8, f"PRESSURE TEST FAILED: Expected {p_expected}, got {ps_modelled[1]} (error: {error_p:.2e})"
    print(f"PRESSURE TEST PASSED: {ps_modelled[1]} ≈ {p_expected} (error: {error_p:.2e})")

    # Salinity Unit Test
    # Use the pressure result as input for salinity
    ts, xs_modelled = x_backward_euler(a, ps_modelled, b_ocean, p_ocean, dt, n, q0, v, x0, c_ocean, rho, show_plot=False)

    # Manual salinity step calculation
    p1 = ps_modelled[1]
    denom_x = 1 - dt * (q0 / (v * rho))
    numer_x = x0 - dt * (b_ocean / (a * rho)) * (p1 - p_ocean) * c_ocean
    x_expected = numer_x / denom_x #=-109.986 from hand calculations

    error_x = abs(xs_modelled[1] - x_expected)
    assert error_x < 1e-8, f"SALINITY TEST FAILED: Expected {x_expected}, got {xs_modelled[1]} (error: {error_x:.2e})"
    print(f"SALINITY TEST PASSED: {xs_modelled[1]} ≈ {x_expected} (error: {error_x:.2e})")

# This function creates two figures, with two subplots, side by side, first pressure and salinity, and then a plot for error percentages.
def benchmark_plots(ts, xa, v, xs, pa, ps):
    """
    Plotting numerical and analytical solutions for both pressure and salinity
    """    

    fig1, ax1 = plt.subplots(1, 2, figsize=(12, 6))  # 1 row, 2 columns

    # First subplot: Pressure
    ax1[0].plot(ts, pa, 'r--',markersize=6, label="Analytical Solution (P)")
    ax1[0].plot(ts, ps, 'b.', markersize=2, label="Numerical Solution (P)")
    ax1[0].set_xlabel("Time [yr]")
    ax1[0].set_ylabel("Pressure in the aquifer [kPa]")
    ax1[0].set_title("Benchmarking Pressure with a Constant Pump Rate")
    ax1[0].legend()

    # Second subplot: Salinity
    ax1[1].plot(ts, np.array(xa)/v, 'r--', markersize=6, label="Analytical Solution (x)")
    ax1[1].plot(ts, np.array(xs)/v, 'b.', markersize=2, label="Numerical Solution (x)")
    ax1[1].set_xlabel("Time [yr]")
    ax1[1].set_ylabel("Salinity in the aquifer [kg/m³]")
    ax1[1].set_title("Benchmarking Salinity with P < P_ocean")
    ax1[1].legend()

    plt.tight_layout()
    plt.show()

    """Determining and plotting error percentages on each plot"""
    # Convert salinity values to concentrations
    xa_conc = np.array(xa) / v
    xs_conc = np.array(xs) / v

    pa = np.array(pa)
    ps = np.array(ps)

    # Compute percentage errors
    pressure_error = np.abs((ps - pa) / pa) * 100
    salinity_error = np.abs((xs_conc - xa_conc) / xa_conc) * 100

    # Clean any divide-by-zero or nan
    pressure_error = np.nan_to_num(pressure_error)
    salinity_error = np.nan_to_num(salinity_error)

    # Create the figure
    fig2, ax2 = plt.subplots(1, 2, figsize=(12, 6))  # 1 row, 2 columns

    # First subplot: Pressure x error 
    ax2[0].plot(ts, pa, 'r--', label="Analytical Solution (P)")
    ax2[0].plot(ts, ps, 'b.', markersize=2, label="Numerical Solution (P)")
    # The segment of code below is to show fewer labels to reduce the plots clutter   
    for i in range(0, len(ts), max(1, len(ts)//20)):  
        ax2[0].annotate(f"{pressure_error[i]:.1f}%", (ts[i], ps[i]),
                       textcoords="offset points", xytext=(0, 5), ha='center', fontsize=6, color='black')
    ax2[0].set_xlabel("Time [yr]")
    ax2[0].set_ylabel("Pressure in the aquifer [kPa]")
    ax2[0].set_title("Benchmarking Pressure with a Constant Pump Rate (% Error Annotated)", fontsize=10)
    ax2[0].legend()

    # Second subplot: Salinity x error 
    ax2[1].plot(ts, xa_conc, 'r--', label="Analytical Solution (x)")
    ax2[1].plot(ts, xs_conc, 'b.', markersize=2, label="Numerical Solution (x)")
    # The segment of code below is to show fewer labels to reduce the plots clutter   
    for i in range(0, len(ts), max(1, len(ts)//20)):
        ax2[1].annotate(f"{salinity_error[i]:.1f}%", (ts[i], xs_conc[i]),
                       textcoords="offset points", xytext=(0, 5), ha='center', fontsize=6, color='black')
    ax2[1].set_xlabel("Time [yr]")
    ax2[1].set_ylabel("Salinity in the aquifer [kg/m³]")
    ax2[1].set_title("Benchmarking Salinity with P < P_ocean (% Error Annotated)", fontsize=10)
    ax2[1].legend()

    plt.tight_layout()
    plt.show()

"""------------------------------------------------------------------------------------------------------------------------------------"""
"""CALIBRATION"""

def main_cali():

    # STAGE 1 OF CALIBRATING THE MODEL TO PRESSURE
    tp, po = np.genfromtxt('si_pres.txt', delimiter=',', skip_header=1).T

    # Only calibrating a, b_alps, b_ocean, p_alps
    initial_guess = [2e-15, 0.5, 0.8, 2]  # a, b_alps, b_ocean, p0
    opt_params, _ = curve_fit(model_for_curve_fit, tp, po, p0=initial_guess)

    # Unpack results
    a_best, b_alps_best, b_ocean_best, p0_best = opt_params

    p_ocean_best = p0_best #assuming 2011 SS intrusion where x=0 and P=0
    p_alps_best = p_ocean_best + 3

    # Optional: plot fit
    v_err = 0.05

    pm_fit = model_for_curve_fit(tp, a_best, b_alps_best, b_ocean_best, p0_best)
    S = np.sum((po - pm_fit) ** 2) / v_err ** 2

    plt.figure()
    plt.plot(tp, pm_fit, 'b-', label='Model (Fitted)')
    plt.errorbar(tp, po, yerr=v_err, fmt='ro', label='Observed')
    plt.xlabel('Time (yrs)')
    plt.ylabel('Pressure [MPa]')
    plt.title(f'Calibrated Best Fit: Objective Function S = {S:.2f}')
    plt.legend()
    plt.show()

    print("Optimized Parameters (for pressure):")
    print(f"  a        = {a_best:.3e}")
    print(f"  b_alps   = {b_alps_best:.3e}")
    print(f"  b_ocean  = {b_ocean_best:.3e}")
    print(f"  p0       = {p0_best:.3f}")
    print(f"  p_alps   = {p_alps_best:.3f}")
    print(f"  p_ocean  = {p_ocean_best:.3f}")

    #STAGE 2 OF CALIBRATING THE MODEL TO SALINITY
    tsal, s = np.genfromtxt('si_salt.txt', delimiter=',', skip_header=1).T
    # Fixed parameters from pressure calibration
    a_fixed = a_best
    b_alps_fixed = b_alps_best
    p_alps_fixed = p_alps_best
    p0_fixed = p0_best
    p_ocean_fixed = p_alps_fixed - 3 #MPa
    c_ocean_fixed = 35  


    # Fit only volume
    v_guess_initial  = [v]

    opt_params_salt, _ = curve_fit(
        lambda ts, v_guess: model_for_curve_fit_salt(ts, b_ocean_best, p0_fixed, v_guess, p_ocean_fixed, c_ocean_fixed, a_fixed, pm_fit),tsal, s, p0=v_guess_initial)

    v_best_salt = opt_params_salt[0]

    # Evaluate model with optimal b_ocean
    sm_fit = model_for_curve_fit_salt(
        tsal, b_ocean_best, p0_fixed, v_best_salt, p_ocean_fixed, c_ocean_fixed, a_fixed, pm_fit)
    
    S = np.sum((s - sm_fit)**2) / v_err**2

    # Plotting
    plt.figure()
    plt.plot(tsal, sm_fit, 'b-', label='Model (Fitted)')
    plt.errorbar(tsal, s, yerr=v_err, fmt='ro', label='Observed')
    plt.xlabel('Time (yrs)')
    plt.ylabel('Salinity [g/L]')
    plt.title(f'Calibrated Best Fit: Objective Function S = {S:.2f}')
    plt.legend()
    plt.show()

    print("Optimized Parameters (for salinity):")
    print(f"  volume  = {v_best_salt:.3f}")

    return a_best, b_alps_best, b_ocean_best, p0_fixed, p_alps_fixed, p_ocean_fixed, v_best_salt

def model_for_curve_fit(tp, a, b_alps, b_ocean, p0_best):
    p_ocean_best = p0_best + 0.01  #assuming 2011 SS intrusion where x=0 and P=0
    p_alps_best = p_ocean_best + 3
    return solve_lpm(tp, a, b_alps, b_ocean, p0_best, p_alps_best, p_ocean_best, tp)
       
def model_for_curve_fit_salt(ts, b_ocean, p0, v_best_salt, p_ocean, c_ocean, a_fixed, pm_fit):
    rho = 997
    return solve_lpm_salt(ts, a_fixed, b_ocean, p0, p_ocean, c_ocean, v_best_salt, rho, tsal, pm_fit)

def lpm(P, ts, a, b_alps, b_ocean, p0, p_alps, p_ocean):
    ''' ODE for lumped parameter model

        Parameters:
        -----------
        p0 : float
            Pressure change from initial.
        ts : float
            Time.
        a : float
            Parameter controlling drawdown response.
        b_alps : float
            Parameter controlling recharge response.
        b_ocean : float
            Parameter controlling slow drainage.

        Returns:
        --------
        dpdt : float
            Rate of change of pressure in the reservoir.

    '''

    qi = np.interp(ts,tq,q)           # interpolate (piecewise linear) flow rate

    return a*qi - b_alps*(P-(p0 + p_alps)) - b_ocean*(P-(p0 - p_ocean))    # compute derivative

def solve_lpm(ts, a, b_alps, b_ocean, p0, p_alps, p_ocean, tp):
    ''' Solve the pressure lumped parameter ODE.
    '''   
    pm = [p[0],]                                # initial value
    for t0,t1 in zip(tp[:-1],tp[1:]):           # solve at pressure steps
        dpdt1 = lpm(pm[-1]-p[0], t0, a, b_alps, b_ocean, p0, p_alps, p_ocean)   # predictor gradient
        pp = pm[-1] + dpdt1*(t1-t0)             # predictor step
        dpdt2 = lpm(pp-p[0], t1, a, b_alps, b_ocean, p0, p_alps, p_ocean)       # corrector gradient
        pm.append(pm[-1] + 0.5*(t1-t0)*(dpdt2+dpdt1))  # corrector step
    return np.interp(ts, tp, pm)             # interp onto requested times

def lpm_salt(x, ts, a, b_ocean, p0, p_ocean, c_ocean, v, rho):
    """
    salt lpm
    """    
    
    qi = np.interp(ts,tq,q)           # interpolate (piecewise linear) flow rate
    pi = np.interp(ts,tp,p)

    if pi > 0:  # Water flowing out to the ocean (outflow)
        concentration = x / v  # Salt concentration in aquifer
    else:       # Water flowing in from the ocean (inflow)
        concentration = c_ocean 
    
    return (qi/rho)*(x/v) - (b_ocean/(a*rho))*(pi-(p0 - p_ocean)) * concentration    # compute derivative

def solve_lpm_salt(ts, a, b_ocean, p0, p_ocean, c_ocean, v_best_salt, rho, tsal, P):
    ''' Solve the salinity lumped parameter ODE.
    '''
    tq,q = np.genfromtxt('si_mass.txt', delimiter = ',', skip_header = 1).T
    q_func = interp1d(tq, q, fill_value="extrapolate")
    if len(tp) ==  len(P):
        pm_func = interp1d(tp, P, fill_value="extrapolate")
    else:
        pm_func = interp1d(ts, P, fill_value="extrapolate")
    xm = [0] # initial salt conc
    for t0, t1 in zip(tsal[:-1], tsal[1:]):
        c_aq = max(xm[-1] / v_best_salt, 0) # limit concentration to be non-negative
        q = q_func(t0)
        p_current = pm_func(t0)
        c = c_ocean if p_current < 0 else c_aq

        dxdt1 = lpm_salt(xm[-1], t0, a, b_ocean, p0, p_ocean, c_ocean, v, rho)
        xp = xm[-1] + dxdt1 * (t1 - t0)
        c_aq = xp / v_best_salt
        dxdt2 = lpm_salt(xm[-1], t0, a, b_ocean, p0, p_ocean, c_ocean, v, rho)

        xm.append(xm[-1] + 0.5 * (t1 - t0) * (dxdt1 + dxdt2))


    return np.interp(ts, tsal, xm) / v_best_salt

def fit_mvn(parspace, dist):
    """Finds the parameters of a multivariate normal distribution that best fits the data

    Parameters:
    -----------
        parspace : array-like
            list of meshgrid arrays spanning parameter space
        dist : array-like 
            PDF over parameter space
    Returns:
    --------
        mean : array-like
            distribution mean
        cov : array-like
            covariance matrix		
    """
    
    # dimensionality of parameter space
    N = len(parspace)
    
    # flatten arrays
    parspace = [p.flatten() for p in parspace]
    dist = dist.flatten()
    
    # compute means
    mean = [np.sum(dist*par)/np.sum(dist) for par in parspace]
    
    # compute covariance matrix
        # empty matrix
    cov = np.zeros((N,N))
        # loop over rows
    for i in range(0,N):
            # loop over upper triangle
        for j in range(i,N):
                # compute covariance
            cov[i,j] = np.sum(dist*(parspace[i] - mean[i])*(parspace[j] - mean[j]))/np.sum(dist)
                # assign to lower triangle
            if i != j: cov[j,i] = cov[i,j]
            
    return np.array(mean), np.array(cov)

def get_familiar_with_model(q0, a, b_alps, b_ocean, p_alps, p_ocean, p0, v, c_ocean):
    ''' This function runs and plots the lumped parameter model for your selection of 
        model parameters.
    '''

    # get data and run model
        # po = pressure observation
    tp,po = np.genfromtxt('si_pres.txt', delimiter = ',', skip_header=1).T
        # pm = pressure model
    pm = solve_lpm(tp, a, b_alps, b_ocean, p0, p_alps, p_ocean, tp)

    tx,xo = np.genfromtxt('si_salt.txt', delimiter = ',', skip_header=1).T
        # xm = salinity model
    
    rho=997
    xm = solve_lpm_salt(tx, a, b_ocean, p0, p_ocean, c_ocean, v, rho, tsal, pm)
    # error variance - 2 bar
    v = 0.025

    # 2. calculate the sum-of-squares objective function (= 0. just a placeholder)
    S_pres = np.sum((po - pm)**2) / v**2   # calculation involves variables 'po' and 'pm'
    S_salt = np.sum((xo - xm)**2) / v**2   # calculation involves variables 'po' and 'pm'

def grid_search(a_best, b_alps_best, b_ocean_best_salt, p0, p_alps, p_ocean):
    ''' This function implements a grid search to compute the posterior over a and b.

        Returns:
        --------
        a : array-like
            Vector of 'a' parameter values.
        b : array-like
            Vector of 'b' parameter values.
        P : array-like
            Posterior probability distribution.
    '''

    # number of values considered for each parameter within a given interval
    N = 20

    # vectors of parameter values
    a = np.linspace(a_best/2,a_best*1.5, N)
    b_alps = np.linspace(b_alps_best/2,b_alps_best*1.5, N)
    b_ocean = np.linspace(b_ocean_best_salt/2,b_ocean_best_salt*1.5, N)

    # grid of parameter values: returns every possible combination of parameters in a and b
    A, Ba, Bo = np.meshgrid(a, b_alps,b_ocean, indexing='ij')

    # empty 2D matrix for objective function
    S = np.zeros(A.shape)

    # data for calibration
    tp,pO = np.genfromtxt('si_pres.txt', delimiter = ',', skip_header=1).T

    # error variance - 2 bar
    v = 0.025
           
    for i in range(len(a)):
        for j in range(len(b_alps)):
            for k in range(len(b_ocean)):
                # compute the sum of squares objective function at each value
                pm = solve_lpm(tp, a[i], b_alps[j], b_ocean[k], p0, p_alps, p_ocean, tp)
                diff = pO - pm
                S[i, j, k] = np.sum(diff**2) 

    P = np.exp(-S/2)  # Shift to avoid underflow/overflow
 
    # normalize to a probability density function
    Pint = np.sum(P)*(a[1]-a[0])*(b_alps[1]-b_alps[0])*(b_ocean[1]-b_ocean[0])
    P = P/Pint

    # plot posterior parameter distribution
    plot_posterior(a, b_alps, b_ocean, P=P)

    return a,b_alps, b_ocean, P

def construct_samples(a,b_alps, b_ocean, p0, p_alps, p_ocean, P, N_samples):
    ''' This function constructs samples from a multivariate normal distribution
        fitted to the data.
    '''


    # compute properties (fitting) of multivariate normal distribution
    # mean = a vector of parameter means
    # covariance = a matrix of parameter variances and correlations
    A, Ba, Bo = np.meshgrid(a,b_alps,b_ocean,indexing='ij')
    mean, covariance = fit_mvn([A, Ba, Bo], P)
  
    # 1. create samples using numpy function multivariate_normal (Google it)
    samples = np.random.multivariate_normal(mean, covariance, N_samples)
    
    # plot samples and predictions
    plot_samples(a, b_alps, b_ocean, p0=p0, p_alps=p_alps, p_ocean=p_ocean, P=P, samples=samples)
    return samples

def model_ensemble(samples, p0, p_alps, p_ocean):
    ''' Runs the model for given parameter samples and plots the results.

        Parameters:
        -----------
        samples : array-like
            parameter samples from the multivariate normal
    '''
    # get the data
    tp,po = np.genfromtxt('si_pres.txt', delimiter = ',', skip_header=1).T	

    # 1. choose a time vector to evaluate your model between 1990 and 2020
    t = np.linspace(1990, 2020, num=30)
    
    # 2. create a figure and axes (see TASK 1)
    f,ax = plt.subplots(1,1)
    
    # 3. for each sample, solve and plot the model  (see TASK 1)
    for a, b_alps, b_ocean in samples:
        pm = solve_lpm(t, a, b_alps, b_ocean, p0, p_alps, p_ocean, tp)
        ax.plot(t, pm, 'b-', lw=0.5, alpha=0.4)

    # this command just adds a line to the legend
    ax.plot([],[],'k-', lw=0.5,alpha=0.4, label='pressure model ensemble')

    ax.axvline(1990, color='b', linestyle=':', label='calibration/forecast')
 
    v = 0.05

    # 2. calculate the sum-of-squares objective function (= 0. just a placeholder)
    S = np.sum((po - pm)**2) / v**2   
    
    # plotting commands
    ax.plot(tp,pm,'b-', label='model')
    ax.errorbar(tp,po,yerr=v,fmt='ro', label='data')
    ax.set_xlabel('Years')
    ax.set_ylabel('Pressure [MPa]')
    ax.set_title('Ensemble Objective Function: S={:3.2f}'.format(S))
    ax.legend()
    plt.show()

def model_ensemble_salt(samples, a_best, b_alps_best, b_ocean_best, p0, p_alps, v, p_ocean, t):
    ''' Runs the model for given parameter samples and plots the results.

        Parameters:
        -----------
        samples : array-like
            parameter samples from the multivariate normal
    '''

    # for each sample, solve and plot the model  
    rho = 997
    sm_all = []
    pm_all = []
    for a, b_alps, b_ocean in samples:

        pm = solve_lpm(t, a, b_alps, b_ocean, p0, p_alps, p_ocean, t)
        pm_all.append(pm)
        sm = solve_lpm_salt(t, a, b_ocean, p0, p_ocean, c_ocean, v, rho, t, pm)
        sm_all.append(sm)

    pm_best = solve_lpm(t, a_best, b_alps_best, b_ocean_best, p0, p_alps, p_ocean, t)
    sm_best = solve_lpm_salt(t, a_best, b_ocean_best, p0, p_ocean, c_ocean, v, rho, t, pm_best)    
    
    sm_all = np.array(sm_all)
    pm_all = np.array(pm_all)

    return t, sm_all, pm_all, pm_best, sm_best
                        
def model_forecast(samples, a_best, b_alps_best, b_ocean_best, p_alps, p_ocean, p0, v):
    """Predict salinity 30 years into the future based on pumping rate scenarios"""

    global tq, q

    # tq, q_increase, q_decrease, q_constant, q_stopped = np.genfromtxt('forecast_q.csv', delimiter=',', skip_header=1).T

    data = np.genfromtxt('forecast_q.csv', delimiter=',', skip_header=1)

    # Handle possible 1D result if there's only one row
    if data.ndim == 1:
        data = data.reshape(1, -1)

    # Keep only first 5 columns
    data = data[:, :5]

    tq, q_increase, q_decrease, q_constant, q_stopped = data.T
    
    scenarios = [q_increase, q_decrease, q_constant, q_stopped]
    labels = ['Increasing to 2.80E+15 kg/yr ', 'Decreasing linearly to 0 kg/yr', 'Constant at 8.36E+14 kg/yr', 'Stopped']
    colors = ['red', 'blue', 'green', 'orange']

    t_fore = np.linspace(1990, 2050, num = 60)
    fig, (ax2, ax1)  = plt.subplots(2,1, figsize = (12,6), sharex = True)
    ax1.set_title('Forecasted Salinity Ensembles under Pumping Scenarios')
    ax2.set_title('Forecasted Pressure Ensembles under Pumping Scenarios')
    ax1.set_ylabel('Salinity [g/L]')
    ax2.set_ylabel('Pressure (MPa)')
    ax1.set_xlabel('Year')


    for q_scenario, label, color in zip(scenarios, labels, colors):
        q = - q_scenario  # updates global q used in solve_lpm

        t, sm_all, pm_all, pm_best, sm_best = model_ensemble_salt(samples, a_best, b_alps_best, b_ocean_best, p0, p_alps, v, p_ocean, t_fore)

        # Plot all ensemble members
        for sm in sm_all:
            ax1.plot(t, sm, color=color, alpha=0.09)

        for pm in pm_all:
            ax2.plot(t, pm, color=color, alpha = 0.09)

        ax1.plot(t, sm_best, color = color, lw = 2, label = f"{label} Flow Scenario Best Fit")
        ax2.plot(t, pm_best, color = color, lw = 2, label = f"{label} Flow Scenario Best Fit")


    ax1.grid(True)
    ax2.grid(True)
    ax1.legend(loc="best")
    ax2.legend(loc="best")
    plt.tight_layout()
    plt.show()

    ###Producing posterior distribution of salt concentrations
    sm_values = []
    rho = 997

    for a, b_alps, b_ocean in samples:
        pm = solve_lpm(t, a, b_alps, b_ocean, p0, p_alps, p_ocean, tp)
        sm = solve_lpm_salt(t, a, b_ocean, p0, p_ocean, c_ocean, v, rho, t, pm)
        sm_values.append(sm[-1])

    p5, p95 = np.percentile(sm_values, [5, 95])
    mean_val = np.mean(sm_values)

    # Plotting the posterior distribution histogram for the specified year
    plt.figure()
    plt.hist(sm_values, bins=20, density=True, edgecolor='k', alpha=0.7)
    plt.axvline(p5, color='red', linestyle='--', label=f'5th percentile ({p5:.2f} g/L)')
    plt.axvline(p95, color='blue', linestyle='--', label=f'95th percentile ({p95:.2f} g/L)')
    plt.axvline(mean_val, color='green', linestyle='-', label=f'Mean ({mean_val:.2f} g/L)')
    plt.xlabel('Salinity [g/L]')
    plt.ylabel('Probability Density')
    plt.title('Posterior Distribution of Salt Concentration')
    plt.legend(loc="best")
    plt.show()

"""------------------------------------------------------------------------------------------------------------------------------------"""
"""PLOTTING"""

def plot_lpm(lpm, theta):
    """Plot lpm model

    Args:
        lpm (callable): lumped parameter model
        theta (numpy array): lpm parameters vector
    """
    
    # write label for model
    a = str(round(theta[0]*1.e-8*365*24*3600, 2))
    b = str(round(theta[1], 2))
    label = 'Model for: \n'
    label += r'$a='+a+'years^{-1}$'
    label += '\n'+r'$b='+b+'10^{-5}m^{-1}s^{-2}$'
    if len(theta) == 3:	# check model dimension 
        c = str(round(theta[2], 2))
        label += '\n'+r'$c='+c+'10^{3}m^{-1}s^{-1}$'
        
    # plot parameters
    year_end_calib = 2020.	# interval used for calibration: observations till this date
    
    text_size = 12

    # load pressure history
    table = np.loadtxt(open('si_pres.txt','rb'), delimiter = ',')
    t = table[:,0]		# time vector [years]
    p_real = table[:,1]	# observed reservoir pressure vector [bars]
    imax = np.argmin(abs(t-year_end_calib))	# index of the final year used in calibration
    
    # load model
    p_model = lpm(theta)	# simulated reservoir pressure vector [bars]
    
    # plotting
    fig = plt.figure(figsize = [10., 7.])			# open figure
    ax1 = plt.axes()								# create axes
    ax1.plot(t[:imax+1], p_real[:imax+1], 'bx', mew = 1.5, zorder = 10, lw = 1.5, label = 'Observations')	# show observations
    ax1.plot(t[:imax+1], p_model[:imax+1], color ='k', lw = 1.5, label = label)					# show model
    
    # plotting upkeep
    ax1.set_xlim(t[0], t[imax])
    ylim = [.4*min(p_real), 1.1*max(p_real)]
    ax1.set_ylim(ylim)
    ax1.set_xlabel('Time (years)', fontsize = text_size)
    ax1.set_ylabel('Pressure (bars)', fontsize = text_size)
    ax1.legend(loc = 'upper right', fontsize = text_size, framealpha = 0.)
    ax1.tick_params(labelsize=text_size)
    
    # save and show
    if len(theta) == 2: save_name = 'lab4_plot1.png'
    elif len(theta) == 3: save_name = 'lab4_plot4.png'
    plt.savefig(save_name, bbox_inches = 'tight')
    plt.show()
    
def plot_posterior(a,b_alps,b_ocean,P=None):
    if b_ocean is None:
        plot_posterior2D(a,b_alps,P)
    else:
        plot_posterior3D(a,b_alps,b_ocean,P)

def plot_posterior2D(a, b_alps, P):	
    """Plot posterior distribution

    Args:
        a (numpy array): a distribution vector
        b (numpy array): b distribution vector
        P (numpy array): posterior matrix
    """
    
    # grid of parameter values: returns every possible combination of parameters in a and b
    A, Ba = np.meshgrid(a, b_alps)
    
    # plotting
    fig = plt.figure(figsize=[10., 7.])				# open figure
    ax1 = fig.add_subplot(111, projection='3d')		# create 3D axes
    ax1.plot_surface(A, Ba, P, rstride=1, cstride=1,cmap=cm.Oranges, lw = 0.5,edgecolor='k')	# show surface
    
    text_size = 12 
    # plotting upkeep
    ax1.set_xlabel('a', fontsize = text_size)
    ax1.set_ylabel('b', fontsize = text_size)
    ax1.set_zlabel('P', fontsize = text_size)
    ax1.set_xlim([a[0], a[-1]])
    ax1.set_ylim([b_alps[0], b_alps[-1]])
    ax1.set_zlim(0., )
    ax1.view_init(40, 100.)
    
    # save and show
    plt.show()
    
def plot_posterior3D(a, b_alps, b_ocean, P):	
    """Plot posterior distribution for each parameter combination

    Args:
        a (numpy array): a distribution vector
        b (numpy array): b distribution vector
        P (numpy array): posterior matrix
    """
    
    # plotting variables
    azim = 15.		# azimuth at which surfaces are shown
    
    # a and b combination
    Ab, Ba = np.meshgrid(a, b_alps, indexing='ij')
    Pab = np.zeros(Ab.shape)
    for i in range(len(a)):
        for j in range(len(b_alps)):
            Pab[i][j] = sum([P[i][j][k] for k in range(len(b_ocean))])

    # a and c combination			
    Ac, Ca = np.meshgrid(a, b_ocean, indexing='ij')
    Pac = np.zeros(Ac.shape)
    for i in range(len(a)):
        for k in range(len(b_ocean)):
            Pac[i][k] = sum([P[i][j][k] for j in range(len(b_alps))])
    
    # b and c combination		
    Bc, Cb = np.meshgrid(b_alps, b_ocean, indexing='ij')
    Pbc = np.zeros(Bc.shape)
    for j in range(len(b_alps)):
        for k in range(len(b_ocean)):
            Pbc[j][k] = sum([P[i][j][k] for i in range(len(a))])
            
    # plotting
    fig = plt.figure(figsize=[20.0,15.])
    ax1 = fig.add_subplot(221, projection='3d')
    ax1.plot_surface(Ab, Ba, Pab, rstride=1, cstride=1, cmap=cm.Oranges, lw = 0.5)
    ax1.set_xlabel('a')
    ax1.set_ylabel('b')
    ax1.set_zlabel('P')
    ax1.set_xlim([a[0], a[-1]])
    ax1.set_ylim([b_alps[0], b_alps[-1]])
    ax1.set_zlim(0., )
    ax1.view_init(40, azim)
    
    ax1 = fig.add_subplot(222, projection='3d')
    ax1.plot_surface(Ac, Ca, Pac, rstride=1, cstride=1,cmap=cm.Oranges, lw = 0.5)
    ax1.set_xlabel('a')
    ax1.set_ylabel('c')
    ax1.set_zlabel('P')
    ax1.set_xlim([a[0], a[-1]])
    ax1.set_ylim([b_ocean[0], b_ocean[-1]])
    ax1.set_zlim(0., )
    ax1.view_init(40, azim)

    ax1 = fig.add_subplot(223, projection='3d')
    ax1.plot_surface(Bc, Cb, Pbc, rstride=1, cstride=1,cmap=cm.Oranges, lw = 0.5)
    ax1.set_xlabel('b')
    ax1.set_ylabel('c')
    ax1.set_zlabel('P')
    ax1.set_xlim([b_alps[0], b_alps[-1]])
    ax1.set_ylim([b_ocean[0], b_ocean[-1]])
    ax1.set_zlim(0., )
    ax1.view_init(40, azim)
    
    # save and show
    plt.show()

def plot_samples(a, b_alps, b_ocean, p0, p_alps, p_ocean, P, samples):
    if b_ocean is None:
        plot_samples2D(a,b_alps,P,samples)
    else:
        plot_samples3D(a,b_alps,b_ocean,p0,p_alps,p_ocean,P,samples)

def plot_samples2D(a, b_alps, P, samples):
    # plotting
    fig = plt.figure(figsize=[10., 7.])				# open figure
    ax1 = fig.add_subplot(111, projection='3d')		# create 3D axes
    A, B = np.meshgrid(a, b_alps, indexing='ij')
    ax1.plot_surface(A, B, P, rstride=1, cstride=1,cmap=cm.Oranges, lw = 0.5)	# show surface
    
    tp,po = np.genfromtxt('si_pres.txt', delimiter = ',')[:28,:].T
    v = 2
    s = np.array([np.sum((solve_lpm(tp,a,b)-po)**2)/v for a,b in samples])
    p = np.exp(-s/2.)
    p = p/np.max(p)*np.max(P)*1.2
        
    ax1.plot(*samples.T,p,'k.')

    text_size = 12
    # plotting upkeep
    ax1.set_xlabel('a', fontsize = text_size)
    ax1.set_ylabel('b', fontsize = text_size)
    ax1.set_zlabel('P', fontsize = text_size)
    ax1.set_zlim(0., )
    ax1.view_init(40, 100.)
    
    # save and show
    plt.show()

def plot_samples3D(a, b_alps, b_ocean, p0,p_alps,p_pocean, P, samples):
    # plotting variables
    azim = 15.		# azimuth at which surfaces are shown


    # a and b combination
    Ab, Ba = np.meshgrid(a, b_alps, indexing='ij')
    Pab = np.zeros(Ab.shape)
    for i in range(len(a)):
        for j in range(len(b_alps)):
            Pab[i][j] = sum([P[i][j][k] for k in range(len(b_ocean))])

    # a and c combination			
    Ac, Ca = np.meshgrid(a, b_ocean, indexing='ij')
    Pac = np.zeros(Ac.shape)
    for i in range(len(a)):
        for k in range(len(b_ocean)):
            Pac[i][k] = sum([P[i][j][k] for j in range(len(b_alps))])
    
    # b and c combination		
    Bc, Cb = np.meshgrid(b_alps, b_ocean, indexing='ij')
    Pbc = np.zeros(Bc.shape)
    for j in range(len(b_alps)):
        for k in range(len(b_ocean)):
            Pbc[j][k] = sum([P[i][j][k] for i in range(len(a))])

    tp,po = np.genfromtxt('si_pres.txt', delimiter = ',', skip_header=1).T
    v = 0.05
    s = np.array([np.sum((solve_lpm(tp,a,b_alps,b_ocean,p0,p_alps,p_pocean,tp)-po)**2)/v for a,b_alps,b_ocean in samples])
    p = np.exp(-s/2.)
    p = p/np.max(p)*np.max(P)*1.2

    # plotting
    fig = plt.figure(figsize=[20.0,15.])
    ax1 = fig.add_subplot(221, projection='3d')
    ax1.plot_surface(Ab, Ba, Pab, rstride=1, cstride=1, cmap=cm.Oranges, lw = 0.5)
    ax1.set_xlabel('a')
    ax1.set_ylabel('b_alps')
    ax1.set_zlabel('P')
    ax1.set_xlim([a[0], a[-1]])
    ax1.set_ylim([b_alps[0], b_alps[-1]])
    ax1.set_zlim(0., )
    ax1.view_init(40, azim)
    ax1.plot(samples[:,0],samples[:,1],p,'k.')
    
    ax1 = fig.add_subplot(222, projection='3d')
    ax1.plot_surface(Ac, Ca, Pac, rstride=1, cstride=1,cmap=cm.Oranges, lw = 0.5)
    ax1.set_xlabel('a')
    ax1.set_ylabel('b_ocean')
    ax1.set_zlabel('P')
    ax1.set_xlim([a[0], a[-1]])
    ax1.set_ylim([b_ocean[0], b_ocean[-1]])
    ax1.set_zlim(0., )
    ax1.view_init(40, azim)
    ax1.plot(samples[:,0],samples[:,-1],p,'k.')

    ax1 = fig.add_subplot(223, projection='3d')
    ax1.plot_surface(Bc, Cb, Pbc, rstride=1, cstride=1,cmap=cm.Oranges, lw = 0.5)
    ax1.set_xlabel('b_alps')
    ax1.set_ylabel('b_ocean')
    ax1.set_zlabel('P')
    ax1.set_xlim([b_alps[0], b_alps[-1]])
    ax1.set_ylim([b_ocean[0], b_ocean[-1]])
    ax1.set_zlim(0., )
    ax1.view_init(40, azim)
    ax1.plot(samples[:,1],samples[:,-1],p,'k.')
    
    # save and show
    plt.show()

if __name__ == "__main__":
    main()
