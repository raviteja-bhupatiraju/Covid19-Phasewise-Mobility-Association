# User-defined function file for python fitVirusXX
# Date: 08.05.2020
# Version 1

# Notes
# Helper function: Internal functions for executing small modules within Major functions
# Major function: External function that user will call from main python file
# Function names ending with 'sw' keyword indicated single wave
# graph with phases are not added now since that will take unnecessary time and resource

# Load necessary modules
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from math import floor, log, exp, ceil, sqrt, isnan, isinf, nan
#import math
from scipy import optimize
from sklearn.metrics import mean_squared_error, r2_score
from gekko import GEKKO
#from pyloess.Loess import Loess # user defined library
from sklearn.neighbors.kde import KernelDensity # For pyloess function
import statsmodels.api as sm
from lmfit import Parameters, fit_report, minimize

########################## BEGIN FUNCTION DEFINTIONS ######################################

# Master function to do all the caluclation stuff. 
# Input will be state data, state name, 
# output will be array of phase locations adjusted to week scale used by Dr. Satya
def master_fn(manual_end,df, week_array_file, state_name, debug, plot_res, size_fig, scale_term, chk_method):
    
    # Step 0 Define working variable and constants
    min_ndat = 7 # This is emperical, may be subject to change or determination using another technique
    tol_single_wave = 0.90
    week_agg = []
    timestamp_week = []
    bunpack = [] # Array to hold actual, predicted and timespans for two waves
    bunfit = [] # Array to fitted parameters for two waves

    # Step 1 Load weekly data and time index
    week_agg, timestamp_week = choose_state(df, state_name)
    
    #timestamp_week = week_array_file # Test with dr satya's array
    
    max_ndat = len(week_agg) # This is a global variable telling how many week of data we have
    if(debug == True):
        print(f"\nState --> {state_name}")
        # print(f"Number of weeks data to fit is {max_ndat}\n")
        # print(week_agg)
    
    # Step 2 Test if we have multiple waves (depricated)
    b0_init = [] # We don't have any initial values to pass for first wave test
    b_wave1, rr2_wave1, rmse_wave1 = check_multiwave(b0_init, week_agg, timestamp_week, nx_test= 0, plot_res= False, method = chk_method)
    
    if(debug == True):
        print(f"R2 score for test for 1 wave data --> {rr2_wave1}")
        print(f"RMSE score for test for 1 wave data --> {rmse_wave1}\n")
        print(f"Fit for test for 1 wave data --> {b_wave1}\n")
    #return rr2_wave1, rmse_wave1
    
    # # Step 3 Find probable ending of wave 1
    if (manual_end == 0):
        nxx_wave1, bb_w1 = find_wave1(week_tot=week_agg, time_tot=timestamp_week, min_ndat = min_ndat, b0_init=b0_init, plot_res = False, max_ndat = max_ndat, chk_method=chk_method)
        # # Step 4 : Find ending of week 1
        wave1_end, b00_w2 = second_round_wave1(tol_prob = 0.97, rmse_tol = 5000, week_tot = week_agg, time_tot = timestamp_week, bb_w1 = bb_w1, max_ndat = max_ndat, nxx_wave1 = nxx_wave1, print_res = False, plot_res= False)
    
    # Step X -- manual override wave1_end based on visual inspection
    if (manual_end > 0):
        wave1_end = manual_end
        print(f"Wave 1 ends in week --> {wave1_end}")
    
    # # Step 5: Finalize wave 1
    bb_fit1, y_pred_w1, y_true_w1, t_span_w1 = finalize_wave1(wave1_end, week_agg, timestamp_week, method = 'leastsq', show_res = False)

    # # Create variables for 2nd wave fit
    tt_w2 = timestamp_wave_2(t_span_w1, timestamp_week)
    slice_index = len(t_span_w1)

    # # Step 6 Finalize wave 2
    bb_fit2, y_true_w2, y_pred_w2, t_span_w2 = finalize_wave2(bb_fit1,tt_w2, week_agg, slice_index, print_debug= debug)

    # # Step 7: Combine outputs in an array for convenience
    bunfit = [bb_fit1, bb_fit2]
    bunpack = [y_true_w1, y_pred_w1, t_span_w1, y_true_w2, y_pred_w2, t_span_w2]

    # # Step 7a: Record max week values for wave 1 and wave 2, for scaling to week scale used by Dr. Satya
    max_w1 = max(t_span_w1)
    max_w2 = max(t_span_w2)

    # # Step 7b : Send out values of rate of change per wave
    w1_rate = bb_fit1[1]
    w2_rate = bb_fit2[1]

    if(debug == True):
        print(f"\nWave 1 rate of change --> {w1_rate} cases per week")
        print(f"Wave 2 rate of change --> {w2_rate} cases per week\n")    

    # # Step 8 -- Prepare variables for cumulative curve
    t_cumu = week_array_file # x scale for correct figure
    y_act = week_agg
    y_pred = np.round(np.concatenate((bunpack[1], bunpack[4]), axis=0))

    # Step 9: Find phases
    # w1_loc_ls = []
    # w2_loc_ls = []
    # w1_loc_ls, w2_loc_ls = find_phases_multi(bb_fit1,bb_fit2,t_span_w1,t_span_w2, print_res = False, scale_val=scale_term)

    # Step 10: plot cumulative waves
    # if (plot_res == True):
    #     pname = state_name + "_cumulative"
    #     plot_fit(pname,t_cumu,y_act,y_pred, size_fig)
    #     max_y_act = max(y_pred)
    #     plot_phase_lines(w1_loc_ls, w2_loc_ls, max_y_act)

    # Step 11: Calculate actual day-over-day change and predicted day-over-day change
    I_act = []
    I_pred = []
    I_act, I_pred = fit_incidence_multiwave(bunpack, bunfit, False)
    # print(len(I_act))
    # print(len(I_pred))

    # Step 12: plot cumulative waves
    # if (plot_res == True):
    #     pname = state_name + "_incidence"
    #     plot_fit(pname,t_cumu,I_act,I_pred, size_fig)
    #     max_y_act = max(I_pred)
    #     plot_phase_lines(w1_loc_ls, w2_loc_ls, max_y_act)
    # Step 13: Plot phase lines

    # ---------------------------End of calculations -------------------------------
    print(f"\n-------------------End of state calculation -----------------------\n")
    
    return y_act, y_pred, I_act, I_pred, bb_fit1, bb_fit2, t_span_w1, t_span_w2, t_cumu
# ------------------------------------------------------------------------------------------------------

# Helper function to find phase starting and ending point locations
def find_phase_loc(nume_scale, w_K, w_r, w_A, t_span, print_res):
    tpeak, tp2, tp3, tp4, tpend = 0,0,0,0,0
    loc_ls = []
    start_loc = t_span[0]
    
    tpeak = int(log(w_A)/w_r)
    tside = int(nume_scale/w_r)
    tp2 = int(tpeak - tside)

    if (tp2>0):
        # Calculate tp3
        tp3 = int(tpeak + tside)
        # Is tp3 fully developed?
        # max(t_span_#) --> last week value
        if(tp3 >= max(t_span)):
            tp3 = max(t_span)
            # Not enough data to find tp4 and tend
            tp4, tpend = 0,0
        if(tp3 < max(t_span)):
            # Phase 3 developed, find tp4
            tp4 = int(2 * tpeak)
            # Check if tp4 is fully developed
            if (tp4 >= max(t_span)):
                tp4 = max(t_span)
                # Not enough points to find tpend, Phase 4 not fully developed
                tpend = 0
            else:
                # Phase 4 fully developed, Phase 5 can be found
                tpend = max(t_span)
    # Pack values into an array
    loc_ls = [tp2,tpeak,tp3,tp4,tpend] #tp2,tpeak,,tp3,tp4,tpend
    # Print statistics
    if (print_res == True):
        print(f"\nStart loc wave --> {start_loc}")
        print(f"tp2 --> {tp2}")
        print(f"tpeak --> {tpeak}")
        #print(f"tside --> {tside}")
        print(f"tp3 --> {tp3}")
        print(f"tp4 --> {tp4}")
        print(f"tpend --> {tpend}\n")
    # Send back locations points
    return loc_ls
# -----------------------------------------------------------------------------------------------------

# Helper function to find phase starting and ending point locations
def find_phase_loc_2(w2_start, force_peak, nume_scale, w_K, w_r, w_A, t_span, print_res):
    tpeak, tp2, tp3, tp4, tpend = 0,0,0,0,0
    loc_ls = []
    start_loc = t_span[0]
    
    tpeak = int(log(w_A)/w_r)
    if (tpeak > max(t_span)):
        tpeak = max(t_span)
    
    # Forcing tpeak based on manual judgement
    if (force_peak > 0):
        tpeak = force_peak
    
    tside = int(nume_scale/w_r)
    tp2 = int(tpeak - tside)

    # Prevent tp2 crossing over to wave1
    if(tp2 < w2_start):
        tp2 = w2_start

    if (tp2>0):
        # Calculate tp3
        tp3 = int(tpeak + tside)
        # Is tp3 fully developed?
        # max(t_span_#) --> last week value
        if(tp3 >= max(t_span)):
            tp3 = max(t_span)
            # Not enough data to find tp4 and tend
            
            if (tp3 == tpeak):
                tp3 = 0
            
            tp4, tpend = 0,0
        if(tp3 < max(t_span)):
            # Phase 3 developed, find tp4
            
            if (tp3 == tpeak):
                tp3 = 0
            
            tp4 = int(2 * tpeak)
            # Check if tp4 is fully developed
            if (tp4 >= max(t_span)):
                tp4 = max(t_span)
                
                if (tp4 == tpeak):
                    tp3 = 0
                    tp4 = 0
                
                # Not enough points to find tpend, Phase 4 not fully developed
                tpend = 0
            else:
                # Phase 4 fully developed, Phase 5 can be found
                tpend = max(t_span)
    # Pack values into an array
    loc_ls = [tp2,tpeak,tp3,tp4,tpend] #tp2,tpeak,,tp3,tp4,tpend
    # Print statistics
    if (print_res == True):
        print(f"\nStart loc wave --> {start_loc}")
        print(f"tp2 --> {tp2}")
        print(f"tpeak --> {tpeak}")
        #print(f"tside --> {tside}")
        print(f"tp3 --> {tp3}")
        print(f"tp4 --> {tp4}")
        print(f"tpend --> {tpend}\n")
    # Send back locations points
    return loc_ls

# -----------------------------------------------------------------------------------------------

# Function that takes in the primary dataframe, name of a state and outputs week level cases and an array for counting week
def choose_state(df, state_name):
    sampleC = np.asarray(df[state_name])
    
    # We assume that we have at least 1 infected person in first day of outbreak
    if (sampleC[0] <= 0):
        sampleC[0] = 1
    
    week_agg = sampleC
    timestamp_week = np.asarray([(i+1) for i in range(len(sampleC))])
    return week_agg, timestamp_week

# -----------------------------------------------------------------------------------

# Corrected week-level aggregation
# week_agg -- array of cumulative cases in 'week' level
# timestamp_week -- array of time in week level
# note -- days which don't form a complete week are truncated
def week_aggregate(sampleC):
    num_dat = len(sampleC)

    week_count = int(num_dat / 7) 
    whole_week = week_count * 7

    # Generate array for week count
    tt = [(i+1) for i in range(week_count)]
    
    # Slice out number of days for which we have full weeks
    P = sampleC[:whole_week]

    # Find daily incidence (daily change in cases)
    I = np.diff(P)
    # Repeat first value again I_act_w1 = np.insert(I_act_w1, [0], [I_act_w1[0]])
    I = np.insert(I,[0],[0])

    # Copy first day's cumulative case as staring seed number
    P_sum = P[0]
    P_ls = [] # Array to hold weekly cumulative
    iterr = 0 # To match python's indexing

    for kk in I:
        P_sum = P_sum + kk
        #print(f"Iteration -->{iterr}, Sum --> {P_sum}")
        iterr = iterr+1
        if (iterr==7):
            P_ls.append(P_sum)
            iterr = 0
    
    # print(P_ls)
    # print(tt)
    P_ls = np.asarray(P_ls)
    tt = np.asarray(tt)
    return P_ls, tt

# Function to get prevelance data from selected file and parish
# Input
# strr -- Name of parish in f_name file
# f_name - Name of the file to work on
# Output -- P_week, tspan
def get_Data(strr, f_name):
    col_name = strr
    ff_name = f_name + ".csv"
    df = pd.read_csv(ff_name, header = 0)
    #print(df.head())
    
    num_dat = len(df[col_name]) # Find number of days 
    sampleC = df[col_name].values # Prevelance, cumulative number of cases, all data

    # We assume that we have at least 1 infected person in first day of outbreak
    if (sampleC[0] <= 0):
        df.at[0,col_name] = 1
    #print(sampleC[0]) # To test if we got the correct update
    
    #print(df.head()) # Sanity check

    week_agg, timestamp_week = week_aggregate(sampleC)
    return week_agg, timestamp_week

# Old Initial guess formula
# calculate initial K, r, A using data from three equidistant points
# Input C -- Actual cumulative case

# Make timestamp array for 2nd wave
# Input
# t_span_w1 - timestamp array of wave 1
# t_actual - timestamp array of full dataset
def timestamp_wave_2(t_span_w1, t_actual):
    aa = len(t_actual) - len(t_span_w1)
    ax = t_span_w1[-1]
    
    ttt = np.asarray([(i+1) for i in range(aa)])
    tdx = ttt + ax # Python broadcasting?
    
    # Debug 
    #print(len(t_span_w1))
    #print(tdx)
    return tdx

# Output bo -- inital guesses for K, r and A or [], empty set
# Major funciton
def iniGuessX(C):
    # k1, k2 and k3 are the tk, tk-m and tk - 2m from Dr. Batista's paper
    k1 = 0
    k2 = 0
    k3 = 0
    b0 = [0,0,0] # Assign an empty array outputs to hold
    n = len(C)
    # print(f"Total samples --> {n}")
    nmax = max(1,ceil(0.5 * n))
    #print(f"Number of elements to consider for 3 equidistant point --> {nmax}")
    
    # calculate time interval for equidistant points: k-2*m, k-m, k
    # Here m is the interval size
    # In MATLAB, index starts from 1, but in python its 0
    # Hence we need to subtract 1 from len(C) to match python's index

    nindex = n -1

    # Dr Batista's equidistant point schema makes no sense
    # According to the paper, we take the first, middle and the last datapoint
    # In this software, I am hardcoding the search indexes
    k1 = 0
    k3 = nindex
    k2 = int((k1 + k3)/2)
    m = k2 - k1 - 1     
    
    # print(f"k1 -- {k1}, k2 -- {k2}, k3 -- {k3}, m -- {m}")
    # print("Number of cases at chosen three points")
    # print(f"k1 -- {C[k1]}, k2 -- {C[k2]}, k3 -- {C[k3]}")

    # Calculate numenator of eqn 11
    p = (C[k1] * C[k2]) - 2 * C[k1] * C[k3] + C[k2]*C[k3]
    if (p<=0):
        p = 0
    
    # Calculate denomenator of eqn 11
    q = (C[k2]*C[k2]) - C[k3]*C[k1]
    if (q<=0):
        q = 0
    
    # Population number cannot be float
    try:
        K = int(C[k2] * (p/q))
    except:
        #K = max(C)
        K = max(C) 
    
    if (K<0 or K == float("inf")):
        K = max(C)
    
    # Calculate r using eqn 12
    r1 = C[k3] * (C[k2] - C[k1])
    r2 = C[k1] * (C[k3] - C[k2])
    try:
        r = (1/m) * log(r1/r2)
    except:
        r = 0.5
    
    if (r<0 or r == float("inf")):
        r = 0.5
    
    # Calculate r using eqn 13
    try:
        A1 = ((C[k3] - C[k2])*(C[k2] - C[k1])) / ((C[k2]*C[k2])-C[k3]*C[k1])
        A2 = (C[k3] * (C[k2] - C[k1])) / (C[k1] * (C[k3] - C[k2]))
        #print (A1, A2)
        #A2 = A2 ** ((k3 - m)/m)
        A22 = (k3 - m)/m
        A2 = pow(A2, A22)
        A = A1 * A2
    except:
        A = max(C)
    
    if (A < 0 or A == float("inf")):
        A = max(C)
    
    if (isnan(A) == True):
        A = max(C)
    
    # Max capacity of a population cannot be a float number
    A = int(A)

    # Simplify value of A
    #A = K / (A + 1)

    # Print debug report
    # print(f"p -- {p}")
    # print(f"q -- {q}")
    # print(f"r -- {r}")
    # print(f"A -- {A}")

    # print(f"Initial K value -- {K}")
    # print(f"Initial r value -- {r}")
    # print(f"Initial A value -- {A}")

    # Load initial guesses
    b0[0] = K
    b0[1] = r
    b0[2] = A
    
    return b0

# Perform week level aggregation
# Inputs
# sampleC -- array of daily cumulative cases corresponding to each entry in 'timestamp' array
# timestamp -- array of time (individual day or individual week)


# Major function whose job is to perform non-linear fit on the given set of data
# Input
# nx_test -- number of elements in the subsample
# week_agg -- numpy array of actual cumulative cases
# tspan -- numpy array of number of weeks
# show_rep -- boolean - Set True to print out fit report
# iniGuess -- array of K, r and A from a previous calculation or user supplied
# method

# Output
# b -- python list containing values of K, r and A sequentially for a particular wave
# ff -- predicted value calculated using logisticFun function
# t_span -- array of weeks for which ff was calculated

def fit_wave(ndat, week_agg, tspan, b00, show_rep, method):
    # Define working variables
    b = []
    method_user = None
    
    # Check if ndat is positive
    if (ndat <= 0):
        return "Subsample value negative or floating"
    
    # Covert to integer if ndat was given as float
    if (isinstance(ndat, float) == True):
        ndat = int(ndat)
    
    P_week = np.asarray(week_agg[:ndat])
    tt_span = np.asarray(tspan[:ndat])
    fit_params = Parameters() # Define Parameter class from LMFIT

    if len(b00) == 0:
        fit_params.add('K', value= max(P_week), max = 10000000)
        fit_params.add('r', value= 0.5, min = 0.1, max = 50.0)
        #fit_params.add('r', value= 0.5)
        fit_params.add('A', value = max(P_week), max = 10000000)
    else:
        #print("b00 not empty, need to have an unpacking script here later")
        KK = b00[0]
        rr = b00[1]
        AA = b00[2]

        fit_params.add('K', value= KK, max = 10000000)
        fit_params.add('r', value= rr, min = 0.1, max = 50.0)
        #fit_params.add('r', value= 0.5)
        fit_params.add('A', value = AA, max = 10000000)

    # Check if the fitting method is defined by user
    # https://lmfit.github.io/lmfit-py/fitting.html
    if (method == 'default'):
        method_user = 'least_squares'
    else:
        method_user = method

    # Here we will get the fit
    # method -- nedler, monte-carlo, leastsq
    fit_yy = minimize(objFun, fit_params, args=(tt_span,), kws={'data': P_week}, method = method_user)
    
    if (show_rep == True):
        print(fit_report(fit_yy))
    
    # Extract results
    Kopt = fit_yy.params['K'].value
    ropt = fit_yy.params['r'].value
    Aopt = fit_yy.params['A'].value

    b_fit = [Kopt, ropt, Aopt]
    
    # Calcuate predicted values
    yy_hat = logisticFun(tt_span,b_fit)
    yy = P_week

    return b_fit, yy_hat, yy, tt_span

# Function to try and fit data to 1 wave
# Output R2 score

def check_multiwave(b00, week_agg, tspan, nx_test, plot_res, method):
    # Load working variables
    P = week_agg # To avoid local variable referenced before problem
    size_tup = (15,5) # Size of graph

    if nx_test == 0:
        ndat = len(P) # Take all the data points given in P
    else:
        ndat = nx_test
    
    P = np.asarray(week_agg[:ndat])
    t = np.asarray(tspan[:ndat])
    #print(len(P))
    
    # Pass initial guess values or take values fitted from last wave
    if len(b00) == 0:
        b01 = iniGuessX(P)
        #print(f"Initial guess from Iniguess X --> {b00}")
    else:
        b01 = b00 # Take values from last wave
        #print("In check_multiwave -- Fitted values of last wave passed")
    
    # Perform fit
    b0, y_pred, y_true, t_span = fit_wave(ndat = ndat, week_agg= P, tspan=t, b00 = b01, show_rep = False, method = method)
    #print(b0) # Testing to see if the fit is working properly

    # Calcualte scores
    y_true = P # For sake of matching function defintions
    rr2, rmse_score = calcStats(y_true, y_pred, print_debug = False)

    # Optional, graph result
    if (plot_res == True):
        visualize_result(y_true,y_pred,t,size_tup)

    return b0, rr2, rmse_score

# Objective function, new version for lmfit
# params -- values of K, r and A [array with three elements]
#  t -- timestamp [array]
# data -- actual value at t [array]
def objFun(params,tspan, data):
    # amp = params['amp']
    # phaseshift = params['phase']
    
    # Unpack the variables
    K = params['K']
    r = params['r']
    A = params['A']
    
    yy = np.asarray(data) # Actual values
    t = np.array(tspan) # To solve can't multiply by sequence error?
    yy1 = K / (1 + A * np.exp(-1 * (r*t)))
    
    #yy1 = A * np.exp(-1 * (r * t))

    #resid = yy - yy1
    resid = np.power((yy - yy1),2)
    return resid
# ------------------------------------------------------------------------

# Function to calculate logisitic growth for given parameters and timeseries
def logisticFun(t, b):
    # Unpack parameters
    K = b[0]
    r = b[1]
    A = b[2]
    
    #A = K/A - 1; # This simiplification has been omitted

    # Set delay value if we have more than two waves (not used here)
    if (len(b) > 3):
        tdel = b[3]
        t = t - tdel # This has to be broadcasted
    
    # Calculate value of C using logitic growth equation (2 in refe to Dr. Batista's paper)
    # y = a/ (1 + c * exp(- (b * t)))
    
    # Vectorize the helper function using numpy.vectorize
    vfunc = np.vectorize(logit)
    f = vfunc(t,K,r,A)
    #print(np.shape(f))
    return f

# Generalized logistic growht model, eqn (2) from Dr. Batista's first paper
# Helper fucntion for logisticFun(t,b)
def logit(t,K,r,A):
    # K = b[0]
    # r = b[1]
    # I0 = b[2]
    f = K/ (1 + A * exp(- (r * t)))
    return f

# Vectorized version for calculating incidence
def logisticFun2(t, b, C):
    # Unpack parameters
    K = b[0]
    r = b[1]
    A = b[2]

    #vfunc = np.vectorize(logit_incidence)
    #del_f = vfunc(t,K,r,A,C)
    vfunc = np.vectorize(logit_incidence_2)
    del_f = vfunc(t,K,r,A)
    #print(np.shape(f))
    return del_f

# Helper function for logisiticFun2
def logit_incidence(t,K,r,A,C):
    dy1 = r * C
    dy2 = (1 - (C/K))
    dy = dy1 * dy2
    return dy

# Helper function for logisiticFun2
def logit_incidence_2(t,K,r,I0):
    #A = K/I0
    A = I0
    dy1 = K * r * A * exp(-r*t)
    dy2 = (1 + A * exp(-r*t) )**2
    dy = dy1/dy2 
    dy = int(round(dy,2))

    # Guard against negative numbers
    if (dy < 0):
        dy = 0 

    # dy1 = A * r * K * exp(-r*t)
    # dy2 = (1 + K * exp(-r*t))**2
    # dy = dy1/dy2 
    return dy

# Major function to search and find end of wave 1
def find_wave1(week_tot, time_tot, min_ndat, b0_init, plot_res, max_ndat, chk_method):
    nxx_test = min_ndat
    #nxx_test = 7
    nxx_wave1 = 0
    while True:
        # Slice out nnx_test number of weeks and time array
        P_test = week_tot[:nxx_test]
        t_test = time_tot[:nxx_test]

        # Print debug
        #print(f"Number of weeks in this subsample --> {nxx_test}")

        # Try fit with nx_test number of weeks, b0_init is empty cause we have no valid fit at this stage
        bb_w1, rr2_w1, rmse = check_multiwave(b0_init, P_test, t_test, nx_test= 0, plot_res=plot_res, method=chk_method)
        #print(f"R2 score for wave 1 search --> {rr2_w1}")

        # Check if R2 score is between 0.98 ~ 0.999
        if ((rr2_w1 >= 0.975 and rr2_w1 <= 0.999) or (rr2_w1 == 1)):
            nxx_wave1 = nxx_test
            break
        else:
            nxx_test= nxx_test + 1
            if (nxx_test > max_ndat):
                nxx_test = max_ndat
                print("Exceeding number of weeks available, model bad fit")
                # Set somesort of flag here later
    # End of while loop    
    #print(f"Probable wave 1 ending found at week --> {nxx_test}")
    
    # Return probable week 1 end number and fitted parameters for that number of weeks
    return nxx_test, bb_w1

# Major fucntion, iteratively finds the ending of wave-1
def second_round_wave1(tol_prob, rmse_tol, week_tot, time_tot, bb_w1, max_ndat, nxx_wave1, print_res, plot_res):
    # Set variables
    #tol_prob = 0.95 # Emperical
    # rmse_tol = 5000 # conservative emperical tolerance
    iterr = 1
    b00_w1 = bb_w1 # Fit from first round is our initialguess for 2nd round
    
    # Calculate remaining number of weeks
    nxx_left = max_ndat - nxx_wave1

    # Clearing for 2nd round of search
    nxx_test = 0 
    bb_w2 = []
    P_test = []
    t_test = []
    rr2_w2 = 0
    rmse = 0
    
    # Run forward to see how many weeks from first week, did we loose the good fit
    while True:
        nxx_test = nxx_wave1 + iterr
        #print(f"Number of weeks in the 2nd round of sampling --> {nxx_test}")

        # Slice out week and time array
        P_test = week_tot[:nxx_test]
        t_test = time_tot[:nxx_test]

        # Perform fit
        bb_w2, rr2_w2, rmse = check_multiwave(b00_w1, P_test, t_test, nx_test= 0, plot_res=plot_res)

        # Print out results
        if (print_res == True):
            print("")
            print("------------------------------------------------------")
            print(f"R2 score for week {nxx_test} --> {rr2_w2}")
            print(f"RMSE for week {nxx_test} --> {rmse}")
            print(f"Fitted parameters for week {nxx_test} --> {bb_w2}")
        
        # Test if the current fit fell below 0.97
        if(rr2_w2 <= tol_prob or rmse >= rmse_tol):
            wave1_end = (nxx_test - 1)
            if (print_res == True):
                print("\n--------------------------------------------------------")
                print(f"Found fit less than {tol_prob} for week --> {nxx_test}")
                print(f"Wave 1 ending value --> {wave1_end}\n")
            # Return value back to function       
            return wave1_end, bb_w2 
            break
        
        # (max_ndat-1) to match python's indexing
        iterr+=1

        if(nxx_test >= (max_ndat-1)):
            break
    # end of While loop

# Function to finalize the values of wave 1
def finalize_wave1(wave1_end, week_agg, timestamp_week, method, show_res):
    # Clear working variables
    bb_w1 = []
    y_pred_w1 = []
    y_true_w1 = []
    t_span_w1 = []

    # Check if fitting model is defined by the user or not
    if (method == 'default'):
        method_user = 'least_squares'
    else:
        method_user = method

    bb_w1, y_pred_w1, y_true_w1, t_span_w1 = fit_wave(wave1_end, week_agg, timestamp_week, [], show_rep = False, method=method)
    
    # Calcualte scores
    rr2_w1, rmse_score = calcStats(y_pred_w1, y_true_w1, print_debug = False)
    
    if (show_res == True):
        # Print out wave 1 result
        print("\n-------------------------------------------------------------------")
        print(f"Wave 1 ended in {wave1_end} weeks and the nonlinear fit has R2 score of --> {rr2_w1}")
        #print(f"Number of samples for debug --> {len(y_pred_w1)}")
    
    return bb_w1, y_pred_w1, y_true_w1, t_span_w1 

# This function is different from first fit_wave() code
def fit_wave2(b00, y_w2, t_w2, show_rep, method):
    y_w2 = np.asarray(y_w2)
    t_w2 = np.asarray(t_w2)

    #print(t_w2) # Sanity check
    
    # Unpack variables
    KK = b00[0]
    rr = b00[1]
    AA = b00[2]

    fit_params = Parameters() # Define Parameter class from LMFIT
    fit_params.add('K', value= KK, min = 1, max = 10000000)
    fit_params.add('r', value= rr, min = 0.1, max = 1000)
    #fit_params.add('r', value= 0.5)
    fit_params.add('A', value = AA, min = 1, max = 10000000)

    # Here we will get the fit
    # method -- nedler, monte-carlo, leastsq
    fit_yy = minimize(objFun, fit_params, args=(t_w2,), kws={'data': y_w2}, method = method)
    
    if (show_rep == True):
        print(fit_report(fit_yy))
    
    # Extract results
    Kopt = fit_yy.params['K'].value
    ropt = fit_yy.params['r'].value
    Aopt = fit_yy.params['A'].value

    b_fit = [Kopt, ropt, Aopt]
    
    # Calcuate predicted values
    yy_hat = logisticFun(t_w2,b_fit)

    return b_fit, yy_hat

# Major function to finalize nonlinear fit for wave 2
def finalize_wave2(bb_w1,tt_w2, week_agg, slice_index, print_debug):
    # Set working variables
    y_pred_w2 = []
    bb_fit2 = []
    
    b00_w2 = bb_w1 # Initial guess for bb_w2 will be final fit for wave 1
    t_span_w2 = tt_w2 # Load the timespan array for wave 2
    #print(t_span_w2)
    
    w2_idx = slice_index
    y_true_w2 = week_agg[w2_idx:] # Slice out remaining data portion

    method_names = ['least_sqaures', 'nelder', 'leastsq', 'cg', 'bfgs', 'dual_annealing', 'lbfgsb']
    mm_final = [] # This variable will contain the name of the method we will use to chose the best fitting method

    for kk in method_names:
        y_pred_w2 = []

        bb_fit2, y_pred_w2 = fit_wave2(b00_w2, y_true_w2, t_span_w2, show_rep = False, method = kk)
        rr_temp = calcR2(y_true_w2, y_pred_w2)
        rr_temp = round(rr_temp, 2)

        if (print_debug == True):
            print(f"\nMethod --> {kk}, bb_fit2 --> {bb_fit2}, R2 --> {rr_temp}\n")
            #visualize_result(y_true_w2, y_pred_w2, t_span_w2, (15,5))
        
        if ((rr_temp> 0.80 and rr_temp<=0.99) or (rr_temp == 1.0)):
            #print(f"Method {kk} gives acceptable fit")
            mm_final = kk
            #print(f"\nMethod chosen --> {kk}\n")
            break
        else:
             # We don't have a choice so go with least_squares
             mm_final = 'dual_annealing'

    # Perform final fit with the best chosen model
    y_pred_w2 = []
    bb_fit2 = []
    bb_fit2, y_pred_w2 = fit_wave2(b00_w2, y_true_w2, t_span_w2, show_rep = False, method = mm_final)
    
    # Calcualte scores
    rr2_w2, rmse_score_w2 = calcStats(y_true_w2, y_pred_w2, print_debug = False)

    if (print_debug == True):
        print(f"\nNumber of week data to fit in 2nd wave --> {len(y_true_w2)}") # Sanity check
        print(f"2nd wave fit R2 score --> {rr2_w2} and RMSE --> {rmse_score_w2}")

    return bb_fit2, y_true_w2, y_pred_w2, t_span_w2

# Calculate R2 Score
def calcR2(y_true, y_pred):
    r2score = r2_score(y_true, y_pred)
    return r2score

# Calculate Root Mean Squared Error
def calcRMSE(y_true, y_pred):
    rmse = sqrt(mean_squared_error(y_true, y_pred))
    return rmse

# Calculate statistics
# Input 
# y_true, y_pred
# Output
# rr2 -- r2 score (0 <= score <=1 )
# rmse_score -- Room mean squared error
# Explanation of R2 is here https://www.youtube.com/watch?v=hdKrUoeUQjE
def calcStats(y_true, y_pred, print_debug):

    rr2 = calcR2(y_true, y_pred)
    rmse_score = calcRMSE(y_true,y_pred)
    
    # Rounding to 2 decimal places
    rr2 = round(rr2, 2)

    if (print_debug == True):
        print(f"R2 score --> {rr2}")
        print(f"RMSE score --> {rmse_score}")

    return rr2, rmse_score

# Visualize fitted plot
# Function to visualize fitted plot
def visualize_result(y_true,y_pred,tspan,size_tup):
    fig = plt.figure(figsize=size_tup)
    plt.scatter(tspan, y_true, color = 'red')
    plt.plot(tspan, y_pred, color = 'blue')
    plt.show()
# ------------------------------------------------------------------

# ---------------------------------------------------------------------------------- 

# Function which scales my timestamp_week to match the week scale used by Dr. Satya
def scale_to_dat_week(w_loc_ls, const_add, max_w):
    new_loc_ls = []
    new_end_val = max_w + const_add # The final week value also needs to shift by 3
    # print(f"New end val --> {new_end_val}")
    idx = range(len(w_loc_ls))
    for k in idx:
        # print(f"Current week before scaling --> {w_loc_ls[k]}")
        lls = w_loc_ls[k]
        lls = lls + const_add
        # print(f"Value after scaling --> {lls}")
        
        # Over the final week value
        if (lls > new_end_val):
            lls = new_end_val
        # Was it zero?
        if (w_loc_ls[k] == 0):
            lls = 0
        # Add to the new array
        
        # print(f"Value before new list --> {lls}\n")
        new_loc_ls.append(lls)
    return new_loc_ls

# --------------------------------------------------------------------

def translate(value, leftMin, leftMax, rightMin, rightMax):
    # Figure out how 'wide' each range is
    leftSpan = leftMax - leftMin
    rightSpan = rightMax - rightMin

    # Convert the left range into a 0-1 range (float)
    valueScaled = float(value - leftMin) / float(leftSpan)

    # Convert the 0-1 range into a value in the right range.
    return rightMin + (valueScaled * rightSpan)

# ----------------------------------------------------------------------

def row_map(lss, l_min, l_max, r_min, r_max):
    out_row = []
    for k in lss:
        val = translate(k, l_min, l_max, r_min, r_max)
        val = int(val) # We need integers
        out_row.append(val)
    return out_row

# ------------------------------------------------------------------------


# ----------------------------------------------------------------------------------

def phase_2_correct(w2_start, l_min, l_max, r_min, r_max):
    pass

# Main function to determine phase location points from two logistic waves
def find_phases_multi(bb_fit1, bb_fit2, t_span_w1, t_span_w2, print_res, scale_val):
    
    # Initiate all phase variables to avoid local variable 'referenced before assignment' before assigned
    w1_tpeak, w1_tp2, w1_tp3, w1_tp4, w1_tpend = 0,0,0,0,0
    w2_tpeak, w2_tp2, w2_tp3, w2_tp4, w2_tpend = 0,0,0,0,0
    # w1_phase1_x, w1_phase2_x, w1_phase3_x, w1_phase4_x, w1_phase5_x = 0,0,0,0,0
    # w2_phase1_x, w2_phase2_x, w2_phase3_x, w2_phase4_x, w2_phase5_x = 0,0,0,0,0
    w1_loc_ls = []
    w2_loc_ls = []
    max_t_span_1 = max(t_span_w1)
    max_t_span_2 = max(t_span_w2)

    # --------------------------------------------------------------------

    # Unpack variables for wave 1
    w1_K = bb_fit1[0]
    w1_r = bb_fit1[1]
    w1_A = bb_fit1[2]

    # Unpack variables for wave 2
    w2_K = bb_fit2[0]
    w2_r = bb_fit2[1]
    w2_A = bb_fit2[2]

    # Find locations from each two waves
    # array structure [tp2,tpeak,tp3,tp4,tend]
    w1_loc_ls = find_phase_loc(2, w1_K, w1_r, w1_A, t_span_w1, print_res = print_res)
    l_min = 0
    l_max = 0
    #w2_loc_ls = phase_2_correct()
    
    w2_loc_ls = find_phase_loc(1, w2_K, w2_r, w2_A, t_span_w2, print_res = print_res)

    # print(w1_loc_ls)

    # Brint to scale
    # w1_loc_ls = scale_to_dat_week(w1_loc_ls, scale_val, max_t_span_1)
    # w2_loc_ls = scale_to_dat_week(w2_loc_ls, scale_val, max_t_span_1)
    # print(w1_loc_ls)

    return w1_loc_ls, w2_loc_ls

# ----------------------------------------------------------------------

# Helper function to ensure the values of phase 2 are not crossing over to phase 1
# Unpack w2_loc_ls to test if some of the values are crossing over to wave 1 time span
def test_phase2(w2_loc_ls, bbfit_2, max_t_span_1):
    # Upack variables
    tp2 = w2_loc_ls[0]
    tpeak = w2_loc_ls[1]
    tp3 = w2_loc_ls[2]
    tp4 = w2_loc_ls[3]
    tpend = w2_loc_ls[4]

    # Unpack variables for wave 2
    w2_K = bb_fit2[0]
    w2_r = bb_fit2[1]
    w2_A = bb_fit2[2]

    # We will resume from here after getting some more information (10/10/2020)


# Master function to visualize phase plot lines on a plot which has same x axis lenght
def plot_phase_lines(w1_loc_ls, w2_loc_ls, max_y_act):
    
    yyline = max_y_act / 2 # To place texts in middle of the vertical lines
    
    # --------------------------------------------------- Wave 1 Phases ------------------------------------------
    if (w1_loc_ls[0] > 0):
        plt.axvline(x = w1_loc_ls[0], lw = 2.5, color = 'black')
        plt.text(w1_loc_ls[0], yyline, 'w1_tp2', ha='center', va='center',rotation='horizontal', backgroundcolor='white')
    
    if (w1_loc_ls[1] > 0):
        plt.axvline(x = w1_loc_ls[1], lw = 2.5, color = 'red')
        plt.text(w1_loc_ls[1], yyline, 'w1_tpeak', ha='center', va='center',rotation='horizontal', backgroundcolor='white')
    
    if (w1_loc_ls[2] > 0):
        plt.axvline(x = w1_loc_ls[2], lw = 2.5, color = 'black')
        plt.text(w1_loc_ls[2], yyline, 'w1_tp3', ha='center', va='center',rotation='horizontal', backgroundcolor='white')

    if (w1_loc_ls[3] > 0):
        plt.axvline(x = w1_loc_ls[3], lw = 2.5, color = 'black')
        plt.text(w1_loc_ls[3], yyline, 'w1_tp4', ha='center', va='center',rotation='horizontal', backgroundcolor='white')

    if (w1_loc_ls[4] > 0):
        plt.axvline(x = w1_loc_ls[4], lw = 2.5, color = 'black')
        plt.text(w1_loc_ls[4], yyline, 'w1_tpend', ha='center', va='center',rotation='horizontal', backgroundcolor='white')

    # --------------------------------------------------- Wave 2 Phases ------------------------------------------

    if (w2_loc_ls[0] > 0):
        plt.axvline(x = w2_loc_ls[0], lw = 2.5, color = 'black')
        plt.text(w2_loc_ls[0], yyline, 'w2_tp2', ha='center', va='center',rotation='horizontal', backgroundcolor='white')

    if (w2_loc_ls[1] > 0):
        plt.axvline(x = w2_loc_ls[1], lw = 2.5, color = 'red')
        plt.text(w2_loc_ls[1], yyline, 'w2_tpeak', ha='center', va='center',rotation='horizontal', backgroundcolor='white')

    if (w2_loc_ls[2] > 0):
        plt.axvline(x = w2_loc_ls[2], lw = 2.5, color = 'black')
        plt.text(w2_loc_ls[2], yyline, 'w2_tp3', ha='center', va='center',rotation='horizontal', backgroundcolor='white')

    if (w2_loc_ls[3] > 0):
        plt.axvline(x = w2_loc_ls[3], lw = 2.5, color = 'black')
        plt.text(w2_loc_ls[3], yyline, 'w2_tp4', ha='center', va='center',rotation='horizontal', backgroundcolor='white')

    if (w2_loc_ls[4] > 0):
        plt.axvline(x = w2_loc_ls[4], lw = 2.5, color = 'black')
        plt.text(w2_loc_ls[4], yyline, 'w2_tpend', ha='center', va='center',rotation='horizontal', backgroundcolor='white')


# Master plotting function
# Also has 
def plot_fit(plot_title, t_cumu, y_act, y_pred, size_tup):
    # plot-title is a string
    
    fig = plt.figure(figsize=size_tup)

    plt.scatter(t_cumu, y_act, color = 'red', label = 'Actual')
    plt.plot(t_cumu, y_pred, color = 'blue', label = 'Model')
    plt.title(plot_title)
    plt.legend()
    #plt.show()
    # Plot legends will go in here later

# Major function which calculates incidence of a wave
# Modified in version 2.5
def fit_incidence_multiwave(bunpack, bunfit, ret_indv_wave):
    # P_act_w1 = Prevelance for week 1 -- numpy array
    # P_tt_w1 = Timespan for Prevelance for week 1 -- numpy array
    #bunfit = [bb_fit1, bb_fit2]
    #bunpack = [y_true_w1, y_pred_w1, t_span_w1, y_true_w2, y_pred_w2, t_span_w2]
    # ret_indv_wave -- Boolean which tells to return individual wave incidence instead of total
    
    # Unravel working variables
    P_act_w1 = bunpack[0]
    P_act_w2 = bunpack[3]
    tt_w1 =  bunpack[2]
    tt_w2 = bunpack[5]
    param_w1 = bunfit[0]
    param_w2 = bunfit[1]
    
    # Calculate actual change in cases for multiple waves
    # # Trick from Dr. Batista's work, copy the first element twice to match sample size
    I_act_w1 = np.diff(P_act_w1)    
    I_act_w1 = np.insert(I_act_w1, [0], [I_act_w1[0]])
    
    I_act_w2 = np.diff(P_act_w2)
    I_act_w2 = np.insert(I_act_w2, [0], [I_act_w2[0]])

    # Calcualte rate of changes using model
    I_pred_w1 = logisticFun2(tt_w1, param_w1, P_act_w1)
    I_pred_w2 = logisticFun2(tt_w2, param_w2, P_act_w2)
    
    if (ret_indv_wave == True):
        return I_pred_w1, I_pred_w2

    # Recombine piecewised actual change in cases array
    I_act = np.concatenate((I_act_w1, I_act_w2))

    # Combine predictions into one array
    I_pred = np.concatenate((I_pred_w1, I_pred_w2)) # Arrays to concatenate must be inside a tuple
    
    # print(I_act)
    # print(I_pred)

    # Recombine piecewised time arrays into one array
    t_cumu = np.concatenate((tt_w1,tt_w2))

    #return t_cumu, I_act, I_pred
    return I_act, I_pred


########################## END FUNCTION DEFINTIONS ######################################
