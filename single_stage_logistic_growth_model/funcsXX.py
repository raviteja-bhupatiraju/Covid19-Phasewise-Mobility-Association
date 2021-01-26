# User-defined function file for python fitVirusXX
# Date Modified: 01.26.2021
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

# -----------------------------------------------------------------------------------
def master_fn(df, state_name, const_add, debug, plot_res):
    # Master function to do all the caluclation stuff. 
    # Input will be state data, state name,
    # Step 0 Define working variable and constants
    min_ndat = 7 # This is emperical, may be subject to change or determination using another technique
    tol_single_wave = 0.90
    week_agg = []
    timestamp_week = []
    bunpack = [] # Array to hold actual, predicted and timespans for two waves
    bunfit = [] # Array to fitted parameters for two waves

    # Step 1 Load weekly data and time index
    week_agg, timestamp_week = choose_state(df, state_name)
    max_ndat = len(week_agg) # This is a global variable telling how many week of data we have
    
    if(debug == True):
        print(f"\nState --> {state_name}")
        print(f"Number of weeks data to fit is {max_ndat}\n")
        # print(week_agg)
    
    # Step 2 Perform non linear fit
    b0_init = [] # We don't have any initial values to pass for first wave test
    b_wave1, rr2_wave1, rmse_wave1, y_pred = fit_wave(b0_init, week_agg, timestamp_week, nx_test= 0, plot_res= False)
    
    # Print Debug
    if(debug == True):
        print(f"R2 score for test for singe wave fit --> {rr2_wave1}")
        print(f"RMSE score for test for single wave fit --> {rmse_wave1}\n")
    
    # Return output
    return b_wave1, timestamp_week, week_agg, y_pred, rr2_wave1
# -----------------------------------------------------------------------------------

# -----------------------------------------------------------------------------------
def scale_to_dat_week(w_loc_ls, const_add, max_w):
    # Function which scales my timestamp_week to match the week scale used by Dr. Satya
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
# -----------------------------------------------------------------------------------

# -----------------------------------------------------------------------------------
def fit_single_phase(bb_fit1, t_span_w1, print_res):
    w1_tpeak, w1_tp2, w1_tp3, w1_tp4, w1_tpend = 0,0,0,0,0
    w1_loc_ls = []

    # Unpack variables for wave 1
    w1_K = bb_fit1[0]
    w1_r = bb_fit1[1]
    w1_A = bb_fit1[2]

    w1_loc_ls = find_phase_loc(w1_K, w1_r, w1_A, t_span_w1, print_res = print_res)
    return w1_loc_ls
# -----------------------------------------------------------------------------------

# -----------------------------------------------------------------------------------
def fit_incidence_single(y_act, t_cumu, param_w1):
    P_act_w1 = y_act
    tt_w1 =  t_cumu
    I_act_w1 = np.diff(P_act_w1)
    I_act_w1 = np.insert(I_act_w1, [0], [I_act_w1[0]])
    I_pred_w1 = logisticFun2(tt_w1, param_w1, P_act_w1)

    return I_pred_w1
# -----------------------------------------------------------------------------------


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
def week_aggregate(sampleC):
    # Corrected week-level aggregation
    # week_agg -- array of cumulative cases in 'week' level
    # timestamp_week -- array of time in week level
    # note -- days which don't form a complete week are truncated
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
# -----------------------------------------------------------------------------------

# -----------------------------------------------------------------------------------
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
# -----------------------------------------------------------------------------------

# -----------------------------------------------------------------------------------
def perform_fit(ndat, week_agg, tspan, b00, show_rep, method):
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
# -----------------------------------------------------------------------------------

# -----------------------------------------------------------------------------------
def fit_wave(b00, week_agg, tspan, nx_test, plot_res):
    # Redundant function
    # Output R2 score
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
    b0, y_pred, y_true, t_span = perform_fit(ndat = ndat, week_agg= P, tspan=t, b00 = b01, show_rep = False, method = 'bfgs')
    #print(b0) # Testing to see if the fit is working properly

    # Calcualte scores
    y_true = P # For sake of matching function defintions
    rr2, rmse_score = calcStats(y_true, y_pred, print_debug = False)

    # Optional, graph result
    if (plot_res == True):
        visualize_result(y_true,y_pred,t,size_tup)

    return b0, rr2, rmse_score, y_pred
# -----------------------------------------------------------------------------------

# -----------------------------------------------------------------------------------
def objFun(params,tspan, data):
    # Objective function, new version for lmfit
    # params -- values of K, r and A [array with three elements]
    #  t -- timestamp [array]
    # data -- actual value at t [array]
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
# -----------------------------------------------------------------------------------

# -----------------------------------------------------------------------------------
def logisticFun(t, b):
    # Function to calculate logisitic growth for given parameters and timeseries
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
# -----------------------------------------------------------------------------------

# -----------------------------------------------------------------------------------
def logit(t,K,r,A):
    # Generalized logistic growht model, eqn (2) from Dr. Batista's first paper
    # Helper fucntion for logisticFun(t,b)
    # K = b[0]
    # r = b[1]
    # I0 = b[2]
    f = K/ (1 + A * exp(- (r * t)))
    return f
# -----------------------------------------------------------------------------------

# -----------------------------------------------------------------------------------
def logisticFun2(t, b, C):
    # Vectorized version for calculating incidence
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
# -----------------------------------------------------------------------------------

# -----------------------------------------------------------------------------------
def logit_incidence(t,K,r,A,C):
    # Helper function for logisiticFun2
    dy1 = r * C
    dy2 = (1 - (C/K))
    dy = dy1 * dy2
    return dy
# -----------------------------------------------------------------------------------

#-----------------------------------------------------------------------------------
def logit_incidence_2(t,K,r,I0):
    # Helper function for logisiticFun2
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
#-----------------------------------------------------------------------------------

# Calculate R2 Score
def calcR2(y_true, y_pred):
    r2score = r2_score(y_true, y_pred)
    return r2score

# Calculate Root Mean Squared Error
def calcRMSE(y_true, y_pred):
    rmse = sqrt(mean_squared_error(y_true, y_pred))
    return rmse

#-----------------------------------------------------------------------------------
def calcStats(y_true, y_pred, print_debug):
    # Calculate statistics
    # Input 
    # y_true, y_pred
    # Output
    # rr2 -- r2 score (0 <= score <=1 )
    # rmse_score -- Room mean squared error
    # Explanation of R2 is here https://www.youtube.com/watch?v=hdKrUoeUQjE
    rr2 = calcR2(y_true, y_pred)
    rmse_score = calcRMSE(y_true,y_pred)
    
    # Rounding to 2 decimal places
    rr2 = round(rr2, 2)

    if (print_debug == True):
        print(f"R2 score --> {rr2}")
        print(f"RMSE score --> {rmse_score}")

    return rr2, rmse_score
#-----------------------------------------------------------------------------------

#-----------------------------------------------------------------------------------
def find_phase_loc(w_K, w_r, w_A, t_span, print_res):
    # Helper function to find phase starting and ending point locations
    tpeak, tp2, tp3, tp4, tpend = 0,0,0,0,0
    loc_ls = []
    start_loc = t_span[0]
    tpeak = int(log(w_A)/w_r)
    aa = (2/w_r)
    tp2 = int(tpeak - (2/w_r))

    if (tp2>0):
        # Calculate tp3
        tp3 = int(tpeak + (2/w_r))
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
        print(f"tp3 --> {tp3}")
        print(f"tp4 --> {tp4}")
        print(f"tpend --> {tpend}\n")
    # Send back locations points
    return loc_ls
#-----------------------------------------------------------------------------------

#-----------------------------------------------------------------------------------
def plot_cumu_phases(const_add, w1_loc_ls, params, timestamp, sampleC, yypred,I_change, fontSIZE):
    ## Plot figure with phases described by Dr Batista
    Kopt = params[0]
    ropt = params[1]
    Aopt = params[2]

    neg_lim = -4 # To make sure we can see the yaxis scale clearly
    hhh = max(sampleC) + 10

    fig = plt.figure(figsize=(15,7))
    
    # Fix x and y limits
    #plt.xlim(0,60)
    #plt.ylim(0,1)

    #plt.xlim(neg_lim,(len(timestamp)+1))
    #plt.ylim(neg_lim,hhh, max(sampleC) + 20)
    
    ax = fig.add_subplot(1,1,1) # one row, one column, first plot

    # Perform min-max normalization
    y_act_scaled = min_max_norm(sampleC)
    y_pred_scaled = min_max_norm(yypred)
    I_pred_scaled = min_max_norm(I_change)

    # HARDCODED, change to change color of all curve elements i.e. logistic curve, cumulative curve
    # Draw data points, logistic growth curve fit and logistic growth rate
    #plt.scatter(timestamp, y_act_scaled, 'ko', label = 'Data')
    ax.plot(timestamp, y_act_scaled, 'o', color='black', label = 'Data', lw = 2.5)
    ax.plot(timestamp, y_pred_scaled, 'b-', color='#A5C8E4', label = 'Cumulative fit',lw = 2.5)
    ax.plot(timestamp, I_pred_scaled, 'r-', color='black', label = 'Rate of change fit',lw = 2.5)

    # tpeak = int(log(Aopt)/ropt)
    tpeak = w1_loc_ls[1] + const_add
    #print(tpeak)

    # tp2 = int(tpeak - (2/ropt))
    # tp3 = int(tpeak + (2/ropt))
    # tp4 = int(tpeak + 2 * (2/ropt))

    # Adding const_add to bring to correct scale
    tp2 = w1_loc_ls[0] + const_add
    tp3 = w1_loc_ls[2] + const_add
    tp4 = w1_loc_ls[3] + const_add

    # Calculate phases
    phase1_x = abs(0 - tp2)
    phase2_x = abs(tp2 - tpeak)
    phase3_x = abs(tpeak - tp3)
    phase4_x = abs(tp3 - tp4)
    phase5_x = abs(tp4 - len(timestamp))

    hhh_adjust = hhh + (-1 * neg_lim) # To make sure all the rectangles are of the same height

    # HARDCODED, change hex values to change color
    # https://www.schemecolor.com/pastel-calm.php
    # RGB color values for each rectangles
    # Hex values found using google's rgb to hex converter
    # Type "rgb to hex" in a google search bar to bring up the pallet

    # Color schema similar to Dr. Batista's work
    # phase1_color = '#d8d4d9'
    # phase2_color = '#ed8a58'
    # phase3_color = '#e173eb'
    # phase4_color = '#f7f494'
    # phase5_color = '#b4f781'

    # To reproduce colors from paper
    phase1_color = '#FFFFFF'
    phase2_color = '#E0E0E0'
    phase3_color = '#FFFFFF'
    phase4_color = '#FFFFFF'
    phase5_color = '#FFFFFF'

    # Create rectagle objects
    phase_rect1 = patches.Rectangle((0,neg_lim), phase1_x, hhh_adjust, color=phase1_color)
    phase_rect2 = patches.Rectangle((tp2,neg_lim), phase2_x, hhh_adjust, color=phase2_color)
    phase_rect3 = patches.Rectangle((tpeak,neg_lim), phase3_x, hhh_adjust, color=phase3_color)
    phase_rect4 = patches.Rectangle((tp3,neg_lim), phase4_x, hhh_adjust, color=phase4_color)
    phase_rect5 = patches.Rectangle((tp4,neg_lim), phase5_x, hhh_adjust, color=phase5_color)

    # Add rectangle objects
    ax.add_patch(phase_rect1)
    ax.add_patch(phase_rect2)
    ax.add_patch(phase_rect3)
    ax.add_patch(phase_rect4)
    ax.add_patch(phase_rect5)


    # Add textbox to name each rectangle
    color_list = [phase1_color,phase2_color,phase3_color,phase4_color,phase5_color] # List to hold each rectangle's color

    xxx_text_box = [(phase1_x/2),(tpeak - phase2_x/2),(tpeak + phase3_x/2),(tp4 - phase4_x/2),(tp4 + phase5_x/2)] # X coordinates for each textbox

    phase_str_list = ['1','2','3','4','5'] # Strings to show in the textbox

    # HARDCODED
    hhh_text_box = max(y_act_scaled) + 0.1

    # Loop to print textboxes
    # for kk in range(5):
    #     #textstr = 'Phase' + " " + str(kk) # Doesn't fit in the rectangles
    #     textstr = phase_str_list[kk]
    #     props = dict(boxstyle='round', facecolor='azure', alpha=0.5)
    #     ax.text(xxx_text_box[kk],hhh_text_box, textstr, fontsize=15, verticalalignment='top', bbox=props)
    
    # Print a red line which shows at which day, the peak growth rate occured 
    # HARDCODED, label
    plt.axvline(x = tpeak, lw = 1, linestyle="dashed", color = 'black', label = 'Max growth rate at week {} '.format(tpeak))
    ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.1), shadow=True, ncol=4)

    plt.axvline(x = tp2, lw = 1, color = 'black', linestyle="dashed", label = 'tp2')
    ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.1), shadow=True, ncol=4)

    plt.axvline(x = tp3, lw = 1, color = 'black', linestyle="dashed", label = 'tp3')
    ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.1), shadow=True, ncol=4)

    plt.axvline(x = tp4, lw = 1, color = 'black', linestyle="dashed", label = 'tp4')
    ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.1), shadow=True, ncol=4, prop={'size': fontSIZE})

    # Add axis label
    plt.xlabel("Time")
    plt.ylabel("Normalized Prevelance / Incidence")
    
    # Add plot label
    #plt.title("Logistic Growth fit for Arizona")

    # Save figure
    # font = {'family' : 'normal',
    #     'weight' : 'bold',
    #     'size'   : 22}

    #plt.rc('font', **font)

    plt.savefig('combined_phase.png', bbox_inches='tight')
    plt.savefig("combined_phase.svg", format = 'png', dpi=600, bbox_inches='tight')
    plt.show()
#-----------------------------------------------------------------------------------

#-----------------------------------------------------------------------------------
def min_max_norm(x):
    # Reference - https://learn.64bitdragon.com/articles/computer-science/data-processing/min-max-normalization
    min_num = np.min(x)
    max_num = np.max(x)
    range_num = max_num - min_num
    aa = [((a - min_num) / range_num) for a in x]
    aa = np.asarray(aa)
    return aa
#-----------------------------------------------------------------------------------



########################## END OF FILE ######################################
