'''
Get PMF parameters

Contains snippets of code originally written by Lukas Oesch and Joao Couto.
Adapted by Gabriel Rojas Bowe.

April 2023
'''


#%% Import modules
from os import path, makedirs
from glob import glob
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from utils import *

#%% Get .h5 files
file_names = pick_files_multi_session("chipmunk", "*.h5") #first we're gonna do this for just one session, and then we'll scale up to multisession analyses
session_data = pd.read_hdf(file_names[0]) # the task data are in the "triallog.h5" files

#%% Print out session metadata
ntrials = len(session_data)
print('The subject was doing the task for {0} trials.\n'.format(ntrials))  # the print function lets you display stuff, it is very useful for debugging 

ntrials_with_choice = len(session_data[session_data['outcome_record'].isin([0, 1])])
print('Out of {0} trials, the subject responded left or right in {1} trials.\n'.format(ntrials,ntrials_with_choice))

freq_presented = set(len(trial_stim_events) for trial_stim_events in session_data.stimulus_event_timestamps)
print(f'There are {len(freq_presented)} stimuli conditions: {sorted(freq_presented)}')

#%% Plot performance data
sel = session_data[session_data.response_side.isin([0,1])] # select only trials where the subject responded
sel = sel.reset_index(drop=True)
responded_right = np.array(sel.response_side == 1).astype(int) # select the response = 1 (i.e. the left side) and cast to integer datatype (number)   

rate_list = np.array([len(timestamps) for timestamps in sel.stimulus_event_timestamps])

frequencies = np.array(list(freq_presented))  # the stimulus intensity values
p_right = np.zeros_like(frequencies,dtype=float)     # pre-allocate the array (fill with zeros when you know the size)
# note that p_left is cast to float so it can take fractional numbers
# this is the part where we estimate the probability of left lick
for i,frequency in enumerate(frequencies):
    p_right[i] = np.sum(responded_right[rate_list == frequency])/np.count_nonzero(rate_list == frequency)

fig = plt.figure()
ax = fig.add_axes([0.2,0.2,0.7,0.7])
ax.plot(frequencies,p_right,'ko')
ax.set_xticks([4, 8, 12, 16, 20])

ax.vlines(12,0,1,color = 'k',lw = 0.3) # plot a vertical line as reference at zero
ax.hlines(0.5,np.min(frequencies),np.max(frequencies),color = 'k',lw = 0.3) # plot an horizontal line as reference for chance performance

ax.set_ylabel('P$_{right}$',fontsize = 14)  # set the y-axis label with latex nomenclature
ax.set_xlabel('Stimulus frequency (event rate)', fontsize = 14); # set the x-axis label

#%% Fit PMF using Wilson-Binomial fit

# first get the average points like above
frequencies = np.sort(np.array(list(freq_presented)).astype(float))
p_right = np.zeros_like(frequencies,dtype=float) 
ci_right = np.zeros((len(frequencies),2),dtype=float)

from statsmodels.stats.proportion import proportion_confint
for i,frequency in enumerate(frequencies):
    cnt = np.sum(responded_right[rate_list == frequency]) # number of times the subject licked left 
    nobs = np.count_nonzero(rate_list == frequency) # number of observations (ntrials)
    p_right[i] = cnt/nobs
    ci_right[i] = proportion_confint(cnt,nobs,method='wilson') # 95% confidence interval

def cumulative_gaussian(alpha,beta,gamma,lmbda, X):
    '''
    Evaluate the cumulative gaussian psychometric function.
       alpha is the bias (left or right)
       beta is the slope
       gamma is the left handside offset
       lmbda is the right handside offset
      
    Adapted from the Palamedes toolbox 
    Joao Couto - Jan 2022    
    '''
        
    from scipy.special import erfc # import the complementary error function
    return  gamma + (1 - gamma - lmbda)*0.5*erfc(-beta*(X-alpha)/np.sqrt(2))+1e-9    

from scipy.optimize import minimize

def neg_log_likelihood_error(func, parameters, X, Y):
    '''
    Compute the log likelihood

    'func' is the (psychometric) function 
    'parameters' are the input parameters to 'func'
    'Y' is the binary response (correct = 1; incorrect=0)
    '''

    pX = func(*parameters, X)*0.99 + 0.005  # the predicted performance for X from the PMF
    # epsilon to prevent error in log(0)
    val = np.nansum(Y*np.log(pX) + (1-Y)*np.log(1-pX))
    return -1*val
#
rate_list = np.array(rate_list)
func = lambda pars: neg_log_likelihood_error(cumulative_gaussian, pars, rate_list.astype(float),responded_right.astype(float))
# x0 is the initial guess for the fit, it is an important parameter
x0 = [12.,0.1,p_right[0],1 - p_right[-1]]

bounds = [(frequencies[0],frequencies[-1]),(0.0001,10),(0,0.7),(0,0.7)]

res = minimize(func, x0, options = dict(maxiter = 500*len(x0),adaptive=True),
               bounds = bounds, method='Nelder-Mead') # method = 'L-BFGS-B'

bias = res.x[0] #horizontal displacement
sensitivity = res.x[1] #slope
left_lapse = res.x[2] #gamma
right_lapse = res.x[3] #lambda

biases = []
sensitivities = []
left_lapses = []
right_lapses =[]

biases.append(bias)
sensitivities.append(sensitivity)
left_lapses.append(left_lapse)
right_lapses.append(right_lapse)

fig = plt.figure()
ax = fig.add_axes([0.2,0.2,0.7,0.7])
ax.vlines(12,0,1,color = 'k',lw = 0.3) # plot a vertical line as reference at zero
ax.hlines(0.5,np.min(frequencies),np.max(frequencies),color = 'k',lw = 0.3) # plot an horizontal line as reference for chance performance
ax.set_xticks([4, 8, 12, 16, 20])

# plot the fit
nx = np.linspace(np.min(frequencies),np.max(frequencies),100)
ax.plot(nx,cumulative_gaussian(*res.x,nx),'k')

# plot the observed data and confidence intervals
for i,e in zip(frequencies,ci_right):  # plot the confidence intervals
    ax.plot(i*np.array([1,1]),e,'_-',lw=0.5,color = 'black')
    
ax.plot(frequencies,p_right,'ko',markerfacecolor = 'lightgray',markersize = 6)

ax.set_ylabel('P$_{right}$',fontsize = 18)  # set the y-axis label with latex nomenclature
ax.set_xlabel('Stimulus rate', fontsize = 14); # set the x-axis label
print('Estimated parameters: alpha {0:2.2f} beta {1:2.2f} gamma {2:2.2f} lambda {3:2.2f}'.format(*res.x))

#%% Get PMF parameters from multiple sessions

def get_PMF_parameters_multi_session(file_names):
    biases = []
    sensitivities = []
    left_lapses = []
    right_lapses =[]

    for file in file_names:
        session_data = pd.read_hdf(file)
        sel = session_data[session_data.response_side.isin([0,1])] # select only trials where the subject responded
        sel = sel.reset_index(drop=True)
        responded_right = np.array(sel.response_side == 1).astype(int) # select the response = 1 (i.e. the left side) and cast to integer datatype (number)   

        rate_list = np.array([len(timestamps) for timestamps in sel.stimulus_event_timestamps])
        freq_presented = set(len(trial_stim_events) for trial_stim_events in session_data.stimulus_event_timestamps)

        if len(freq_presented)<6: #QC: plots sessions with [4 6 8 10 12 14 16 18 20] Hz
            continue
        else:
            frequencies = np.sort(np.array(list(freq_presented)).astype(float))
            p_right = np.zeros_like(frequencies,dtype=float) 
            ci_right = np.zeros((len(frequencies),2),dtype=float)

            from statsmodels.stats.proportion import proportion_confint
            for i,frequency in enumerate(frequencies):
                cnt = np.sum(responded_right[rate_list == frequency]) # number of times the subject licked left 
                nobs = np.count_nonzero(rate_list == frequency) # number of observations (ntrials)
                p_right[i] = cnt/nobs
                ci_right[i] = proportion_confint(cnt,nobs,method='wilson') # 95% confidence interval

            def cumulative_gaussian(alpha,beta,gamma,lmbda, X):
                '''
                Evaluate the cumulative gaussian psychometric function.
                alpha is the bias (left or right)
                beta is the slope
                gamma is the left handside offset
                lmbda is the right handside offset
                
                Adapted from the Palamedes toolbox 
                Joao Couto - Jan 2022    
                '''
                    
                from scipy.special import erfc # import the complementary error function
                return  gamma + (1 - gamma - lmbda)*0.5*erfc(-beta*(X-alpha)/np.sqrt(2))+1e-9    

            from scipy.optimize import minimize

            def neg_log_likelihood_error(func, parameters, X, Y):
                '''
                Compute the log likelihood

                'func' is the (psychometric) function 
                'parameters' are the input parameters to 'func'
                'Y' is the binary response (correct = 1; incorrect=0)
                '''

                pX = func(*parameters, X)*0.99 + 0.005  # the predicted performance for X from the PMF
                # epsilon to prevent error in log(0)
                val = np.nansum(Y*np.log(pX) + (1-Y)*np.log(1-pX))
                return -1*val
            #
            rate_list = np.array(rate_list)
            func = lambda pars: neg_log_likelihood_error(cumulative_gaussian, pars, rate_list.astype(float),responded_right.astype(float))
            # x0 is the initial guess for the fit, it is an important parameter
            x0 = [12.,0.1,p_right[0],1 - p_right[-1]]

            bounds = [(frequencies[0],frequencies[-1]),(0.0001,10),(0,0.7),(0,0.7)]

            res = minimize(func, x0, options = dict(maxiter = 500*len(x0),adaptive=True),
                        bounds = bounds, method='Nelder-Mead') # method = 'L-BFGS-B'

            bias = res.x[0]-12. #horizontal displacement
            sensitivity = res.x[1] #slope
            left_lapse = res.x[2] #gamma
            right_lapse = res.x[3] #lambda

            biases.append(bias)
            sensitivities.append(sensitivity)
            left_lapses.append(left_lapse)
            right_lapses.append(right_lapse)

            fig = plt.figure()
            ax = fig.add_axes([0.2,0.2,0.7,0.7])
            ax.vlines(12,0,1,color = 'k',lw = 0.3) # plot a vertical line as reference at zero
            ax.hlines(0.5,np.min(frequencies),np.max(frequencies),color = 'k',lw = 0.3) # plot an horizontal line as reference for chance performance
            ax.set_xticks([4, 8, 12, 16, 20])

            # plot the fit
            nx = np.linspace(np.min(frequencies),np.max(frequencies),100)
            ax.plot(nx,cumulative_gaussian(*res.x,nx),'k')

            # plot the observed data and confidence intervals
            for i,e in zip(frequencies,ci_right):  # plot the confidence intervals
                ax.plot(i*np.array([1,1]),e,'_-',lw=0.5,color = 'black')
                
            ax.plot(frequencies,p_right,'ko',markerfacecolor = 'lightgray',markersize = 6)

            ax.set_ylabel('P$_{right}$',fontsize = 18)  # set the y-axis label with latex nomenclature
            ax.set_xlabel('Stimulus rate', fontsize = 14); # set the x-axis label
            print('Estimated parameters: alpha {0:2.2f} beta {1:2.2f} gamma {2:2.2f} lambda {3:2.2f}'.format(*res.x))

    return biases, sensitivities, left_lapses, right_lapses

file_names = pick_files_multi_session("chipmunk", "*.h5")
biases, sensitivities, left_lapses, right_lapses = get_PMF_parameters_multi_session(file_names)

#%% Plot parameters

# Define the x-axis values as indices of the lists
x_values = range(len(biases))

# Create subplots
fig, axs = plt.subplots(4, 1, figsize=(8, 10))

# Plot each list on a separate subplot
axs[0].plot(x_values, biases, color='red')
axs[0].set_title('Biases')
# axs[0].hlines(0.0,np.min(x_values),np.max(x_values),color = 'k',lw = 0.3) # plot an horizontal line as reference for chance performance

axs[1].plot(x_values, sensitivities, color='green')
axs[1].set_title('Sensitivities')

axs[2].plot(x_values, left_lapses, color='blue')
axs[2].set_title('Left Lapses')

axs[3].plot(x_values, right_lapses, color='purple')
axs[3].set_title('Right Lapses')

# Set x tick marks to integers
for ax in axs:
    ax.set_xticks(range(len(x_values)))
    ax.set_xticklabels([str(i) for i in range(len(x_values))])

# Add a shared x-axis label
fig.text(0.5, 0.04, 'Index', ha='center')
fig.text(0.04, 0.5, 'Value', va='center', rotation='vertical')

# Adjust the spacing between subplots
plt.subplots_adjust(hspace=0.4)

# Show the plot
plt.show()

