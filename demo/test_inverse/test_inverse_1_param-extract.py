import argparse
import json
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import os
from pathlib import Path
import random
import sys
import tensorflow as tf                                                         
from tensorflow.keras.models import load_model
import transistor_extract as TE

dir_path = os.path.dirname(os.path.abspath(__file__))

# plot settings
mpl.rcParams['lines.linewidth'] = 1.5
mpl.rcParams['lines.markeredgewidth'] = 1
mpl.rcParams['axes.linewidth'] = 0.7
mpl.rcParams['xtick.major.width'] = 0.4
mpl.rcParams['ytick.major.width'] = 0.4
mpl.rcParams['xtick.minor.width'] = 0.2
mpl.rcParams['ytick.minor.width'] = 0.2
plt.rcParams['xtick.minor.visible'] = True
plt.rcParams['ytick.minor.visible'] = True
plt.rcParams["xtick.top"] = True
plt.rcParams["ytick.right"] = True
mpl.rcParams['xtick.direction'] = 'in'
mpl.rcParams['ytick.direction'] = 'in'
mpl.rcParams['lines.markersize'] = 5

# Make the results folder if it doesn't exist
os.makedirs(os.path.join(dir_path, 'inverse_results'), exist_ok=True)

repo_root = Path(__file__).resolve().parents[2]
processed_data_loc = str(repo_root / 'data' / 'processed')
config_path = os.path.join(repo_root, "config.json")

tf.random.set_seed(19700101)
np.random.seed(19700101)
random.seed(19700101)

with open(config_path, "r") as f:
    cfg = json.load(f)

# model from command line
parser = argparse.ArgumentParser(description="Evaluate a trained inverse model.")
parser.add_argument("--model_inv", required=True, help="Path to the trained inverse model (.keras/.h5)")
args = parser.parse_args()
model_name_inverse = args.model_inv

fontsize = 9
mpl.rcParams.update({'font.size': fontsize})

blue = '#19546d'
red = '#bd2b49'
purple = '#192a6d'



model_inverse = load_model(                                             
                model_name_inverse, 
                custom_objects={'surrogate_loss': TE.surrogate_loss}
                ) 

X_test = np.load(processed_data_loc + '/X_test.npy')
Y_test = np.load(processed_data_loc + '/Y_test.npy')

Xscaling = np.loadtxt(processed_data_loc + '/Xscaling.dat')
Xmins = Xscaling[0,:]
Xmaxs = Xscaling[1,:]

Yscaling = np.loadtxt(processed_data_loc + '/Yscaling.dat')
# Handle Yscaling which is now 1D for single parameter
if Yscaling.ndim == 1:
    Ymins = [Yscaling[0]]
    Ymaxs = [Yscaling[1]]
else:
    Ymins = Yscaling[0,:]
    Ymaxs = Yscaling[1,:]

# Y_pred = np.array(model_inverse.predict(X_test))[:, 0:cfg["data"]["num_params"]]

# fig, axs = plt.subplots(1,3, figsize = (7,2))

# ticks = [
#           [0, 15, 30], 
#           [0, 125, 250, 375, 500], 
#           [0,0.5,1],
#           [0,1,2,3],
#           [0, 50, 100, 150, 200],
#           [0, 50, 100, 150, 200],
#           [0, 1, 2, 3],
#           [50, 175, 300],
#           ]
# # subset = range(250)
# subset = range(len(X_test))


# variables = [
#             'Mobility (cm$^2$  V$^{-1}$ s $^{-1}$)',
#             'Schottky barrier height (meV)',
#             'Effective density of states ($\\times 10^{13}$ cm$^{-2}$)',
#             'Peak donor density ($\\times$ 10$^{13}$ cm$^{-2}$ eV$^{-1}$)',
#             'Donor energy mid (meV below conduction band edge)', 
#             'Donor energy width (meV)',
#             'Peak acceptor band tail density ($\\times$ 10$^{13}$ cm$^{-2}$ eV$^{-1}$)',
#             'Acceptor band tail energy width (meV)',
#             ]

# variable_names = np.loadtxt(dir_path + '/../variable_names.txt', dtype = 'str')

# for j in range(8):
#     fig, axs = plt.subplots(1,2, figsize = (3.5, 2.25))
# ... (previous code)

Y_pred = np.array(model_inverse.predict(X_test))[:, 0:cfg["data"]["num_params"]]

fig, axs = plt.subplots(1,1, figsize = (3.5,2.25))

# Updated ticks for 1 parameter - only key values
ticks = [
    [300, 5000, 10000],
]

subset = range(len(X_test))

# Updated variable names for 1 parameter
variables = [
    'Gate length (nm)',
]

variable_names = np.loadtxt(dir_path + '/../variable_names.txt', dtype = 'str')
# Ensure variable_names is always an array, even for single parameter
if variable_names.ndim == 0:
    variable_names = np.array([variable_names])

# Updated loop to use the number of parameters from the config file
for j in range(cfg["data"]["num_params"]):
    fig, axs = plt.subplots(1,2, figsize = (3.5, 2.25))
    plt.subplots_adjust(left = 0.13, top = 0.71, right = 0.9, bottom = 0.175, hspace = 0.5, wspace = 0.7)
    Ymin = Ymins[j]
    Ymax = Ymaxs[j]

    # For Lg_nm (j=0), no special scaling needed
    Y_test[:,j] = TE.unscale_vector(Y_test[:,j], Ymin, Ymax)
    Y_pred[:,j] = TE.unscale_vector(Y_pred[:,j], Ymin, Ymax)



    axs[0].plot(
            Y_test[subset,j], 
            Y_pred[subset,j], 
            marker = 'o', 
            ls = 'None',
            markersize = 3,
            color = 'k',
            markerfacecolor = purple,
            markeredgewidth = 0.4
            )
    
    axs[0].plot(
            [-10000, 10000], 
            [-10000, 10000], 
            color = red, 
            ls = '--'
            )
    
    axs[0].set_xlim([Ymin, Ymax])
    axs[0].set_ylim([Ymin, Ymax])
    axs[0].set_xticks(ticks[j])
    axs[0].set_yticks(ticks[j])
    errors = (Y_test[:,j] - Y_pred[:,j])
    
    MAE = np.median(np.abs(errors))
    std = np.std(errors)
    binmin = -4*std
    binmax = 4*std
    binwidth = (binmax - binmin)/25
    bins = np.arange(binmin, binmax, binwidth)
    axs[1].hist(
                errors, 
                bins = bins,
                color = purple,
                edgecolor = 'k',
                linewidth = 0.15
                )
    
    axs[0].set_xlabel('Actual')
    axs[0].set_ylabel('Predicted')
    axs[1].set_xlabel('Error')
    axs[1].set_ylabel('Counts')

    fig.text(0.5, 0.88, variables[j], ha='center', fontsize=10.5)     
    fig.suptitle('\n \n Median absolute error = {} \n Standard deviation of error = {}'.format(
                 round(MAE, 3),
                 round(std, 3)),
             fontsize=fontsize)
    
    plt.savefig(dir_path + '/inverse_results/{}.png'.format(variable_names[j]), transparent = False)
    plt.close()


