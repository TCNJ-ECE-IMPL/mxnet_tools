#!/usr/bin/python

import sys
import os
import re
import subprocess
import numpy as np
import argparse
import scipy.io
import matplotlib.pyplot as plt
import matplotlib as mpl
import pylab

# Add arguments for when plot_progress.py is called 
parser = argparse.ArgumentParser()
parser.add_argument('-k',      metavar='POSITION', help='add plot legend (Position: C,N,S,E,W,NW,NE,SE,SW)')
parser.add_argument('-t',      action='store_true', help='add PyPlot interactive toolbar')
parser.add_argument('logfile', help='path of input file, e.g. ~/exps/resnet50/checkpoint/progress.log' )
args = parser.parse_args()

# Usage: log2csv.py -s -tstl -tsta -trnl -trna <pattern> <csv_filename_root>

# Tell matplotlib to not set a toolbar if argument "-t" isn't supplied
if (args.t == 0):
    mpl.rcParams['toolbar'] = 'None'


mode_lgnd       = args.k # Add legend if "-k" argument is included
logfile         = args.logfile # logfile name is supplied from argument
        
fd = open( logfile, "r" ) # Read logfile

# Create arrays for the accuracy/loss values that are being graphed
tsta = np.array([])
tstl = np.array([])
trna = np.array([])
trnl = np.array([])

while (1):
    x = fd.readline()
    if x == "": # Stop reading at end of file
        break
    
    # Search for validation accuracy strings and append values to "tsta" array
    # tsta: Test Accuracy
    pattern = "Validation\-accuracy\=([0-9\.]+)"
    m = re.search( pattern, x )
    if m:
        val = float(m.group(1))
        #print( "%f" % val )
        tsta=np.append(tsta, val)
    
    # Search for validation cross-entropy strings and append values to "tstl" array
    # tstl: Test Loss
    pattern = "Validation\-cross\-entropy\=([0-9\.]+)"
    m = re.search( pattern, x )
    if m:
        val = float(m.group(1))
        #print( "%f" % val )
        tstl=np.append(tstl, val)
    
    # Search for training accuracy strings and append values to "trna" array
    # trna: Training Accuracy
    pattern = "Train\-accuracy\=([0-9\.]+)"
    m = re.search( pattern, x )
    if m:
        val = float(m.group(1))
        #print( "%f" % val )
        trna=np.append(trna, val)
    
    # Search for training cross-entropy strings and append values to "trnl" array
    # trnl: Training Loss
    pattern = "Train\-cross\-entropy\=([0-9\.]+)"
    m = re.search( pattern, x )
    if m:
        val = float(m.group(1))
        #print( "%f" % val )
        trnl=np.append(trnl, val)

# Find maximum number of epochs to be considered when plotting x-axis
num_epoch = max( len(tsta), len(tstl), len(trna), len(trnl) )

# 12 x 8 inch figure size
plt.rcParams["figure.figsize"] = (12,8)

# Validation vs Training Accuracy
ax1 = plt.subplot(211)
ax1.plot(range(1,num_epoch+1), trna, 'o--')
ax1.plot(range(1,num_epoch+1), tsta, 'o-')
ax1.set_xlabel('Epoch')
ax1.set_ylabel('Accuracy')
ax1.tick_params('y')

# Place legend based on position argument
if (mode_lgnd == 'C'):
    plt.legend(('Train','Validation'), loc='center', bbox_to_anchor=(0.5, -0.25), shadow=True)
elif (mode_lgnd == 'N'):
    plt.legend(('Train','Validation'), loc='center', bbox_to_anchor=(0.5, 0.8), shadow=True)
elif (mode_lgnd == 'E'):
    plt.legend(('Train','Validation'), loc='center', bbox_to_anchor=(0.935, -0.25), shadow=True)
elif (mode_lgnd == 'W'):
    plt.legend(('Train','Validation'), loc='center', bbox_to_anchor=(0.065, -0.25), shadow=True)
elif (mode_lgnd == 'NE'):
    plt.legend(('Train','Validation'), loc='center', bbox_to_anchor=(0.935, 0.8), shadow=True)
elif (mode_lgnd == 'NW'):
    plt.legend(('Train','Validation'), loc='center', bbox_to_anchor=(0.065, 0.8), shadow=True)
# Default position to center
elif (mode_lgnd != 'S' and mode_lgnd != 'SE' and mode_lgnd != 'SW'):
    plt.legend(('Train','Validation'), loc='center', bbox_to_anchor=(0.5, -0.25), shadow=True)

# Validation vs Training Loss
ax2 = plt.subplot(212, sharex=ax1)
ax2.semilogy(range(1,num_epoch+1), tstl, 'o-')
ax2.semilogy(range(1,num_epoch+1), trnl, 'o--')
ax2.set_xlabel('Epoch')
ax2.set_ylabel('Cross-Entropy Loss')
ax2.tick_params('y')

# Place legend based on position argument
if (mode_lgnd == 'S'):
    plt.legend(('Train','Validation'), loc='center', bbox_to_anchor=(0.5, 0.1), shadow=True)
elif (mode_lgnd == 'SE'):
    plt.legend(('Train','Validation'), loc='center', bbox_to_anchor=(0.935, 0.1), shadow=True)
elif (mode_lgnd == 'SW'):
    plt.legend(('Train','Validation'), loc='center', bbox_to_anchor=(0.065, 0.1), shadow=True)

# Present Plot
plt.tight_layout()
plt.show()
