"""
ANN for calculating how many years since 2035. This script plots training
results for all of the different regions.

Author      : Zachary M. Labe
Date        : 25 September 2022
Version     : 4 - selected 10x10 and only showing 1 seed for each high L2
Environment : source activate env-tf2.4
"""

### Import packages
import sys
import matplotlib.pyplot as plt
import matplotlib.colors as c
import numpy as np
from sklearn.metrics import mean_squared_error,mean_absolute_error
import scipy.stats as sts

### Plotting defaults 
plt.rc('text',usetex=True)
plt.rc('font',**{'family':'sans-serif','sans-serif':['Avant Garde']}) 

###############################################################################
###############################################################################
###############################################################################
### Data preliminaries 
directorydata = '/Users/zlabe/Data/SAI/'
###############################################################################
###############################################################################
modelGCMs = ['ARISE']
datasetsingle = ['ARISE']
seasons = ['annual']
monthlychoice = seasons[0]
variq = 'PRECT'
###############################################################################
###############################################################################
land_only = True
ocean_only = False
###############################################################################
###############################################################################
yearsall = np.arange(2035,2069+1,1)
numOfEns = 10
dataset_obs = '20CRv3'
###############################################################################
###############################################################################
num_of_class = len(modelGCMs)
ensTypeExperi = 'ENS'
###############################################################################
###############################################################################
###############################################################################
###############################################################################
### Create sample class labels for each model for my own testing
if seasons != 'none':
    classesl = np.empty((num_of_class,numOfEns,len(yearsall)))
    for i in range(num_of_class):
        classesl_row = np.arange(2035,2069+1,1) - 2035 # years since 2035
        classesl[i,:,:] = np.tile(classesl_row,(numOfEns,1))
        
    if ensTypeExperi == 'ENS':
        classeslnew = np.swapaxes(classesl,0,1)

###############################################################################
###############################################################################
###############################################################################
###############################################################################     
### Begin ANN and the entire script - loop through these parameters
letters = ["a","b","c","d","e","f","g","h","i","j","k","l","m"]
reg_nameq = ['Globe','NH','SH','Arctic','Antarctic','narrowTropics','SEAsia','NorthAfrica','Amazon']
directorydata = '/Users/zlabe/Documents/Research/SolarIntervention/Data/yearsSinceSAI/ONLYARISE/'
directoryfigure = '/Users/zlabe/Documents/Research/SolarIntervention/Figures/'

### Read in hyperparameters
trainpre = np.empty((len(reg_nameq),7*yearsall.shape[0]))
trainactual = np.empty((len(reg_nameq),7*yearsall.shape[0]))
trainIndices = np.empty((len(reg_nameq),7))
for rr in range(len(reg_nameq)):
    
    ### Select how to open files
    reg_name = reg_nameq[rr]
    if land_only == True:
        saveData = seasons[0] + '_LAND' + '_GCMarise_YearsSinceSAI_ARISEONLY' + '_' + variq + '_' + reg_name  + '_' + 'NumOfGCMS-' + str(num_of_class)
    elif ocean_only == True:
        saveData = seasons[0] + '_OCEAN' + '_GCMarise_YearsSinceSAI_ARISEONLY' + '_' + variq + '_' + reg_name + '_' + 'NumOfGCMS-' + str(num_of_class)
    else:
        saveData = seasons[0] + '_GCMarise_YearsSinceSAI_ARISEONLY' + '_' + variq + '_' + reg_name + '_' + 'NumOfGCMS-' + str(num_of_class)
    print('*Filename == < %s >' % saveData) 
    
    trainIndices[rr,:] = np.genfromtxt(directorydata + 'trainingEnsIndices_' + saveData + '.txt',unpack=True)
    trainactual[rr] = np.genfromtxt(directorydata + 'trainingTrueLabels_' + saveData + '.txt',unpack=True)
    trainpre[rr] = np.genfromtxt(directorydata + 'trainingPredictions_' + saveData + '.txt',unpack=True)
    
### Reshape predictions
trainpredictions = trainpre.reshape(len(reg_nameq),len(trainIndices[0]),yearsall.shape[0])
trainactual = trainactual.reshape(len(reg_nameq),len(trainIndices[0]),yearsall.shape[0])
   
###############################################################################
###############################################################################
###############################################################################
### Graph for accuracy
labels = ['Globe','N. Hemisphere','S. Hemisphere','Arctic','Antarctic','Tropics','Southeast Asia','Central Africa','Amazon']

def adjust_spines(ax, spines):
    for loc, spine in ax.spines.items():
        if loc in spines:
            spine.set_position(('outward', 5))
        else:
            spine.set_color('none')  
    if 'left' in spines:
        ax.yaxis.set_ticks_position('left')
    else:
        ax.yaxis.set_ticks([])

    if 'bottom' in spines:
        ax.xaxis.set_ticks_position('bottom')
    else:
        ax.xaxis.set_ticks([])

fig = plt.figure()
for plo in range(len(labels)):
    ax = plt.subplot(3,3,plo+1)
    
    adjust_spines(ax, ['left', 'bottom'])
    ax.spines['top'].set_color('none')
    ax.spines['right'].set_color('none')
    ax.spines['left'].set_color('dimgrey')
    ax.spines['bottom'].set_color('dimgrey')
    ax.spines['left'].set_linewidth(2)
    ax.spines['bottom'].set_linewidth(2)
    ax.tick_params('both',length=4,width=2,which='major',color='dimgrey')
    
    if variq == 'TREFHT':
        cc1 = 'teal'
    elif variq == 'PRECT':
        cc1 = 'saddlebrown'
    
    ### Read in data
    oneToOne = trainactual[plo,0,:]
    plotSAI = trainpredictions[plo,:,:]

    
    plt.plot(yearsall,oneToOne,linestyle='-',linewidth=2,color='k',clip_on=False)
    
    for ensm in range(len(plotSAI)):
        plt.plot(yearsall,plotSAI[ensm,:],linestyle='-',linewidth=1,
                 color=cc1,label=r'\textbf{SAI}',clip_on=False,alpha=0.4)
    
    plt.yticks(np.arange(0,36,5),list(map(str,np.round(np.arange(0,36,5),2))),fontsize=5)    
    plt.xticks(np.arange(2035,2070+1,5),list(map(str,np.arange(2035,2070+1,5))),fontsize=5)
    if any([plo==3]):
        plt.ylabel(r'\textbf{Years Since Injection}',color='k',fontsize=7)
    if any([plo==7]):
        plt.xlabel(r'\textbf{Actual Year}',color='k',fontsize=7)
            
    if any([plo==0,plo==1,plo==2,plo==3,plo==4,plo==5]):
        ax.axes.xaxis.set_ticklabels([])
    if any([plo==1,plo==2,plo==4,plo==5,plo==7,plo==8]):
        ax.axes.yaxis.set_ticklabels([])
        
    if plo == 1:
        plt.text(2022,47.5,r'\textbf{SAI -- TRAINING ENSEMBLE MEMBERS}',color=cc1,fontsize=9)
    plt.text(2035,35,r'\textbf{[%s] %s}' % (letters[plo],labels[plo]),color='dimgrey',fontsize=9)
          
    plt.xlim([2035,2070])
    plt.ylim([0,35])

plt.subplots_adjust(wspace=0.3,hspace=0.5)
plt.savefig(directoryfigure + 'ANN_Predictions_YearsSAI_Regions_%s_ARISE_corr_training.png' % variq,dpi=300)
