"""
ANN for calculating how many years since 2035. This script plots testing
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
variq = 'TREFHT'
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
testpre = np.empty((len(reg_nameq),yearsall.shape[0]))
testactual = np.empty((len(reg_nameq),yearsall.shape[0]))
testIndices = np.empty((len(reg_nameq)))
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
    
    testIndices[rr] = np.genfromtxt(directorydata + 'testingEnsIndices_' + saveData + '.txt',unpack=True)
    testactual[rr] = np.genfromtxt(directorydata + 'testingTrueLabels_' + saveData + '.txt',unpack=True)
    testpre[rr] = np.genfromtxt(directorydata + 'testingPredictions_' + saveData + '.txt',unpack=True)
    
### Reshape predictions
testpredictions = testpre.reshape(len(reg_nameq),1,yearsall.shape[0])
testactual = testactual.reshape(len(reg_nameq),1,yearsall.shape[0])

### Evaluate each region
mae_sai = np.empty((len(reg_nameq)))
for rr in range(len(reg_nameq)):
    mae_sai[rr] = mean_absolute_error(testactual[rr,0,:],testpredictions[rr,0,:])
   
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
    oneToOne = testactual[plo,0,:]
    plotSAI = testpredictions[plo,0,:]
    
    ### Calculate trend line 
    slope,intercept,r_value,p_value,std_err = sts.linregress(yearsall,plotSAI)
    lines = slope*yearsall + intercept
    
    plt.plot(yearsall,oneToOne,linestyle='-',linewidth=2,color='k',clip_on=False)
    plt.plot(yearsall,plotSAI,linestyle='--',linewidth=1.5,dashes=(1,0.3),color=cc1,label=r'\textbf{SAI}',clip_on=False)
    plt.plot(yearsall,lines,linestyle='-',linewidth=0.8,color=cc1,clip_on=False)
    
    plt.yticks(np.arange(0,36,5),list(map(str,np.round(np.arange(0,36,5),2))),fontsize=5)    
    plt.xticks(np.arange(2035,2070+1,5),list(map(str,np.arange(2035,2070+1,5))),fontsize=5)
    if any([plo==3]):
        plt.ylabel(r'\textbf{Years Since Injection}',color='k',fontsize=7)
    if any([plo==7]):
        plt.xlabel(r'\textbf{Actual Year}',color='k',fontsize=7)
    if any([plo==1]):
        leg = plt.legend(shadow=False,fontsize=15,loc='upper center',
              bbox_to_anchor=(0.5,1.8),fancybox=True,ncol=4,frameon=False,
              handlelength=1,handletextpad=0.5)
        for line,text in zip(leg.get_lines(), leg.get_texts()):
            text.set_color(line.get_color())
            
    if any([plo==0,plo==1,plo==2,plo==3,plo==4,plo==5]):
        ax.axes.xaxis.set_ticklabels([])
    if any([plo==1,plo==2,plo==4,plo==5,plo==7,plo==8]):
        ax.axes.yaxis.set_ticklabels([])
        
    plt.text(2035,35,r'\textbf{[%s] %s}' % (letters[plo],labels[plo]),color='dimgrey',fontsize=9)
    plt.text(2068,3,r'\textbf{MAE}',fontsize=5,color='k',ha='right')
    plt.text(2070,-1,r'%s years' % np.round(mae_sai[plo],2),fontsize=5,color=cc1,ha='right')
          
    plt.xlim([2035,2070])
    plt.ylim([0,35])

plt.subplots_adjust(wspace=0.3,hspace=0.5)
plt.savefig(directoryfigure + 'ANN_Predictions_YearsSAI_Regions_%s_ARISE_corr.png' % variq,dpi=300)
