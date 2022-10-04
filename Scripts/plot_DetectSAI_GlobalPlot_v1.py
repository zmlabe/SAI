"""
Make plot of predictions for detecting SAI using logistic regression for the 
models identifyied through tuning

Author     : Zachary M. Labe
Date       : 14 April 2022
Version    : 1
"""

### Import packages
import sys
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import accuracy_score

### Plotting defaults 
plt.rc('text',usetex=True)
plt.rc('font',**{'family':'sans-serif','sans-serif':['Avant Garde']})

### Hyperparamters for files of the ANN model
yearsall = np.arange(2035,2069+1,1)
yearsobs = np.arange(1900,2015+1,1)
variqall = ['TREFHT','PRECT']
hiddensList = [[0]]
actFun = 'relu'
NNType = 'logreg'
reg_nameq = ['Globe','NH','SH','Arctic','Antarctic','narrowTropics','SEAsia','NorthAfrica','Amazon']
labels = variqall
monthlychoice = 'annual'
seasons = monthlychoice
land_only = True
ocean_only = False
num_of_class = 2

### Loop through the different regions
fig = plt.figure(figsize=(9,3))
for rr in range(len(variqall)):
    reg_name = reg_nameq[0]
    variq = variqall[rr]
    if land_only == True:
        saveData = seasons + '_LAND' + '_GCMarise_LOGREG' + '_' + variq + '_' + reg_name  + '_' + 'NumOfGCMS-' + str(num_of_class)
    elif ocean_only == True:
        saveData = seasons + '_OCEAN' + '_GCMarise_LOGREG' + '_' + variq + '_' + reg_name + '_' + 'NumOfGCMS-' + str(num_of_class)
    else:
        saveData = seasons + '_GCMarise_LOGREG' + '_' + variq + '_' + reg_name + '_' + 'NumOfGCMS-' + str(num_of_class)
    print('*Filename == < %s >' % saveData) 
                      
    ### Directories to save files
    directoryoutput = '/Users/zlabe/Documents/Research/SolarIntervention/Data/DetectSAI_ActualModel/'
    directoryfigure = '/Users/zlabe/Documents/Research/SolarIntervention/Figures/'
    
    ###############################################################################
    ###############################################################################
    ###############################################################################
    ### Read in data for predictions
    testIndices = np.genfromtxt(directoryoutput + 'testingEnsIndices_' + saveData + '.txt')
    
    testingTrue = np.genfromtxt(directoryoutput + 'testingTrueLabels_' + saveData + '.txt').reshape(2,yearsall.shape[0])
    testingPred = np.genfromtxt(directoryoutput + 'testingPredictedLabels_' + saveData + '.txt').reshape(2,yearsall.shape[0])
    
    testingCONF = np.genfromtxt(directoryoutput + 'testingPredictedConfidence_' + saveData + '.txt').reshape(2,yearsall.shape[0],2)
    
    ###############################################################################
    ###############################################################################
    ###############################################################################
    ### Create arrays for plotting
    def adjust_spines(ax, spines):
        for loc, spine in ax.spines.items():
            if loc in spines:
                spine.set_position(('outward', 20))
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
                
    if variq == 'TREFHT':
        cc1 = 'teal'
        cc2 = 'maroon'
    elif variq == 'PRECT':
        cc1 = 'saddlebrown'
        cc2 = 'darkolivegreen'
    
    length = np.arange(yearsall.shape[0])
    
    ax = plt.subplot(2,1,rr+1)
    adjust_spines(ax, ['left', 'bottom'])
    
    acc = accuracy_score(testingTrue.ravel(),testingPred.ravel())*100
    print(acc)
    
    if rr == 1:
        ax.spines['top'].set_color('none')
        ax.spines['right'].set_color('none')
        ax.spines['left'].set_color('none')
        ax.spines['bottom'].set_color('dimgrey')
        ax.spines['left'].set_linewidth(2)
        ax.spines['bottom'].set_linewidth(2)
        ax.tick_params('both',length=4,width=2,which='major',color='dimgrey')
        ax.tick_params(axis='y',which='both',length=0)
    else:
        ax.spines['top'].set_color('none')
        ax.spines['right'].set_color('none')
        ax.spines['left'].set_color('none')
        ax.spines['bottom'].set_color('none')
        ax.spines['left'].set_linewidth(0)
        ax.spines['bottom'].set_linewidth(0)
        ax.tick_params('both',length=0,width=0,which='major',color='dimgrey')
        ax.tick_params(axis='y',which='both',length=0)
    
    for i in range(testingPred.shape[0]):
        for yr in range(testingPred.shape[1]):
            if testingPred[i,yr] == 0:
                cc = cc1
                label = 'SAI'
            elif testingPred[i,yr] == 1:
                cc = cc2
                label = 'SSP2-4.5'
                
            if testingPred[i,yr] == 0:
                conf = testingCONF[i,yr,0]
            elif testingPred[i,yr] == 1:
                conf = testingCONF[i,yr,1]
    
            plotdata = testingPred.copy()
            plotdata[i,yr] = 0+i
            
            plt.scatter(yearsall[yr],plotdata[i,yr],s=225,color=cc,clip_on=False,
                        zorder=3,edgecolor=cc,linewidth=0.75,label=label,
                        alpha=(conf-0.5)/(1-0.5))
        
    plt.xticks(np.arange(2035,2101,5),map(str,np.arange(2035,2101,5)),size=8)
    
    if rr == 1:
        plt.xlabel(r'\textbf{Years}',color='dimgrey',fontsize=9)
        plt.yticks(np.arange(0,testingPred.shape[0],1),['SAI','SSP2-4.5'],size=11)
        plt.xticks(np.arange(2035,2101,5),map(str,np.arange(2035,2101,5)),size=8)
    else:
        plt.yticks(np.arange(0,testingPred.shape[0],1),['SAI','SSP2-4.5'],size=11)
        plt.xticks(np.arange(2035,2101,5),map(str,np.arange(2035,2101,5)),size=8,
                   color='w')
    plt.text(2070.8,0.5,r'\textbf{%s [%s\%%]}' % (labels[rr],np.round(acc,1)),ha='center',va='center',
             color='k',rotation=270,fontsize=11)
    
    plt.xlim([2035,2070])   
    plt.ylim([0,1])
    plt.tight_layout()

plt.savefig(directoryfigure + 'Predictions_DetectSAI_GLOBAL_TREFHT-PRECT.png',dpi=300)
