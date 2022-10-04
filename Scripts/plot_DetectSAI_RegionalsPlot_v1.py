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

### Plotting defaults 
plt.rc('text',usetex=True)
plt.rc('font',**{'family':'sans-serif','sans-serif':['Avant Garde']})

### Hyperparamters for files of the ANN model
yearsall = np.arange(2035,2069+1,1)
yearsobs = np.arange(1900,2015+1,1)
variqall = ['TREFHT','PRECT','TREFHT','PRECT','TREFHT','PRECT','TREFHT','PRECT',
            'TREFHT','PRECT','TREFHT','PRECT','TREFHT','PRECT','TREFHT','PRECT']
hiddensList = [[0]]
actFun = 'relu'
NNType = 'logreg'
reg_nameq = ['NH','NH',
             'SH','SH',
             'Arctic','Arctic',
             'Antarctic','Antarctic',
             'narrowTropics','narrowTropics',
             'SEAsia','SEAsia',
             'NorthAfrica','NorthAfrica',
             'Amazon','Amazon']
labels = np.repeat(['N. Hemisphere','S. Hemisphere','Arctic','Antarctic','Tropics','Southeast Asia','Central Africa','Amazon'],2)
monthlychoice = 'annual'
seasons = monthlychoice
land_only = True
ocean_only = False
num_of_class = 2

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

### Loop through the different regions
fig = plt.figure(figsize=(8,6))
for rr in range(len(variqall)):
    reg_name = reg_nameq[rr]
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
    if variq == 'TREFHT':
        cc1 = 'teal'
        cc2 = 'maroon'
    elif variq == 'PRECT':
        cc1 = 'saddlebrown'
        cc2 = 'darkolivegreen'
    
    length = np.arange(yearsall.shape[0])
    
    ax = plt.subplot(8,2,rr+1)
    adjust_spines(ax, ['left', 'bottom'])
    
    if any([rr==14,rr==15]):
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
        
    # ax.yaxis.grid(zorder=1,color='dimgrey',alpha=0.35,clip_on=False)
    
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
                
            if any([rr==1,rr==3,rr==5,rr==7,rr==9,rr==11,rr==13,rr==15]):
                plt.axhline(y=0,xmin=0,xmax=0.9715,color='k',zorder=0,
                            clip_on=False,linewidth=0.35)
                plt.axhline(y=1,xmin=0,xmax=0.9715,color='k',zorder=0,
                            clip_on=False,linewidth=0.35)
            else:
                plt.axhline(y=0,xmin=-0.09,xmax=1.2,color='k',zorder=0,
                            clip_on=False,linewidth=0.35)
                plt.axhline(y=1,xmin=-0.09,xmax=1.2,color='k',zorder=0,
                            clip_on=False,linewidth=0.35)
    
            plotdata = testingPred.copy()
            plotdata[i,yr] = 0+i
            
            plt.scatter(yearsall[yr],plotdata[i,yr],s=35,color=cc,clip_on=False,
                        zorder=3,edgecolor=cc,linewidth=0.2,label=label,
                        alpha=(conf-0.5)/(1-0.5))
    
    if any([rr==14,rr==15]):
        plt.xlabel(r'\textbf{Years}',color='dimgrey')
        plt.xticks(np.arange(2035,2101,5),map(str,np.arange(2035,2101,5)),size=8)
    else:
        plt.xticks(np.arange(2035,2101,5),map(str,np.arange(2035,2101,5)),size=8,
                   color='w')
        
    if any([rr==1,rr==3,rr==5,rr==7,rr==9,rr==11,rr==13,rr==15]):
        plt.text(2076,0.45,r'\textbf{\underline{%s}}' % (labels[rr]),ha='center',va='center',
                  color='k',rotation=0,fontsize=9.5)
        plt.yticks(np.arange(0,testingPred.shape[0],1),[],size=9,
                   color='w')
    else:
        plt.yticks(np.arange(0,testingPred.shape[0],1),[r'\textbf{SAI}',r'\textbf{SSP2-4.5}'],size=9,
                   color='k')
    
    plt.xlim([2035,2070])   
    # plt.ylim([0,1])
plt.tight_layout()
plt.subplots_adjust(hspace=1.5)

plt.savefig(directoryfigure + 'Predictions_DetectSAI_REGIONS_TREFHT-PRECT.png',dpi=300)
