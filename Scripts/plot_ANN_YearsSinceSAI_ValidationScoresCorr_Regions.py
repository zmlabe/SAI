"""
Logistic regression for evaluating differences in ARISE vs. CONTROL. This 
script looks at the optimized models for each region and plots the 
validation scores.

Author     : Zachary M. Labe
Date       : 13 April 2022
Version    : 1 - testing ANN architectures for detecting SAI
"""

### Import packages
import sys
import matplotlib.pyplot as plt
import matplotlib.colors as c
import numpy as np

### Plotting defaults 
plt.rc('text',usetex=True)
plt.rc('font',**{'family':'sans-serif','sans-serif':['Avant Garde']}) 

###############################################################################
###############################################################################
modelGCMs = ['ARISE','WACCM']
datasetsingle = ['all_saiComparison']
seasons = ['annual']
monthlychoice = seasons[0]
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
        classesl[i,:,:] = np.full((numOfEns,len(yearsall)),i)  
        
    if ensTypeExperi == 'ENS':
        classeslnew = np.swapaxes(classesl,0,1)

###############################################################################
###############################################################################
###############################################################################
###############################################################################     
### Begin ANN and the entire script - loop through these parameters
letters = ["a","b","c","d","e","f","g","h","i","j","k","l","m"]
ridge_penaltyq = [0.01,0.1,0.25,0.5,0.75,1,1.5,2,3,5,10]
reg_nameq = ['Globe','NH','SH','Arctic','Antarctic','narrowTropics','SEAsia','NorthAfrica','Amazon']
NCOMBOS = 20
variq = 'PRECT'
# ANNn = 'Hiddens_10x10'
# ANNname = '2-LAYERS$_{10}$'
# ANNn = 'Hiddens_5x5'
# ANNname = '2-LAYERS$_{5}$'
ANNn = 'Hiddens_5'
ANNname = '1-LAYER$_{5}$'
directorydata = '/Users/zlabe/Documents/Research/SolarIntervention/Data/YearsSinceSAI/ONLYARISE/Loop/%s/' % ANNn
directoryfigure = '/Users/zlabe/Documents/Research/SolarIntervention/Figures/'

### Read in hyperparameters
segmentseeds = np.empty((len(reg_nameq),len(ridge_penaltyq),NCOMBOS))
networkseeds = np.empty((len(reg_nameq),len(ridge_penaltyq),NCOMBOS))
trainacc = np.empty((len(reg_nameq),len(ridge_penaltyq),NCOMBOS))
testacc = np.empty((len(reg_nameq),len(ridge_penaltyq),NCOMBOS))
valacc = np.empty((len(reg_nameq),len(ridge_penaltyq),NCOMBOS))
latitudes = []
longitudes = []
trainindices = np.empty((len(reg_nameq),len(ridge_penaltyq),NCOMBOS,7))
testindices = np.empty((len(reg_nameq),len(ridge_penaltyq),NCOMBOS,1))
valindices = np.empty((len(reg_nameq),len(ridge_penaltyq),NCOMBOS,2))
for rr in range(len(reg_nameq)):
    saveSeedsq = 'YearsSinceSAI_ARISEONLY' + '_ANN_' +variq+'_' + reg_nameq[rr] + '_' + monthlychoice
    saveAccuracyq = 'YearsSinceSAI_ARISEONLY' + '_ANN_' +variq+'_' + reg_nameq[rr] + '_' + monthlychoice
    saveLatsq = 'YearsSinceSAI_ARISEONLY' + '_ANN_' +variq+'_' + reg_nameq[rr] + '_' + monthlychoice
    saveLonsq = 'YearsSinceSAI_ARISEONLY' + '_ANN_'+variq+'_' + reg_nameq[rr]+ '_' + monthlychoice
    saveIndicesq = 'YearsSinceSAI_ARISEONLY' + '_ANN_' +variq+'_' + reg_nameq[rr] + '_' + monthlychoice
    
    segmentseeds[rr] = np.load(directorydata + saveSeedsq + '_SegmentSeeds.npy')
    networkseeds[rr] = np.load(directorydata + saveSeedsq + '_NetworkSeeds.npy')
    trainacc[rr] = np.load(directorydata + saveAccuracyq + '_TrainAcc.npy')
    testacc[rr] = np.load(directorydata + saveAccuracyq + '_TestAcc.npy')
    valacc[rr] = np.load(directorydata + saveAccuracyq + '_ValAcc.npy')
    trainindices[rr] = np.load(directorydata + saveIndicesq + '_TrainIndices.npy')
    testindices[rr] = np.load(directorydata + saveIndicesq + '_TestIndices.npy')
    valindices[rr] = np.load(directorydata + saveIndicesq + '_ValIndices.npy')
    
    latitudesq = np.load(directorydata + saveLatsq + '_Latitudes.npy')
    longitudesq = np.load(directorydata + saveLonsq + '_Longitudes.npy')
    latitudes.append(latitudesq)
    longitudes.append(longitudesq)
    
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
    
    plotdata = valacc[plo].transpose()
    
    adjust_spines(ax, ['left', 'bottom'])
    ax.spines['top'].set_color('none')
    ax.spines['right'].set_color('none')
    ax.spines['left'].set_color('dimgrey')
    ax.spines['bottom'].set_color('none')
    ax.spines['left'].set_linewidth(2)
    ax.tick_params('both',length=4,width=2,which='major',color='dimgrey')
    ax.tick_params(axis="x",which="both",bottom = False,top=False,
                    labelbottom=True,pad=-5)
    
    # ax.yaxis.grid(zorder=2,color='darkgrey',alpha=0.7,clip_on=False,linewidth=0.5)
    ax.xaxis.grid(zorder=2,color='darkgrey',alpha=1,clip_on=False,linewidth=1)
    
    def set_box_color(bp, color):
        plt.setp(bp['boxes'],color='w')
        plt.setp(bp['whiskers'], color='w',linewidth=1.5)
        plt.setp(bp['caps'], color='w',alpha=0)
        plt.setp(bp['medians'], color=color,linewidth=2)
    
    positionsq = np.arange(len(ridge_penaltyq))
    bpl = plt.boxplot(plotdata,positions=positionsq,widths=0.6,
                      patch_artist=True,sym='',zorder=1)
    
    # Modify boxes
    cp= 'maroon'
    set_box_color(bpl,cp)
    plt.plot([], c=cp, label=r'\textbf{ACCURACY}',clip_on=False)
        
    for i in range(plotdata.shape[1]):
        y = plotdata[:,i]
        x = np.random.normal(positionsq[i], 0.04, size=len(y))
        plt.plot(x, y,color='teal', alpha=0.8,zorder=10,marker='.',linewidth=0,markersize=5,markeredgewidth=0,clip_on=False)
     
    if any([plo==0,plo==3,plo==6]):
        plt.yticks(np.arange(-1,1.1,0.2),list(map(str,np.round(np.arange(-1,1.1,0.2),2))),
                    fontsize=6) 
        if variq == 'TREFHT':
            plt.ylim([-0.4,1])
        elif variq == 'PRECT':
            plt.ylim([-0.4,1])
    else:
        plt.yticks(np.arange(-1,1.1,0.2),list(map(str,np.round(np.arange(-1,1.1,0.2),2))),
                    fontsize=6) 
        if variq == 'TREFHT':
            plt.ylim([-0.4,1])
        elif variq == 'PRECT':
            plt.ylim([-0.4,1])
        ax.axes.yaxis.set_ticklabels([])
        
    if any([plo==6,plo==7,plo==8]):
        plt.xticks(np.arange(0,len(ridge_penaltyq)+1,1),list(map(str,np.array(ridge_penaltyq))),fontsize=5)
        plt.ylim([-0.4,1])
    else:
        ax.axes.xaxis.set_ticklabels([])
        
    if plo == 7:
        plt.xlabel(r'\textbf{Ridge Penalty (L$_{2}$) -- %s}' % variq,color='k',fontsize=6)
        
    plt.xlim([-0.5,len(ridge_penaltyq)-0.5])
    
        
    plt.text(4.4,1.1,r'\textbf{%s}' % labels[plo],fontsize=11,color='dimgrey',
              ha='center',va='center')
    plt.text(len(ridge_penaltyq)-1,1.1,r'\textbf{[%s]}' % letters[plo],color='k',fontsize=6)

plt.tight_layout()   
plt.text(-30,0.95,r'\textbf{Correlation Coefficient for %s}' % ANNname,color='k',fontsize=7,
         rotation=90)

plt.savefig(directoryfigure + 'AccuracyScores_ANN_YearsSinceSAI_ValidationScoresCorr_%s_%s.png' % (variq,ANNn),dpi=300)

###############################################################################
###############################################################################
###############################################################################
### Sort networks  
whichL2 = []
whichSeed = []
for i in range(len(reg_nameq)):
    valregions = valacc[i,:,:]
    
    ### Calculate median
    medregions = np.median(valregions,axis=1)
    whichL2q = np.argmax(medregions)
    
    ### Find seed with highest correlation
    percN = -3
    valregions_l2 = valregions[whichL2q]
    valregions_l2_sort = np.sort(valregions_l2)
    valregions_seed = valregions_l2_sort[percN]
    whichSeedq = np.argwhere(valregions_l2 == valregions_seed)[0][0]
    
    ### Save location of model
    whichL2.append(whichL2q)
    whichSeed.append(whichSeedq)
    
### Find model to use accuracy for validation
valacc_model = np.empty((len(reg_nameq)))
l2_model = np.empty((len(reg_nameq)))
segSeed_model = np.empty((len(reg_nameq)))
netSeed_model = np.empty((len(reg_nameq)))
testindices_model = np.empty((len(reg_nameq),1))
valindices_model = np.empty((len(reg_nameq),2))
for i in range(len(reg_nameq)):
    valacc_model[i] = valacc[i,whichL2[i],whichSeed[i]]
    
    l2_model[i] = ridge_penaltyq[whichL2[i]]
    
    segSeed_model[i] = segmentseeds[i,whichL2[i],whichSeed[i]]
    netSeed_model[i] = networkseeds[i,whichL2[i],whichSeed[i]]
    
    testindices_model[i,:] = testindices[i,whichL2[i],whichSeed[i]]
    valindices_model[i,:] = valindices[i,whichL2[i],whichSeed[i]]
    
### Save output
directoryModels = '/Users/zlabe/Documents/Research/SolarIntervention/Data/YearsSinceSAI_ActualModel/'
np.savetxt(directoryModels + 'L2_ANN_YearsSinceSAI_%s_corr_%s.txt' % (variq,ANNn),l2_model)
np.savetxt(directoryModels + 'SegSeed_ANN_YearsSinceSAI_%s_corr_%s.txt' % (variq,ANNn),segSeed_model)
np.savetxt(directoryModels + 'NetSeed_ANN_YearsSinceSAI_%s_corr_%s.txt' % (variq,ANNn),netSeed_model)