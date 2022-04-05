"""
Make plot of predictions for detecting SAI

Author     : Zachary M. Labe
Date       : 5 April 2022
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
variq = 'TREFHT'
random_segment_seed = int(np.genfromtxt('/Users/zlabe/Documents/Research/GmstTrendPrediction/Data/SelectedSegmentSeed.txt',unpack=True))
random_network_seed = 87750
hiddensList = [[20,20]]
batch_size = 32
lr_here = 0.001
ridge_penalty = [1]
iterations = [500]
actFun = 'relu'
NNType = 'ANN'
reg_name = 'Globe'
monthlychoice = 'annual'
seasons = monthlychoice
land_only = True
ocean_only = False
num_of_class = 2
if land_only == True:
    saveData = seasons + '_LAND' + '_GCMarise' + '_' + variq + '_' + reg_name  + '_' + 'NumOfGCMS-' + str(num_of_class)
elif ocean_only == True:
    saveData = seasons + '_OCEAN' + '_GCMarise' + '_' + variq + '_' + reg_name + '_' + 'NumOfGCMS-' + str(num_of_class)
else:
    saveData = seasons + '_GCMarise' + '_' + variq + '_' + reg_name + '_' + 'NumOfGCMS-' + str(num_of_class)
print('*Filename == < %s >' % saveData) 

### Naming conventions for files
dirname = '/Users/zlabe/Documents/Research/SolarIntervention/savedModels/'
savename = 'DETECTSAI' + '_' + variq + '_' + reg_name + '_' + monthlychoice + '_L2'+ str(ridge_penalty[0])+ '_LR' + str(lr_here)+ '_Batch'+ str(batch_size)+ '_Iters' + str(iterations[0]) + '_' + NNType + str(hiddensList[0][0]) + 'x' + str(hiddensList[0][-1]) + '_SegSeed' + str(random_segment_seed) + '_NetSeed'+ str(random_network_seed) 
savenameModelTestTrain = 'DETECTSAI' + '_'+variq+'_modelTrainTest_SegSeed'+str(random_segment_seed)+'_NetSeed'+str(random_network_seed)
                        
### Directories to save files
directoryoutput = '/Users/zlabe/Documents/Research/SolarIntervention/Data/'
directoryfigure = '/Users/zlabe/Desktop/SAI/detectSAI/'

###############################################################################
###############################################################################
###############################################################################
### Read in data for predictions
trainIndices = np.genfromtxt(directoryoutput + 'trainingEnsIndices_' + saveData + '.txt')
testIndices = np.genfromtxt(directoryoutput + 'testingEnsIndices_' + saveData + '.txt')

trainingTrue = np.genfromtxt(directoryoutput + 'trainingTrueLabels_' + saveData + '.txt')
testingTrue = np.genfromtxt(directoryoutput + 'testingTrueLabels_' + saveData + '.txt').reshape(2,yearsall.shape[0])

trainingPred = np.genfromtxt(directoryoutput + 'trainingPredictedLabels_' + saveData + '.txt')
testingPred = np.genfromtxt(directoryoutput + 'testingPredictedLabels_' + saveData + '.txt').reshape(2,yearsall.shape[0])

trainingCONF = np.genfromtxt(directoryoutput + 'trainingPredictedConfidence_' + saveData + '.txt')
testingCONF = np.genfromtxt(directoryoutput + 'testingPredictedConfidence_' + saveData + '.txt').reshape(2,yearsall.shape[0],2)

### Save predictions for observations
obslabels = np.genfromtxt(directoryoutput + 'obsLabels_' + saveData + '.txt')
obsconf = np.genfromtxt(directoryoutput + 'obsConfid_' + saveData + '.txt')

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

length = np.arange(yearsall.shape[0])

fig = plt.figure(figsize=(9,1.5))
ax = plt.subplot(111)
adjust_spines(ax, ['left', 'bottom'])
ax.spines['top'].set_color('none')
ax.spines['right'].set_color('none')
ax.spines['left'].set_color('none')
ax.spines['bottom'].set_color('dimgrey')
ax.spines['left'].set_linewidth(2)
ax.spines['bottom'].set_linewidth(2)
ax.tick_params('both',length=4,width=2,which='major',color='dimgrey')
ax.tick_params(axis='y',which='both',length=0)
# ax.yaxis.grid(zorder=1,color='dimgrey',alpha=0.35,clip_on=False)

for i in range(testingPred.shape[0]):
    for yr in range(testingPred.shape[1]):
        if testingPred[i,yr] == 0:
            cc = 'deepskyblue'
            label = 'SAI'
        elif testingPred[i,yr] == 1:
            cc = 'crimson'
            label = 'Control'
            
        if testingPred[i,yr] == 0:
            conf = testingCONF[i,yr,0]
        elif testingPred[i,yr] == 1:
            conf = testingCONF[i,yr,1]

        plotdata = testingPred.copy()
        plotdata[i,yr] = 0+i
        
        plt.scatter(yearsall[yr],plotdata[i,yr],s=200,color=cc,clip_on=False,
                    zorder=3,edgecolor='dimgrey',linewidth=0.5,label=label,
                    alpha=conf)
    
plt.xticks(np.arange(2035,2101,5),map(str,np.arange(2035,2101,5)),size=7)
plt.yticks(np.arange(0,testingPred.shape[0],1),['SAI','CONTROL'],size=7)
plt.xlim([2035,2070])   
plt.ylim([0,1])
plt.xlabel(r'\textbf{Years - Global}')
# plt.ylabel(r'\textbf{Simulations}')
plt.tight_layout()

plt.savefig(directoryfigure + 'Predictions_DetectSAI_LAND_%s.png' % reg_name,dpi=300)


