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
from mpl_toolkits.basemap import Basemap, addcyclic, shiftgrid
import palettable.cubehelix as cm
import cmocean as cmocean
import cmasher as cmr

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
ridge_penaltyq = [0.01,0.1,0.25,0.5,0.75,1,1.5,5]
reg_nameq = ['Globe']
NCOMBOS = 10
directorydata = '/Users/zlabe/Documents/Research/SolarIntervention/Data/DetectSAI/'
directoryfigure = '/Users/zlabe/Documents/Research/SolarIntervention/Figures/'

### Read in weights for temperature
latshape = 96
lonshape = 144
mapweights_t = np.empty((len(reg_nameq),yearsall.shape[0]*2,latshape*lonshape))
latitudes_t = np.empty((len(reg_nameq),latshape))
longitudes_t = np.empty((len(reg_nameq),lonshape))
truelabels_t = np.empty((len(reg_nameq),yearsall.shape[0]*2))
predlabels_t = np.empty((len(reg_nameq),yearsall.shape[0]*2))
for rr in range(len(reg_nameq)):
    reg_name = reg_nameq[rr]
    variq = 'TREFHT'
    ### Select how to save files
    if land_only == True:
        saveData = seasons[0] + '_LAND' + '_GCMarise_LOGREG' + '_' + variq + '_' + reg_name  + '_' + 'NumOfGCMS-' + str(num_of_class)
    elif ocean_only == True:
        saveData = seasons[0] + '_OCEAN' + '_GCMarise_LOGREG' + '_' + variq + '_' + reg_name + '_' + 'NumOfGCMS-' + str(num_of_class)
    else:
        saveData = seasons[0] + '_GCMarise_LOGREG' + '_' + variq + '_' + reg_name + '_' + 'NumOfGCMS-' + str(num_of_class)
    print('*Filename == < %s >' % saveData) 
    
    mapweights_t[rr] = np.load(directorydata + 'WeightsInputs-LOGREG_' + saveData + '.npy')
    latitudes_t[rr] = np.load(directorydata + 'Latitudes-LOGREG_' + saveData + '.npy')
    longitudes_t[rr] = np.load(directorydata + 'Longitudes-LOGREG_' + saveData + '.npy')
    truelabels_t[rr] = np.genfromtxt(directorydata + 'testingTrueLabels_' + saveData + '.txt')
    predlabels_t[rr] = np.genfromtxt(directorydata + 'testingPredictedLabels_' + saveData + '.txt')
    
### Read in weights for precipitation
latshape = 96
lonshape = 144
mapweights_p = np.empty((len(reg_nameq),yearsall.shape[0]*2,latshape*lonshape))
latitudes_p = np.empty((len(reg_nameq),latshape))
longitudes_p = np.empty((len(reg_nameq),lonshape))
truelabels_p = np.empty((len(reg_nameq),yearsall.shape[0]*2))
predlabels_p = np.empty((len(reg_nameq),yearsall.shape[0]*2))
for rr in range(len(reg_nameq)):
    reg_name = reg_nameq[rr]
    variq = 'PRECT'
    ### Select how to save files
    if land_only == True:
        saveData = seasons[0] + '_LAND' + '_GCMarise_LOGREG' + '_' + variq + '_' + reg_name  + '_' + 'NumOfGCMS-' + str(num_of_class)
    elif ocean_only == True:
        saveData = seasons[0] + '_OCEAN' + '_GCMarise_LOGREG' + '_' + variq + '_' + reg_name + '_' + 'NumOfGCMS-' + str(num_of_class)
    else:
        saveData = seasons[0] + '_GCMarise_LOGREG' + '_' + variq + '_' + reg_name + '_' + 'NumOfGCMS-' + str(num_of_class)
    print('*Filename == < %s >' % saveData) 
    
    mapweights_p[rr] = np.load(directorydata + 'WeightsInputs-LOGREG_' + saveData + '.npy')
    latitudes_p[rr] = np.load(directorydata + 'Latitudes-LOGREG_' + saveData + '.npy')
    longitudes_p[rr] = np.load(directorydata + 'Longitudes-LOGREG_' + saveData + '.npy')
    truelabels_p[rr] = np.genfromtxt(directorydata + 'testingTrueLabels_' + saveData + '.txt')
    predlabels_p[rr] = np.genfromtxt(directorydata + 'testingPredictedLabels_' + saveData + '.txt')
    
### Prepare weights for evaluation
weights_t_sai = mapweights_t.squeeze().reshape(2,yearsall.shape[0],latshape,lonshape)[0]
weights_t_con = mapweights_t.squeeze().reshape(2,yearsall.shape[0],latshape,lonshape)[1]
weights_p_sai = mapweights_p.squeeze().reshape(2,yearsall.shape[0],latshape,lonshape)[0]
weights_p_con = mapweights_p.squeeze().reshape(2,yearsall.shape[0],latshape,lonshape)[1]

### Prepare classes for composites:
labels_t_sai = truelabels_t.squeeze().reshape(2,35)[0]
labels_t_con = truelabels_t.squeeze().reshape(2,35)[1]
labels_p_sai = truelabels_p.squeeze().reshape(2,35)[0]
labels_p_con = truelabels_p.squeeze().reshape(2,35)[1]

predss_t_sai = predlabels_t.squeeze().reshape(2,35)[0]
predss_t_con = predlabels_t.squeeze().reshape(2,35)[1]
predss_p_sai = predlabels_p.squeeze().reshape(2,35)[0]
predss_p_con = predlabels_p.squeeze().reshape(2,35)[1]


### Average across time
comp_t_sai = []
comp_t_con = []
comp_p_sai = []
comp_p_con = []
for i in range(len(yearsall)):
    if predss_t_sai[i] == 0.:
        comp_t_sai.append(weights_t_sai[i])
    if predss_t_con[i] == 1.:
        comp_t_con.append(weights_t_con[i])
    if predss_p_sai[i] == 0.:
        comp_p_sai.append(weights_p_sai[i])
    if predss_p_con[i] == 1.:
        comp_p_con.append(weights_p_con[i])
        
comp_t_sai_ready = np.nanmean(np.asarray(comp_t_sai),axis=0)
comp_t_con_ready = np.nanmean(np.asarray(comp_t_con),axis=0)
comp_p_sai_ready = np.nanmean(np.asarray(comp_p_sai),axis=0)
comp_p_con_ready = np.nanmean(np.asarray(comp_p_con),axis=0)

###############################################################################
###############################################################################
###############################################################################
### Plot subplot of observations
lon = longitudes_p
lat = latitudes_p

letters = ["a","b","c","d","e","f","g","h","i","j","k","l","m","n"]
limit = np.arange(-0.005,0.00501,0.0001)
barlim = np.round(np.arange(-0.005,0.00501,0.005),3)
cmap = cmocean.cm.balance
label = r'\textbf{[TREFHT] Input*Weights}'
limitd = np.arange(-0.005,0.00501,0.0001)
barlimd = np.round(np.arange(-0.005,0.00501,0.005),3)
cmapd = cmocean.cm.tarn
labeld = r'\textbf{[PRECT] Input*Weights}'

fig = plt.figure(figsize=(10,8))
###############################################################################
ax1 = plt.subplot(221)
m = Basemap(projection='moll',lon_0=0,resolution='l',area_thresh=10000)
m.drawcoastlines(color='k',linewidth=0.4,zorder=30)
    
### Variable
x, y = np.meshgrid(lon,lat)
   
circle = m.drawmapboundary(fill_color='dimgrey',color='dimgray',
                  linewidth=0.7)
circle.set_clip_on(False)

cs1 = m.contourf(x,y,comp_t_sai_ready,limit,extend='both',latlon=True)
cs1.set_cmap(cmap) 
m.drawlsmask(land_color=(0,0,0,0),ocean_color='dimgrey',lakes=False,zorder=11)

plt.title(r'\textbf{TREFHT - SAI}',fontsize=13,color='dimgrey')     
ax1.annotate(r'\textbf{[%s]}' % letters[0],xy=(0,0),xytext=(0.98,0.84),
              textcoords='axes fraction',color='k',fontsize=9,
              rotation=0,ha='center',va='center')

###############################################################################
###############################################################################
###############################################################################
ax2 = plt.subplot(223)
m = Basemap(projection='moll',lon_0=0,resolution='l',area_thresh=10000)
m.drawcoastlines(color='k',linewidth=0.4,zorder=30)

### Variable
circle = m.drawmapboundary(fill_color='dimgrey',color='dimgray',
                  linewidth=0.7)
circle.set_clip_on(False)

cs2 = m.contourf(x,y,comp_t_con_ready,limit,extend='both',latlon=True)
cs2.set_cmap(cmap) 
m.drawlsmask(land_color=(0,0,0,0),ocean_color='dimgrey',lakes=False,zorder=11)

plt.title(r'\textbf{TREFHT - CONTROL}',fontsize=13,color='dimgrey')         
ax2.annotate(r'\textbf{[%s]}' % letters[2],xy=(0,0),xytext=(0.98,0.84),
              textcoords='axes fraction',color='k',fontsize=9,
              rotation=0,ha='center',va='center')

###############################################################################
###############################################################################
###############################################################################
ax2 = plt.subplot(222)
m = Basemap(projection='moll',lon_0=0,resolution='l',area_thresh=10000)
m.drawcoastlines(color='k',linewidth=0.4,zorder=30)

### Variable
circle = m.drawmapboundary(fill_color='dimgrey',color='dimgray',
                  linewidth=0.7)
circle.set_clip_on(False)

cs2 = m.contourf(x,y,comp_p_sai_ready,limitd,extend='both',latlon=True)
cs2.set_cmap(cmapd) 
m.drawlsmask(land_color=(0,0,0,0),ocean_color='dimgrey',lakes=False,zorder=11)

plt.title(r'\textbf{PRECT - SAI}',fontsize=13,color='dimgrey')         
ax2.annotate(r'\textbf{[%s]}' % letters[1],xy=(0,0),xytext=(0.98,0.84),
              textcoords='axes fraction',color='k',fontsize=9,
              rotation=0,ha='center',va='center')

###############################################################################
cbar_ax1 = fig.add_axes([0.155,0.1,0.2,0.025])                
cbar1 = fig.colorbar(cs1,cax=cbar_ax1,orientation='horizontal',
                    extend='both',extendfrac=0.07,drawedges=False)
cbar1.set_label(label,fontsize=6,color='dimgrey',labelpad=1.4)  
cbar1.set_ticks(barlim)
cbar1.set_ticklabels(list(map(str,barlim)))
cbar1.ax.tick_params(axis='x', size=.01,labelsize=4)
cbar1.outline.set_edgecolor('dimgrey')

###############################################################################
###############################################################################
###############################################################################
ax2 = plt.subplot(224)
m = Basemap(projection='moll',lon_0=0,resolution='l',area_thresh=10000)
m.drawcoastlines(color='k',linewidth=0.4,zorder=30)

### Variable
circle = m.drawmapboundary(fill_color='dimgrey',color='dimgray',
                  linewidth=0.7)
circle.set_clip_on(False)

cs4 = m.contourf(x,y,comp_p_con_ready,limitd,extend='both',latlon=True)
cs4.set_cmap(cmapd) 
m.drawlsmask(land_color=(0,0,0,0),ocean_color='dimgrey',lakes=False,zorder=11)

plt.title(r'\textbf{PRECT - CONTROL}',fontsize=13,color='dimgrey')         
ax2.annotate(r'\textbf{[%s]}' % letters[3],xy=(0,0),xytext=(0.98,0.84),
              textcoords='axes fraction',color='k',fontsize=9,
              rotation=0,ha='center',va='center')

###############################################################################
cbar_ax1 = fig.add_axes([0.155,0.1,0.2,0.025])                
cbar1 = fig.colorbar(cs1,cax=cbar_ax1,orientation='horizontal',
                    extend='both',extendfrac=0.07,drawedges=False)
cbar1.set_label(label,fontsize=10,color='dimgrey',labelpad=1.4)  
cbar1.set_ticks(barlim)
cbar1.set_ticklabels(list(map(str,barlim)))
cbar1.ax.tick_params(axis='x', size=.01,labelsize=7)
cbar1.outline.set_edgecolor('dimgrey')

###############################################################################
cbar_axd1 = fig.add_axes([0.65,0.1,0.2,0.025])                
cbard1 = fig.colorbar(cs4,cax=cbar_axd1,orientation='horizontal',
                    extend='both',extendfrac=0.07,drawedges=False)
cbard1.set_label(labeld,fontsize=10,color='dimgrey',labelpad=1.4)  
cbard1.set_ticks(barlimd)
cbard1.set_ticklabels(list(map(str,barlimd)))
cbard1.ax.tick_params(axis='x', size=.01,labelsize=7)
cbard1.outline.set_edgecolor('dimgrey')

plt.tight_layout()
plt.subplots_adjust(hspace=-0.4)

plt.savefig(directoryfigure + 'LogReg_DetectSAI_OverallWeightsMaps_Globe.png',dpi=300)
    
    