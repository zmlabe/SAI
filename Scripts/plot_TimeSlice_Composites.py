"""
Script for creating composites to look at timeslice differences between WACCM
and ARISE

Author     : Zachary M. Labe
Date       : 28 February 2022
Version    : 1
"""

### Import packages
import sys
import math
import time
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy.stats as stats
from mpl_toolkits.basemap import Basemap, addcyclic, shiftgrid
import palettable.cubehelix as cm
import palettable.scientific.sequential as sss
import cmocean as cmocean
import calc_Utilities as UT
import calc_dataFunctions as df
import calc_Stats as dSS
import scipy.stats as sts
import matplotlib
import cmasher as cmr

### Plotting defaults 
plt.rc('text',usetex=True)
plt.rc('font',**{'family':'sans-serif','sans-serif':['Avant Garde']}) 

###############################################################################
###############################################################################
###############################################################################
### Data preliminaries 
experiments = ['ARISE minus CONTROL','ARISE-PRE','ARISE-POST','DIFF-ARISE','WACCM-PRE','WACCM-POST','DIFF-WACCM']
dataset_obs = 'ERA5BE'
allDataLabels = ['ARISE','WACCM4.5']
letters = ["a","b","c","d","e","f","g","h","i","j","k","l","m","n"]
monthlychoiceq = ['JFM','AMJ','JAS','OND','annual']
variables = ['TREFHT']
variq = variables[0]
reg_name = 'Globe'
level = 'surface'
###############################################################################
###############################################################################
randomalso = False
timeper = 'historical'
shuffletype = 'GAUSS'
###############################################################################
###############################################################################
land_only = False
ocean_only = False
###############################################################################
###############################################################################
window = 0
yearswaccm = np.arange(2015,2069+1,1)
yearsarise = np.arange(2035,2069+1,1)
###############################################################################
###############################################################################
numOfEns = 10
###############################################################################
###############################################################################
lat_bounds,lon_bounds = UT.regions(reg_name)
###############################################################################
###############################################################################
ravelyearsbinary = False
ravelbinary = False
lensalso = True
###############################################################################
###############################################################################
###############################################################################
###############################################################################
### Read in model and observational/reanalysis data
def read_primary_dataset(variq,dataset,monthlychoice,numOfEns,lensalso,randomalso,ravelyearsbinary,ravelbinary,shuffletype,timeper,lat_bounds=lat_bounds,lon_bounds=lon_bounds):
    data,lats,lons = df.readFiles(variq,dataset,monthlychoice,numOfEns,lensalso,randomalso,ravelyearsbinary,ravelbinary,shuffletype,timeper)
    datar,lats,lons = df.getRegion(data,lats,lons,lat_bounds,lon_bounds)
    print('\nOur dataset: ',dataset,' is shaped',data.shape)
    return datar,lats,lons
###############################################################################
###############################################################################
###############################################################################
###############################################################################
### Read in WACCM
waccm = np.empty((len(monthlychoiceq),numOfEns,yearswaccm.shape[0],96,144))
for i in range(len(monthlychoiceq)):
    waccm[i],lats,lons = read_primary_dataset(variq,'WACCM',monthlychoiceq[i],numOfEns,
                                            lensalso,randomalso,ravelyearsbinary,
                                            ravelbinary,shuffletype,timeper,
                                            lat_bounds,lon_bounds)

arise = np.empty((len(monthlychoiceq),numOfEns,yearsarise.shape[0],96,144))
for i in range(len(monthlychoiceq)):
    arise[i],lats,lons = read_primary_dataset(variq,'ARISE',monthlychoiceq[i],numOfEns,
                                            lensalso,randomalso,ravelyearsbinary,
                                            ravelbinary,shuffletype,timeper,
                                            lat_bounds,lon_bounds)
    
### Meshgrid
lon2,lat2 = np.meshgrid(lons,lats)
    
### Concatenate ARISE to get prior to injection
yearsall = np.arange(2015,2069+1,1)
lengthdiff = yearswaccm.shape[0] - yearsarise.shape[0]
injectionyear = lengthdiff

priorSAI = waccm[:,:,:lengthdiff,:,:]
allarise = np.append(priorSAI,arise,axis=2)

### Take ensemble mean
waccmm = np.nanmean(waccm[:,:,:,:,:],axis=1)
arisem = np.nanmean(allarise[:,:,:,:,:],axis=1)

### Calculate global means
waccm_globalmean = UT.calc_weightedAve(waccm,lat2)
arise_globalmean = UT.calc_weightedAve(allarise,lat2)
waccm_globalmeanm = np.nanmean(waccm_globalmean,axis=1)
arise_globalmeanm = np.nanmean(arise_globalmean,axis=1)

fig = plt.figure()
a=waccm_globalmean[-1,:,:]
b=arise_globalmean[-1,:,:]
plt.plot(a.transpose(),color='r')
plt.plot(b.transpose(),color='b')
plt.ylim([14.8,17])


sys.exit()
### Composites of before/after injections
typeOfSlice = 'direct'
if typeOfSlice == 'direct':
    timeslice = 10
    beforeSAI = np.nanmean(arisem[:,injectionyear-timeslice:injectionyear,:,:],axis=1)
    afterSAI = np.nanmean(arisem[:,injectionyear:injectionyear+timeslice,:,:],axis=1)
    beforeCON = np.nanmean(waccmm[:,injectionyear-timeslice:injectionyear,:,:],axis=1)
    afterCON = np.nanmean(waccmm[:,injectionyear:injectionyear+timeslice,:,:],axis=1)
elif typeOfSlice == 'later':
    timeslice = 10
    futurelater = 20
    beforeSAI = np.nanmean(arisem[:,injectionyear-timeslice:injectionyear,:,:],axis=1)
    afterSAI = np.nanmean(arisem[:,injectionyear+futurelater:injectionyear+timeslice+futurelater,:,:],axis=1)
    beforeCON = np.nanmean(waccmm[:,injectionyear-timeslice:injectionyear,:,:],axis=1)
    afterCON = np.nanmean(waccmm[:,injectionyear+futurelater:injectionyear+timeslice+futurelater,:,:],axis=1)

diffSAIrun = afterSAI - beforeSAI
diffCONrun = afterCON - beforeCON
differenceSAI = afterSAI - afterCON

###############################################################################
###############################################################################
###############################################################################
### Plotting parameters
directoryfigure = '//Users/zlabe/Desktop/SAI/'
limitd = np.arange(-1,1.01,0.01)
barlimd = np.round(np.arange(-1,1.1,1),2)
limitc = np.arange(0,10.01,0.01)
barlimc = np.round(np.arange(0,11,2),2)
arangeForPlotting = [differenceSAI,beforeSAI,afterSAI,diffSAIrun,
                     beforeCON,afterCON,diffCONrun]
cmapq = [cmocean.cm.tarn,cmocean.cm.rain,cmocean.cm.rain,cmocean.cm.tarn,
         '',cmocean.cm.rain,cmocean.cm.rain,cmocean.cm.tarn]
limitq = [limitd,limitc,limitc,limitd,
          '',limitc,limitc,limitd]
barlimq = [barlimd,barlimc,barlimc,barlimd,
           '',barlimc,barlimc,barlimd]
loc = [0,1,2,3,5,6,7]
    
###############################################################################
###############################################################################
###############################################################################
### Plot subplot of different SAI analysis

for mo in range(len(monthlychoiceq)):                                                                                                                         
    label = r'\textbf{%s -- %s -- [mm/day]}' % (monthlychoiceq[mo],typeOfSlice)
    
    fig = plt.figure(figsize=(8,4))
    for r in range(len(arangeForPlotting)):
        var = arangeForPlotting[r][mo]
        
        ax1 = plt.subplot(2,4,loc[r]+1)
        m = Basemap(projection='moll',lon_0=0,resolution='l',area_thresh=10000)
        m.drawcoastlines(color='dimgrey',linewidth=0.27)
            
        var, lons_cyclic = addcyclic(var, lons)
        var, lons_cyclic = shiftgrid(180., var, lons_cyclic, start=False)
        lon2d, lat2d = np.meshgrid(lons_cyclic, lats)
        x, y = m(lon2d, lat2d)
        circle = m.drawmapboundary(fill_color='white',color='dimgray',
                          linewidth=0.7)
        circle.set_clip_on(False)
        
        cs1 = m.contourf(x,y,var,limitq[loc[r]],extend='both')
        cs1.set_cmap(cmapq[loc[r]]) 
                
        ax1.annotate(r'\textbf{%s}' % experiments[r],xy=(0,0),xytext=(0.5,1.10),
                      textcoords='axes fraction',color='dimgrey',fontsize=8,
                      rotation=0,ha='center',va='center')
        ax1.annotate(r'\textbf{[%s]}' % letters[r],xy=(0,0),xytext=(0.86,0.97),
                      textcoords='axes fraction',color='k',fontsize=6,
                      rotation=330,ha='center',va='center')
        
        cbar1 = plt.colorbar(cs1,orientation='horizontal',
                            extend='both',extendfrac=0.07,drawedges=False)
        cbar1.set_label(label,fontsize=5,color='dimgrey',labelpad=1.4)  
        cbar1.set_ticks(barlimq[loc[r]])
        cbar1.set_ticklabels(list(map(str,barlimq[loc[r]])))
        cbar1.ax.tick_params(axis='x', size=.01,labelsize=5)
        cbar1.outline.set_edgecolor('dimgrey')
    
    plt.tight_layout()
    
    plt.savefig(directoryfigure + 'SAI_Composites_PRECT_%s_%s.png' % (monthlychoiceq[mo],typeOfSlice),dpi=300)