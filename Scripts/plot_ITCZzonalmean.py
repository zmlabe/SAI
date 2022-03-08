"""
Script for creating ITCZ metric to look at zonal mean changes in precipitation

Author     : Zachary M. Labe
Date       : 1 March 2022
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
directoryfigure = '/Users/zlabe/Desktop/SAI/zonalMean/'

###############################################################################
###############################################################################
###############################################################################
### Data preliminaries 
experiments = ['ARISE minus CONTROL','ARISE-PRE','ARISE-POST','DIFF-ARISE','WACCM-PRE','WACCM-POST','DIFF-WACCM']
dataset_obs = 'ERA5BE'
allDataLabels = ['ARISE','WACCM4.5']
letters = ["a","b","c","d","e","f","g","h","i","j","k","l","m","n"]
monthlychoiceq = ['JFM','AMJ','JAS','OND','annual']
# monthlychoiceq = ['annual']
variables = ['PRECT']
variq = variables[0]
reg_name = 'eqAtlantic'
level = 'surface'
if reg_name == 'eqPacific':
    latq = 22
    lonq = 57
    miny = -2.5
    maxy = 2.5
if reg_name == 'eqAtlantic':
    latq = 22
    lonq = 16
    miny = -1.5
    maxy = 1.5
if reg_name == 'ENSO':
    latq = 22
    lonq = 39
    miny = -3
    maxy = 3
if reg_name == 'Globe':
    latq = 96
    lonq = 144
    miny = -1.5
    maxy = 1.5
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
waccm = np.empty((len(monthlychoiceq),numOfEns,yearswaccm.shape[0],latq,lonq))
for i in range(len(monthlychoiceq)):
    waccm[i],lats,lons = read_primary_dataset(variq,'WACCM',monthlychoiceq[i],numOfEns,
                                            lensalso,randomalso,ravelyearsbinary,
                                            ravelbinary,shuffletype,timeper,
                                            lat_bounds,lon_bounds)

arise = np.empty((len(monthlychoiceq),numOfEns,yearsarise.shape[0],latq,lonq))
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

### Composites of before/after injections
typeOfSlice = 'earlyLater'
if typeOfSlice == 'direct':
    timeslice = 10
    beforeSAI = np.nanmean(allarise[:,:,injectionyear-timeslice:injectionyear,:,:],axis=2)
    afterSAI = np.nanmean(allarise[:,:,injectionyear:injectionyear+timeslice,:,:],axis=2)
    beforeCON = np.nanmean(waccm[:,:,injectionyear-timeslice:injectionyear,:,:],axis=2)
    afterCON = np.nanmean(waccm[:,:,injectionyear:injectionyear+timeslice,:,:],axis=2)
elif typeOfSlice == 'later':
    timeslice = 10
    futurelater = 20
    beforeSAI = np.nanmean(allarise[:,:,injectionyear-timeslice:injectionyear,:,:],axis=2)
    afterSAI = np.nanmean(allarise[:,:,injectionyear+futurelater:injectionyear+timeslice+futurelater,:,:],axis=2)
    beforeCON = np.nanmean(waccm[:,:,injectionyear-timeslice:injectionyear,:,:],axis=2)
    afterCON = np.nanmean(waccm[:,:,injectionyear+futurelater:injectionyear+timeslice+futurelater,:,:],axis=2)
elif typeOfSlice == 'earlyLater':
    timeslice = 10
    futurelater = 20
    beforeSAI = np.nanmean(allarise[:,:,:timeslice,:,:],axis=2)
    afterSAI = np.nanmean(allarise[:,:,injectionyear+futurelater:injectionyear+timeslice+futurelater,:,:],axis=2)
    beforeCON = np.nanmean(waccm[:,:,:timeslice,:,:],axis=2)
    afterCON = np.nanmean(waccm[:,:,injectionyear+futurelater:injectionyear+timeslice+futurelater,:,:],axis=2)

diffSAIrun = afterSAI - beforeSAI
diffCONrun = afterCON - beforeCON
differenceSAI = afterSAI - afterCON

### Calculate zonal mean
sai_zonal = np.nanmean(diffSAIrun,axis=3)
con_zonal = np.nanmean(diffCONrun,axis=3)
all_zonal = np.nanmean(differenceSAI,axis=3)

### Calculate ensemble mean
sai_zonalm = np.nanmean(sai_zonal,axis=1)
con_zonalm = np.nanmean(con_zonal,axis=1)
all_zonalm = np.nanmean(all_zonal,axis=1)

###############################################################################
### Adjust axes in time series plots 
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
###############################################################################
###############################################################################
###############################################################################    
for i in range(len(monthlychoiceq)):
    fig = plt.figure()
    ax = plt.subplot(111) 
        
    adjust_spines(ax, ['left', 'bottom'])            
    ax.spines['top'].set_color('none')
    ax.spines['right'].set_color('none')
    ax.spines['bottom'].set_linewidth(2)
    ax.spines['left'].set_linewidth(2)
    ax.tick_params('both',length=3,width=2,which='major')
    ax.tick_params(axis='x',labelsize=6.5,pad=1.5)
    ax.tick_params(axis='y',labelsize=6.5,pad=1.5)
    
    plt.plot(lats,con_zonalm[i],linestyle='-',color='mediumspringgreen',linewidth=1,label=r'\textbf{Climate Change}')
    plt.fill_between(lats,0,con_zonalm[i],color='mediumspringgreen')
    
    plt.plot(lats,sai_zonalm[i],linestyle='--',color='k',linewidth=2,dashes=(1,0.3),label=r'\textbf{Change from SAI}')
    plt.plot(lats,all_zonalm[i],linestyle='-',color='dodgerblue',linewidth=1,label=r'\textbf{Effect of SAI in future}')
  
    plt.xticks(np.arange(-90,91,15),np.arange(-90,91,15),rotation=0)
    plt.yticks(np.arange(-3,3.1,0.5),np.round(np.arange(-3,3.1,0.5),2),rotation=0)
    plt.xlim([-90,90])
    plt.ylim([miny,maxy])
    
    plt.ylabel(r'\textbf{$\Delta$PRECT [mm/day]}',fontsize=8,
                         color='dimgrey')
    plt.xlabel(r'\textbf{Latitude [$^{\circ}$]}',fontsize=8,
                         color='dimgrey')
    plt.title(r'\textbf{Zonal Mean - %s - %s}' % (monthlychoiceq[i],reg_name),
                        color='k',fontsize=17)
    
    leg = plt.legend(shadow=False,fontsize=6,loc='upper center',
                bbox_to_anchor=(0.5, 0.05),fancybox=True,ncol=24,frameon=False,
                handlelength=1,handletextpad=1)
    for line,text in zip(leg.get_lines(), leg.get_texts()):
        text.set_color(line.get_color())
    
    plt.savefig(directoryfigure + 'ZonalMean_%s_EnsembleMean_%s.png' % (reg_name,monthlychoiceq[i]),dpi=300)