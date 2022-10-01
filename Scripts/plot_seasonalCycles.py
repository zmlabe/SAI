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
directoryfigure = '//Users/zlabe/Desktop/SAI/'

###############################################################################
###############################################################################
###############################################################################
### Data preliminaries 
experiments = ['ARISE minus CONTROL','ARISE-PRE','ARISE-POST','DIFF-ARISE','WACCM-PRE','WACCM-POST','DIFF-WACCM']
dataset_obs = 'ERA5BE'
allDataLabels = ['ARISE','WACCM4.5']
letters = ["a","b","c","d","e","f","g","h","i","j","k","l","m","n"]
monq = [r'JAN',r'FEB',r'MAR',r'APR',r'MAY',r'JUN',r'JUL',r'AUG',r'SEP',r'OCT',r'NOV',r'DEC']
monthlychoiceq = ['none']
variables = ['PRECT']
variq = variables[0]
regionslist = ['SEAsia','Tropics','ENSO','CentralAfrica','Amazon','Globe','Indonesia','eqPacific','narrowTropics','nPacific','sPacific','LowerArctic','HighArctic']
regionslabels = ['Southeast Asia','Tropics','ENSO','Central Africa','Amazon','Global','Indonesia','Equatorial Pacific','Narrow Tropics','North Eq Pacific','South Eq Pacific','Lower Arctic','High Arctic']
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
ravelyearsbinary = False
ravelbinary = False
lensalso = True
###############################################################################
###############################################################################
###############################################################################
###############################################################################
### Read in model and observational/reanalysis data
def read_primary_dataset(variq,dataset,monthlychoice,numOfEns,lensalso,randomalso,ravelyearsbinary,ravelbinary,shuffletype,timeper,lat_bounds,lon_bounds):
    data,lats,lons = df.readFiles(variq,dataset,monthlychoice,numOfEns,lensalso,randomalso,ravelyearsbinary,ravelbinary,shuffletype,timeper)
    datar,lats,lons = df.getRegion(data,lats,lons,lat_bounds,lon_bounds)
    print('\nOur dataset: ',dataset,' is shaped',data.shape)
    return datar,lats,lons
###############################################################################
###############################################################################
###############################################################################
###############################################################################
### Read in WACCM
saveAllData_ARISE = []
saveAllData_WACCM = []
saveAllData_before = []
latsALL = []
lonsALL = []
latsALL2 = []
lonsALL2 = []
for rr in range(len(regionslist)):
    lat_bounds,lon_bounds = UT.regions(regionslist[rr])
    waccmq,lats,lons = read_primary_dataset(variq,'WACCM',monthlychoiceq[0],numOfEns,
                                            lensalso,randomalso,ravelyearsbinary,
                                            ravelbinary,shuffletype,timeper,
                                            lat_bounds,lon_bounds)
    ariseq,lats,lons = read_primary_dataset(variq,'ARISE',monthlychoiceq[0],numOfEns,
                                            lensalso,randomalso,ravelyearsbinary,
                                            ravelbinary,shuffletype,timeper,
                                            lat_bounds,lon_bounds)
    
    ### Reshape
    waccm = waccmq.reshape(numOfEns,waccmq.shape[1]//12,12,lats.shape[0],lons.shape[0])
    arise = ariseq.reshape(numOfEns,ariseq.shape[1]//12,12,lats.shape[0],lons.shape[0])
    
    ### Meshgrid
    lons2,lats2 = np.meshgrid(lons,lats)
        
    ### Concatenate ARISE to get prior to injection
    yearsall = np.arange(2015,2069+1,1)
    lengthdiff = yearswaccm.shape[0] - yearsarise.shape[0]
    injectionyear = lengthdiff
    
    priorSAI = waccm[:,:lengthdiff,:,:,:]
    allarise = np.append(priorSAI,arise,axis=1)
    
    ### Composites of before/after injections
    typeOfSlice = 'earlyLater'
    if typeOfSlice == 'direct':
        timeslice = 10
        beforeSAI = np.nanmean(allarise[:,injectionyear-timeslice:injectionyear,:,:,:],axis=1)
        afterSAI = np.nanmean(allarise[:,injectionyear:injectionyear+timeslice,:,:,:],axis=1)
        beforeCON = np.nanmean(waccm[:,injectionyear-timeslice:injectionyear,:,:,:],axis=1)
        afterCON = np.nanmean(waccm[:,injectionyear:injectionyear+timeslice,:,:,:],axis=1)
    elif typeOfSlice == 'later':
        timeslice = 10
        futurelater = 20
        beforeSAI = np.nanmean(allarise[:,injectionyear-timeslice:injectionyear,:,:,:],axis=1)
        afterSAI = np.nanmean(allarise[:,injectionyear+futurelater:injectionyear+timeslice+futurelater,:,:,:],axis=1)
        beforeCON = np.nanmean(waccm[:,injectionyear-timeslice:injectionyear,:,:,:],axis=1)
        afterCON = np.nanmean(waccm[:,injectionyear+futurelater:injectionyear+timeslice+futurelater,:,:,:],axis=1)
    elif typeOfSlice == 'earlyLater':
        timeslice = 10
        futurelater = 20
        beforeSAI = np.nanmean(allarise[:,:timeslice,:,:,:],axis=1)
        afterSAI = np.nanmean(allarise[:,injectionyear+futurelater:injectionyear+timeslice+futurelater,:,:,:],axis=1)
        beforeCON = np.nanmean(waccm[:,:timeslice,:,:,:],axis=1)
        afterCON = np.nanmean(waccm[:,injectionyear+futurelater:injectionyear+timeslice+futurelater,:,:,:],axis=1)
    
    diffSAIrun = afterSAI - beforeSAI
    diffCONrun = afterCON - beforeCON
    differenceSAI = afterSAI - afterCON
    
    ### Calculate regional means
    beforeSAIq = UT.calc_weightedAve(beforeSAI,lats2)
    afterSAIq = UT.calc_weightedAve(afterSAI,lats2)
    afterCONq = UT.calc_weightedAve(afterCON,lats2)
    
    ### Save files
    latsALL.append(lats)
    lonsALL.append(lons)
    latsALL2.append(lats2)
    lonsALL2.append(lons2)
    saveAllData_ARISE.append(afterSAIq) 
    saveAllData_WACCM.append(afterCONq)
    saveAllData_before.append(beforeSAIq)
    
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
minall = [0,3,4,0,3,2.5,4,4,3,2,2,1,0]
maxall = [12,4,6,4,9,3.5,10,6,5,9,9,2.2,2]
        
for i in range(len(regionslist)):
    fig = plt.figure()
    ax = plt.subplot(111) 
    
    varplota = saveAllData_ARISE[i]
    varplotw = saveAllData_WACCM[i]
    varbeforeplot = saveAllData_before[i]
        
    adjust_spines(ax, ['left', 'bottom'])            
    ax.spines['top'].set_color('none')
    ax.spines['right'].set_color('none')
    ax.spines['bottom'].set_linewidth(2)
    ax.spines['left'].set_linewidth(2)
    ax.tick_params('both',length=3,width=2,which='major')
    ax.tick_params(axis='x',labelsize=6.5,pad=1.5)
    ax.tick_params(axis='y',labelsize=6.5,pad=1.5)
    
    plt.plot(varbeforeplot.transpose(),linewidth=0.3,alpha=0.3,color='k',linestyle='-',clip_on=False)
    if typeOfSlice == 'earlyLater':
        plt.plot(np.nanmean(varbeforeplot,axis=0),linewidth=1.5,alpha=1,color='k',linestyle='-',clip_on=False,label=r'\textbf{2015-2024}')
    else: 
        plt.plot(np.nanmean(varbeforeplot,axis=0),linewidth=1.5,alpha=1,color='k',linestyle='-',clip_on=False,label=r'\textbf{2026-2035}')
    plt.plot(varplota.transpose(),linewidth=0.3,alpha=0.3,color='darkblue',linestyle='-',clip_on=False)
    plt.plot(np.nanmean(varplota,axis=0),linewidth=1.5,alpha=1,color='darkblue',linestyle='-',clip_on=False,label=r'\textbf{ARISE}')
    plt.plot(varplotw.transpose(),linewidth=0.3,alpha=0.3,color='crimson',linestyle='-',clip_on=False)
    plt.plot(np.nanmean(varplotw,axis=0),linewidth=1.5,alpha=1,color='crimson',linestyle='-',clip_on=False,label=r'\textbf{WACCM}')
    
    plt.xticks(np.arange(0,12,1),monq,rotation=0)
    plt.yticks(np.arange(0,30,1),np.arange(0,30,1),rotation=0)
    plt.xlim([0,11])
    plt.ylim([minall[i],maxall[i]])
    
    plt.ylabel(r'\textbf{PRECT [mm/day]}',fontsize=8,
                         color='dimgrey')
    plt.title(r'\textbf{Seasonal Cycle - %s}' % regionslabels[i],
                        color='k',fontsize=22)
    
    leg = plt.legend(shadow=False,fontsize=10,loc='upper center',
                bbox_to_anchor=(0.5, 1.035),fancybox=True,ncol=24,frameon=False,
                handlelength=1,handletextpad=1)
    for line,text in zip(leg.get_lines(), leg.get_texts()):
        text.set_color(line.get_color())
    
    plt.savefig(directoryfigure + 'Seasonalcycle_%s_%s.png' % (typeOfSlice,regionslist[i]),dpi=300)