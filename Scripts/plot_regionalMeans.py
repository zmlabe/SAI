"""
Script for plotting regional means

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
directoryfigure = '/Users/zlabe/Desktop/SAI/globalMeans/'

###############################################################################
###############################################################################
###############################################################################
### Data preliminaries 
dataset_obs = 'ERA5BE'
allDataLabels = ['ARISE','WACCM4.5']
letters = ["a","b","c","d","e","f","g","h","i","j","k","l","m","n"]
monthlychoiceq = ['JFM','AMJ','JAS','OND','annual']
# monthlychoiceq = ['annual']
variables = ['PRECT']
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
ocean_only = True
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
waccm = []
for i in range(len(monthlychoiceq)):
    waccmq,lats,lons = read_primary_dataset(variq,'WACCM',monthlychoiceq[i],numOfEns,
                                            lensalso,randomalso,ravelyearsbinary,
                                            ravelbinary,shuffletype,timeper,
                                            lat_bounds,lon_bounds)
    waccm.append(waccmq)

arise = []
for i in range(len(monthlychoiceq)):
    ariseq,lats,lons = read_primary_dataset(variq,'ARISE',monthlychoiceq[i],numOfEns,
                                            lensalso,randomalso,ravelyearsbinary,
                                            ravelbinary,shuffletype,timeper,
                                            lat_bounds,lon_bounds)
    arise.append(ariseq)
    
### Create arrays
arise = np.asarray(arise)
waccm = np.asarray(waccm)
    
### Meshgrid
lon2,lat2 = np.meshgrid(lons,lats)
    
### Concatenate ARISE to get prior to injection
yearsall = np.arange(2015,2069+1,1)
lengthdiff = yearswaccm.shape[0] - yearsarise.shape[0]
injectionyear = lengthdiff

priorSAI = waccm[:,:,:lengthdiff,:,:]
allarise = np.append(priorSAI,arise,axis=2)
allwaccm = waccm

### Mask ocean out                  
if land_only == True:
    allwaccm = UT.remove_ocean(allwaccm,lat_bounds,lon_bounds) 
    allarise = UT.remove_ocean(allarise,lat_bounds,lon_bounds) 
    allwaccm[np.where(allwaccm==0.)] = np.nan
    allarise[np.where(allarise==0.)] = np.nan 
    print('\n*Removed ocean data*')
    
### Mask land out
if ocean_only == True:
    allwaccm = UT.remove_land(allwaccm,lat_bounds,lon_bounds) 
    allarise = UT.remove_land(allarise,lat_bounds,lon_bounds) 
    allwaccm[np.where(allwaccm==0.)] = np.nan
    allarise[np.where(allarise==0.)] = np.nan 
    print('\n*Removed land data*') 
    
### Calculate global means
mean_waccm = UT.calc_weightedAve(allwaccm,lat2)
mean_arise = UT.calc_weightedAve(allarise,lat2)

minwaccm = np.percentile(mean_waccm,10,axis=1)
maxwaccm = np.percentile(mean_waccm,90,axis=1)
meanwaccm = np.nanmean(mean_waccm,axis=1)

minarise = np.percentile(mean_arise,10,axis=1)
maxarise = np.percentile(mean_arise,90,axis=1)
meanarise = np.nanmean(mean_arise,axis=1)

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
   
fig = plt.figure(figsize=(3,8))     
for i in range(len(monthlychoiceq)):
    ax = plt.subplot(5,1,i+1) 
        
    adjust_spines(ax, ['left', 'bottom'])            
    ax.spines['top'].set_color('none')
    ax.spines['right'].set_color('none')
    ax.spines['left'].set_color('dimgrey')
    ax.spines['bottom'].set_color('dimgrey')
    ax.spines['bottom'].set_linewidth(2)
    ax.spines['left'].set_linewidth(2)
    ax.tick_params('both',length=3,width=2,which='major')
    ax.tick_params(axis='x',labelsize=6.5,pad=1.5,color='dimgrey')
    ax.tick_params(axis='y',labelsize=6.5,pad=1.5,color='dimgrey')
    
    plt.fill_between(yearsall,minwaccm[i],maxwaccm[i],facecolor='deepskyblue',alpha=0.5)
    plt.plot(yearsall,meanwaccm[i],linestyle='-',color='deepskyblue',linewidth=1,
             clip_on=False,alpha=1,label=r'\textbf{WACCM}')
    plt.fill_between(yearsall[injectionyear:],minarise[i,injectionyear:],maxarise[i,injectionyear:],facecolor='crimson',alpha=0.5)
    plt.plot(yearsall[injectionyear:],meanarise[i,injectionyear:],linestyle='-',color='crimson',linewidth=1,
             clip_on=False,alpha=1,label=r'\textbf{ARISE}')
    
    plt.xticks(np.arange(2015,2080,20),np.arange(2015,2080,20),rotation=0)
    plt.yticks(np.arange(0,30,0.1),np.round(np.arange(0,30,0.1),2),rotation=0)
    plt.xlim([2015,2070])
    
    if land_only == True:
        plt.ylim([2.0,2.7])
        plt.text(2015,2.0,r'\textbf{%s}' % monthlychoiceq[i],fontsize=13,color='k')
    elif ocean_only == True:
        plt.ylim([3,3.5])
        plt.text(2015,3,r'\textbf{%s}' % monthlychoiceq[i],fontsize=13,color='k')
    else:
        plt.ylim([2.8,3.1])
        plt.text(2015,2.8,r'\textbf{%s}' % monthlychoiceq[i],fontsize=13,color='k')
    
    if i == 2:
        plt.ylabel(r'\textbf{PRECT [mm/day]}',fontsize=8,
                             color='dimgrey')
    
    if i ==0:
        leg = plt.legend(shadow=False,fontsize=10,loc='upper center',
                    bbox_to_anchor=(0.5, 1.2),fancybox=True,ncol=24,frameon=False,
                    handlelength=1,handletextpad=1)
        for line,text in zip(leg.get_lines(), leg.get_texts()):
            text.set_color(line.get_color())
    
plt.tight_layout()
plt.savefig(directoryfigure + 'Mean_%s_%s_land-%s_ocean-%s.png' % (reg_name,monthlychoiceq[i],land_only,ocean_only),dpi=300)