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
directoryfigure = '/Users/zlabe/Desktop/SAI/Composites/'

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

### Calculate statistical significance
alpha_f = 0.1
pruns = np.empty((len(monthlychoiceq),lats.shape[0],lons.shape[0]))
for i in range(len(monthlychoiceq)):
    pruns[i,:,:] = UT.calc_FDR_ttest(afterSAI[i],afterCON[i],alpha_f)
    
###############################################################################
###############################################################################
###############################################################################
### Plot subplot of different SAI analysis
limit = np.arange(-0.5,0.501,0.01)
barlim = np.round(np.arange(-0.5,0.51,0.1),2)

for mo in range(len(monthlychoiceq)):                                                                                                                         
    label = r'\textbf{%s -- 2055-2065 -- [mm/day]}' % (monthlychoiceq[mo])
    
    fig = plt.figure()
    ax = plt.subplot(111)
    
    var = np.nanmean(afterSAI[mo] - afterCON[mo],axis=0)
    pvar = pruns[mo]

    m = Basemap(projection='robin',lon_0=0,resolution='l',area_thresh=10000)
    m.drawcoastlines(color='dimgrey',linewidth=0.5)
        
    var, lons_cyclic = addcyclic(var, lons)
    var, lons_cyclic = shiftgrid(180., var, lons_cyclic, start=False)
    pvar, lons_cyclic = addcyclic(pvar, lons)
    pvar, lons_cyclic = shiftgrid(180., pvar, lons_cyclic, start=False)
    lon2d, lat2d = np.meshgrid(lons_cyclic, lats)
    x, y = m(lon2d, lat2d)
    
    circle = m.drawmapboundary(fill_color='white',color='dimgray',
                      linewidth=0.7)
    circle.set_clip_on(False)
    
    cs1 = m.contourf(x,y,var,limit,extend='both')
    cs2 = m.contourf(x,y,pvar,colors='None',hatches=['///////'])
    
    cs1.set_cmap(cmocean.cm.balance)
    
    cbar1 = plt.colorbar(cs1,orientation='horizontal',
                        extend='both',extendfrac=0.07,drawedges=False,
                        fraction=0.03)
    cbar1.set_label(label,fontsize=10,color='dimgrey',labelpad=2)  
    cbar1.set_ticks(barlim)
    cbar1.set_ticklabels(list(map(str,barlim)))
    cbar1.ax.tick_params(axis='x', size=.01,labelsize=5)
    cbar1.outline.set_edgecolor('dimgrey')
    
    plt.tight_layout()
    
    plt.savefig(directoryfigure + 'endOfCenturyCompositeStatisticalSignificances_PRECT_%s_%s.png' % (monthlychoiceq[mo],typeOfSlice),dpi=300)