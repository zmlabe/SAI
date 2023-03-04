"""
Graphic showing spread in ensemble trends over two periods between WACCM/ARISE

Author     : Zachary M. Labe
Date       : 27 July 2022
Version    : 1 - testing ANN architectures for calculating years since SAI
"""

### Import packages
import sys
import matplotlib.pyplot as plt
import matplotlib.colors as c
from mpl_toolkits.basemap import Basemap, addcyclic, shiftgrid
import cmocean
import cmasher as cmr
import numpy as np
import calc_Utilities as UT
import calc_Stats as dSS
import calc_dataFunctions as df
import scipy.stats as sts

### Plotting defaults 
plt.rc('text',usetex=True)
plt.rc('font',**{'family':'sans-serif','sans-serif':['Avant Garde']}) 

###############################################################################
###############################################################################
###############################################################################
### Data preliminaries 
letters = ["a","b","c","d","e","f","g","h","i","j","k","l","m","o","p"]
reg_nameq = ['Globe','NH','SH','Arctic','Antarctic','narrowTropics','SEAsia','NorthAfrica','Amazon']
labels = ['Globe','N. Hemisphere','S. Hemisphere','Arctic','Antarctic','Tropics','Southeast Asia','Central Africa','Amazon']
regionboxes = ['Arctic','Antarctic','narrowTropics','SEAsia','NorthAfrica','Amazon']
directorydata = '/Users/zlabe/Data/SAI/'
directoryfigure = '/Users/zlabe/Documents/Research/SolarIntervention/Figures/'
###############################################################################
###############################################################################
modelGCMs = ['ARISE']
datasetsingle = ['ARISE']
seasons = ['annual']
monthlychoice = seasons[0]
###############################################################################
###############################################################################
land_only = True
ocean_only = False
ravelyearsbinary = False
ravelbinary = False
lensalso = True
randomalso = False
ravel_modelens = False
ravelmodeltime = False
timeper = 'historical'
shuffletype = 'GAUSS'
###############################################################################
###############################################################################
yearsall = np.arange(2035,2069+1,1)
yearsarise = np.arange(2035,2069+1,1)
numOfEns = 10
dataset_obs = 'ERA5'
###############################################################################
###############################################################################
num_of_class = len(modelGCMs)
ensTypeExperi = 'ENS'
###############################################################################
###############################################################################
###############################################################################  
### Read in data
reg_name = 'Globe'
lat_bounds,lon_bounds = UT.regions(reg_name)
###############################################################################   
###############################################################################   
###############################################################################   
def read_primary_dataset(variq,dataset,numOfEns,lensalso,randomalso,ravelyearsbinary,ravelbinary,shuffletype,lat_bounds=lat_bounds,lon_bounds=lon_bounds):
    data,lats,lons = df.readFiles(variq,dataset,monthlychoice,numOfEns,lensalso,randomalso,ravelyearsbinary,ravelbinary,shuffletype,timeper)
    data = data[np.newaxis,:,:,:,:] # add model dimension
    datar,lats,lons = df.getRegion(data,lats,lons,lat_bounds,lon_bounds)
    print('\nOur dataset: ',dataset,' is shaped',data.shape)
    return datar,lats,lons
  
def read_obs_dataset(variq,dataset_obs,numOfEns,lensalso,randomalso,ravelyearsbinary,ravelbinary,shuffletype,lat_bounds=lat_bounds,lon_bounds=lon_bounds):
    if variq == 'TREFHT':
        variq = 'T2M'
    elif variq == 'TREFHT':
        variq = 'P'
    data_obs,lats_obs,lons_obs = df.readFiles(variq,dataset_obs,monthlychoice,numOfEns,lensalso,randomalso,ravelyearsbinary,ravelbinary,shuffletype,timeper)
    data_obs,lats_obs,lons_obs = df.getRegion(data_obs,lats_obs,lons_obs,
                                            lat_bounds,lon_bounds)     
    print('our OBS dataset: ',dataset_obs,' is shaped',data_obs.shape)
    return data_obs,lats_obs,lons_obs
###############################################################################   
###############################################################################   
###############################################################################  
### Calculate linear trends
def calcTrend(data):
    if data.ndim == 3:
        slopes = np.empty((data.shape[1],data.shape[2]))
        x = np.arange(data.shape[0])
        for i in range(data.shape[1]):
            for j in range(data.shape[2]):
                mask = np.isfinite(data[:,i,j])
                y = data[:,i,j]
                
                if np.sum(mask) == y.shape[0]:
                    xx = x
                    yy = y
                else:
                    xx = x[mask]
                    yy = y[mask]      
                if np.isfinite(np.nanmean(yy)):
                    slopes[i,j],intercepts, \
                    r_value,p_value,std_err = sts.linregress(xx,yy)
                else:
                    slopes[i,j] = np.nan
    elif data.ndim == 4:
        slopes = np.empty((data.shape[0],data.shape[2],data.shape[3]))
        x = np.arange(data.shape[1])
        for ens in range(data.shape[0]):
            for i in range(data.shape[2]):
                for j in range(data.shape[3]):
                    mask = np.isfinite(data[ens,:,i,j])
                    y = data[ens,:,i,j]
                    
                    if np.sum(mask) == y.shape[0]:
                        xx = x
                        yy = y
                    else:
                        xx = x[mask]
                        yy = y[mask]      
                    if np.isfinite(np.nanmean(yy)):
                        slopes[ens,i,j],intercepts, \
                        r_value,p_value,std_err = sts.linregress(xx,yy)
                    else:
                        slopes[ens,i,j] = np.nan
            print('-- Completed ensemble member #%s! --' % (ens+1))
    
    dectrend = slopes * 10.   
    print('Completed: Finished calculating trends!')     
    return dectrend
###############################################################################   
###############################################################################   
###############################################################################
### Read in data for temperature
data_arise,lats,lons = read_primary_dataset('TREFHT','ARISE',
                                          numOfEns,lensalso,
                                          randomalso,
                                          ravelyearsbinary,
                                          ravelbinary,
                                          shuffletype,
                                          lat_bounds,
                                          lon_bounds)
data_waccmall,lats,lons = read_primary_dataset('TREFHT','WACCM',
                                          numOfEns,lensalso,
                                          randomalso,
                                          ravelyearsbinary,
                                          ravelbinary,
                                          shuffletype,
                                          lat_bounds,
                                          lon_bounds)
data_obs = np.empty((data_arise.shape))
data_obs[:] = np.nan
lon2,lat2 = np.meshgrid(lons,lats)

### Only include 2035-2069 for comparing trends
data_waccm = data_waccmall[:,:,-yearsall.shape[0]:,:,:].squeeze()
data_arise = data_arise.squeeze()

### Calculate two trend periods
yearq1 = np.where((yearsall >= 2035) & (yearsall <= 2044))[0]
yearq2 = np.where((yearsall >= 2045) & (yearsall <= 2069))[0]

per1_trend_waccm = calcTrend(data_waccm[:,yearq1,:,:])
per2_trend_waccm = calcTrend(data_waccm[:,yearq2,:,:])
per1_trend_arise = calcTrend(data_arise[:,yearq1,:,:])
per2_trend_arise = calcTrend(data_arise[:,yearq2,:,:])

###############################################################################   
###############################################################################   
###############################################################################   
### Remove ocean
per1_trend_waccm, data_obs = dSS.remove_ocean(per1_trend_waccm,data_obs,lat_bounds,lon_bounds) 
per2_trend_waccm, data_obs = dSS.remove_ocean(per2_trend_waccm,data_obs,lat_bounds,lon_bounds)
per1_trend_arise, data_obs = dSS.remove_ocean(per1_trend_arise,data_obs,lat_bounds,lon_bounds) 
per2_trend_arise, data_obs = dSS.remove_ocean(per2_trend_arise,data_obs,lat_bounds,lon_bounds)

### Calculate max trend
max1_waccm = np.nanmax(per1_trend_waccm,axis=0)
max2_waccm = np.nanmax(per2_trend_waccm,axis=0)
max1_arise = np.nanmax(per1_trend_arise,axis=0)
max2_arise = np.nanmax(per2_trend_arise,axis=0)

### Calculate min trend
min1_waccm = np.nanmin(per1_trend_waccm,axis=0)
min2_waccm = np.nanmin(per2_trend_waccm,axis=0)
min1_arise = np.nanmin(per1_trend_arise,axis=0)
min2_arise = np.nanmin(per2_trend_arise,axis=0)

### Calculate spread
spread1_waccm = max1_waccm - min1_waccm
spread2_waccm = max2_waccm - min2_waccm
spread1_arise = max1_arise - min1_arise
spread2_arise = max2_arise - min2_arise

###############################################################################   
###############################################################################   
###############################################################################
###############################################################################   
###############################################################################   
###############################################################################
###############################################################################   
###############################################################################   
###############################################################################
###############################################################################   
###############################################################################   
###############################################################################
### Read in data for precipitation 
datap_arise,lats,lons = read_primary_dataset('PRECT','ARISE',
                                          numOfEns,lensalso,
                                          randomalso,
                                          ravelyearsbinary,
                                          ravelbinary,
                                          shuffletype,
                                          lat_bounds,
                                          lon_bounds)
datap_waccmall,lats,lons = read_primary_dataset('PRECT','WACCM',
                                          numOfEns,lensalso,
                                          randomalso,
                                          ravelyearsbinary,
                                          ravelbinary,
                                          shuffletype,
                                          lat_bounds,
                                          lon_bounds)
datap_obs = np.empty((datap_arise.shape))
datap_obs[:] = np.nan
lon2,lat2 = np.meshgrid(lons,lats)

### Only include 2035-2069 for comparing trends
datap_waccm = datap_waccmall[:,:,-yearsall.shape[0]:,:,:].squeeze()
datap_arise = datap_arise.squeeze()

### Calculate two trend periods
yearq1 = np.where((yearsall >= 2035) & (yearsall <= 2044))[0]
yearq2 = np.where((yearsall >= 2045) & (yearsall <= 2069))[0]

per1_trendp_waccm = calcTrend(datap_waccm[:,yearq1,:,:])
per2_trendp_waccm = calcTrend(datap_waccm[:,yearq2,:,:])
per1_trendp_arise = calcTrend(datap_arise[:,yearq1,:,:])
per2_trendp_arise = calcTrend(datap_arise[:,yearq2,:,:])

###############################################################################   
###############################################################################   
###############################################################################   
### Remove ocean
per1_trendp_waccm, datap_obs = dSS.remove_ocean(per1_trendp_waccm,datap_obs,lat_bounds,lon_bounds) 
per2_trendp_waccm, datap_obs = dSS.remove_ocean(per2_trendp_waccm,datap_obs,lat_bounds,lon_bounds)
per1_trendp_arise, datap_obs = dSS.remove_ocean(per1_trendp_arise,datap_obs,lat_bounds,lon_bounds) 
per2_trendp_arise, datap_obs = dSS.remove_ocean(per2_trendp_arise,datap_obs,lat_bounds,lon_bounds)

### Calculate max trend
max1p_waccm = np.nanmax(per1_trendp_waccm,axis=0)
max2p_waccm = np.nanmax(per2_trendp_waccm,axis=0)
max1p_arise = np.nanmax(per1_trendp_arise,axis=0)
max2p_arise = np.nanmax(per2_trendp_arise,axis=0)

### Calculate min trend
min1p_waccm = np.nanmin(per1_trendp_waccm,axis=0)
min2p_waccm = np.nanmin(per2_trendp_waccm,axis=0)
min1p_arise = np.nanmin(per1_trendp_arise,axis=0)
min2p_arise = np.nanmin(per2_trendp_arise,axis=0)

### Calculate spread
spread1p_waccm = max1p_waccm - min1p_waccm
spread2p_waccm = max2p_waccm - min2p_waccm
spread1p_arise = max1p_arise - min1p_arise
spread2p_arise = max2p_arise - min2p_arise

### Create lists for plotting
plotvar = [spread1_arise,spread1_waccm,spread1p_arise,spread1p_waccm,
           spread2_arise,spread2_waccm,spread2p_arise,spread2p_waccm]
colormaps = [cmr.sunburst,cmr.sunburst,cmr.savanna,cmr.savanna,
             cmr.sunburst,cmr.sunburst,cmr.savanna,cmr.savanna]
limits = [np.arange(0,3.01,0.01),np.arange(0,3.01,0.01),np.arange(0,3.01,0.01),np.arange(0,3.01,0.01),
          np.arange(0,3.01,0.01),np.arange(0,3.01,0.01),np.arange(0,3.01,0.01),np.arange(0,3.01,0.01)]
barlims = [np.round(np.arange(0,3.01,0.5),2),np.round(np.arange(0,3.01,0.5),2),np.round(np.arange(0,3.01,0.5),2),np.round(np.arange(0,3.01,0.5),2),
           np.round(np.arange(0,3.01,0.5),2),np.round(np.arange(0,3.01,0.5),2),np.round(np.arange(0,3.01,0.5),2),np.round(np.arange(0,3.01,0.5),2)]
titles = ['SAI-1.5','SSP2-4.5','SAI-1.5','SSP2-4.5']

###############################################################################
###############################################################################
###############################################################################
### Graphs
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
def setcolor(x, color):
      for m in x:
          for t in x[m][1]:
              t.set_color(color)
        
fig = plt.figure(figsize=(8,3))
for i in range(len(plotvar)):
    ax = plt.subplot(2,4,i+1)
    
    var = plotvar[i]
    limit = limits[i]
    barlim = barlims[i]

    m = Basemap(projection='robin',lon_0=0,resolution='l',area_thresh=10000)
    m.drawcoastlines(color='darkgrey',linewidth=0.4)
        
    var, lons_cyclic = addcyclic(var, lons)
    var, lons_cyclic = shiftgrid(180., var, lons_cyclic, start=False)
    lon2d, lat2d = np.meshgrid(lons_cyclic, lats)
    x, y = m(lon2d, lat2d)
    
    parallels = np.arange(-90,91,30)
    meridians = np.arange(-180,180,60)
    par=m.drawparallels(parallels,labels=[False,False,False,False],linewidth=0.3,
                    color='w',fontsize=4,zorder=40)
    mer=m.drawmeridians(meridians,labels=[False,False,False,False],linewidth=0.3,
                        fontsize=4,color='w',zorder=40)
    
    circle = m.drawmapboundary(fill_color='dimgray',color='dimgray',
                      linewidth=1)
    circle.set_clip_on(False)
    
    if any([i==0,i==1,i==4,i==5]):
        cs1 = m.contourf(x,y,var,limit,extend='max')
        cs1.set_cmap(colormaps[i])
    else:
        cs2 = m.contourf(x,y,var,limit,extend='max')
        cs2.set_cmap(colormaps[i])        
    
    m.drawlsmask(land_color=(0,0,0,0),ocean_color='dimgray',lakes=False,zorder=11)

    plt.annotate(r'\textbf{[%s]}' % letters[i],
              textcoords='axes fraction',
              xy=(0,0), xytext=(0.96,0.92),
              fontsize=8,color='k',alpha=1,
              ha='center',va='center',zorder=30)
    
    if i == 0:
        plt.annotate(r'\textbf{2035-2044}',
                      textcoords='axes fraction',
                      xy=(0,0), xytext=(-0.08,0.5),
                      fontsize=12,color='dimgrey',alpha=1,
                      ha='center',va='center',zorder=30,rotation=90)
    if i == 4:
        plt.annotate(r'\textbf{2045-2069}',
                      textcoords='axes fraction',
                      xy=(0,0), xytext=(-0.08,0.5),
                      fontsize=12,color='dimgrey',alpha=1,
                      ha='center',va='center',zorder=30,rotation=90)

    if i < 4:
        plt.title(r'\textbf{%s}' % titles[i],color='dimgrey',fontsize=18)

cbar_ax = fig.add_axes([0.12,0.105,0.31,0.026])                
cbar = fig.colorbar(cs1,cax=cbar_ax,orientation='horizontal',
                    extend='max',extendfrac=0.07,drawedges=False)
cbar.set_label(r'\textbf{ENSEMBLE SPREAD -- Temperature [$^{\circ}$C/decade]}',fontsize=9,color='dimgrey',labelpad=1.4)  
cbar.set_ticks(barlims[0])
cbar.set_ticklabels(list(map(str,barlims[0])))
cbar.ax.tick_params(axis='x', size=.01,labelsize=7)
cbar.outline.set_edgecolor('dimgrey')

cbar_ax2 = fig.add_axes([0.60,0.105,0.31,0.026])                
cbar2 = fig.colorbar(cs2,cax=cbar_ax2,orientation='horizontal',
                    extend='max',extendfrac=0.07,drawedges=False)
cbar2.set_label(r'\textbf{ENSEMBLE SPREAD -- Precipitation [mm/day/decade]}',fontsize=9,color='dimgrey',labelpad=1.4)  
cbar2.set_ticks(barlims[-1])
cbar2.set_ticklabels(list(map(str,barlims[-1])))
cbar2.ax.tick_params(axis='x', size=.01,labelsize=7)
cbar2.outline.set_edgecolor('dimgrey')
    
plt.tight_layout()
plt.subplots_adjust(bottom=0.14)
plt.savefig(directoryfigure + 'Map_SpreadTrends_TREFHT-PRECT.png',dpi=500)
