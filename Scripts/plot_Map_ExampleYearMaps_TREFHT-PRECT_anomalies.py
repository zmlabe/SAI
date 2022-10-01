"""
Graphic showing example map of one year-one ensemble for ARISE/WACCM for TREFHT/PRECT

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
    elif variq == 'PRECT':
        variq = 'P'
    data_obs,lats_obs,lons_obs = df.readFiles(variq,dataset_obs,monthlychoice,numOfEns,lensalso,randomalso,ravelyearsbinary,ravelbinary,shuffletype,timeper)
    data_obs,lats_obs,lons_obs = df.getRegion(data_obs,lats_obs,lons_obs,
                                            lat_bounds,lon_bounds)     
    print('our OBS dataset: ',dataset_obs,' is shaped',data_obs.shape)
    return data_obs,lats_obs,lons_obs
###############################################################################   
###############################################################################   
###############################################################################
### Read in data  
datat_arise,lats,lons = read_primary_dataset('TREFHT','ARISE',
                                          numOfEns,lensalso,
                                          randomalso,
                                          ravelyearsbinary,
                                          ravelbinary,
                                          shuffletype,
                                          lat_bounds,
                                          lon_bounds)
datat_waccmall,lats,lons = read_primary_dataset('TREFHT','WACCM',
                                          numOfEns,lensalso,
                                          randomalso,
                                          ravelyearsbinary,
                                          ravelbinary,
                                          shuffletype,
                                          lat_bounds,
                                          lon_bounds)
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
data_obs = np.empty((datat_arise.shape))
data_obs[:] = np.nan
lon2,lat2 = np.meshgrid(lons,lats)

### Only include 2035-2069 for comparing trends
datat_waccm = datat_waccmall[:,:,-yearsall.shape[0]:,:,:].squeeze()
datap_waccm = datap_waccmall[:,:,-yearsall.shape[0]:,:,:].squeeze()
datat_arise = datat_arise.squeeze()
datap_arise = datap_arise.squeeze()

### Slice same year and ensemble member
ensindext = 9
ensindexp = 9
yearindex = 10

### Calculate climatology
climot = np.nanmean(datat_waccmall[:,ensindext,:yearsall.shape[0],:,:],axis=1).squeeze()
climop = np.nanmean(datap_waccmall[:,ensindexp,:yearsall.shape[0],:,:],axis=1).squeeze()

### Calculate anomalies for selected
waccmt = datat_waccm[ensindext,yearindex,:,:] - climot
waccmp = datap_waccm[ensindexp,yearindex,:,:] - climop
ariset = datat_arise[ensindext,yearindex,:,:] - climot
arisep = datap_arise[ensindexp,yearindex,:,:] - climop

###############################################################################   
###############################################################################   
###############################################################################   
### Remove ocean
waccmt, data_obs = dSS.remove_ocean(waccmt,data_obs,lat_bounds,lon_bounds) 
waccmp, data_obs = dSS.remove_ocean(waccmp,data_obs,lat_bounds,lon_bounds)
ariset, data_obs = dSS.remove_ocean(ariset,data_obs,lat_bounds,lon_bounds) 
arisep, data_obs = dSS.remove_ocean(arisep,data_obs,lat_bounds,lon_bounds)

### Create lists for plotting
plotvar = [ariset,waccmt,
            arisep,waccmp]
colormaps = [cmocean.cm.balance,cmocean.cm.balance,
              cmocean.cm.curl,cmocean.cm.curl]
limits = [np.arange(-4,4.01,0.01),np.arange(-4,4.01,0.01),
          np.arange(-2,2.01,0.01),np.arange(-2,2.01,0.01)]
barlims = [np.round(np.arange(-4,5,2),2),np.round(np.arange(-4,5,2),2),
          np.round(np.arange(-2,3,1),2),np.round(np.arange(-2,3,1),2)]
titles = ['SAI -- YEAR %s' % yearsall[yearindex],'SSP2-4.5 -- YEAR %s' % yearsall[yearindex]]

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
        
fig = plt.figure(figsize=(8,4))
for i in range(len(plotvar)):
    ax = plt.subplot(2,2,i+1)
    
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
    
    if i<2:
        cs1 = m.contourf(x,y,var,limit,extend='both')
        cs1.set_cmap(colormaps[i])
    else:
        cs2 = m.contourf(x,y,var,limit,extend='both')
        cs2.set_cmap(colormaps[i])
    
    m.drawlsmask(land_color=(0,0,0,0),ocean_color='dimgray',lakes=False,zorder=11)

    plt.annotate(r'\textbf{[%s]}' % letters[i],
              textcoords='axes fraction',
              xy=(0,0), xytext=(0.96,0.9),
              fontsize=8,color='k',alpha=1,
              ha='center',va='center',zorder=30)

    if i < 2:
        plt.title(r'\textbf{%s}' % titles[i],color='dimgrey',fontsize=18)
    
    if i == 0:
        plt.annotate(r'\textbf{TREFHT -- \#%s}' % (ensindext+1),
                      textcoords='axes fraction',
                      xy=(0,0), xytext=(-0.08,0.5),
                      fontsize=12,color='dimgrey',alpha=1,
                      ha='center',va='center',zorder=30,rotation=90)
    if i == 2:
        plt.annotate(r'\textbf{PRECT -- \#%s}' % (ensindexp+1),
                      textcoords='axes fraction',
                      xy=(0,0), xytext=(-0.08,0.5),
                      fontsize=12,color='dimgrey',alpha=1,
                      ha='center',va='center',zorder=30,rotation=90)

cbar_ax = fig.add_axes([0.94,0.584,0.015,0.2])                
cbar = fig.colorbar(cs1,cax=cbar_ax,orientation='vertical',
                    extend='both',extendfrac=0.07,drawedges=False)
cbar.set_label(r'\textbf{TREFHT [$^{\circ}$C]}',fontsize=8,color='dimgrey',labelpad=3)  
cbar.set_ticks(barlims[0])
cbar.set_ticklabels(list(map(str,barlims[0])))
cbar.ax.tick_params(axis='y', size=.01,labelsize=7)
cbar.outline.set_edgecolor('dimgrey')

cbar_ax2 = fig.add_axes([0.94,0.146,0.015,0.2])                
cbar2 = fig.colorbar(cs2,cax=cbar_ax2,orientation='vertical',
                    extend='both',extendfrac=0.07,drawedges=False)
cbar2.set_label(r'\textbf{PRECT [mm/day]}',fontsize=8,color='dimgrey',labelpad=3)  
cbar2.set_ticks(barlims[-1])
cbar2.set_ticklabels(list(map(str,barlims[-1])))
cbar2.ax.tick_params(axis='y', size=.01,labelsize=7)
cbar2.outline.set_edgecolor('dimgrey')


plt.subplots_adjust(wspace=-0.24)
plt.tight_layout()
plt.savefig(directoryfigure + 'MAP_ExampleYearMaps_TREFHT-PRECT_anomalies.png',dpi=300)
