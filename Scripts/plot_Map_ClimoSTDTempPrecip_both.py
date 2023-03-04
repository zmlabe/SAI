"""
Graph showing regional temperature anomaly projections between SAI and WACCM

Author     : Zachary M. Labe
Date       : 24 July 2022
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
import calc_DetrendData as DT

### Plotting defaults 
plt.rc('text',usetex=True)
plt.rc('font',**{'family':'sans-serif','sans-serif':['Avant Garde']}) 

###############################################################################
###############################################################################
###############################################################################
### Data preliminaries 
letters = ["a","b","c","d","e","f","g","h","i","j","k","l","m","o","p"]
reg_nameq = ['Globe','NH','SH','Arctic','Antarctic','narrowTropics','SEAsia','NorthAfrica','Amazon']
labels = ['Globe','N. Hemisphere','S. Hemisphere','Arctic','Antarctic','Tropics','Southeast Asia','North Africa','Amazon']
regionboxes = ['Arctic','Antarctic','narrowTropics','SEAsia','NorthAfrica','Amazon']
directorydata = '/Users/zlabe/Data/SAI/'
directoryfigure = '/Users/zlabe/Documents/Research/SolarIntervention/Figures/'
###############################################################################
###############################################################################
modelGCMs = ['WACCM']
datasetsingle = ['WACCM']
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
yearswaccm = np.arange(2015,2069+1,1)
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
### Read in data
directorydata = '/Users/zlabe/Documents/Research/SolarIntervention/Data/'
arise = np.load(directorydata + 'ARISE_STDTempPrecip.npz')
arise_temp = arise['tempstd'][:]
arise_precip = arise['precipstd'][:]
lats = arise['lat'][:]
lons = arise['lon'][:]

waccm = np.load(directorydata + 'WACCM_STDTempPrecip.npz')
waccm_temp = waccm['tempstd'][:]
waccm_precip = waccm['precipstd'][:]

### Plottin stuff
limit1 = np.arange(0,1.51,0.01)
barlim1 = np.arange(0,1.6,0.5)
label1 = r'\textbf{Std. Dev. -- Temperature [$^{\circ}$C]}'
limit2 = np.arange(0,1.51,0.01)
barlim2 = np.arange(0,1.6,0.5)
label2 = r'\textbf{Std. Dev. -- Precipitation [mm/day]}'

plotdata = [waccm_temp,waccm_precip,arise_temp,arise_precip]
limits = [limit1,limit2,limit1,limit2]
barlim = [barlim1,barlim2,barlim1,barlim2]
label = [label1,label2,label1,label2]

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
        
fig = plt.figure(figsize=(9,6))
for i in range(len(plotdata)):
    ax = plt.subplot(2,2,i+1)
    
    var = plotdata[i]
    
    m = Basemap(projection='robin',lon_0=0,resolution='l',area_thresh=10000)
    m.drawcoastlines(color='darkgrey',linewidth=0.4)
    
    parallels = np.arange(-90,91,30)
    meridians = np.arange(-180,180,60)
    par=m.drawparallels(parallels,labels=[True,True,True,True],linewidth=0.3,
                    color='w',fontsize=0,zorder=40)
    mer=m.drawmeridians(meridians,labels=[True,True,True,True],linewidth=0.3,
                        fontsize=0,color='w',zorder=40)
    setcolor(mer,'dimgrey')
    setcolor(par,'dimgrey')
        
    var, lons_cyclic = addcyclic(var, lons)
    var, lons_cyclic = shiftgrid(180., var, lons_cyclic, start=False)
    lon2d, lat2d = np.meshgrid(lons_cyclic, lats)
    x, y = m(lon2d, lat2d)
    
    circle = m.drawmapboundary(fill_color='dimgray',color='dimgray',
                      linewidth=1)
    circle.set_clip_on(False)
    
    cs1 = m.contourf(x,y,var,limits[i],extend='max')
    
    cs1.set_cmap(cmr.eclipse)
    m.drawlsmask(land_color=(0,0,0,0),ocean_color='dimgray',lakes=False,zorder=11)
    ax.annotate(r'\textbf{[%s]}' % letters[i],xy=(0,0),xytext=(0.98,0.96),
              textcoords='axes fraction',color='k',fontsize=9,
              rotation=0,ha='center',va='center')
    

###############################################################################
cbar_ax1 = fig.add_axes([0.15,0.08,0.2,0.025])                
cbar1 = fig.colorbar(cs1,cax=cbar_ax1,orientation='horizontal',
                    extend='max',extendfrac=0.07,drawedges=False)
cbar1.set_label(label1,fontsize=10,color='dimgrey',labelpad=1.8)  
cbar1.set_ticks(barlim1)
cbar1.set_ticklabels(list(map(str,barlim1)))
cbar1.ax.tick_params(axis='x', size=.01,labelsize=8)
cbar1.outline.set_edgecolor('dimgrey')

###############################################################################
cbar_axd1 = fig.add_axes([0.655,0.08,0.2,0.025])                
cbard1 = fig.colorbar(cs1,cax=cbar_axd1,orientation='horizontal',
                    extend='max',extendfrac=0.07,drawedges=False)
cbard1.set_label(label2,fontsize=10,color='dimgrey',labelpad=1.8)  
cbard1.set_ticks(barlim2)
cbard1.set_ticklabels(list(map(str,barlim2)))
cbard1.ax.tick_params(axis='x', size=.01,labelsize=8)
cbard1.outline.set_edgecolor('dimgrey')

plt.tight_layout()
plt.subplots_adjust(hspace=-0.3,wspace=0.1)

plt.annotate(r'\textbf{SSP2-4.5}',
             textcoords='figure fraction',
             xy=(0,0), xytext=(0.025,0.635),
             fontsize=15,color='k',alpha=1,ha='right',rotation=90)
plt.annotate(r'\textbf{SAI-1.5}',
             textcoords='figure fraction',
             xy=(0,0), xytext=(0.025,0.256),
             fontsize=15,color='k',alpha=1,ha='right',rotation=90)

plt.savefig(directoryfigure + 'Map_ClimoSTDTempPrecip_RegionBoxes_detrended_both.png',dpi=300)
