"""
Graphic shows a sample comparison of contribution maps for SAI vs. SNR of TREFHT

Author     : Zachary M. Labe
Date       : 11 August 2022
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
directorydata = '/Users/zlabe/Documents/Research/SolarIntervention/Data/'
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

cont = np.load(directorydata + 'TREFHT-SAI_period2Contributions.npz')
contribution = cont['cont'][:]
lats = cont['lat'][:]
lons = cont['lon'][:]
snr = np.load(directorydata + 'TREFHT-SAI_period2SNR.npz')
snrall = snr['snr']

lon2,lat2 = np.meshgrid(lons,lats)

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
        
fig = plt.figure(figsize=(9,3))
ax = plt.subplot(121)

var = contribution
limit = np.arange(-0.005,0.00501,0.0001)
barlim = np.round(np.arange(-0.005,0.00501,0.005),3)
label = r'\textbf{SAI-TREFHT [Input$\times$Weights]}'

m = Basemap(projection='robin',lon_0=0,resolution='l',area_thresh=10000)
m.drawcoastlines(color='darkgrey',linewidth=0.4)

parallels = np.arange(-90,91,30)
meridians = np.arange(-180,180,60)
par=m.drawparallels(parallels,labels=[False,False,False,False],linewidth=0.3,
                color='w',fontsize=4,zorder=40)
mer=m.drawmeridians(meridians,labels=[False,False,False,False],linewidth=0.3,
                    fontsize=4,color='w',zorder=40)

circle = m.drawmapboundary(fill_color='dimgray',color='dimgray',
                  linewidth=1)
circle.set_clip_on(False)

cs1 = m.contourf(lon2,lat2,var,limit,extend='both',latlon=True)

cs1.set_cmap(cmocean.cm.balance)
m.drawlsmask(land_color=(0,0,0,0),ocean_color='dimgray',lakes=False,zorder=11)

cbar1 = plt.colorbar(cs1,orientation='horizontal',
                    extend='both',extendfrac=0.07,drawedges=False,
                    fraction=0.03,pad=0.06)
cbar1.set_label(label,fontsize=10,color='dimgrey',labelpad=2)  
cbar1.set_ticks(barlim)
cbar1.set_ticklabels(list(map(str,barlim)))
cbar1.ax.tick_params(axis='x', size=.01,labelsize=8)
cbar1.outline.set_edgecolor('dimgrey')

###############################################################################
###############################################################################
###############################################################################
ax = plt.subplot(122)

var = snrall
limit = np.arange(0,2.1,0.25)
barlim = np.round(np.arange(0,2.1,1),2)

label = r'\textbf{SAI-TREFHT [Signal-To-Noise]}'

m = Basemap(projection='robin',lon_0=0,resolution='l',area_thresh=10000)
m.drawcoastlines(color='darkgrey',linewidth=0.4)

parallels = np.arange(-90,91,30)
meridians = np.arange(-180,180,60)
par=m.drawparallels(parallels,labels=[False,False,False,False],linewidth=0.3,
                color='w',fontsize=4,zorder=40)
mer=m.drawmeridians(meridians,labels=[False,False,False,False],linewidth=0.3,
                    fontsize=4,color='w',zorder=40)

circle = m.drawmapboundary(fill_color='dimgray',color='dimgray',
                  linewidth=1)
circle.set_clip_on(False)

cs1 = m.contourf(lon2,lat2,var,limit,extend='max',latlon=True)

cs1.set_cmap(cmr.torch)
m.drawlsmask(land_color=(0,0,0,0),ocean_color='dimgray',lakes=False,zorder=11)

cbar1 = plt.colorbar(cs1,orientation='horizontal',
                    extend='max',extendfrac=0.07,drawedges=False,
                    fraction=0.03,pad=0.06)
cbar1.set_label(label,fontsize=10,color='dimgrey',labelpad=1.1)  
cbar1.set_ticks(barlim)
cbar1.set_ticklabels(list(map(str,barlim)))
cbar1.ax.tick_params(axis='x', size=.01,labelsize=8)
cbar1.outline.set_edgecolor('dimgrey')

plt.tight_layout()
###############################################################################
###############################################################################
###############################################################################
### Add text
plt.annotate(r'\textbf{[%s]}' % letters[0],
             textcoords='figure fraction',
             xy=(0,0), xytext=(0.485,0.87),
             fontsize=10,color='k',alpha=1,ha='right')
plt.annotate(r'\textbf{[%s]}' % letters[1],
             textcoords='figure fraction',
             xy=(0,0), xytext=(0.98,0.87),
             fontsize=10,color='k',alpha=1,ha='right')

plt.savefig(directoryfigure + 'SampleMaps_SNR-Contributions_TREFHT_period2.png',dpi=1000)
