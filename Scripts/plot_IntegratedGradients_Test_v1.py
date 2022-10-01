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
directorydata = '/Users/zlabe/Documents/Research/SolarIntervention/Data/YearsSinceSAI/ONLYARISE/'
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

### Read in XAI
ensindexp = 7
yearq1 = np.where((yearsall >= 2035) & (yearsall <= 2049))[0]
yearq2 = np.where((yearsall >= 2050) & (yearsall <= 2069))[0]

xai = np.load(directorydata + 'integratedGradients.npz')['XAI']
data_arise,lats,lons = read_primary_dataset('TREFHT','ARISE',
                                          numOfEns,lensalso,
                                          randomalso,
                                          ravelyearsbinary,
                                          ravelbinary,
                                          shuffletype,
                                          lat_bounds,
                                          lon_bounds)

xai1 = np.nanmean(xai[yearq1,:,:],axis=0)
xai2 = np.nanmean(xai[yearq2,:,:],axis=0)

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

var = xai1
limit = np.arange(-0.01,0.0101,0.0001)
barlim = np.round(np.arange(-0.01,0.011,0.01),3)
label = r'\textbf{Relevance}'

m = Basemap(projection='robin',lon_0=0,resolution='l',area_thresh=10000)
m.drawcoastlines(color='darkgrey',linewidth=0.4)

# parallels = np.arange(-90,91,30)
# meridians = np.arange(-180,180,60)
# par=m.drawparallels(parallels,labels=[True,True,True,True],linewidth=0.3,
#                 color='w',fontsize=4,zorder=40)
# mer=m.drawmeridians(meridians,labels=[True,True,True,True],linewidth=0.3,
#                     fontsize=4,color='w',zorder=40)
# setcolor(mer,'dimgrey')
# setcolor(par,'dimgrey')
    
var, lons_cyclic = addcyclic(var, lons)
var, lons_cyclic = shiftgrid(180., var, lons_cyclic, start=False)
lon2d, lat2d = np.meshgrid(lons_cyclic, lats)
x, y = m(lon2d, lat2d)

circle = m.drawmapboundary(fill_color='dimgray',color='dimgray',
                  linewidth=1)
circle.set_clip_on(False)

cs1 = m.contourf(x,y,var,limit,extend='both')

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

plt.title(r'\textbf{2035-2049}',color='dimgrey',fontsize=19)

###############################################################################
###############################################################################
###############################################################################
ax = plt.subplot(122)

var = xai2
limit = np.arange(-0.01,0.0101,0.0001)
barlim = np.round(np.arange(-0.01,0.011,0.01),3)
label = r'\textbf{Relevance}'

m = Basemap(projection='robin',lon_0=0,resolution='l',area_thresh=10000)
m.drawcoastlines(color='darkgrey',linewidth=0.4)

# parallels = np.arange(-90,91,30)
# meridians = np.arange(-180,180,60)
# par=m.drawparallels(parallels,labels=[True,True,True,True],linewidth=0.3,
#                 color='w',fontsize=4,zorder=40)
# mer=m.drawmeridians(meridians,labels=[True,True,True,True],linewidth=0.3,
#                     fontsize=4,color='w',zorder=40)
# setcolor(mer,'dimgrey')
# setcolor(par,'dimgrey')
    
var, lons_cyclic = addcyclic(var, lons)
var, lons_cyclic = shiftgrid(180., var, lons_cyclic, start=False)
lon2d, lat2d = np.meshgrid(lons_cyclic, lats)
x, y = m(lon2d, lat2d)

circle = m.drawmapboundary(fill_color='dimgray',color='dimgray',
                  linewidth=1)
circle.set_clip_on(False)

cs1 = m.contourf(x,y,var,limit,extend='both')

cs1.set_cmap(cmocean.cm.balance)
m.drawlsmask(land_color=(0,0,0,0),ocean_color='dimgray',lakes=False,zorder=11)

cbar1 = plt.colorbar(cs1,orientation='horizontal',
                    extend='both',extendfrac=0.07,drawedges=False,
                    fraction=0.03,pad=0.06)
cbar1.set_label(label,fontsize=10,color='dimgrey',labelpad=1.1)  
cbar1.set_ticks(barlim)
cbar1.set_ticklabels(list(map(str,barlim)))
cbar1.ax.tick_params(axis='x', size=.01,labelsize=8)
cbar1.outline.set_edgecolor('dimgrey')

plt.title(r'\textbf{2050-2069}',color='dimgrey',fontsize=19)

plt.tight_layout()
plt.subplots_adjust(top=0.85)
###############################################################################
###############################################################################
###############################################################################
### Add text
plt.annotate(r'\textbf{[%s]}' % letters[0],
             textcoords='figure fraction',
             xy=(0,0), xytext=(0.46,0.87),
             fontsize=10,color='k',alpha=1,ha='right')
plt.annotate(r'\textbf{[%s]}' % letters[1],
             textcoords='figure fraction',
             xy=(0,0), xytext=(0.95,0.87),
             fontsize=10,color='k',alpha=1,ha='right')

plt.savefig(directoryfigure + 'XAI_Test_v1.png',dpi=300)
