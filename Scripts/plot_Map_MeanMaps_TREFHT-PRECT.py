"""
Graphic showing difference in the multi-model means at the end of the century

Author     : Zachary M. Labe
Date       : 5 March 2023
Version    : 1 
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
numOfEns = 10
dataset_obs = 'ERA5'
###############################################################################
###############################################################################
num_of_class = 2
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

### Only include 2035-2069 for comparing 
data_waccm = data_waccmall[:,:,-yearsall.shape[0]:,:,:].squeeze()
data_arise = data_arise.squeeze()

### Calculate two trend periods
# yearq1 = np.where((yearsall >= 2035) & (yearsall <= 2044))[0]
yearq2 = np.where((yearsall >= 2045) & (yearsall <= 2069))[0]

### Slice end of century
waccm_t = data_waccm[:,yearq2,:,:]
arise_t = data_arise[:,yearq2,:,:]

### Calculate year and ensemble mean
waccm_tm = np.nanmean(np.nanmean(waccm_t,axis=1),axis=0)
arise_tm = np.nanmean(np.nanmean(arise_t,axis=1),axis=0)

### Calculate difference
diff_t = arise_tm - waccm_tm

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

### Only include 2035-2069 for comparing 
datap_waccm = datap_waccmall[:,:,-yearsall.shape[0]:,:,:].squeeze()
datap_arise = datap_arise.squeeze()

### Slice end of century
waccm_p = datap_waccm[:,yearq2,:,:]
arise_p = datap_arise[:,yearq2,:,:]

### Calculate year and ensemble mean
waccm_pm = np.nanmean(np.nanmean(waccm_p,axis=1),axis=0)
arise_pm = np.nanmean(np.nanmean(arise_p,axis=1),axis=0)

### Calculate difference
diff_p = arise_pm - waccm_pm

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

var = diff_t
limit = np.arange(-2,2.01,0.01)
barlim = np.arange(-2,3,1)
label = r'\textbf{Temperature [$^{\circ}$C] -- SAI-1.5 minus SSP2-4.5}'

m = Basemap(projection='robin',lon_0=0,resolution='l',area_thresh=10000)
m.drawcoastlines(color='darkgrey',linewidth=0.4)

parallels = np.arange(-90,91,30)
meridians = np.arange(-180,180,60)
par=m.drawparallels(parallels,labels=[True,True,True,True],linewidth=0.3,
                color='w',fontsize=4,zorder=40)
mer=m.drawmeridians(meridians,labels=[True,True,True,True],linewidth=0.3,
                    fontsize=4,color='w',zorder=40)
setcolor(mer,'dimgrey')
setcolor(par,'dimgrey')
    
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

###############################################################################
###############################################################################
###############################################################################
ax = plt.subplot(122)

var = diff_p
limit = np.arange(-0.8,0.801,0.001)
barlim = np.round(np.arange(-0.8,0.81,0.4),2)
label = r'\textbf{Precipitation [mm/day] -- SAI-1.5 minus SSP2-4.5}'

m = Basemap(projection='robin',lon_0=0,resolution='l',area_thresh=10000)
m.drawcoastlines(color='darkgrey',linewidth=0.4)

parallels = np.arange(-90,91,30)
meridians = np.arange(-180,180,60)
par=m.drawparallels(parallels,labels=[True,True,True,True],linewidth=0.3,
                color='w',fontsize=4,zorder=40)
mer=m.drawmeridians(meridians,labels=[True,True,True,True],linewidth=0.3,
                    fontsize=4,color='w',zorder=40)
setcolor(mer,'dimgrey')
setcolor(par,'dimgrey')
    
var, lons_cyclic = addcyclic(var, lons)
var, lons_cyclic = shiftgrid(180., var, lons_cyclic, start=False)
lon2d, lat2d = np.meshgrid(lons_cyclic, lats)
x, y = m(lon2d, lat2d)

circle = m.drawmapboundary(fill_color='dimgray',color='dimgray',
                  linewidth=1)
circle.set_clip_on(False)

cs1 = m.contourf(x,y,var,limit,extend='both')

cs1.set_cmap(cmocean.cm.tarn)
m.drawlsmask(land_color=(0,0,0,0),ocean_color='dimgray',lakes=False,zorder=11)

cbar1 = plt.colorbar(cs1,orientation='horizontal',
                    extend='both',extendfrac=0.07,drawedges=False,
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
             xy=(0,0), xytext=(0.46,0.87),
             fontsize=10,color='k',alpha=1,ha='right')
plt.annotate(r'\textbf{[%s]}' % letters[1],
             textcoords='figure fraction',
             xy=(0,0), xytext=(0.95,0.87),
             fontsize=10,color='k',alpha=1,ha='right')

plt.savefig(directoryfigure + 'Map_MeanMaps_TREFHT-PRECT.png',dpi=500)