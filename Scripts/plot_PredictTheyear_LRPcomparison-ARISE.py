"""
LRP for ARISE epochs to see if the regions change

Author     : Zachary M. Labe
Date       : 3 March 2022
Version    : 1
"""

### Import packages
import sys
import matplotlib.pyplot as plt
import numpy as np
from netCDF4 import Dataset
from mpl_toolkits.basemap import Basemap, addcyclic, shiftgrid
import palettable.cubehelix as cm
import cmocean as cmocean
import cmasher as cmr
import calc_Utilities as UT

### Plotting defaults 
plt.rc('text',usetex=True)
plt.rc('font',**{'family':'sans-serif','sans-serif':['Avant Garde']})

### Parameters
directorydata = '/Users/zlabe/Documents/Research/SolarIntervention/Data/'
years = np.arange(2035,2069+1,1)
testingn = 2
variq = 'TREFHT'
directoryfigure = '/Users/zlabe/Desktop/sAI/LRP/%s/' % variq

###############################################################################
### Read in LRP after training on ARISE
data = Dataset(directorydata + 'LRPMap_Z_Testing_ARISE_%s.nc' % variq)
lat = data.variables['lat'][:]
lon = data.variables['lon'][:]                             
lrp_arise_zq = data.variables['LRP'][:].reshape(testingn,years.shape[0],96,144)
data.close()

### Change longitudes
lon = np.where(lon >180,lon-360,lon)

data = Dataset(directorydata + 'LRPMap_E_Testing_ARISE_%s.nc' % variq)
lrp_arise_eq = data.variables['LRP'][:].reshape(testingn,years.shape[0],96,144)
data.close()

data = Dataset(directorydata + 'LRPMap_IG_Testing_ARISE_%s.nc' % variq)
lrp_arise_igq = data.variables['LRP'][:].reshape(testingn,years.shape[0],96,144)
data.close()

### Calculate ensemble mean 
lrp_arise_z = np.nanmean(lrp_arise_zq,axis=0)
lrp_arise_e = np.nanmean(lrp_arise_eq,axis=0)
lrp_arise_ig = np.nanmean(lrp_arise_igq,axis=0)

### Take means across all years
lrp_arise_z1 = np.nanmean(lrp_arise_z[:10,:,:],axis=0)
lrp_arise_z2 = np.nanmean(lrp_arise_z[10:,:,:],axis=0)
lrp_arise_e1 = np.nanmean(lrp_arise_e[:10,:,:],axis=0)
lrp_arise_e2 = np.nanmean(lrp_arise_e[10:,:,:],axis=0)
lrp_arise_ig1 = np.nanmean(lrp_arise_ig[:10,:,:],axis=0)
lrp_arise_ig2 = np.nanmean(lrp_arise_ig[10:,:,:],axis=0)

###############################################################################
###############################################################################
###############################################################################
### Plot subplot of observations
letters = ["a","b","c","d","e","f","g","h","i","j","k","l","m","n"]
limit = np.arange(0,0.81,0.005)
barlim = np.round(np.arange(0,0.81,0.1),2)
cmap = cm.cubehelix2_16.mpl_colormap
label = r'\textbf{LRP-ARISE [Relevance]}'

fig = plt.figure(figsize=(10,5))
###############################################################################
ax1 = plt.subplot(231)
m = Basemap(projection='robin',lon_0=-180,resolution='l',area_thresh=10000)
m.drawcoastlines(color='darkgrey',linewidth=0.5)
    
### Variable
lrp_arise_z1 = lrp_arise_z1/np.max(lrp_arise_z1)

lon = np.where(lon >180,lon-360,lon)
x, y = np.meshgrid(lon,lat)
   
circle = m.drawmapboundary(fill_color='dimgrey',color='dimgray',
                  linewidth=0.7)
circle.set_clip_on(False)

cs1 = m.contourf(x,y,lrp_arise_z1,limit,extend='max',latlon=True)
cs1.set_cmap(cmap) 
   
plt.title(r'\textbf{LRP$_{z}$ Rule}',fontsize=20,color='dimgrey')
ax1.annotate(r'\textbf{[%s]}' % letters[0],xy=(0,0),xytext=(0.98,0.84),
              textcoords='axes fraction',color='k',fontsize=9,
              rotation=0,ha='center',va='center')
ax1.annotate(r'\textbf{2035-2044}',xy=(0,0),xytext=(-0.04,0.5),
              textcoords='axes fraction',color='k',fontsize=9,
              rotation=90,ha='center',va='center')

###############################################################################
###############################################################################
###############################################################################
ax2 = plt.subplot(232)
m = Basemap(projection='robin',lon_0=-180,resolution='l',area_thresh=10000)
m.drawcoastlines(color='darkgrey',linewidth=0.5)

### Variable
lrp_arise_e1 = lrp_arise_e1/np.max(lrp_arise_e1)

circle = m.drawmapboundary(fill_color='dimgrey',color='dimgray',
                  linewidth=0.7)
circle.set_clip_on(False)

cs1 = m.contourf(x,y,lrp_arise_e1,limit,extend='max',latlon=True)
cs1.set_cmap(cmap) 
 
plt.title(r'\textbf{LRP$_{\epsilon}$ Rule}',fontsize=20,color='dimgrey')
ax2.annotate(r'\textbf{[%s]}' % letters[1],xy=(0,0),xytext=(0.98,0.84),
              textcoords='axes fraction',color='k',fontsize=9,
              rotation=0,ha='center',va='center')

###############################################################################
ax1 = plt.subplot(233)
m = Basemap(projection='robin',lon_0=-180,resolution='l',area_thresh=10000)
m.drawcoastlines(color='darkgrey',linewidth=0.5)
    
### Variable
lrp_arise_ig1 = lrp_arise_ig1/np.max(lrp_arise_ig1)

lon = np.where(lon >180,lon-360,lon)
x, y = np.meshgrid(lon,lat)
   
circle = m.drawmapboundary(fill_color='dimgrey',color='dimgray',
                  linewidth=0.7)
circle.set_clip_on(False)

cs1 = m.contourf(x,y,lrp_arise_ig1,limit,extend='max',latlon=True)
cs1.set_cmap(cmap) 
   
plt.title(r'\textbf{Integrated Gradients}',fontsize=20,color='dimgrey')
ax1.annotate(r'\textbf{[%s]}' % letters[2],xy=(0,0),xytext=(0.98,0.84),
              textcoords='axes fraction',color='k',fontsize=9,
              rotation=0,ha='center',va='center')

###############################################################################
###############################################################################
###############################################################################
ax2 = plt.subplot(234)
m = Basemap(projection='robin',lon_0=-180,resolution='l',area_thresh=10000)
m.drawcoastlines(color='darkgrey',linewidth=0.5)

### Variable
lrp_arise_z2 = lrp_arise_z2/np.max(lrp_arise_z2)

circle = m.drawmapboundary(fill_color='dimgrey',color='dimgray',
                  linewidth=0.7)
circle.set_clip_on(False)

cs2 = m.contourf(x,y,lrp_arise_z2,limit,extend='max',latlon=True)
cs2.set_cmap(cmap) 
     
ax2.annotate(r'\textbf{[%s]}' % letters[3],xy=(0,0),xytext=(0.98,0.84),
              textcoords='axes fraction',color='k',fontsize=9,
              rotation=0,ha='center',va='center')
ax2.annotate(r'\textbf{2045-2069}',xy=(0,0),xytext=(-0.04,0.5),
              textcoords='axes fraction',color='k',fontsize=9,
              rotation=90,ha='center',va='center')

###############################################################################
###############################################################################
###############################################################################
ax2 = plt.subplot(235)
m = Basemap(projection='robin',lon_0=-180,resolution='l',area_thresh=10000)
m.drawcoastlines(color='darkgrey',linewidth=0.5)

### Variable
lrp_arise_e2 = lrp_arise_e2/np.max(lrp_arise_e2)

circle = m.drawmapboundary(fill_color='dimgrey',color='dimgray',
                  linewidth=0.7)
circle.set_clip_on(False)

cs1 = m.contourf(x,y,lrp_arise_e2,limit,extend='max',latlon=True)
cs1.set_cmap(cmap) 
     
ax2.annotate(r'\textbf{[%s]}' % letters[4],xy=(0,0),xytext=(0.98,0.84),
              textcoords='axes fraction',color='k',fontsize=9,
              rotation=0,ha='center',va='center')

###############################################################################
###############################################################################
###############################################################################
ax2 = plt.subplot(236)
m = Basemap(projection='robin',lon_0=-180,resolution='l',area_thresh=10000)
m.drawcoastlines(color='darkgrey',linewidth=0.5)

### Variable
lrp_arise_ig2 = lrp_arise_ig2/np.max(lrp_arise_ig2)

circle = m.drawmapboundary(fill_color='dimgrey',color='dimgray',
                  linewidth=0.7)
circle.set_clip_on(False)

cs1 = m.contourf(x,y,lrp_arise_ig2,limit,extend='max',latlon=True)
cs1.set_cmap(cmap) 
     
ax2.annotate(r'\textbf{[%s]}' % letters[5],xy=(0,0),xytext=(0.98,0.84),
              textcoords='axes fraction',color='k',fontsize=9,
              rotation=0,ha='center',va='center')

###############################################################################
cbar_ax1 = fig.add_axes([0.395,0.1,0.2,0.025])                
cbar1 = fig.colorbar(cs1,cax=cbar_ax1,orientation='horizontal',
                    extend='max',extendfrac=0.07,drawedges=False)
cbar1.set_label(label,fontsize=10,color='dimgrey',labelpad=1.4)  
cbar1.set_ticks(barlim)
cbar1.set_ticklabels(list(map(str,barlim)))
cbar1.ax.tick_params(axis='x', size=.01,labelsize=7)
cbar1.outline.set_edgecolor('dimgrey')

plt.tight_layout()
plt.subplots_adjust(hspace=-0.4)

plt.savefig(directoryfigure + 'PredictTheYear_LRPcomparison-ARISE_%s.png' % variq,dpi=300)