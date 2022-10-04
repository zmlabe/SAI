"""
XAI results for the ANN in detecting how many years it has been since 
injection in ARISE-1.5

Author     : Zachary M. Labe
Date       : 3 October 2022
Version    : 1 - testing ANN architectures for detecting SAI
"""

### Import packages
import sys
import matplotlib.pyplot as plt
import matplotlib.colors as c
import numpy as np
from mpl_toolkits.basemap import Basemap, addcyclic, shiftgrid
import palettable.cubehelix as cm
import cmocean as cmocean
import cmasher as cmr

### Plotting defaults 
plt.rc('text',usetex=True)
plt.rc('font',**{'family':'sans-serif','sans-serif':['Avant Garde']}) 

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
###############################################################################
###############################################################################
yearsall = np.arange(2035,2069+1,1)
numOfEns = 10
dataset_obs = '20CRv3'
###############################################################################
###############################################################################
num_of_class = len(modelGCMs)
ensTypeExperi = 'ENS'
###############################################################################
###############################################################################
###############################################################################
###############################################################################
### Create sample class labels for each model for my own testing
if seasons != 'none':
    classesl = np.empty((num_of_class,numOfEns,len(yearsall)))
    for i in range(num_of_class):
        classesl[i,:,:] = np.full((numOfEns,len(yearsall)),i)  
        
    if ensTypeExperi == 'ENS':
        classeslnew = np.swapaxes(classesl,0,1)

###############################################################################
###############################################################################
###############################################################################
###############################################################################     
### Begin ANN and the entire script - loop through these parameters
letters = ["a","b","c","d","e","f","g","h","i","j","k","l","m"]
ridge_penaltyq = [0.01,0.1,0.25,0.5,0.75,1,1.5,5]
reg_nameq = ['Globe','narrowTropics']
NCOMBOS = 20
directorydata = '/Users/zlabe/Documents/Research/SolarIntervention/Data/YearsSinceSAI/ONLYARISE/'
directoryfigure = '/Users/zlabe/Documents/Research/SolarIntervention/Figures/'

def normalize_negative_one(img):
    normalized_input = (img - np.nanmin(img)) / (np.nanmax(img) - np.nanmin(img))
    return 2*normalized_input - 1

### XAI reshape
latshapeg = 96
lonshapeg = 144
latshapet = 22
lonshapet = 144  

### Read in xai for temperature
mapweights_t = []
latitudes_t = []
longitudes_t = []
for rr in range(len(reg_nameq)):
    reg_name = reg_nameq[rr]
    variq = 'TREFHT'      

    latitudes_tn = np.load('/Users/zlabe/Documents/Research/SolarIntervention/Data/DetectSAI_ActualModel/' + 'Latitudes-LOGREG_annual_LAND_GCMarise_LOGREG_%s_%s_NumOfGCMS-2.npy' % (variq,reg_name))
    longitudes_tn = np.load('/Users/zlabe/Documents/Research/SolarIntervention/Data/DetectSAI_ActualModel/' + 'Longitudes-LOGREG_annual_LAND_GCMarise_LOGREG_%s_%s_NumOfGCMS-2.npy' % (variq,reg_name))
    XAI_t = np.load(directorydata + 'InputsxGradients_annual_LAND_GCMarise_YearsSinceSAI_ARISEONLY_%s_%s_NumOfGCMS-1.npz' % (variq,reg_name))['XAI']
    
    latitudes_t.append(latitudes_tn)
    longitudes_t.append(longitudes_tn)
    mapweights_t.append(XAI_t)
    
### Read in xai for precipitation
mapweights_p = []
latitudes_p = []
longitudes_p = []
for rr in range(len(reg_nameq)):
    reg_name = reg_nameq[rr]
    variq = 'PRECT'      

    latitudes_pn = np.load('/Users/zlabe/Documents/Research/SolarIntervention/Data/DetectSAI_ActualModel/' + 'Latitudes-LOGREG_annual_LAND_GCMarise_LOGREG_%s_%s_NumOfGCMS-2.npy' % (variq,reg_name))
    longitudes_pn = np.load('/Users/zlabe/Documents/Research/SolarIntervention/Data/DetectSAI_ActualModel/' + 'Longitudes-LOGREG_annual_LAND_GCMarise_LOGREG_%s_%s_NumOfGCMS-2.npy' % (variq,reg_name))
    XAI_p = np.load(directorydata + 'InputsxGradients_annual_LAND_GCMarise_YearsSinceSAI_ARISEONLY_%s_%s_NumOfGCMS-1.npz' % (variq,reg_name))['XAI']
    
    latitudes_p.append(latitudes_pn)
    longitudes_p.append(longitudes_pn)
    mapweights_p.append(XAI_p)
    
### Reshape everything
xai_globe_t = mapweights_t[0].reshape(yearsall.shape[0],latshapeg,lonshapeg)
xai_globe_p = mapweights_p[0].reshape(yearsall.shape[0],latshapeg,lonshapeg)
lat_g = latitudes_t[0]
lon_g = longitudes_t[0]

xai_tropic_t = mapweights_t[1].reshape(yearsall.shape[0],latshapet,lonshapet)
xai_tropic_p = mapweights_p[1].reshape(yearsall.shape[0],latshapet,lonshapet)
lat_t = latitudes_t[1]
lon_t = longitudes_t[1]

### Calculate means
mean_globe_t = np.nanmean(xai_globe_t,axis=0)
mean_globe_p = np.nanmean(xai_globe_p,axis=0)

mean_tropics_t = np.nanmean(xai_tropic_t,axis=0)
mean_tropics_p = np.nanmean(xai_tropic_p,axis=0)

### Scale the map means
scale_globe_t = mean_globe_t/np.nanmax(abs(mean_globe_t))
scale_globe_p = mean_globe_p/np.nanmax(abs(mean_globe_p))

scale_tropics_t = mean_tropics_t/np.nanmax(abs(mean_tropics_t))
scale_tropics_p = mean_tropics_p/np.nanmax(abs(mean_tropics_p))

### Read in SNR data
directorydata2 = '/Users/zlabe/Documents/Research/SolarIntervention/Data/'
snr_t = np.load(directorydata2 + 'TREFHT-SAI_allSNR.npz')['snr']
snr_p = np.load(directorydata2 + 'PRECT-SAI_allSNR.npz')['snr']

###############################################################################
###############################################################################
###############################################################################
### Plot subplot of observations
lon = longitudes_p
lat = latitudes_p

letters = ["a","b","c","d","e","f","g","h","i","j","k","l","m","n"]
limit = np.arange(-0.4,0.41,0.01)
barlim = np.round(np.arange(-0.4,0.5,0.2),2)
cmap = cmocean.cm.balance
label = r'\textbf{Input*Gradient}'

limits = np.arange(0,2.1,0.25)
barlims = np.round(np.arange(0,2.1,1),2)
labelss = r'\textbf{Signal-To-Noise}'
cmaps = cmr.torch

fig = plt.figure(figsize=(10,3.7))
###############################################################################
ax1 = plt.subplot(321)
m = Basemap(projection='robin',lon_0=0,resolution='l',area_thresh=10000)
m.drawcoastlines(color='k',linewidth=0.4,zorder=30)
    
### Variable
x, y = np.meshgrid(lon_g,lat_g)
   
circle = m.drawmapboundary(fill_color='dimgrey',color='dimgray',
                  linewidth=0.7)
circle.set_clip_on(False)

parallels = np.arange(-90,91,30)
meridians = np.arange(-180,180,60)
par=m.drawparallels(parallels,labels=[False,False,False,False],linewidth=0.3,
                color='w',fontsize=4,zorder=40)
mer=m.drawmeridians(meridians,labels=[False,False,False,False],linewidth=0.3,
                    fontsize=4,color='w',zorder=40)

cs1 = m.contourf(x,y,scale_globe_t,limit,extend='both',latlon=True)
cs1.set_cmap(cmap) 
m.drawlsmask(land_color=(0,0,0,0),ocean_color='dimgrey',lakes=False,zorder=11)

plt.title(r'\textbf{TREFHT}',fontsize=20,color='dimgrey')     
ax1.annotate(r'\textbf{[%s]}' % letters[0],xy=(0,0),xytext=(0.98,0.84),
              textcoords='axes fraction',color='k',fontsize=9,
              rotation=0,ha='center',va='center')

###############################################################################
###############################################################################
###############################################################################
ax2 = plt.subplot(322)
m = Basemap(projection='robin',lon_0=0,resolution='l',area_thresh=10000)
m.drawcoastlines(color='k',linewidth=0.4,zorder=30)

### Variable
x, y = np.meshgrid(lon_g,lat_g)

circle = m.drawmapboundary(fill_color='dimgrey',color='dimgray',
                  linewidth=0.7)
circle.set_clip_on(False)

parallels = np.arange(-90,91,30)
meridians = np.arange(-180,180,60)
par=m.drawparallels(parallels,labels=[False,False,False,False],linewidth=0.3,
                color='w',fontsize=4,zorder=40)
mer=m.drawmeridians(meridians,labels=[False,False,False,False],linewidth=0.3,
                    fontsize=4,color='w',zorder=40)

cs2 = m.contourf(x,y,scale_globe_p,limit,extend='both',latlon=True)
cs2.set_cmap(cmap) 
m.drawlsmask(land_color=(0,0,0,0),ocean_color='dimgrey',lakes=False,zorder=11)

plt.title(r'\textbf{PRECT}',fontsize=20,color='dimgrey')         
ax2.annotate(r'\textbf{[%s]}' % letters[1],xy=(0,0),xytext=(0.98,0.84),
              textcoords='axes fraction',color='k',fontsize=9,
              rotation=0,ha='center',va='center')

###############################################################################
###############################################################################
###############################################################################
ax2 = plt.subplot(323)
m = Basemap(projection='cea',llcrnrlat=-20.1,urcrnrlat=20.1,\
            llcrnrlon=-179.99,urcrnrlon=179.99,resolution='l',area_thresh=10000)
m.drawcoastlines(color='k',linewidth=0.4,zorder=30)

### Variable
x, y = np.meshgrid(lon_t,lat_t)

circle = m.drawmapboundary(fill_color='dimgrey',color='dimgray',
                  linewidth=0.7)
circle.set_clip_on(False)

parallels = np.arange(-90,91,30)
meridians = np.arange(-180,180,60)
par=m.drawparallels(parallels,labels=[True,False,False,False],linewidth=0.3,
                color='w',fontsize=4,zorder=40)
mer=m.drawmeridians(meridians,labels=[False,False,False,True],linewidth=0.3,
                    fontsize=4,color='w',zorder=40)

cs2 = m.contourf(x,y,scale_tropics_t,limit,extend='both',latlon=True)
cs2.set_cmap(cmap) 
m.drawlsmask(land_color=(0,0,0,0),ocean_color='dimgrey',lakes=False,zorder=11)
       
ax2.annotate(r'\textbf{[%s]}' % letters[2],xy=(0,0),xytext=(0.99,1.14),
              textcoords='axes fraction',color='k',fontsize=9,
              rotation=0,ha='center',va='center')

###############################################################################
###############################################################################
###############################################################################
ax2 = plt.subplot(324)
m = Basemap(projection='cea',llcrnrlat=-20.1,urcrnrlat=20.1,\
            llcrnrlon=-179.99,urcrnrlon=179.99,resolution='l',area_thresh=10000)
m.drawcoastlines(color='k',linewidth=0.4,zorder=30)

### Variable
x, y = np.meshgrid(lon_t,lat_t)

circle = m.drawmapboundary(fill_color='dimgrey',color='dimgray',
                  linewidth=0.7)
circle.set_clip_on(False)

parallels = np.arange(-90,91,30)
meridians = np.arange(-180,180,60)
par=m.drawparallels(parallels,labels=[False,True,False,False],linewidth=0.3,
                color='w',fontsize=4,zorder=40)
mer=m.drawmeridians(meridians,labels=[False,False,False,True],linewidth=0.3,
                    fontsize=4,color='w',zorder=40)

cs4 = m.contourf(x,y,scale_tropics_p,limit,extend='both',latlon=True)
cs4.set_cmap(cmap) 
m.drawlsmask(land_color=(0,0,0,0),ocean_color='dimgrey',lakes=False,zorder=11)
       
ax2.annotate(r'\textbf{[%s]}' % letters[3],xy=(0,0),xytext=(0.99,1.14),
              textcoords='axes fraction',color='k',fontsize=9,
              rotation=0,ha='center',va='center')

###############################################################################
###############################################################################
###############################################################################
ax2 = plt.subplot(325)
m = Basemap(projection='cea',llcrnrlat=-20.1,urcrnrlat=20.1,\
            llcrnrlon=-179.99,urcrnrlon=179.99,resolution='l',area_thresh=10000)
m.drawcoastlines(color='k',linewidth=0.4,zorder=30)

### Variable
x, y = np.meshgrid(lon_g,lat_g)

circle = m.drawmapboundary(fill_color='dimgrey',color='dimgray',
                  linewidth=0.7)
circle.set_clip_on(False)

parallels = np.arange(-90,91,30)
meridians = np.arange(-180,180,60)
par=m.drawparallels(parallels,labels=[True,False,False,False],linewidth=0.3,
                color='w',fontsize=4,zorder=40)
mer=m.drawmeridians(meridians,labels=[False,False,True,False],linewidth=0.3,
                    fontsize=4,color='w',zorder=40)

cs5 = m.contourf(x,y,snr_t,limits,extend='max',latlon=True)
cs5.set_cmap(cmaps) 
m.drawlsmask(land_color=(0,0,0,0),ocean_color='dimgrey',lakes=False,zorder=11)
       
ax2.annotate(r'\textbf{[%s]}' % letters[4],xy=(0,0),xytext=(0.99,1.14),
              textcoords='axes fraction',color='k',fontsize=9,
              rotation=0,ha='center',va='center')

###############################################################################
###############################################################################
###############################################################################
ax2 = plt.subplot(326)
m = Basemap(projection='cea',llcrnrlat=-20.1,urcrnrlat=20.1,\
            llcrnrlon=-179.99,urcrnrlon=179.99,resolution='l',area_thresh=10000)
m.drawcoastlines(color='k',linewidth=0.4,zorder=30)

### Variable
x, y = np.meshgrid(lon_g,lat_g)

circle = m.drawmapboundary(fill_color='dimgrey',color='dimgray',
                  linewidth=0.7)
circle.set_clip_on(False)

parallels = np.arange(-90,91,30)
meridians = np.arange(-180,180,60)
par=m.drawparallels(parallels,labels=[False,True,False,False],linewidth=0.3,
                color='w',fontsize=4,zorder=40)
mer=m.drawmeridians(meridians,labels=[False,False,True,False],linewidth=0.3,
                    fontsize=4,color='w',zorder=40)

cs6 = m.contourf(x,y,snr_p,limits,extend='max',latlon=True)
cs6.set_cmap(cmaps) 
m.drawlsmask(land_color=(0,0,0,0),ocean_color='dimgrey',lakes=False,zorder=11)
       
ax2.annotate(r'\textbf{[%s]}' % letters[5],xy=(0,0),xytext=(0.99,1.14),
              textcoords='axes fraction',color='k',fontsize=9,
              rotation=0,ha='center',va='center')

###############################################################################
cbar_ax1 = fig.add_axes([0.426,0.705,0.15,0.025])                
cbar1 = fig.colorbar(cs1,cax=cbar_ax1,orientation='horizontal',
                    extend='both',extendfrac=0.07,drawedges=False)
cbar1.set_label(label,fontsize=10,color='dimgrey',labelpad=1.4)  
cbar1.set_ticks(barlim)
cbar1.set_ticklabels(list(map(str,barlim)))
cbar1.ax.tick_params(axis='x', size=.01,labelsize=8)
cbar1.outline.set_edgecolor('dimgrey')

###############################################################################
cbar_ax2 = fig.add_axes([0.43,0.09,0.15,0.025])                
cbar2 = fig.colorbar(cs5,cax=cbar_ax2,orientation='horizontal',
                    extend='max',extendfrac=0.07,drawedges=False)
cbar2.set_label(labelss,fontsize=10,color='dimgrey',labelpad=1.4)  
cbar2.set_ticks(barlims)
cbar2.set_ticklabels(list(map(str,barlims)))
cbar2.ax.tick_params(axis='x', size=.01,labelsize=8)
cbar2.outline.set_edgecolor('dimgrey')

plt.tight_layout()
plt.subplots_adjust(hspace=-0.2)

plt.savefig(directoryfigure + 'YearsSinceSAI_XAI-SNR_Globe-Tropics.png',dpi=300)
    
    