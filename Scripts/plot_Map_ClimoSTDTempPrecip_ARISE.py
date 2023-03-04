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
data_temp,lats,lons = read_primary_dataset('TREFHT','ARISE',
                                          numOfEns,lensalso,
                                          randomalso,
                                          ravelyearsbinary,
                                          ravelbinary,
                                          shuffletype,
                                          lat_bounds,
                                          lon_bounds)
data_precip,lats,lons = read_primary_dataset('PRECT','ARISE',
                                          numOfEns,lensalso,
                                          randomalso,
                                          ravelyearsbinary,
                                          ravelbinary,
                                          shuffletype,
                                          lat_bounds,
                                          lon_bounds)
data_obs = np.empty((data_precip.shape))
data_obs[:] = np.nan
lon2,lat2 = np.meshgrid(lons,lats)

###############################################################################   
###############################################################################   
###############################################################################   
### Remove ocean
data_temp, data_obs = dSS.remove_ocean(data_temp.squeeze(),data_obs,lat_bounds,lon_bounds) 
data_precip, data_obs = dSS.remove_ocean(data_precip.squeeze(),data_obs,lat_bounds,lon_bounds)

### Calculate climatologies for 2045-2069
yearmin = 2045
yearmax = 2069
yearq = np.where((yearsarise >= yearmin) & (yearsarise <= yearmax))[0]

tempyr = data_temp[:,yearq,:,:]
precyr = data_precip[:,yearq,:,:]

### Detrend data before calculating variability
data_tempdt = DT.detrendData(tempyr,'surface','monthly')
data_precdt = DT.detrendData(precyr,'surface','monthly')

data_tempdt[np.where(data_tempdt==0.)] = np.nan 
data_precdt[np.where(data_precdt==0.)] = np.nan 

### Calculate variability
tempens = np.nanstd(data_tempdt[:,:,:,:],axis=1)
precens = np.nanstd(data_precdt[:,:,:,:],axis=1)

### Calculate ensemble means
tempmean = np.nanmean(tempens,axis=0)
precmean = np.nanmean(precens,axis=0)

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

var = tempmean
limit = np.arange(0,1.51,0.05)
barlim = np.arange(0,1.6,0.5)
label = r'\textbf{Std. Dev. -- Temperature [$^{\circ}$C]}'

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

cs1 = m.contourf(x,y,var,limit,extend='max')

# for r in range(len(regionboxes)):
#     lat_bounds,lon_bounds = UT.regions(regionboxes[r])
#     la1 = lat_bounds[0]
#     la2 = lat_bounds[1]
#     lo1 = lon_bounds[0]
#     lo2 = lon_bounds[1]
#     if r < 3:
#         lonsslice = np.linspace(lo1,lo2,lo2-lo1+1)
#         latsslice = np.ones(len(lonsslice))*la2
#         m.plot(lonsslice, latsslice, color='gold', linewidth=0.75, latlon=True,zorder=20,clip_on=False)
#         latsslice = np.ones(len(lonsslice))*la1
#         m.plot(lonsslice, latsslice, color='gold', linewidth=0.75, latlon=True,zorder=20,clip_on=False)
#         m.drawgreatcircle(180, la1, 180, la2,linewidth=0.75,color='gold',zorder=20,clip_on=False)
#         m.drawgreatcircle(-179.9, la1, -179.9, la2,linewidth=0.75,color='gold',zorder=20,clip_on=False)
#     else:
#         lonsslice = np.linspace(lo1,lo2,lo2-lo1+1)
#         latsslice = np.ones(len(lonsslice))*la2
#         m.plot(lonsslice, latsslice, color='gold', linewidth=0.75, latlon=True,zorder=20,clip_on=False)
#         latsslice = np.ones(len(lonsslice))*la1
#         m.plot(lonsslice, latsslice, color='gold', linewidth=0.75, latlon=True,zorder=20,clip_on=False)
#         m.drawgreatcircle(lo1, la1, lo1, la2,linewidth=0.75,color='gold',zorder=20,clip_on=False)
#         m.drawgreatcircle(lo2, la2, lo2, la1,linewidth=0.75,color='gold',zorder=20,clip_on=False)

cs1.set_cmap(cmr.eclipse)
m.drawlsmask(land_color=(0,0,0,0),ocean_color='dimgray',lakes=False,zorder=11)

cbar1 = plt.colorbar(cs1,orientation='horizontal',
                    extend='max',extendfrac=0.07,drawedges=False,
                    fraction=0.03,pad=0.06)
cbar1.set_label(label,fontsize=10,color='dimgrey',labelpad=2)  
cbar1.set_ticks(barlim)
cbar1.set_ticklabels(list(map(str,barlim)))
cbar1.ax.tick_params(axis='x', size=.01,labelsize=8)
cbar1.outline.set_edgecolor('dimgrey')

# plt.annotate(r'\textbf{Tropics}',
#              textcoords='axes fraction',
#              xy=(0,0), xytext=(0.09,0.5),
#              fontsize=10,color='gold',alpha=1,
#              ha='center',va='center',zorder=30)
# plt.annotate(r'\textbf{Arctic}',
#              textcoords='axes fraction',
#              xy=(0,0), xytext=(0.5,0.92),
#              fontsize=10,color='gold',alpha=1,
#              ha='center',va='center',zorder=30)
# plt.annotate(r'\textbf{Antarctic}',
#              textcoords='axes fraction',
#              xy=(0,0), xytext=(0.5,0.06),
#              fontsize=10,color='gold',alpha=1,
#              ha='center',va='center',zorder=30)
# plt.annotate(r'\textbf{Amazon}',
#              textcoords='axes fraction',
#              xy=(0,0), xytext=(0.347,0.49),
#              fontsize=8,color='gold',alpha=1,
#              ha='center',va='center',zorder=30)
# plt.annotate(r'\textbf{Central}',
#              textcoords='axes fraction',
#              xy=(0,0), xytext=(0.5565,0.57),
#              fontsize=8,color='gold',alpha=1,
#              ha='center',va='center',zorder=30)
# plt.annotate(r'\textbf{Africa}',
#              textcoords='axes fraction',
#              xy=(0,0), xytext=(0.5565,0.522),
#              fontsize=8,color='gold',alpha=1,
#              ha='center',va='center',zorder=30)
# plt.annotate(r'\textbf{SE}',
#              textcoords='axes fraction',
#              xy=(0,0), xytext=(0.78,0.687),
#              fontsize=8,color='gold',alpha=1,
#              ha='center',va='center',zorder=30)
# plt.annotate(r'\textbf{Asia}',
#              textcoords='axes fraction',
#              xy=(0,0), xytext=(0.78,0.646),
#              fontsize=8,color='gold',alpha=1,
#              ha='center',va='center',zorder=30)

###############################################################################
###############################################################################
###############################################################################
ax = plt.subplot(122)

var = precmean
limit = np.arange(0,1.51,0.05)
barlim = np.arange(0,1.6,0.5)
label = r'\textbf{Std. Dev. -- Precipitation [mm/day]}'

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

cs1 = m.contourf(x,y,var,limit,extend='max')

# for r in range(len(regionboxes)):
#     lat_bounds,lon_bounds = UT.regions(regionboxes[r])
#     la1 = lat_bounds[0]
#     la2 = lat_bounds[1]
#     lo1 = lon_bounds[0]
#     lo2 = lon_bounds[1]
#     if r < 3:
#         lonsslice = np.linspace(lo1,lo2,lo2-lo1+1)
#         latsslice = np.ones(len(lonsslice))*la2
#         m.plot(lonsslice, latsslice, color='gold', linewidth=0.75, latlon=True,zorder=20,clip_on=False)
#         latsslice = np.ones(len(lonsslice))*la1
#         m.plot(lonsslice, latsslice, color='gold', linewidth=0.75, latlon=True,zorder=20,clip_on=False)
#         m.drawgreatcircle(180, la1, 180, la2,linewidth=0.75,color='gold',zorder=20,clip_on=False)
#         m.drawgreatcircle(-179.9, la1, -179.9, la2,linewidth=0.75,color='gold',zorder=20,clip_on=False)
#     else:
#         lonsslice = np.linspace(lo1,lo2,lo2-lo1+1)
#         latsslice = np.ones(len(lonsslice))*la2
#         m.plot(lonsslice, latsslice, color='gold', linewidth=0.75, latlon=True,zorder=20,clip_on=False)
#         latsslice = np.ones(len(lonsslice))*la1
#         m.plot(lonsslice, latsslice, color='gold', linewidth=0.75, latlon=True,zorder=20,clip_on=False)
#         m.drawgreatcircle(lo1, la1, lo1, la2,linewidth=0.75,color='gold',zorder=20,clip_on=False)
#         m.drawgreatcircle(lo2, la2, lo2, la1,linewidth=0.75,color='gold',zorder=20,clip_on=False)

cs1.set_cmap(cmr.eclipse)
m.drawlsmask(land_color=(0,0,0,0),ocean_color='dimgray',lakes=False,zorder=11)

cbar1 = plt.colorbar(cs1,orientation='horizontal',
                    extend='max',extendfrac=0.07,drawedges=False,
                    fraction=0.03,pad=0.06)
cbar1.set_label(label,fontsize=10,color='dimgrey',labelpad=1.1)  
cbar1.set_ticks(barlim)
cbar1.set_ticklabels(list(map(str,barlim)))
cbar1.ax.tick_params(axis='x', size=.01,labelsize=8)
cbar1.outline.set_edgecolor('dimgrey')

# plt.annotate(r'\textbf{Tropics}',
#              textcoords='axes fraction',
#              xy=(0,0), xytext=(0.09,0.5),
#              fontsize=10,color='gold',alpha=1,
#              ha='center',va='center',zorder=30)
# plt.annotate(r'\textbf{Arctic}',
#              textcoords='axes fraction',
#              xy=(0,0), xytext=(0.5,0.92),
#              fontsize=10,color='gold',alpha=1,
#              ha='center',va='center',zorder=30)
# plt.annotate(r'\textbf{Antarctic}',
#              textcoords='axes fraction',
#              xy=(0,0), xytext=(0.5,0.06),
#              fontsize=10,color='gold',alpha=1,
#              ha='center',va='center',zorder=30)
# plt.annotate(r'\textbf{Amazon}',
#              textcoords='axes fraction',
#              xy=(0,0), xytext=(0.347,0.49),
#              fontsize=8,color='gold',alpha=1,
#              ha='center',va='center',zorder=30)
# plt.annotate(r'\textbf{Central}',
#              textcoords='axes fraction',
#              xy=(0,0), xytext=(0.5565,0.57),
#              fontsize=8,color='gold',alpha=1,
#              ha='center',va='center',zorder=30)
# plt.annotate(r'\textbf{Africa}',
#              textcoords='axes fraction',
#              xy=(0,0), xytext=(0.5565,0.522),
#              fontsize=8,color='gold',alpha=1,
#              ha='center',va='center',zorder=30)
# plt.annotate(r'\textbf{SE}',
#              textcoords='axes fraction',
#              xy=(0,0), xytext=(0.78,0.687),
#              fontsize=8,color='gold',alpha=1,
#              ha='center',va='center',zorder=30)
# plt.annotate(r'\textbf{Asia}',
#              textcoords='axes fraction',
#              xy=(0,0), xytext=(0.78,0.646),
#              fontsize=8,color='gold',alpha=1,
#              ha='center',va='center',zorder=30)


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

plt.savefig(directoryfigure + 'Map_ClimoSTDTempPrecip_RegionBoxes_detrended_ARISE.png',dpi=500)

### Save composite for figure
directoryoutput = '/Users/zlabe/Documents/Research/SolarIntervention/Data/'
np.savez(directoryoutput + 'ARISE_STDTempPrecip.npz',
         tempstd=tempmean,precipstd=precmean,lat=lats,lon=lons)
