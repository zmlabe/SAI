"""
Graph showing regional precipitation anomaly projections between SAI and WACCM

Author     : Zachary M. Labe
Date       : 24 July 2022
Version    : 1 - testing ANN architectures for calculating years since SAI
"""

### Import packages
import sys
import matplotlib.pyplot as plt
import matplotlib.colors as c
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
directorydata = '/Users/zlabe/Data/SAI/'
directoryfigure = '/Users/zlabe/Documents/Research/SolarIntervention/Figures/'
###############################################################################
###############################################################################
modelGCMs = ['ARISE']
datasetsingle = ['ARISE']
seasons = ['annual']
monthlychoice = seasons[0]
variq = 'PRECT'
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
fig = plt.figure()
for i in range(len(reg_nameq)):
    reg_name = reg_nameq[i]
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
    data_wacc,lats,lons = read_primary_dataset(variq,'WACCM',
                                              numOfEns,lensalso,
                                              randomalso,
                                              ravelyearsbinary,
                                              ravelbinary,
                                              shuffletype,
                                              lat_bounds,
                                              lon_bounds)
    data_arise,lats,lons = read_primary_dataset(variq,'ARISE',
                                              numOfEns,lensalso,
                                              randomalso,
                                              ravelyearsbinary,
                                              ravelbinary,
                                              shuffletype,
                                              lat_bounds,
                                              lon_bounds)
    data_obs = np.empty((data_arise.shape))
    data_obs[:] = np.nan
    ###############################################################################   
    ###############################################################################   
    ###############################################################################   
    ### Remove ocean
    data_wacc, data_obs = dSS.remove_ocean(data_wacc.squeeze(),data_obs,lat_bounds,lon_bounds) 
    data_arise, data_obs = dSS.remove_ocean(data_arise.squeeze(),data_obs,lat_bounds,lon_bounds)
    
    data_wacc[np.where(data_wacc==0.)] = np.nan 
    data_arise[np.where(data_arise==0.)] = np.nan 
    
    ### Calculate anomalies
    climo_ens = np.nanmean(data_wacc[:,:(len(yearswaccm)-len(yearsarise)),:,:],axis=1)
    climo = np.nanmean(climo_ens,axis=0)
    
    data_wacca = data_wacc - climo
    data_arisea = data_arise - climo
 
    ### Calculate global means
    lon2,lat2 = np.meshgrid(lons,lats)
    mean_waccm = UT.calc_weightedAve(data_wacca,lat2)
    mean_arise = UT.calc_weightedAve(data_arisea,lat2)
    
    # minwaccm = np.percentile(mean_waccm,10,axis=0)
    # maxwaccm = np.percentile(mean_waccm,90,axis=0)
    minwaccm = np.nanmin(mean_waccm,axis=0)
    maxwaccm = np.nanmax(mean_waccm,axis=0)
    meanwaccm = np.nanmean(mean_waccm,axis=0)
    
    # minarise = np.percentile(mean_arise,10,axis=0)
    # maxarise = np.percentile(mean_arise,90,axis=0)
    minarise = np.nanmin(mean_arise,axis=0)
    maxarise = np.nanmax(mean_arise,axis=0)
    meanarise = np.nanmean(mean_arise,axis=0)

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
    
    ax = plt.subplot(3,3,i+1) 
    adjust_spines(ax, ['left', 'bottom'])            
    ax.spines['top'].set_color('none')
    ax.spines['right'].set_color('none')
    ax.spines['left'].set_color('dimgrey')
    ax.spines['bottom'].set_color('dimgrey')
    ax.spines['bottom'].set_linewidth(2)
    ax.spines['left'].set_linewidth(2)
    ax.tick_params('both',length=3,width=2,which='major')
    ax.tick_params(axis='x',labelsize=6.5,pad=1.5,color='dimgrey')
    ax.tick_params(axis='y',labelsize=6.5,pad=1.5,color='dimgrey')
    
    plt.fill_between(yearswaccm,minwaccm,maxwaccm,facecolor='darkolivegreen',alpha=0.5,clip_on=False)
    plt.plot(yearswaccm,meanwaccm,linestyle='-',color='darkolivegreen',linewidth=1,
             clip_on=False,alpha=1,label=r'\textbf{SSP2-4.5}')
    plt.fill_between(yearsarise,minarise,maxarise,facecolor='saddlebrown',alpha=0.5,clip_on=False)
    plt.plot(yearsarise,meanarise,linestyle='--',color='saddlebrown',linewidth=1,
             clip_on=False,alpha=1,label=r'\textbf{SAI}',dashes=(1,0.3))
    
    plt.axvline(x=2035,ymax=0.8,linestyle='-',color='k',linewidth=2)
    plt.axvline(x=2045,ymax=0.8,linestyle='--',color='dimgrey',linewidth=2,
                dashes=(1,0.8))
    
    if i == 6:
        plt.xticks(np.arange(2015,2080,20),np.arange(2015,2080,20),rotation=0)
        plt.yticks(np.arange(-6,6.1,0.6),np.round(np.arange(-6,6.1,0.6),2),rotation=0)
    elif any([i==0,i==3]):
        plt.xticks(np.arange(2015,2080,20),[],rotation=0)
        plt.yticks(np.arange(-6,6.1,0.6),np.round(np.arange(-6,6.1,0.6),2),rotation=0)
    elif any([i==1,i==2,i==4,i==5]):
        plt.xticks(np.arange(2015,2080,20),[],rotation=0)
        plt.yticks(np.arange(-6,6.1,0.6),[],rotation=0)
    elif any([i==7,i==8]):
        plt.xticks(np.arange(2015,2080,20),np.arange(2015,2080,20),rotation=0)
        plt.yticks(np.arange(-6,6.1,0.6),[],rotation=0)
        
    if i < 6:
        plt.xlim([2015,2069])
        plt.ylim([-0.6,0.6])
        plt.text(2014,0.6,r'\textbf{%s}' % labels[i],fontsize=11,color='k')
        plt.text(2071,0.6,r'\textbf{[%s]}' % letters[i],fontsize=7,color='k',ha='right')
    else:
        plt.xlim([2015,2069])
        plt.ylim([-1.2,1.2])
        plt.text(2014,1.2,r'\textbf{%s}' % labels[i],fontsize=11,color='k')
        plt.text(2071,1.2,r'\textbf{[%s]}' % letters[i],fontsize=7,color='k',ha='right')

    
    if i == 3:
        plt.ylabel(r'\textbf{PRECT [mm/day]}',fontsize=8,
                             color='dimgrey')
    
    if i ==1:
        leg = plt.legend(shadow=False,fontsize=15,loc='upper center',
                    bbox_to_anchor=(0.5, 1.7),fancybox=True,ncol=24,frameon=False,
                    handlelength=1,handletextpad=1)
        for line,text in zip(leg.get_lines(), leg.get_texts()):
            text.set_color(line.get_color())
   
plt.subplots_adjust(wspace=0.2,hspace=0.3)
plt.savefig(directoryfigure + 'Regional_PrecipitationProjections.png',dpi=1000)
