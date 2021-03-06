"""
Functions are useful statistical untilities for data processing in the ANN
 
Notes
-----
    Author : Zachary Labe
    Date   : 15 July 2020
    
Usage
-----
    [1] rmse(a,b)
    [2] pickSmileModels(data,modelGCMs,pickSMILE)
    [3] remove_annual_mean(data,data_obs,lats,lons,lats_obs,lons_obs)
    [4] remove_merid_mean(data,data_obs)
    [5] remove_observations_mean(data,data_obs,lats,lons)
    [6] calculate_anomalies(data,data_obs,lats,lons,baseline,yearsall)
    [7] remove_ensemble_mean(data,ravel_modelens,ravelmodeltime,rm_standard_dev,numOfEns)
    [8] remove_ocean(data,data_obs)
    [9] remove_land(data,data_obs)
    [10] standardize_data(Xtrain,Xtest)
    [11] standardize_dataVal(Xtrain,Xtest,Xval)
    [12] rm_standard_dev(var,window,ravelmodeltime,numOfEns)
    [13] rm_variance_dev(var,window)
    [14] smoothedEnsembles(data,lat_bounds,lon_bounds)
    [15] remove_trend_obs(datavar,level)
    [16] convert_fuzzyDecade(data,startYear,classChunk)
    [17] convert_fuzzyDecade_toYear(label,startYear,classChunk)
    [18] invert_year_output(ypred,startYear):
    [19] invert_year_outputChunk(ypred,startYear):d
    [20] convert_to_class(data,startYear):
    [21] create_multiClass(xInput,yOutput):
    [22] create_multiLabel(yClass):
"""

def rmse(a,b):
    """
    Calculates the root mean squared error
    takes two variables, a and b, and returns value
    """
    
    ### Import modules
    import numpy as np
    
    ### Calculate RMSE
    rmse_stat = np.sqrt(np.mean((a - b)**2))
    
    return rmse_stat

###############################################################################
    
def pickSmileModels(data,modelGCMs,pickSMILE):
    """
    Select models to analyze if using a subset
    """
    
    ### Pick return indices of models
    lenOfPicks = len(pickSMILE)
    indModels = [i for i, item in enumerate(modelGCMs) if item in pickSMILE]
    
    ### Slice data
    if data.shape[0] == len(modelGCMs):
        if len(indModels) == lenOfPicks:
            modelSelected = data[indModels]
        else:
            print(ValueError('Something is wrong with the indexing of the models!'))
    else:
        print(ValueError('Something is wrong with the order of the data!'))
    
    return modelSelected

###############################################################################

def remove_annual_mean(data,data_obs,lats,lons,lats_obs,lons_obs):
    """
    Removes annual mean from data set
    """
    
    ### Import modulates
    import numpy as np
    import calc_Utilities as UT
    import sys
    
    ### Create 2d grid
    lons2,lats2 = np.meshgrid(lons,lats)
    lons2_obs,lats2_obs = np.meshgrid(lons_obs,lats_obs)
    
    ### Calculate weighted average and remove mean
    if data.ndim == 4:
        data = data - UT.calc_weightedAve(data,lats2)[:,:,np.newaxis,np.newaxis]
        data_obs = data_obs - UT.calc_weightedAve(data_obs,lats2_obs)[:,np.newaxis,np.newaxis]
    elif data.ndim == 5:
        data = data - UT.calc_weightedAve(data,lats2)[:,:,:,np.newaxis,np.newaxis]
        data_obs = data_obs - UT.calc_weightedAve(data_obs,lats2_obs)[:,np.newaxis,np.newaxis]
    else:
        print(ValueError('WRONG SIZE OF DATA FOR REMOVING ANNUAL MEAN!'))
        sys.exit()
        
    return data,data_obs

###############################################################################

def remove_merid_mean(data,data_obs,lats,lons,lats_obs,lons_obs):
    """
    Removes meridional mean from data set
    """
    
    ### Import modules
    import numpy as np
    
    ### Remove mean of latitude
    data = data - np.nanmean(data,axis=2)[:,:,np.newaxis,:]
    data_obs = data_obs - np.nanmean(data_obs,axis=1)[:,np.newaxis,:]

    return data,data_obs

###############################################################################

def remove_observations_mean(data,data_obs,lats,lons):
    """
    Removes observations to calculate model biases
    """
    
    ### Import modules
    import numpy as np
    
    ### Remove observational data
    databias = data - data_obs[np.newaxis,np.newaxis,:,:,:]

    return databias

###############################################################################

def calculate_anomalies(data,data_obs,lats,lons,baseline,yearsall,yearsobs):
    """
    Calculates anomalies for each model and observational data set. Note that
    it assumes the years at the moment
    """
    
    ### Import modules
    import numpy as np
    
    ### Select years to slice
    minyr = baseline.min()
    maxyr = baseline.max()
    yearq = np.where((yearsall >= minyr) & (yearsall <= maxyr))[0]
    yearobsq = np.where((yearsobs >= minyr) & (yearsobs <= maxyr))[0]
    
    if data.ndim == 5:
        
        ### Slice years
        modelnew = data[:,:,yearq,:,:]
        obsnew = data_obs[yearobsq,:,:]
        
        ### Average climatology
        meanmodel = np.nanmean(modelnew[:,:,:,:,:],axis=2)
        meanobs = np.nanmean(obsnew,axis=0)
        
        ### Calculate anomalies
        modelanom = data[:,:,:,:,:] - meanmodel[:,:,np.newaxis,:,:]
        obsanom = data_obs[:,:,:] - meanobs[:,:]
        
    elif data.ndim == 4:
        
        ### Slice years
        modelnew = data[:,yearq,:,:]
        obsnew = data_obs[yearobsq,:,:]
        
        ### Average climatology
        meanmodel = np.nanmean(modelnew[:,:,:,:],axis=1)
        meanobs = np.nanmean(obsnew,axis=0)
        
        ### Calculate anomalies
        modelanom = data[:,:,:,:] - meanmodel[:,np.newaxis,:,:]
        obsanom = data_obs[:,:,:] - meanobs[:,:]
        
    else:
        obsnew = data_obs[yearobsq,:,:]
        
        ### Average climatology
        meanobs = np.nanmean(obsnew,axis=0)
        
        ### Calculate anomalies
        obsanom = data_obs[:,:,:] - meanobs[:,:]
        modelanom = np.nan
        print('NO MODEL ANOMALIES DUE TO SHAPE SIZE!!!')

    return modelanom,obsanom

###############################################################################

def remove_ensemble_mean(data,ravel_modelens,ravelmodeltime,rm_standard_dev,numOfEns):
    """
    Removes ensemble mean
    """
    
    ### Import modulates
    import numpy as np
    
    ### Remove ensemble mean
    if data.ndim == 4:
        datameangoneq = data - np.nanmean(data,axis=0)
    elif data.ndim == 5:
        ensmeanmodel = np.nanmean(data,axis=1)
        datameangoneq = np.empty((data.shape))
        for i in range(data.shape[0]):
            datameangoneq[i,:,:,:,:] = data[i,:,:,:,:] - ensmeanmodel[i,:,:,:]
            print('Completed: Ensemble mean removed for model %s!' % (i+1))
    
    if ravel_modelens == True:
        datameangone = np.reshape(datameangoneq,(datameangoneq.shape[0]*datameangoneq.shape[1],
                                                 datameangoneq.shape[2],
                                                 datameangoneq.shape[3],
                                                 datameangoneq.shape[4]))
    else: 
        datameangone = datameangoneq
    if rm_standard_dev == False:
        if ravelmodeltime == True:
            datameangone = np.reshape(datameangoneq,(datameangoneq.shape[0]*datameangoneq.shape[1]*datameangoneq.shape[2],
                                                      datameangoneq.shape[3],
                                                      datameangoneq.shape[4]))
        else: 
            datameangone = datameangoneq
    
    return datameangone

###############################################################################

def remove_ocean(data,data_obs,lat_bounds,lon_bounds):
    """
    Masks out the ocean for land_only == True
    """
    
    ### Import modules
    import numpy as np
    from netCDF4 import Dataset
    import calc_dataFunctions as df
    
    ### Read in land mask
    directorydata = '/Users/zlabe/Data/masks/'
    filename = 'lsmask_19x25.nc'
    datafile = Dataset(directorydata + filename)
    maskq = datafile.variables['nmask'][:]
    lats = datafile.variables['latitude'][:]
    lons = datafile.variables['longitude'][:]
    datafile.close()
    
    mask,lats,lons = df.getRegion(maskq,lats,lons,lat_bounds,lon_bounds)
    
    ### Mask out model and observations
    datamask = data * mask
    data_obsmask = data_obs * mask
    
    ### Check for floats
    datamask[np.where(datamask==0.)] = 0
    data_obsmask[np.where(data_obsmask==0.)] = 0
    
    return datamask, data_obsmask

###############################################################################

def remove_land(data,data_obs,lat_bounds,lon_bounds):
    """
    Masks out the ocean for ocean_only == True
    """
    
    ### Import modules
    import numpy as np
    from netCDF4 import Dataset
    import calc_dataFunctions as df
    
    ### Read in ocean mask
    directorydata = '/Users/zlabe/Data/masks/'
    filename = 'ocmask_19x25.nc'
    datafile = Dataset(directorydata + filename)
    maskq = datafile.variables['nmask'][:]
    lats = datafile.variables['latitude'][:]
    lons = datafile.variables['longitude'][:]
    datafile.close()
    
    mask,lats,lons = df.getRegion(maskq,lats,lons,lat_bounds,lon_bounds)
    
    ### Mask out model and observations
    datamask = data * mask
    data_obsmask = data_obs * mask
    
    ### Check for floats
    datamask[np.where(datamask==0.)] = 0
    data_obsmask[np.where(data_obsmask==0.)] = 0
    
    return datamask, data_obsmask

###############################################################################

def standardize_data(Xtrain,Xtest):
    """
    Standardizes training and testing data
    """
    
    ### Import modulates
    import numpy as np

    Xmean = np.mean(Xtrain,axis=0)
    Xstd = np.std(Xtrain,axis=0)
    Xtest = (Xtest - Xmean)/Xstd
    Xtrain = (Xtrain - Xmean)/Xstd
    
    stdVals = (Xmean,Xstd)
    stdVals = stdVals[:]
    
    ### If there is a nan (like for land/ocean masks)
    if np.isnan(np.min(Xtrain)) == True:
        Xtrain[np.isnan(Xtrain)] = 0
        Xtest[np.isnan(Xtest)] = 0
        print('--THERE WAS A NAN IN THE STANDARDIZED DATA!--')
    
    return Xtrain,Xtest,stdVals

###############################################################################

def standardize_dataVal(Xtrain,Xtest,Xval):
    """
    Standardizes training, testing, and validation data
    """
    
    ### Import modulates
    import numpy as np

    Xmean = np.mean(Xtrain,axis=0)
    Xstd = np.std(Xtrain,axis=0)
    
    Xtest = (Xtest - Xmean)/Xstd
    Xtrain = (Xtrain - Xmean)/Xstd
    Xval = (Xval - Xmean)/Xstd
    
    stdVals = (Xmean,Xstd)
    stdVals = stdVals[:]
    
    ### If there is a nan (like for land/ocean masks)
    if np.isnan(np.min(Xtrain)) == True:
        Xtrain[np.isnan(Xtrain)] = 0
        Xtest[np.isnan(Xtest)] = 0
        Xval[np.isnan(Xval)] = 0
        print('--THERE WAS A NAN IN THE STANDARDIZED DATA!--')
    
    return Xtrain,Xtest,Xval,stdVals

###############################################################################
    
def rm_standard_dev(var,window,ravelmodeltime,numOfEns):
    """
    Smoothed standard deviation
    """
    import pandas as pd
    import numpy as np
    
    print('\n\n-----------STARTED: Rolling std!\n\n')
    
    
    if var.ndim == 3:
        rollingstd = np.empty((var.shape))
        for i in range(var.shape[1]):
            for j in range(var.shape[2]):
                series = pd.Series(var[:,i,j])
                rollingstd[:,i,j] = series.rolling(window).std().to_numpy()
    elif var.ndim == 4:
        rollingstd = np.empty((var.shape))
        for ens in range(var.shape[0]):
            for i in range(var.shape[2]):
                for j in range(var.shape[3]):
                    series = pd.Series(var[ens,:,i,j])
                    rollingstd[ens,:,i,j] = series.rolling(window).std().to_numpy()
    elif var.ndim == 5:
        varn = np.reshape(var,(var.shape[0]*var.shape[1],var.shape[2],var.shape[3],var.shape[4]))
        rollingstd = np.empty((varn.shape))
        for ens in range(varn.shape[0]):
            for i in range(varn.shape[2]):
                for j in range(varn.shape[3]):
                    series = pd.Series(varn[ens,:,i,j])
                    rollingstd[ens,:,i,j] = series.rolling(window).std().to_numpy()
    
    newdataq = rollingstd[:,window:,:,:] 
    
    if ravelmodeltime == True:
        newdata = np.reshape(newdataq,(newdataq.shape[0]*newdataq.shape[1],
                                       newdataq.shape[2],newdataq.shape[3]))
    else:
        newdata = np.reshape(newdataq,(numOfEns,newdataq.shape[1],
                                       newdataq.shape[2],newdataq.shape[3]))
    print('-----------COMPLETED: Rolling std!\n\n')     
    return newdata 

###############################################################################
    
def rm_variance_dev(var,window,ravelmodeltime):
    """
    Smoothed variance
    """
    import pandas as pd
    import numpy as np
    
    print('\n\n-----------STARTED: Rolling vari!\n\n')
    
    rollingvar = np.empty((var.shape))
    for ens in range(var.shape[0]):
        for i in range(var.shape[2]):
            for j in range(var.shape[3]):
                series = pd.Series(var[ens,:,i,j])
                rollingvar[ens,:,i,j] = series.rolling(window).var().to_numpy()
    
    newdataq = rollingvar[:,window:,:,:] 
    
    if ravelmodeltime == True:
        newdata = np.reshape(newdataq,(newdataq.shape[0]*newdataq.shape[1],
                                       newdataq.shape[2],newdataq.shape[3]))
    else:
        newdata = newdataq
    print('-----------COMPLETED: Rolling vari!\n\n')     
    return newdata 

###############################################################################

def smoothedEnsembles(data,lat_bounds,lon_bounds):
    """ 
    Smoothes all ensembles by taking subsamples
    """
    ### Import modules
    import numpy as np
    import sys
    print('\n------- Beginning of smoothing the ensembles per model -------')
       
    ### Save MM
    newmodels = data.copy()
    mmean = newmodels[-1,:,:,:,:] # 7 for MMmean
    otherens = newmodels[:7,:,:,:,:]

    newmodeltest = np.empty(otherens.shape)
    for modi in range(otherens.shape[0]):
        for sh in range(otherens.shape[1]):
            ensnum = np.arange(otherens.shape[1])
            slices = np.random.choice(ensnum,size=otherens.shape[0],replace=False)
            modelsmooth = otherens[modi]
            slicenewmodel = np.nanmean(modelsmooth[slices,:,:,:],axis=0)
            newmodeltest[modi,sh,:,:,:] = slicenewmodel
    
    ### Add new class
    smoothClass = np.append(newmodeltest,mmean[np.newaxis,:,:,:],axis=0)
    print('--Size of smooth twin --->',newmodeltest.shape)
    
    print('--NEW Size of smoothedclass--->',smoothClass.shape)
    print('------- Ending of smoothing the ensembles per model -------')
    return smoothClass

###############################################################################

def remove_trend_obs(datavar,level):
    """
    Function removes linear trend

    Parameters
    ----------
    datavar : n-d array
        [year,lat,lon] or [year,month,lat,lon] or [year,month,level,lat,lon]
    level : string
        Height of variable (surface or profile)
    
    Returns
    -------
    datavardt : n-d array
        [year,lat,lon] or [year,month,lat,lon] or [year,month,level,lat,lon]

    Usage
    -----
    datavardt = remove_trend_obs(datavar,level)
    """
    print('\n>>> Using remove_trend_obs function! \n')
    ###########################################################################
    ###########################################################################
    ###########################################################################
    ### Import modules
    import numpy as np
    import scipy.stats as sts
    import sys
    
    ### Detrend data array
    if level == 'surface':
        if datavar.ndim == 4:
            x = np.arange(datavar.shape[0])
            
            slopes = np.empty((datavar.shape[1],datavar.shape[2],datavar.shape[3]))
            intercepts = np.empty((datavar.shape[1],datavar.shape[2],
                                   datavar.shape[3]))
            for mo in range(datavar.shape[1]):
                print('Completed: detrended -- Month %s --!' % (mo+1))
                for i in range(datavar.shape[2]):
                    for j in range(datavar.shape[3]):
                        mask = np.isfinite(datavar[:,mo,i,j])
                        y = datavar[:,mo,i,j]
                        
                        if np.sum(mask) == y.shape[0]:
                            xx = x
                            yy = y
                        else:
                            xx = x[mask]
                            yy = y[mask]
                        
                        if np.isfinite(np.nanmean(yy)):
                            slopes[mo,i,j],intercepts[mo,i,j], \
                            r_value,p_value,std_err = sts.linregress(xx,yy)
                        else:
                            slopes[mo,i,j] = np.nan
                            intercepts[mo,i,j] = np.nan
            print('Completed: Detrended data for each grid point!')
                                
            datavardt = np.empty(datavar.shape)
            for yr in range(datavar.shape[0]):
                datavardt[yr,:,:,:] = datavar[yr,:,:,:] - (slopes*x[yr] + intercepts)
        elif datavar.ndim == 3:
            x = np.arange(datavar.shape[0])
            
            slopes = np.empty((datavar.shape[1],datavar.shape[2]))
            intercepts = np.empty((datavar.shape[1],datavar.shape[2]))
            for i in range(datavar.shape[1]):
                for j in range(datavar.shape[2]):
                    mask = np.isfinite(datavar[:,i,j])
                    y = datavar[:,i,j]
                    
                    if np.sum(mask) == y.shape[0]:
                        xx = x
                        yy = y
                    else:
                        xx = x[mask]
                        yy = y[mask]
                    
                    if np.isfinite(np.nanmean(yy)):
                        slopes[i,j],intercepts[i,j], \
                        r_value,p_value,std_err = sts.linregress(xx,yy)
                    else:
                        slopes[i,j] = np.nan
                        intercepts[i,j] = np.nan
            print('Completed: Detrended data for each grid point!')
                                
            datavardt = np.empty(datavar.shape)
            for yr in range(datavar.shape[0]):
                datavardt[yr,:,:] = datavar[yr,:,:] - (slopes*x[yr] + intercepts)
        else:
            print(ValueError('SOMETHING IS WRONG WITH OBS!!!!!'))
            sys.exit()
                                
    elif level == 'profile':
        x = np.arange(datavar.shape[0])
        
        slopes = np.empty((datavar.shape[1],datavar.shape[2],
                          datavar.shape[3],datavar.shape[4]))
        intercepts = np.empty((datavar.shape[1],datavar.shape[2],
                      datavar.shape[3],datavar.shape[4]))
        for mo in range(datavar.shape[1]):
            print('Completed: detrended -- Month %s --!' % (mo+1))
            for le in range(datavar.shape[2]):
                print('Completed: detrended Level %s!' % (le+1))
                for i in range(datavar.shape[3]):
                    for j in range(datavar.shape[4]):
                        mask = np.isfinite(datavar[:,mo,le,i,j])
                        y = datavar[:,mo,le,i,j]
                        
                        if np.sum(mask) == y.shape[0]:
                            xx = x
                            yy = y
                        else:
                            xx = x[mask]
                            yy = y[mask]
                        
                        if np.isfinite(np.nanmean(yy)):
                            slopes[mo,le,i,j],intercepts[mo,le,i,j], \
                            r_value,p_value,std_err= sts.linregress(xx,yy)
                        else:
                            slopes[mo,le,i,j] = np.nan
                            intercepts[mo,le,i,j] = np.nan
        print('Completed: Detrended data for each grid point!')
                            
        datavardt = np.empty(datavar.shape)
        for yr in range(datavar.shape[1]):
            datavardt[yr,:,:,:,:] = datavar[yr,:,:,:,:] - \
                                    (slopes*x[yr] + intercepts)        
    else:
        print(ValueError('Selected wrong height - (surface or profile!)!')) 
        sys.exit()

    ### Save memory
    del datavar
    
    print('\n>>> Completed: Finished remove_trend_obs function!')
    return datavardt

###############################################################################
###############################################################################
###############################################################################
    
def convert_fuzzyDecade(data,startYear,classChunk):
    ### Import modules
    import numpy as np
    import scipy.stats as stats
    
    years = np.arange(startYear-classChunk*2,2100+classChunk*2)
    chunks = years[::int(classChunk)] + classChunk/2
    
    labels = np.zeros((np.shape(data)[0],len(chunks)))
    
    for iy,y in enumerate(data):
        norm = stats.uniform.pdf(years,loc=y-classChunk/2.,scale=classChunk)
        
        vec = []
        for sy in years[::classChunk]:
            j=np.logical_and(years>sy,years<sy+classChunk)
            vec.append(np.sum(norm[j]))
        vec = np.asarray(vec)
        vec[vec<.0001] = 0. # This should not matter
        
        vec = vec/np.sum(vec)
        
        labels[iy,:] = vec
    return labels, chunks

def convert_fuzzyDecade_toYear(label,startYear,classChunk):
    ### Import modules
    import numpy as np
    
    print('SELECT END YEAR - HARD CODED IN FUNCTION - 2069(WACCM)')
    years = np.arange(startYear-classChunk*2,2069+classChunk*2)
    chunks = years[::int(classChunk)] + classChunk/2
    
    return np.sum(label*chunks,axis=1)

def invert_year_output(ypred,startYear):
    ### Import modules
    import numpy as np
    import scipy.stats as stats
    
    if(option4):
        inverted_years = convert_fuzzyDecade_toYear(ypred,startYear,classChunk)
    else:
        inverted_years = invert_year_outputChunk(ypred,startYear)
    
    return inverted_years

def invert_year_outputChunk(ypred,startYear):
    ### Import modules
    import numpy as np
    import scipy.stats as stats
    
    if(len(np.shape(ypred))==1):
        maxIndices = np.where(ypred==np.max(ypred))[0]
        if(len(maxIndices)>classChunkHalf):
            maxIndex = maxIndices[classChunkHalf]
        else:
            maxIndex = maxIndices[0]

        inverted = maxIndex + startYear - classChunkHalf

    else:    
        inverted = np.zeros((np.shape(ypred)[0],))
        for ind in np.arange(0,np.shape(ypred)[0]):
            maxIndices = np.where(ypred[ind]==np.max(ypred[ind]))[0]
            if(len(maxIndices)>classChunkHalf):
                maxIndex = maxIndices[classChunkHalf]
            else:
                maxIndex = maxIndices[0]
            inverted[ind] = maxIndex + startYear - classChunkHalf
    
    return inverted

def convert_to_class(data,startYear):
    ### Import modules
    import numpy as np
    
    data = np.array(data) - startYear + classChunkHalf
    dataClass = to_categorical(data)
    
    return dataClass

def create_multiClass(xInput,yOutput):
    ### Import modules
    import numpy as np
    import copy as copy
    
    yMulti = copy.deepcopy(yOutput)
    
    for stepVal in np.arange(-classChunkHalf,classChunkHalf+1,1.):
        if(stepVal==0):
            continue
        y = yOutput + stepVal
        
    return xInput, yMulti

def create_multiLabel(yClass):
    ### Import modules
    import numpy as np
    
    youtClass = yClass
    
    for i in np.arange(0,np.shape(yClass)[0]):
        v = yClass[i,:]
        j = np.argmax(v)
        youtClass[i,j-classChunkHalf:j+classChunkHalf+1] = 1
    
    return youtClass