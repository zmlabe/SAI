"""
Function reads in both ARISE and CONTROL

Notes
-----
    Author : Zachary Labe
    Date   : 4 April 2022

Usage
-----
    [1] read_ARISEtogether(directory,vari,sliceperiod,slicebase,sliceshape,addclimo,slicenan,takeEnsMean,numOfEns,timeper)
"""

def read_ARISEtogether(directory,vari,monthlychoice,slicebase,sliceshape,addclimo,slicenan,takeEnsMean,numOfEns,timeper):
    """
    Function reads monthly data from all cesm large ensembles

    Parameters
    ----------
    directory : string
        path for data
    vari : string
        variable for analysis
    sliceperiod : string
        how to average time component of data
    sliceyear : string
        how to slice number of years for data
    sliceshape : string
        shape of output array
    addclimo : binary
        True or false to add climatology
    slicenan : string or float
        Set missing values
    takeEnsMean : binary
        whether to take ensemble mean
    numOfEns : integer
        number of ensemble members to use
    timeper : time period of analysis
        string
    ENSmean : numpy array
        ensemble mean

    Returns
    -------
    lat : 1d numpy array
        latitudes
    lon : 1d numpy array
        longitudes
    var : numpy array
        processed variable

    Usage
    -----
    read_ALLCESM_le(directory,vari,sliceperiod,slicebase,
                    sliceshape,addclimo,slicenan,takeEnsMean,numOfEns,timeper)
    """
    print('\n>>>>>>>>>> STARTING read_ALLCESM_le function!')

    ### Import modules
    import numpy as np
    from netCDF4 import Dataset
    import calc_Utilities as UT
    import read_WACCM as WAC
    import read_ARISE as ARI
    
    ### Parameters 
    directorydataWAC = '/Users/zlabe/Data/SAI/monthly/'
    directorydataARI = '/Users/zlabe/Data/SAI/monthly/'
    
    ### Read in both large ensembles from ARISE
    lat1,lon1,waccm,ENSmeanWAC = WAC.read_WACCM(directorydataWAC,vari,
                                                  monthlychoice,
                                                  slicebase,sliceshape,
                                                  addclimo,slicenan,
                                                  takeEnsMean) 
    lat1,lon1,arise,ENSmeanARI = ARI.read_ARISE(directorydataARI,vari,
                                          monthlychoice,
                                          slicebase,sliceshape,
                                          addclimo,slicenan,
                                          takeEnsMean) 

    ### Combine data 
    models = np.asarray([arise[:,:,:,:],waccm[:,-arise.shape[1]:,:,:]]) # only for 2035-2069

    print('\n\nShape of output FINAL = ', models.shape,[[models.ndim]])
    print('>>>>>>>>>> ENDING read_ARISEtogether function!')    
    return lat1,lon1,models 

# ### Test functions - do not use!
# import numpy as np
# import matplotlib.pyplot as plt
# import calc_Utilities as UT
# directory = '/Users/zlabe/Data/'
# vari = 'SST'
# sliceperiod = 'annual'
# slicebase = np.arange(2035,2069+1,1)
# sliceshape = 4
# slicenan = 'nan'
# addclimo = True
# takeEnsMean = False
# timeper = 'historical'
# numOfEns = 10
# lat,lon,var = read_ARISEtogether(directory,vari,sliceperiod,slicebase,
#                                       sliceshape,addclimo,slicenan,
#                                       takeEnsMean,numOfEns,timeper)
# lon2,lat2 = np.meshgrid(lon,lat)
# ave = UT.calc_weightedAve(var,lat2)
