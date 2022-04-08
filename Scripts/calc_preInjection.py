"""
Function reads data for 2015-2034 for the pre-injection

Notes
-----
    Author : Zachary Labe
    Date   : 8 April 2022

Usage
-----
    [1] read_preInjection(directory,vari,sliceperiod,slicebase,sliceshape,addclimo,slicenan,takeEnsMean,numOfEns,timeper)
"""

def read_preInjection(directory,vari,monthlychoice,slicebase,sliceshape,addclimo,slicenan,takeEnsMean,numOfEns,timeper):
    """
    Function reads monthly data from pre-injection WACCM6

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
    read_preInjection(directory,vari,sliceperiod,slicebase,
                    sliceshape,addclimo,slicenan,takeEnsMean,numOfEns,timeper)
    """
    print('\n>>>>>>>>>> STARTING read_preInjection function!')

    ### Import modules
    import numpy as np
    import read_WACCM as WAC
    
    ### Parameters 
    directorydataWAC = '/Users/zlabe/Data/SAI/monthly/'
    
    ### Read in both large ensembles from ARISE
    lat1,lon1,waccm,ENSmeanWAC = WAC.read_WACCM(directorydataWAC,vari,
                                                  monthlychoice,
                                                  slicebase,sliceshape,
                                                  addclimo,slicenan,
                                                  takeEnsMean) 
    ### Look for years before injection
    yearsWAC = np.arange(2015,2069+1,1)
    yearsqw = np.where((yearsWAC < 2035))[0]

    ### Combine data 
    preinj = waccm[:,yearsqw,:,:] # only for 2015-2034

    print('\n\nShape of output FINAL = ', preinj.shape,[[preinj.ndim]])
    print('>>>>>>>>>> ENDING read_preInjection function!')    
    return lat1,lon1,preinj

# ### Test functions - do not use!
# import numpy as np
# import matplotlib.pyplot as plt
# import calc_Utilities as UT
# directory = '/Users/zlabe/Data/'
# vari = 'TREFHT'
# sliceperiod = 'annual'
# slicebase = np.arange(2035,2069+1,1)
# sliceshape = 4
# slicenan = 'nan'
# addclimo = True
# takeEnsMean = False
# timeper = 'historical'
# numOfEns = 10
# lat,lon,var = read_preInjection(directory,vari,sliceperiod,slicebase,
#                                       sliceshape,addclimo,slicenan,
#                                       takeEnsMean,numOfEns,timeper)
# lon2,lat2 = np.meshgrid(lon,lat)
# ave = UT.calc_weightedAve(var,lat2)
