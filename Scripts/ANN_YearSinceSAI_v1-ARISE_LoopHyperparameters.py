"""
ANN for evaluating how long since 2035 (injection) only for ARISE. This 
script looks at the optimized models for each region

Author     : Zachary M. Labe
Date       : 15 April 2022
Version    : 1 - testing ANN architectures for detecting SAI
"""

### Import packages
import sys
import math
import time
import matplotlib.pyplot as plt
import numpy as np
import keras.backend as K
from keras.layers import Dense, Activation
from keras import regularizers
from keras import metrics
from keras import optimizers
from keras.models import Sequential
import tensorflow.keras as keras
import tensorflow as tf
import pandas as pd
import random
import calc_Utilities as UT
import calc_dataFunctions as df
import calc_Stats as dSS
import scipy.stats as sts
from sklearn.metrics import mean_squared_error,mean_absolute_error

import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
warnings.filterwarnings('ignore', category=DeprecationWarning)

### Prevent tensorflow 2.+ deprecation warnings
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)

### Plotting defaults 
plt.rc('text',usetex=True)
plt.rc('font',**{'family':'sans-serif','sans-serif':['Avant Garde']}) 

###############################################################################
###############################################################################
###############################################################################
### Data preliminaries 
directorydata = '/Users/zlabe/Data/SAI/'
###############################################################################
###############################################################################
modelGCMs = ['ARISE']
datasetsingle = ['ARISE']
seasons = ['annual']
variq = 'TREFHT'
reg_nameq = ['Globe','NH','SH','Arctic','Antarctic','narrowTropics','SEAsia','NorthAfrica','Amazon']
labels = ['Globe','NH','SH','Arctic','Antarctic','Tropics','SE Asia','North Africa','Amazon']
timeper = 'historical'
window = 0
###############################################################################
###############################################################################
land_only = True
ocean_only = False
###############################################################################
###############################################################################
yearsall = np.arange(2035+window,2069+1,1)
numOfEns = 10
dataset_obs = '20CRv3'
###############################################################################
###############################################################################
rm_merid_mean = False
rm_annual_mean = False
rm_ensemble_mean = False
rm_standard_dev = False
###############################################################################
###############################################################################
###############################################################################
###############################################################################
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
lensalso = True
lentime = len(yearsall)
###############################################################################
###############################################################################
ravelyearsbinary = False
ravelbinary = False
num_of_class = len(modelGCMs)
ensTypeExperi = 'ENS'
###############################################################################
###############################################################################
ridge_penaltyq = [[0.01],[0.1],[0.25],[0.5],[0.75],[1],[1.5],[2],[3],[5],[10]]
NCOMBOS = 10
###############################################################################
###############################################################################
###############################################################################
###############################################################################
### Create sample class labels for each model for my own testing
### Appends a twin set of classes for the random noise class 
if seasons != 'none':
    classesl = np.empty((num_of_class,numOfEns,len(yearsall)))
    for i in range(num_of_class):
        classesl_row = np.arange(2035,2069+1,1) - 2035 # years since 2035
        classesl[i,:,:] = np.tile(classesl_row,(numOfEns,1))
        
    if ensTypeExperi == 'ENS':
        classeslnew = np.swapaxes(classesl,0,1)

###############################################################################
###############################################################################
###############################################################################
###############################################################################     
### Begin ANN and the entire script
saveSeeds_rr = []
saveAccuracy_rr = []
saveLats_rr = []
saveLons_rr = []
saveIndices_rr = []
segmentSeed_rr = []
networkSeed_rr = []
TrainAccuracy_rr = []
TestAccuracy_rr = []
ValAccuracy_rr = []
lats_rr = []
lons_rr = []
trainIndices_rr = []
testIndices_rr = []
valIndices_rr = []
for rr in range(len(reg_nameq)):
    saveSeeds_ll = []
    saveAccuracy_ll = []
    saveLats_ll = []
    saveLons_ll = []
    saveIndices_ll = []
    segmentSeed_ll = []
    networkSeed_ll = []
    TrainAccuracy_ll = []
    TestAccuracy_ll = []
    ValAccuracy_ll = []
    lats_ll = []
    lons_ll = []
    trainIndices_ll = []
    testIndices_ll = []
    valIndices_ll = []
    for ll in range(len(ridge_penaltyq)):
        saveSeeds_comb = []
        saveAccuracy_comb = []
        saveLats_comb = []
        saveLons_comb = []
        saveIndices_comb = []
        segmentSeed_comb = []
        networkSeed_comb = []
        TrainAccuracy_comb = []
        TestAccuracy_comb = []
        ValAccuracy_comb = []
        lats_comb = []
        lons_comb = []
        trainIndices_comb = []
        testIndices_comb = []
        valIndices_comb = []
        for n in range(NCOMBOS):
            ###############################################################################
            ###############################################################################
            ###############################################################################
            ### ANN preliminaries
            simuqq = datasetsingle[0]
            monthlychoice = seasons[0]
            reg_name = reg_nameq[rr]
            ###############################################################################
            ###############################################################################
            ###############################################################################
            ###############################################################################
            ### Select how to save files
            if land_only == True:
                saveData = seasons[0] + '_LAND' + '_GCMarise_YearsSinceSAI_ARISEONLY' + '_' + variq + '_' + reg_name  + '_' + 'NumOfGCMS-' + str(num_of_class)
            elif ocean_only == True:
                saveData = seasons[0] + '_OCEAN' + '_GCMarise_YearsSinceSAI_ARISEONLY' + '_' + variq + '_' + reg_name + '_' + 'NumOfGCMS-' + str(num_of_class)
            else:
                saveData = seasons[0] + '_GCMarise_YearsSinceSAI_ARISEONLY' + '_' + variq + '_' + reg_name + '_' + 'NumOfGCMS-' + str(num_of_class)
            print('*Filename == < %s >' % saveData) 
        
            lat_bounds,lon_bounds = UT.regions(reg_name)
            directoryfigure = '/Users/zlabe/Desktop/SAI/yearSinceSAI/ONLYARISE/'
            experiment_result = pd.DataFrame(columns=['actual iters','hiddens','cascade',
                                                      'RMSE Train','RMSE Test',
                                                      'ridge penalty','zero mean',
                                                      'zero merid mean','land only?','ocean only?'])    
            
            ### Define primary dataset to use
            dataset = datasetsingle[0]
            
            ### Whether to test and plot the results using obs data
            if dataset_obs == '20CRv3':
                year_obsall = np.arange(1900,2015+1,1)
            elif dataset_obs == 'ERA5':
                year_obsall = np.arange(1979+window,2019+1,1)
            if monthlychoice == 'DJF':
                obsyearstart = year_obsall.min()+1
                year_obs = year_obsall[1:]
            else:
                obsyearstart = year_obsall.min()
                year_obs = year_obsall
            
            ### Remove the annual mean? True to subtract it from dataset ##########
            if rm_annual_mean == True:
                directoryfigure = '/Users/zlabe/Desktop/SAI/yearSinceSAI/ONLYARISE/Loop/'
            
            ### Rove the ensemble mean? True to subtract it from dataset ##########
            if rm_ensemble_mean == True:
                directoryfigure = '/Users/zlabe/Desktop/SAI/yearSinceSAI/ONLYARISE/Loop/'
            
            ### Split the data into training and testing sets? value of 1 will use all 
            ### data as training
            segment_data_factor = .70
            
            ###############################################################################
            ###############################################################################
            ###############################################################################
            ### Read in model and observational/reanalysis data
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
            ### Select data to test, train on           
            def segment_data(data,classesl,ensTypeExperi,fac = segment_data_factor):
              
                global random_segment_seed,trainIndices,testIndices
                if random_segment_seed == None:
                    random_segment_seed = int(int(np.random.randint(1, 100000)))
                np.random.seed(random_segment_seed)
        
        ############################################################################### 
        ############################################################################### 
        ###############################################################################             
                ###################################################################
                ### Large Ensemble experiment
                if ensTypeExperi == 'ENS':
                    
                    ### Flip GCM and ensemble member axes
                    datanew = np.swapaxes(data,0,1)
                    classeslnew = np.swapaxes(classesl,0,1)
            
                if fac < 1 :
                    nrows = datanew.shape[0]
                    segment_train = int(np.round(nrows * fac))
                    segment_test = 1
                    segment_val = nrows - segment_train - segment_test
                    print('--------------------------------------------------------------------')
                    print('Training on',segment_train,'ensembles, Testing on',segment_test,'ensembles, Validation on',segment_val,'ensembles')
                    print('--------------------------------------------------------------------')
            
                    ### Picking out random ensembles for training/testing/validation
                    i = 0
                    trainIndices = list()
                    while i < segment_train:
                        line = np.random.randint(0, nrows)
                        if line not in trainIndices:
                            trainIndices.append(line)
                            i += 1
                        else:
                            pass
                
                    i = 0
                    testIndices = list()
                    while i < segment_test:
                        line = np.random.randint(0, nrows)
                        if line not in trainIndices:
                            if line not in testIndices:
                                testIndices.append(line)
                                i += 1
                        else:
                            pass
                        
                    i = 0
                    valIndices = list()
                    while i < segment_val:
                        line = np.random.randint(0, nrows)
                        if line not in trainIndices:
                            if line not in testIndices:
                                if line not in valIndices:
                                    valIndices.append(line)
                                    i += 1
                        else:
                            pass
                
            ###############################################################################  
            ###############################################################################  
            ###############################################################################  
                    ### Training segment----------
                    data_train = np.empty((len(trainIndices),datanew.shape[1],
                                            datanew.shape[2],datanew.shape[3],
                                            datanew.shape[4]))
                    Ytrain = np.empty((len(trainIndices),classeslnew.shape[1],
                                        classeslnew.shape[2]))
                    for index,ensemble in enumerate(trainIndices):
                        data_train[index,:,:,:,:] = datanew[ensemble,:,:,:,:]
                        Ytrain[index,:,:] = classeslnew[ensemble,:,:]
                        
                    ### Random ensembles are picked
                    print('\n----------------------------------------')
                    print('Training on ensembles: ',trainIndices)
                    print('Testing on ensembles: ',testIndices)
                    print('Validation on ensembles: ',valIndices)
                    print('----------------------------------------')
                    print('\n----------------------------------------')
                    print('org data - shape', datanew.shape)
                    print('training data - shape', data_train.shape)
                
                    ### Reshape into X and Y
                    Xtrain = data_train.reshape((data_train.shape[0]*data_train.shape[1]*data_train.shape[2]),(data_train.shape[3]*data_train.shape[4]))
                    Ytrain = Ytrain.reshape((Ytrain.shape[0]*Ytrain.shape[1]*Ytrain.shape[2]))
                    Xtrain_shape = (data_train.shape[0],data_train.shape[1])
            
            ###############################################################################  
            ###############################################################################          
            ###############################################################################        
                    ### Testing segment----------
                    data_test = np.empty((len(testIndices),datanew.shape[1],
                                            datanew.shape[2],datanew.shape[3],
                                            datanew.shape[4]))
                    Ytest = np.empty((len(testIndices),classeslnew.shape[1],
                                        classeslnew.shape[2]))
                    for index,ensemble in enumerate(testIndices):
                        data_test[index,:,:,:,:] = datanew[ensemble,:,:,:,:]
                        Ytest[index,:,:] = classeslnew[ensemble,:,:]
                    
                    ### Random ensembles are picked
                    print('----------------------------------------\n')
                    print('----------------------------------------')
                    print('Training on ensembles: count %s' % len(trainIndices))
                    print('Testing on ensembles: count %s' % len(testIndices))
                    print('Validation on ensembles: count %s' % len(valIndices))
                    print('----------------------------------------\n')
                    
                    print('----------------------------------------')
                    print('org data - shape', datanew.shape)
                    print('testing data - shape', data_test.shape)
                    print('----------------------------------------')
            
                    ### Reshape into X and Y
                    Xtest= data_test.reshape((data_test.shape[0]*data_test.shape[1]*data_test.shape[2]),(data_test.shape[3]*data_test.shape[4]))
                    Ytest = Ytest.reshape((Ytest.shape[0]*Ytest.shape[1]*Ytest.shape[2]))
                    Xtest_shape = (data_test.shape[0],data_test.shape[1])
                    
            ###############################################################################  
            ###############################################################################  
            ###############################################################################  
                    ### Validation segment----------
                    data_val = np.empty((len(valIndices),datanew.shape[1],
                                            datanew.shape[2],datanew.shape[3],
                                            datanew.shape[4]))
                    Yval = np.empty((len(valIndices),classeslnew.shape[1],
                                        classeslnew.shape[2]))
                    for index,ensemble in enumerate(valIndices):
                        data_val[index,:,:,:,:] = datanew[ensemble,:,:,:,:]
                        Yval[index,:,:] = classeslnew[ensemble,:,:]
                    
                    ### Random ensembles are picked
                    print('\n----------------------------------------')
                    print('Training on ensembles: count %s' % len(trainIndices))
                    print('Testing on ensembles: count %s' % len(testIndices))
                    print('Validation on ensembles: count %s' % len(valIndices))
                    print('----------------------------------------\n')
                    print('----------------------------------------')
                    print('org data - shape', datanew.shape)
                    print('Validation data - shape', data_val.shape)
                    print('----------------------------------------')
            
                    ### Reshape into X and Y
                    Xval= data_val.reshape((data_val.shape[0]*data_val.shape[1]*data_val.shape[2]),(data_val.shape[3]*data_val.shape[4]))
                    Yval = Yval.reshape((Yval.shape[0]*Yval.shape[1]*Yval.shape[2]))
                    Xval_shape = (data_val.shape[0],data_val.shape[1])
                  
                    ### 'unlock' the random seed
                    np.random.seed(None)
              
                else:
                    print(ValueError('WRONG EXPERIMENT!'))
                return Xtrain,Ytrain,Xtest,Ytest,Xval,Yval,Xtrain_shape,Xtest_shape,Xval_shape,testIndices,trainIndices,valIndices
                        
            ###############################################################################
            ###############################################################################
            ###############################################################################
            ### Neural Network Creation & Training        
            class TimeHistory(keras.callbacks.Callback):
                def on_train_begin(self, logs={}):
                    self.times = []
            
                def on_epoch_begin(self, epoch, logs={}):
                    self.epoch_time_start = time.time()
            
                def on_epoch_end(self, epoch, logs={}):
                    self.times.append(time.time() - self.epoch_time_start)
            
            def defineNN(hidden, input_shape, output_shape, ridgePenalty):        
               
                model = Sequential()
                ### Initialize first layer
                ### Model is a single node with activation function
                model.add(Dense(hidden[0],input_shape=(input_shape,),
                                activation=actFun, use_bias=True,
                                kernel_regularizer=regularizers.l1_l2(l1=0.00,l2=ridgePenalty),
                                bias_initializer=keras.initializers.RandomNormal(seed=random_network_seed),
                                kernel_initializer=keras.initializers.RandomNormal(seed=random_network_seed)))
        
                ### Initialize other layers
                for layer in hidden[1:]:
                    model.add(Dense(layer,activation=actFun,
                                    use_bias=True,
                                    kernel_regularizer=regularizers.l1_l2(l1=0.00,l2=0.00),
                                    bias_initializer=keras.initializers.RandomNormal(seed=random_network_seed),
                                    kernel_initializer=keras.initializers.RandomNormal(seed=random_network_seed)))
                        
                    print('\nTHIS IS AN ANN!\n')
            
                #### Initialize output layer
                model.add(Dense(output_shape,activation=None,use_bias=True,
                                kernel_regularizer=regularizers.l1_l2(l1=0.00, l2=0.00),
                                bias_initializer=keras.initializers.RandomNormal(seed=random_network_seed),
                                kernel_initializer=keras.initializers.RandomNormal(seed=random_network_seed)))
                
                return model
            
            def trainNN(model, Xtrain, Ytrain, Xval, Yval, niter, verbose):
              
                global lr_here, batch_size
                lr_here = 0.001
                model.compile(optimizer=optimizers.Adam(lr=lr_here),  
                              loss = 'mean_absolute_error',
                              metrics=[metrics.mean_absolute_error])
            
                ### Declare the relevant model parameters
                batch_size = 32 
            
                print('----ANN Training: learning rate = '+str(lr_here)+'; activation = '+actFun+'; batch = '+str(batch_size) + '----')    
                
                ### Callbacks
                time_callback = TimeHistory()
                early_stopping = keras.callbacks.EarlyStopping(monitor='val_loss',
                                                               patience=10,
                                                               verbose=1,
                                                               mode='auto',
                                                               restore_best_weights=True)
                
                history = model.fit(Xtrain,Ytrain,batch_size=batch_size,epochs=niter,
                                    shuffle=True,verbose=verbose,
                                    callbacks=[time_callback,early_stopping],
                                    validation_data=(Xval,Yval))
                print('******** done training ***********')
            
                return model, history
            
            def test_train_loopClass(Xtrain,Ytrain,Xtest,Ytest,Xval,Yval,iterations,ridge_penalty,hiddens,plot_in_train=True):
                """or loops to iterate through training iterations, ridge penalty, 
                and hidden layer list
                """
                results = {}
                global nnet,random_network_seed
              
                for niter in iterations:
                    for penalty in ridge_penalty:
                        for hidden in hiddens:
                            
                            ### Check / use random seed
                            if random_network_seed == None:
                              np.random.seed(None)
                              random_network_seed = int(np.random.randint(1, 100000))
                            np.random.seed(random_network_seed)
                            random.seed(random_network_seed)
                            tf.set_random_seed(0)
            
                            # For use later
                            Xtrain,Xtest,Xval,stdVals = dSS.standardize_dataVal(Xtrain,Xtest,Xval)
                            Xmean, Xstd = stdVals    
                            
                            ### Define the model
                            model = defineNN(hidden,
                                              input_shape=np.shape(Xtrain)[1],
                                              output_shape=1,
                                              ridgePenalty=penalty)  
                           
                            ### Train the net
                            model, history = trainNN(model,Xtrain,
                                                      Ytrain,Xval,Yval,niter,
                                                      verbose=1)
            
                            ### After training, use the network with training data to 
                            ### check that we don't have any errors and output RMSE
                            rmse_train = dSS.rmse(Ytrain,model.predict(Xtrain))
                            if type(Ytest) != bool:
                                rmse_test = 0.
                                rmse_test = dSS.rmse(Ytest,model.predict(Xtest))
                            else:
                                rmse_test = False
            
                            this_result = {'iters': niter, 
                                            'hiddens' : hidden, 
                                            'RMSE Train' : rmse_train, 
                                            'RMSE Test' : rmse_test, 
                                            'ridge penalty': penalty, 
                                            'zero mean' : rm_annual_mean,
                                            'zero merid mean' : rm_merid_mean,
                                            'land only?' : land_only,
                                            'ocean only?' : ocean_only,
                                            'Segment Seed' : random_segment_seed,
                                            'Network Seed' : random_network_seed }
                            results.update(this_result)
            
                            global experiment_result
                            experiment_result = experiment_result.append(results,
                                                                          ignore_index=True)
            
                            #'unlock' the random seed
                            np.random.seed(None)
                            random.seed(None)
                            tf.set_random_seed(None)
              
                return experiment_result, model, history
            
            ###############################################################################
            ###############################################################################
            ###############################################################################
            ### Parameters
            if variq == 'TREFHT':
                debug = True
                NNType = 'ANN_regress'
                option4 = True
                biasBool = False
                hiddensList = [[10,10]]
                ridge_penalty = ridge_penaltyq[ll]
                actFun = 'relu'       
                iterations = [500] 
                random_segment = True
                foldsN = 1
            elif variq == 'PRECT':
                debug = True
                NNType = 'ANN_regress'
                option4 = True
                biasBool = False
                hiddensList = [[10,10]]
                ridge_penalty = ridge_penaltyq[ll]
                actFun = 'relu'       
                iterations = [500] 
                random_segment = True
            
            session_conf = tf.ConfigProto(intra_op_parallelism_threads=1,
                                          inter_op_parallelism_threads=1)
            sess = tf.Session(graph=tf.get_default_graph(), config=session_conf)
            K.set_session(sess)
            K.clear_session()
            
            ### Get info about the region
            lat_bounds,lon_bounds = UT.regions(reg_name)
            data_all,lats,lons = read_primary_dataset(variq,dataset,
                                                      numOfEns,lensalso,
                                                      randomalso,
                                                      ravelyearsbinary,
                                                      ravelbinary,
                                                      shuffletype,
                                                      lat_bounds,
                                                      lon_bounds)
            data_obs_all,lats_obs,lons_obs = read_obs_dataset(variq,
                                                              dataset_obs,
                                                              numOfEns,
                                                              lensalso,
                                                              randomalso,
                                                              ravelyearsbinary,
                                                              ravelbinary,
                                                              shuffletype,
                                                              lat_bounds,
                                                              lon_bounds)
        
        ###############################################################################
        ###############################################################################
        ###############################################################################                    
            ### Get the data together
            data, data_obs, = data_all, data_obs_all,
        ###############################################################################                        
            if rm_annual_mean == True:
                data, data_obs = dSS.remove_annual_mean(data,data_obs,
                                                    lats,lons,
                                                    lats_obs,lons_obs)
                print('\n*Removed annual mean*\n')
        ###############################################################################                        
            if rm_merid_mean == True:
                data, data_obs = dSS.remove_merid_mean(data,data_obs,
                                                    lats,lons,
                                                    lats_obs,lons_obs)
                print('\n*Removed meridional mean*\n')
        ###############################################################################                        
            if rm_ensemble_mean == True:
                data = dSS.remove_ensemble_mean(data,ravel_modelens,
                                                  ravelmodeltime,
                                                  rm_standard_dev,
                                                  numOfEns)
                print('\n*Removed ensemble mean*')
        ###############################################################################                        
            if land_only == True:
                data, data_obs = dSS.remove_ocean(data,data_obs,
                                                  lat_bounds,
                                                  lon_bounds) 
                print('\n*Removed ocean data*')
        ###############################################################################
            if ocean_only == True:
                data, data_obs = dSS.remove_land(data,data_obs,
                                                  lat_bounds,
                                                  lon_bounds) 
                print('\n*Removed land data*')     
         
        ###############################################################################
        ###############################################################################
        ###############################################################################
            ### Loop over folds
            K.clear_session()
            #---------------------------
            random_segment_seed = None
            #---------------------------
            Xtrain,Ytrain,Xtest,Ytest,Xval,Yval,Xtrain_shape,Xtest_shape,Xval_shape,testIndices,trainIndices,valIndices = segment_data(data,classesl,ensTypeExperi,segment_data_factor)
        
            YtrainClassMulti = Ytrain  
            YtestClassMulti = Ytest
            YvalClassMulti = Yval
        
            # For use later
            XtrainS,XtestS,XvalS,stdVals = dSS.standardize_dataVal(Xtrain,Xtest,Xval)
            Xmean, Xstd = stdVals      
        
            #---------------------------
            random_network_seed = None
            #---------------------------
        
            # Create and train network
            exp_result,model,histmet = test_train_loopClass(Xtrain,
                                                    YtrainClassMulti,
                                                    Xtest,
                                                    YtestClassMulti,
                                                    Xval,
                                                    YvalClassMulti,
                                                    iterations=iterations,
                                                    ridge_penalty=ridge_penalty,
                                                    hiddens=hiddensList,
                                                    plot_in_train = True)
            model.summary()  
            
            ################################################################################################################################################                
            # save the model
            dirname = '/Users/zlabe/Documents/Research/SolarIntervention/savedModels/YearsSinceSAI/ARISEONLY/'
            savename = 'DETECTSAI_ARISEONLY' + '_'+variq+'_' + reg_name + '_' + monthlychoice + '_L2'+ str(ridge_penalty[0])+ '_LR' + str(lr_here)+ '_Batch'+ str(batch_size)+ '_Iters' + str(iterations[0]) + '_' + NNType + str(hiddensList[0][0]) + 'x' + str(hiddensList[0][-1]) + '_SegSeed' + str(random_segment_seed) + '_NetSeed'+ str(random_network_seed) 
            savenameModelTestTrain = 'yearsSinceSAI_ARISEONLY' + '_'+variq+'_modelTrainTest_SegSeed'+str(random_segment_seed)+'_NetSeed'+str(random_network_seed)
            
            if rm_annual_mean == True:
                savename = savename + '_AnnualMeanRemoved' 
                savenameModelTestTrain = savenameModelTestTrain + '_AnnualMeanRemoved'
            if rm_merid_mean == True:
                savename = savename + '_MeridionalMeanRemoved' 
                savenameModelTestTrain = savenameModelTestTrain + '_MeridionalMeanRemoved'
            if rm_ensemble_mean == True:
                savename = savename + '_EnsembleMeanRemoved' 
                savenameModelTestTrain = savenameModelTestTrain + '_EnsembleMeanRemoved'
            if land_only == True: 
                savename = savename + '_LANDONLY'
                savenameModelTestTrain = savenameModelTestTrain + '_LANDONLY'
            if ocean_only == True:
                savename = savename + '_OCEANONLY'
                savenameModelTestTrain = savenameModelTestTrain + '_OCEANONLY'
            
            ###############################################################
            ### Assess observations
            Xobs = data_obs.reshape(data_obs.shape[0],data_obs.shape[1]*data_obs.shape[2])
            yearsObs = np.arange(data_obs.shape[0]) + obsyearstart
            
            ### Standardize testing 'obs'
            XobsS = (Xobs-Xmean)/Xstd
            XobsS[np.isnan(XobsS)] = 0
                
            ### Standardize training
            XtrainS = (Xtrain-Xmean)/Xstd
            XtrainS[np.isnan(XtrainS)] = 0
            
            ### Standardize testing
            XtestS = (Xtest-Xmean)/Xstd
            XtestS[np.isnan(XtestS)] = 0   
            
            ### Prepare validation again
            xvalpred = (Xval-Xmean)/Xstd
            xvalpred[np.isnan(xvalpred)] = 0
            
            ### Make predictions
            YpredTrain = model.predict(XtrainS)
            YpredTest = model.predict(XtestS)
            YpredObs = model.predict(XobsS)
            YpredVal = model.predict(xvalpred)
            
            ### Get output from model
            trainingout = YpredTrain.squeeze()
            testingout = YpredTest.squeeze()
            valout = YpredVal.squeeze()
            
            ### Calculate metrics
            mse_train= mean_squared_error(Ytrain,trainingout,squared=True)
            mse_test = mean_squared_error(Ytest,testingout,squared=True)
            mse_val = mean_squared_error(Yval,valout,squared=True)
            mae_train= mean_absolute_error(Ytrain,trainingout)
            mae_test = mean_absolute_error(Ytest,testingout)
            mae_val = mean_absolute_error(Yval,valout)
            print('\nValidation MAE = %s' % mae_val)
            print('Testing MAE = %s' % mae_test)
            
            corr1 = sts.spearmanr(Yval[:len(yearsall)],valout[:len(yearsall)])[0]
            corr2 = sts.spearmanr(Yval[len(yearsall):],valout[len(yearsall):])[0]
            corrmeanval = (corr1+corr2)/2
                
            ##############################################################################
            ##############################################################################
            ##############################################################################        
            ### Observations
            obsout = YpredObs
        
            ### Save the output 
            directoryoutput = '/Users/zlabe/Documents/Research/SolarIntervention/Data/YearsSinceSAI/ONLYARISE/Loop/'
           
            ## Define variable for analysis
            print('\n\n------------------------')
            print(variq,'= Variable!')
            print(monthlychoice,'= Time!')
            print(reg_name,'= Region!')
            print(lat_bounds,lon_bounds)
            print(dataset,'= Model!')
            print(dataset_obs,'= Observations!\n')
            print(rm_annual_mean,'= rm_annual_mean') 
            print(rm_merid_mean,'= rm_merid_mean') 
            print(rm_ensemble_mean,'= rm_ensemble_mean') 
            print(land_only,'= land_only')
            print(ocean_only,'= ocean_only')
            
            ### Save information
            saveSeedsq = 'YearsSinceSAI_ARISEONLY' + '_ANN_' +variq+'_' + reg_name + '_' + monthlychoice
            saveAccuracyq = 'YearsSinceSAI_ARISEONLY' + '_ANN_' +variq+'_' + reg_name + '_' + monthlychoice
            saveLatsq = 'YearsSinceSAI_ARISEONLY' + '_ANN_' +variq+'_' + reg_name + '_' + monthlychoice
            saveLonsq = 'YearsSinceSAI_ARISEONLY' + '_ANN_' +variq+'_' + reg_name + '_' + monthlychoice
            saveIndicesq = 'YearsSinceSAI_ARISEONLY' + '_ANN_' +variq+'_' + reg_name + '_' + monthlychoice
            print('Iteration ====> [[%s]-regions, %s-L2, [%s]-combos]' % (reg_nameq[rr],ridge_penaltyq[ll],n))
            
            ############################################
            saveSeeds_comb.append(saveSeedsq)
            saveAccuracy_comb.append(saveAccuracyq)
            saveLats_comb.append(saveLatsq)
            saveLons_comb.append(saveLonsq)
            saveIndices_comb.append(saveIndicesq)
            ############################################
            segmentSeed_comb.append(random_segment_seed)
            networkSeed_comb.append(random_network_seed)
            TrainAccuracy_comb.append(mae_train)
            TestAccuracy_comb.append(mae_test)
            ValAccuracy_comb.append(corrmeanval)
            lats_comb.append(lats)
            lons_comb.append(lons)
            trainIndices_comb.append(trainIndices)
            testIndices_comb.append(testIndices)
            valIndices_comb.append(valIndices)  

        ############################################
        ############################################
        saveSeeds_ll.append(saveSeeds_comb)
        saveAccuracy_ll.append(saveAccuracy_comb)
        saveLats_ll.append(saveLats_comb)
        saveLons_ll.append(saveLons_comb)
        saveIndices_ll.append(saveIndices_comb)
        ############################################
        segmentSeed_ll.append(segmentSeed_comb)
        networkSeed_ll.append(networkSeed_comb)
        TrainAccuracy_ll.append(TrainAccuracy_comb)
        TestAccuracy_ll.append(TestAccuracy_comb)
        ValAccuracy_ll.append(ValAccuracy_comb)
        lats_ll.append(lats_comb)
        lons_ll.append(lons_comb)
        trainIndices_ll.append(trainIndices_comb)
        testIndices_ll.append(testIndices_comb)
        valIndices_ll.append(valIndices_comb)  
    ############################################
    ############################################
    saveSeeds_rr.append(saveSeeds_ll)
    saveAccuracy_rr.append(saveAccuracy_ll)
    saveLats_rr.append(saveLats_ll)
    saveLons_rr.append(saveLons_ll)
    saveIndices_rr.append(saveIndices_ll)
    ############################################
    segmentSeed_rr.append(segmentSeed_ll)
    networkSeed_rr.append(networkSeed_ll)
    TrainAccuracy_rr.append(TrainAccuracy_ll)
    TestAccuracy_rr.append(TestAccuracy_ll)
    ValAccuracy_rr.append(ValAccuracy_ll)
    lats_rr.append(lats_ll)
    lons_rr.append(lons_ll)
    trainIndices_rr.append(trainIndices_ll)
    testIndices_rr.append(testIndices_ll)
    valIndices_rr.append(valIndices_ll)  
    ############################################
    
    np.save(directoryoutput + saveSeedsq + '_SegmentSeeds.npy',segmentSeed_ll)
    np.save(directoryoutput + saveSeedsq + '_NetworkSeeds.npy',networkSeed_ll)
    np.save(directoryoutput + saveAccuracyq + '_TrainAcc.npy',TrainAccuracy_ll)
    np.save(directoryoutput + saveAccuracyq + '_TestAcc.npy',TestAccuracy_ll)
    np.save(directoryoutput + saveAccuracyq + '_ValAcc.npy',ValAccuracy_ll)
    np.save(directoryoutput + saveLatsq + '_Latitudes.npy',lats_ll)
    np.save(directoryoutput + saveLonsq + '_Longitudes.npy',lons_ll)
    np.save(directoryoutput + saveIndicesq + '_TrainIndices.npy',trainIndices_ll)
    np.save(directoryoutput + saveIndicesq + '_TestIndices.npy',testIndices_ll)
    np.save(directoryoutput + saveIndicesq + '_ValIndices.npy',valIndices_ll)