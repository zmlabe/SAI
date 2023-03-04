"""
Logistic regression for evaluating differences in ARISE vs. CONTROL

Author     : Zachary M. Labe
Date       : 7 April 2022
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
from keras import regularizers,metrics,optimizers
from keras.models import Sequential
import tensorflow.keras as keras
import tensorflow as tf
import pandas as pd
import random
import calc_Utilities as UT
import calc_dataFunctions as df
import calc_Stats as dSS
from sklearn.metrics import accuracy_score

import warnings
warnings.simplefilter(action = 'ignore', category = FutureWarning)
warnings.filterwarnings('ignore', category = DeprecationWarning)

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
directoryModels = '/Users/zlabe/Documents/Research/SolarIntervention/Data/detectSAI/Loop/Models/'
###############################################################################
###############################################################################
modelGCMs = ['ARISE','WACCM']
datasetsingle = ['all_saiComparison']
seasons = ['annual']
variq = 'PRECT'
timeper = 'historical'
window = 0
###############################################################################
###############################################################################
reg_nameq = ['Globe','NH','SH','Arctic','Antarctic','narrowTropics','SEAsia','NorthAfrica','Amazon']
ridge_penaltyq = np.genfromtxt(directoryModels + 'L2_LogReg_DetectSAI_%s.txt' % variq,unpack=True)
# random_segment_seedq = np.genfromtxt(directoryModels + 'SegSeed_LogReg_DetectSAI_%s.txt' % variq,unpack=True).astype(int)
# random_network_seedq = np.genfromtxt(directoryModels + 'NetSeed_LogReg_DetectSAI_%s.txt' % variq,unpack=True).astype(int)
sys.exit()
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
### Begin ANN and the entire script
for seas in range(len(reg_nameq)):
    monthlychoice = seasons[0]
    reg_name = reg_nameq[seas]
    ###############################################################################
    ###############################################################################
    ###############################################################################
    ### Select how to save files
    if land_only == True:
        saveData = seasons[0] + '_LAND' + '_GCMarise_LOGREG' + '_' + variq + '_' + reg_name  + '_' + 'NumOfGCMS-' + str(num_of_class)
    elif ocean_only == True:
        saveData = seasons[0] + '_OCEAN' + '_GCMarise_LOGREG' + '_' + variq + '_' + reg_name + '_' + 'NumOfGCMS-' + str(num_of_class)
    else:
        saveData = seasons[0] + '_GCMarise_LOGREG' + '_' + variq + '_' + reg_name + '_' + 'NumOfGCMS-' + str(num_of_class)
    print('*Filename == < %s >' % saveData) 
    ###############################################################################
    ###############################################################################
    ###############################################################################
    lat_bounds,lon_bounds = UT.regions(reg_name)
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
    
    ### Split the data into training and testing sets? value of 1 will 
    ### use all data as training
    segment_data_factor = .70
    
    ###############################################################################
    ###############################################################################
    ###############################################################################
    ### Read in model and observational/reanalysis data
    def read_primary_dataset(variq,dataset,numOfEns,lensalso,randomalso,ravelyearsbinary,ravelbinary,shuffletype,lat_bounds=lat_bounds,lon_bounds=lon_bounds):
        data,lats,lons = df.readFiles(variq,dataset,monthlychoice,numOfEns,lensalso,randomalso,ravelyearsbinary,ravelbinary,shuffletype,timeper)
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
      
        global random_segment_seed,trainIndices,testIndices,valIndices
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
            
            ### One-hot vectors
            Ytrain = keras.utils.to_categorical(Ytrain)
            Ytest = keras.utils.to_categorical(Ytest)  
            Yval = keras.utils.to_categorical(Yval)  
      
        else:
            print(ValueError('WRONG EXPERIMENT!'))
        return Xtrain,Ytrain,Xtest,Ytest,Xval,Yval,Xtrain_shape,Xtest_shape,Xval_shape,testIndices,trainIndices,valIndices
                
    ###############################################################################
    ###############################################################################
    ###############################################################################
    ### Add any callbacks  
    class TimeHistory(keras.callbacks.Callback):
        def on_train_begin(self, logs={}):
            self.times = []
    
        def on_epoch_begin(self, epoch, logs={}):
            self.epoch_time_start = time.time()
    
        def on_epoch_end(self, epoch, logs={}):
            self.times.append(time.time() - self.epoch_time_start)
            
    ###############################################################################
    ###############################################################################
    ###############################################################################
    ### Neural Network Creation & Training          
    def defineLOGREG(hidden, input_shape, output_shape, ridgePenalty):        
        
        ### Model is linear (softmax regression)
        model = Sequential()
        model.add(Dense(2,input_shape=(input_shape,),
                        activation='softmax',use_bias=True,
                        kernel_regularizer=regularizers.l1_l2(l1=0.00,l2=ridgePenalty),
                        bias_initializer=keras.initializers.RandomNormal(seed=random_network_seed),
                        kernel_initializer=keras.initializers.RandomNormal(seed=random_network_seed)))  
        print('\nTHIS IS AN LOGISTIC REGRESSION\n')
        
        return model
    
    def trainLOGREG(model, Xtrain, Ytrain, Xval, Yval, niter, verbose):
      
        global lr_here, batch_size
        lr_here = 0.001
        model.compile(optimizer=optimizers.SGD(lr=lr_here,
                      momentum=0.9,nesterov=True),  
                      loss = 'categorical_crossentropy',
                      metrics=[metrics.categorical_accuracy])
    
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
        """loops to iterate through training iterations, ridge penalty, 
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
                    model = defineLOGREG(hidden,
                                      input_shape=np.shape(Xtrain)[1],
                                      output_shape=np.shape(Ytrain)[1],
                                      ridgePenalty=penalty)  
                   
                    ### Train the net
                    model, history = trainLOGREG(model,Xtrain,
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
                    experiment_result = experiment_result.append(results,ignore_index=True)
                    
                    ### if True to plot each iter's graphs.
                    if plot_in_train == True:
                        plt.figure(figsize=(15,5))
                        plt.subplot(1,2,1)
                        plt.plot(history.history['loss'],label = 'training')
                        plt.plot(history.history['val_loss'], label = 'validation')
                        plt.title('loss')
                        plt.xlabel('epoch')
                        plt.legend()
                        
                        plt.subplot(1,2,2)
                        plt.plot(history.history['categorical_accuracy'],label = 'training')
                        plt.plot(history.history['val_categorical_accuracy'],label = 'validation')
                        plt.title('Accuracy')
                        plt.xlabel('epoch')
                        plt.legend()
    
                    ### 'unlock' the random seed
                    np.random.seed(None)
                    random.seed(None)
                    tf.set_random_seed(None)
      
        return experiment_result, model, history
    
    ###############################################################################
    ###############################################################################
    ###############################################################################
    ### Results
    session_conf = tf.ConfigProto(intra_op_parallelism_threads=1,
                                  inter_op_parallelism_threads=1)
    
    sess = tf.Session(graph=tf.get_default_graph(), config=session_conf)
    K.set_session(sess)
    K.clear_session()
    
    ### Parameters
    NNType = 'logreg'
    hiddensList = [[0]]
    ridge_penalty = [ridge_penaltyq[seas]]
    actFun = 'linear'       
    iterations = [750] 
    random_segment = True

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
    K.clear_session()
    #---------------------------
    random_segment_seed = 16159
    #---------------------------
    Xtrain,Ytrain,Xtest,Ytest,Xval,Yval,Xtrain_shape,Xtest_shape,Xval_shape,testIndices,trainIndices,valIndices = segment_data(data,classesl,ensTypeExperi,segment_data_factor)

    YtrainClassMulti = Ytrain  
    YtestClassMulti = Ytest
    YvalClassMulti = Yval

    # For use later
    XtrainS,XtestS,XvalS,stdVals = dSS.standardize_dataVal(Xtrain,Xtest,Xval)
    Xmean, Xstd = stdVals      

    #---------------------------
    random_network_seed = 87750
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
    dirname = '/Users/zlabe/Documents/Research/SolarIntervention/savedModels/'
    savename = 'DETECTSAI' + '_LOGREG_' +variq+'_' + reg_name + '_' + monthlychoice + '_L2'+ str(ridge_penalty[0])+ '_LR' + str(lr_here)+ '_Batch'+ str(batch_size)+ '_Iters' + str(iterations[0]) + '_' + NNType + str(hiddensList[0][0]) + 'x' + str(hiddensList[0][-1]) + '_SegSeed' + str(random_segment_seed) + '_NetSeed'+ str(random_network_seed) 
    savenameModelTestTrain = 'DETECTSAI' + '_LOGREG_' +variq+'_modelTrainTest_SegSeed'+str(random_segment_seed)+'_NetSeed'+str(random_network_seed)
    
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

    model.save(dirname + savename + '.h5')
    np.savez(dirname + savenameModelTestTrain + '.npz',trainModels=trainIndices,testModels=testIndices,Xtrain=Xtrain,Ytrain=Ytrain,Xtest=Xtest,Ytest=Ytest,Xmean=Xmean,Xstd=Xstd,lats=lats,lons=lons)
    print('Saving ------->' + savename)
    
    ###############################################################
    ### Assess observations
    Xobs = data_obs.reshape(data_obs.shape[0],data_obs.shape[1]*data_obs.shape[2])
    yearsObs = np.arange(data_obs.shape[0]) + obsyearstart
    
    ### Standardize obs
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
    trainingout = YpredTrain
    testingout = YpredTest
    valout = YpredVal
    
    if ensTypeExperi == 'ENS':
        classesltrain = classeslnew[trainIndices,:,:].ravel()
        classesltest = classeslnew[testIndices,:,:].ravel()
        classeslval = classeslnew[valIndices,:,:].ravel()
        
    ###########################################################################
    ###########################################################################
    ###########################################################################
    ### Try preinjection
    prearrayallens,latspre,lonspre = read_primary_dataset(variq,'preInjection',
                                                   numOfEns,lensalso,
                                                   randomalso,
                                                   ravelyearsbinary,
                                                   ravelbinary,
                                                   shuffletype,
                                                   lat_bounds,
                                                   lon_bounds)
    dataNULL = data.copy() # needed for mask function to work - lazy
    dataNULL, prearrayallens = dSS.remove_ocean(dataNULL,prearrayallens,lat_bounds,lon_bounds) 
    
    ### Pick one ensemble member from testing
    prearray = prearrayallens[testIndices,:,:,:].squeeze()
    prearrayn = prearray.reshape(prearray.shape[0],prearray.shape[1]*prearray.shape[2])
    
    ### Prediction on pre-injection data
    prearrayS = (prearrayn-Xmean)/Xstd
    prearrayS[np.isnan(prearrayS)] = 0
    Ypredprearray = model.predict(prearrayS)
 
    ##############################################################################
    ##############################################################################
    ############################################################################## 
    ### Metrics
    def truelabel(data):
        """
        Calculate argmax
        """
        maxindexdata= np.argmax(data[:,:],axis=1)    
        return maxindexdata
    
    def accuracyTotalTime(data_pred,data_true):
        """
        Compute accuracy for the entire time series
        """
        
        data_truer = data_true
        data_predr = data_pred
        accdata_pred = accuracy_score(data_truer,data_predr)
        return accdata_pred

    ##############################################################################
    ##############################################################################
    ##############################################################################        
    indextrain = truelabel(trainingout)
    acctrain = accuracyTotalTime(indextrain,classesltrain)
    indextest = truelabel(testingout)
    acctest = accuracyTotalTime(indextest,classesltest)
    indexval = truelabel(valout)
    accval = accuracyTotalTime(indexval,classeslval)
    print('\n\nAccuracy Training == ',acctrain)
    print('Accuracy Testing == ',acctest)
    print('Accuracy Validation == ',accval)
    
    ### Observations
    obsout = YpredObs
    labelsobs = np.argmax(obsout,axis=1)
    uniqueobs,countobs = np.unique(labelsobs,return_counts=True)
    
    ### Preinjection data
    preout = Ypredprearray
    labelspre = np.argmax(preout,axis=1)
    uniquepre,countpre = np.unique(labelspre,return_counts=True)
    yearspre = np.arange(2015,2034+1,1)
    
    ### Save the output 
    directoryoutput = '/Users/zlabe/Documents/Research/SolarIntervention/Data/DetectSAI_ActualModel/'
    np.savetxt(directoryoutput + 'trainingEnsIndices_' + saveData + '.txt',trainIndices)
    np.savetxt(directoryoutput + 'testingEnsIndices_' + saveData + '.txt',testIndices)
    
    np.savetxt(directoryoutput + 'trainingTrueLabels_' + saveData + '.txt',classesltrain)
    np.savetxt(directoryoutput + 'testingTrueLabels_' + saveData + '.txt',classesltest)
    
    np.savetxt(directoryoutput + 'trainingPredictedLabels_' + saveData + '.txt',indextrain)
    np.savetxt(directoryoutput + 'testingPredictedLabels_' + saveData + '.txt',indextest)
    
    np.savetxt(directoryoutput + 'trainingPredictedConfidence_' + saveData + '.txt',trainingout)
    np.savetxt(directoryoutput + 'testingPredictedConfidence_' + saveData + '.txt',testingout)
    
    ### Save predictions for observations
    np.savetxt(directoryoutput + 'obsLabels_' + saveData + '.txt',labelsobs)
    np.savetxt(directoryoutput + 'obsConfid_' + saveData + '.txt',obsout)

    ### See more more details
    print(model.layers[0].get_config())
   
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
    
    ## Variables for plotting
    lons2,lats2 = np.meshgrid(lons,lats) 
    observations = data_obs
    modeldata = data
    modeldatamean = np.nanmean(modeldata,axis=1)
    
    spatialmean_obs = UT.calc_weightedAve(observations,lats2)
    spatialmean_mod = UT.calc_weightedAve(modeldata,lats2)
    spatialmean_modmean = np.nanmean(spatialmean_mod,axis=1)
    plt.figure()
    plt.plot(spatialmean_modmean.transpose())
    
    ### Look at model weights
    weights,bias = model.get_weights()
    
    ### Spatial weights
    mapweights_sai = weights[:,0].reshape(lats.shape[0],lons.shape[0])
    mapweights_con = weights[:,1].reshape(lats.shape[0],lons.shape[0])
    
    ### Bias terms
    bias_sai = bias[0]
    bias_con = bias[1]
    
    ### Weights times input for testing
    mapsweights = np.empty((XtestS.shape))
    for i in range(mapsweights.shape[0]):
        if indextest[i] == 0:
            mapsweights[i,:] = XtestS[i,:] * weights[:,0] # for SAI predictions
        elif indextest[i] == 1:
            mapsweights[i,:] = XtestS[i,:] * weights[:,1] # for CON predictions
        else:
            print(ValueError('SOMETHING IS WRONG WITH THE CLASS LABELS!'))
            sys.exit()
            
    ### Set nans for ocean
    mapsweights[np.where(mapsweights == 0.)] = np.nan
    
    ### Save weights after doing weights*input
    np.save(directoryoutput + 'WeightsInputs-LOGREG_' + saveData + '.npy',mapsweights)
    np.save(directoryoutput + 'Latitudes-LOGREG_' + saveData + '.npy',np.asarray(lats))
    np.save(directoryoutput + 'Longitudes-LOGREG_' + saveData + '.npy',np.asarray(lons))