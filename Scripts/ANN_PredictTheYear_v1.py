"""
Train ANN to predict the year between WACCM and ARISE

Reference  : Barnes et al. [2020, JAMES]
Author    : Zachary M. Labe
Date      : 2 March 2022
"""

### Import packages
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
import innvestigate
import random
import scipy.stats as stats
from mpl_toolkits.basemap import Basemap, addcyclic, shiftgrid
import palettable.cubehelix as cm
import cmocean as cmocean
import calc_Utilities as UT
import calc_dataFunctions as df
import calc_Stats as dSS
import calc_LRP_old as LRP
import calc_LRPclass as LRP
import sys

import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
warnings.filterwarnings('ignore', category=DeprecationWarning)

### Prevent tensorflow 2.+ deprecation warnings
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)

### LRP param
DEFAULT_NUM_BWO_ITERATIONS = 200
DEFAULT_BWO_LEARNING_RATE = .01

### Plotting defaults 
plt.rc('text',usetex=True)
plt.rc('font',**{'family':'sans-serif','sans-serif':['Avant Garde']}) 

###############################################################################
###############################################################################
###############################################################################
### Data preliminaries 
directorydata = '/Users/zlabe/Data/SAI/'
datasetsingle = ['WACCM']
dataset_obs = 'ARISE'
variq = 'TREFHT'
seasons = ['annual','JFM','AMJ','JAS','OND']
seasons = ['annual']
window = 0

if datasetsingle[0] == 'WACCM':
    simuqq = 'WACCM'
    timewaccm = np.arange(2015+window,2069+1,1)
    yearsall = [timewaccm]
    directoriesall = [directorydata]
    numOfEns = 10
elif datasetsingle[0] == 'ARISE':
    simuqq = 'ARISE'
    timearise = np.arange(2035+window,2069+1,1)
    yearsall = [timearise]
    directoriesall = [directorydata]
    numOfEns = 10
    
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
lrpRule1 = 'z'
lrpRule2 = 'epsilon'
lrpRule3 = 'integratedgradient'
normLRP = True
cascade = True
###############################################################################
###############################################################################

for sis,singlesimulation in enumerate(datasetsingle):
    lrpsns = []
    for seas in range(len(seasons)):
        ###############################################################################
        ###############################################################################
        ###############################################################################
        ### ANN preliminaries
        if variq == 'TREFHT':
            directoryfigure = '/Users/zlabe/Desktop/SAI/predictTheYear/T2M/'
        else:
            directoryfigure = '/Users/zlabe/Desktop/SAI/predictTheYear/%s/' % variq
            
        monthlychoice = seasons[seas]
        reg_name = 'Globe'
        lat_bounds,lon_bounds = UT.regions(reg_name)
        experiment_result = pd.DataFrame(columns=['actual iters','hiddens','cascade',
                                                  'RMSE Train','RMSE Test',
                                                  'ridge penalty','zero mean',
                                                  'zero merid mean','land only?','ocean only?'])
        
        
        ### Define primary dataset to use
        dataset = singlesimulation
        modelType = 'TrainedOn' + dataset
        
        ### Whether to test and plot the results using obs data
        test_on_obs = True
        if dataset_obs == 'ARISE':
            year_obsall = np.arange(2035+window,2069+1,1)
            year_obs = year_obsall
            obsyearstart = year_obsall[0]
        elif dataset_obs == 'WACCM':
            year_obsall = np.arange(2035+window,2069+1,1)
            year_obs = year_obsall
            obsyearstart = year_obsall[0]
        
        ### Remove the annual mean? True to subtract it from dataset ##########
        rm_annual_mean = False ################################################
        if rm_annual_mean == True:
            if variq == 'TREFHT':
                directoryfigure = '/Users/zlabe/Desktop/SAI/predictTheYear/T2M/'
            else:
                directoryfigure = '/Users/zlabe/Desktop/SAI/predictTheYear/%s/' % variq
        
        ### Remove the meridional mean? True to subtract it from dataset ######
        rm_merid_mean = True ##################################################
        if rm_merid_mean == True:
            if variq == 'TREFHT':
                directoryfigure = '/Users/zlabe/Desktop/SAI/predictTheYear/T2M/'
            else:
                directoryfigure = '/Users/zlabe/Desktop/SAI/predictTheYear/%s/' % variq
                
        ### Feed the standard deviations? True to calculate it ################
        rm_standard_dev = False ###############################################
        if rm_standard_dev == True:
            if variq == 'TREFHT':
                directoryfigure = '/Users/zlabe/Desktop/SAI/predictTheYear/T2M/'
            else:
                directoryfigure = '/Users/zlabe/Desktop/SAI/predictTheYear/%s/' % variq
        
        ### Calculate only over land? True if land ############################
        land_only = True ######################################################
        if land_only == True:
            if variq == 'TREFHT':
                directoryfigure = '/Users/zlabe/Desktop/SAI/predictTheYear/T2M/'
            else:
                directoryfigure = '/Users/zlabe/Desktop/SAI/predictTheYear/%s/' % variq
        
        ### Calculate only over ocean? True if ocean ##########################
        ocean_only = False #####################################################
        if ocean_only == True:
            if variq == 'TREFHT':
                directoryfigure = '/Users/zlabe/Desktop/SAI/predictTheYear/T2M/'
            else:
                directoryfigure = '/Users/zlabe/Desktop/SAI/predictTheYear/%s/' % variq
        
        ### Rove the ensemble mean? True to subtract it from dataset ##########
        rm_ensemble_mean = False ##############################################
        if rm_ensemble_mean == True:
            if variq == 'TREFHT':
                directoryfigure = '/Users/zlabe/Desktop/SAI/predictTheYear/T2M/'
            else:
                directoryfigure = '/Users/zlabe/Desktop/SAI/predictTheYear/%s/' % variq
        
        ### Split the data into training and testing sets? value of 1 will use all 
        ### data as training, .8 will use 80% training, 20% testing; etc.
        segment_data_factor = .8
        
        ### Plot within the training loop - may want to set to False when testing out 
        ### larget sets of parameters
        plot_in_train = False
        
        ###############################################################################
        ###############################################################################
        ###############################################################################
        ### Read in climate models
        
        def read_primary_dataset(variq,dataset,monthlychoice,numOfEns,lensalso,randomalso,ravelyearsbinary,ravelbinary,shuffletype,timeper,lat_bounds=lat_bounds,lon_bounds=lon_bounds):
            data,lats,lons = df.readFiles(variq,dataset,monthlychoice,numOfEns,lensalso,randomalso,ravelyearsbinary,ravelbinary,shuffletype,timeper)
            datar,lats,lons = df.getRegion(data,lats,lons,lat_bounds,lon_bounds)
            print('\nOur dataset: ',dataset,' is shaped',data.shape)
            return datar,lats,lons        
        
        ###############################################################################
        ###############################################################################
        ###############################################################################
        ### Select data to test, train on
            
        def segment_data(data,fac = segment_data_factor):
          
            global random_segment_seed,trainIndices, estIndices
            if random_segment_seed == None:
                random_segment_seed = int(int(np.random.randint(1, 100000)))
            np.random.seed(random_segment_seed)
            
            if fac < 1 :
                nrows = data.shape[0]
                segment_train = int(np.round(nrows * fac))
                segment_test = nrows - segment_train
                print('Training on',segment_train,'ensembles, testing on',segment_test)
        
                ### Picking out random ensembles
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
                
                ### Random ensembles are picked
                if debug:
                    print('Training on ensembles: ',trainIndices)
                    print('Testing on ensembles: ',testIndices)
                
                ### Training segment----------
                data_train = ''
                for ensemble in trainIndices:
                    this_row = data[ensemble, :, :, :]
                    this_row = this_row.reshape(-1,data.shape[1],data.shape[2],
                                                data.shape[3])
                    if data_train == '':
                        data_train = np.empty_like(this_row)
                    data_train = np.vstack((data_train,this_row))
                data_train = data_train[1:, :, :, :]
                
                if debug:
                    print('org data - shape', data.shape)
                    print('training data - shape', data_train.shape)
            
                ### Reshape into X and T
                Xtrain = data_train.reshape((data_train.shape[0] * data_train.shape[1]),
                                            (data_train.shape[2] * data_train.shape[3]))
                Ttrain = np.tile((np.arange(data_train.shape[1]) + yearsall[sis].min()).reshape(data_train.shape[1],1),
                                  (data_train.shape[0],1))
                Xtrain_shape = (data_train.shape[0],data_train.shape[1])
                
                
                ### Testing segment----------
                data_test = ''
                for ensemble in testIndices:
                    this_row = data[ensemble, :, :, :]
                    this_row = this_row.reshape(-1,data.shape[1],data.shape[2],
                                                data.shape[3])
                    if data_test == '':
                        data_test = np.empty_like(this_row)
                    data_test = np.vstack((data_test, this_row))
                data_test = data_test[1:, :, :, :]
                
                if debug:
                    print('testing data', data_test.shape)
                  
                ### Reshape into X and T
                Xtest = data_test.reshape((data_test.shape[0] * data_test.shape[1]),
                                          (data_test.shape[2] * data_test.shape[3]))
                Ttest = np.tile((np.arange(data_test.shape[1]) + yearsall[sis].min()).reshape(data_test.shape[1],1),
                                (data_test.shape[0], 1))   
        
            else:
                trainIndices = np.arange(0,np.shape(data)[0])
                testIndices = np.arange(0,np.shape(data)[0])    
                print('Training on ensembles: ',trainIndices)
                print('Testing on ensembles: ',testIndices)
        
                data_train = data
                data_test = data
            
                Xtrain = data_train.reshape((data_train.shape[0] * data_train.shape[1]),
                                            (data_train.shape[2] * data_train.shape[3]))
                Ttrain = np.tile((np.arange(data_train.shape[1]) + yearsall[sis].min()).reshape(data_train.shape[1],1),
                                  (data_train.shape[0],1))
                Xtrain_shape = (data_train.shape[0], data_train.shape[1])
        
            Xtest = data_test.reshape((data_test.shape[0] * data_test.shape[1]),
                                      (data_test.shape[2] * data_test.shape[3]))
            Ttest = np.tile((np.arange(data_test.shape[1]) + yearsall[sis].min()).reshape(data_test.shape[1],1),
                            (data_test.shape[0],1))
        
            Xtest_shape = (data_test.shape[0], data_test.shape[1])
            data_train_shape = data_train.shape[1]
            data_test_shape = data_test.shape[1]
          
            ### 'unlock' the random seed
            np.random.seed(None)
          
            return Xtrain,Ttrain,Xtest,Ttest,Xtest_shape,Xtrain_shape,data_train_shape,data_test_shape,testIndices
        
        ###############################################################################
        ###############################################################################
        ###############################################################################
        ### Data management
            
        def shape_obs(data_obs,year_obs):
            Xtest_obs = np.reshape(data_obs,(data_obs.shape[0],
                                              (data_obs.shape[1]*data_obs.shape[2])))
            Ttest_obs = np.tile(np.arange(data_obs.shape[0])+year_obs[0])
            return Xtest_obs,Ttest_obs
        
        def consolidate_data():
            '''function to delete data and data_obs since we have already sliced other 
            variables from them.  Only run after segment_data and shape_obs!!!
            will delete global variables and clear memory'''
            global data
            global data_obs
            del data
            del data_obs
        
        ###############################################################################
        ###############################################################################
        ###############################################################################
        ### Plotting functions
        
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
        
        def plot_prediction (Ttest, test_output, Ttest_obs, obs_output):
            ### Predictions
            
            plt.figure(figsize=(16,4))
            plt.subplot(1, 2, 1)
            plt.title('Predicted vs Actual Year for Testing')
            plt.xlabel('Actual Year')
            plt.ylabel('Predicted Year')
            plt.plot(Ttest, test_output, 'o', color='black', label='GCM')
              
            if test_on_obs == True:
                plt.plot(Ttest_obs, obs_output,'o',color='deepskyblue',label='obs')
            a = min(min(Ttest), min(test_output))
            b = max(max(Ttest), max(test_output))
            plt.plot((a,b), (a,b), '-', lw=3, alpha=0.7, color='gray')
            #plt.axis('square')
            plt.xlim(a * .995, b * 1.005)
            plt.ylim(a * .995, b * 1.005)
            plt.legend()
            plt.show()
          
        def plot_training_error(nnet):
            ### Training error (nnet)
          
            plt.subplot(1, 2, 2)
            plt.plot(nnet.getErrors(), color='black')
            plt.title('Training Error per Itobstion')
            plt.xlabel('Training Itobstion')
            plt.ylabel('Training Error')
            plt.show()
         
        def plot_rmse(train_output,Ttrain,test_output,Ttest,data_train_shape,data_test_shape):
            ### rmse (train_output, Ttrain, test_output, Ttest, data_train_shape, data_test_shape)
          
            plt.figure(figsize=(16, 4))
            plt.subplot(1, 2, 1)
            rmse_by_year_train = np.sqrt(np.mean(((train_output - Ttrain)**2).reshape(Xtrain_shape),
                                                  axis=0))
            xs_train = (np.arange(data_train_shape) + yearsall[sis].min())
            rmse_by_year_test = np.sqrt(np.mean(((test_output - Ttest)**2).reshape(Xtest_shape),
                                                axis=0))
            xs_test = (np.arange(data_test_shape) + yearsall[sis].min())
            plt.title('RMSE by year')
            plt.xlabel('year')
            plt.ylabel('error')
            plt.plot(xs_train,rmse_by_year_train,label = 'training error',
                      color='gold',linewidth=1.5)
            plt.plot(xs_test,rmse_by_year_test,labe ='test error',
                      color='forestgreen',linewidth=0.7)
            plt.legend()
        
            if test_on_obs == True:
                plt.subplot(1,2,2)
                error_by_year_test_obs = obs_output - Ttest_obs
                plt.plot(Ttest_obs,error_by_year_test_obs,label='obs error',
                      color='deepskyblue',linewidth=2.)            
                plt.title('Error by year for obs')
                plt.xlabel('year')
                plt.ylabel('error')
                plt.legend()
                plt.plot((1979,2020), (0,0), color='gray', linewidth=2.)
                plt.xlim(1979,2020)
            plt.show()
            
        
        def plot_weights(nnet, lats, lons, basemap):
            # plot maps of the NN weights
            plt.figure(figsize=(16, 6))
            ploti = 0
            nUnitsFirstLayer = nnet.layers[0].nUnits
            
            for i in range(nUnitsFirstLayer):
                ploti += 1
                plt.subplot(np.ceil(nUnitsFirstLayer/3), 3, ploti)
                maxWeightMag = nnet.layers[0].W[1:, i].abs().max().item() 
                df.drawOnGlobe(((nnet.layers[0].W[1:, i]).cpu().data.numpy()).reshape(len(lats),
                                                                                      len(lons)),
                                lats,lons,basemap,vmin=-maxWeightMag,vmax=maxWeightMag,
                                cmap=cmocean.cm.balance)
                if(hiddens[0]==0):
                    plt.title('Linear Weights')
                else:
                    plt.title('First Layer, Unit {}'.format(i+1))
              
            if(cascade is True and hiddens[0]!=0):
                plt.figure(figsize=(16, 6))
                ploti += 1
                plt.subplot(np.ceil(nUnitsFirstLayer/3), 3, ploti)
                maxWeightMag = nnet.layers[-1].W[1:Xtrain.shape[1]+1, 0].abs().max().item()
                df.drawOnGlobe(((nnet.layers[-1].W[1:Xtrain.shape[1]+1, 0]).cpu().data.numpy()).reshape(len(lats),
                                                                                                        len(lons)),
                                lats,lons,basemap,vmin=-maxWeightMag,
                                vmax=maxWeightMag,cmap=cmocean.cm.balance)
                plt.title('Linear Weights')
            plt.tight_layout()
          
        def plot_results(plots = 4): 
            ### Calls all our plot functions together
            global nnet,train_output,test_output,obs_output,Ttest,Ttrain,Xtrain_shape,Xtest_shape,data_train_shape,data_test_shape,Ttest_obs,lats,lons,basemap
            
            if plots >=1:
                plot_prediction(Ttest, test_output, Ttest_obs, obs_output)
            if plots >= 2:
                plot_training_error(nnet)
                plot_rmse(train_output, Ttrain, test_output, Ttest, data_train_shape, data_test_shape)
            if plots == 4:
                plot_weights(nnet, lats, lons, basemap)
            plt.show()
           
        def plot_classifier_output(class_prob,test_class_prob,Xtest_shape,Xtrain_shape):
            prob = class_prob[-1].reshape(Xtrain_shape)
            
            plt.figure(figsize=(14, 6))
            plt.plot((np.arange(Xtest_shape[1]) + yearsall[sis].min()),
                      prob[:,:,1].T, '-',alpha = .7)
            plt.plot((np.arange(Xtest_shape[1]) + yearsall[sis].min()),
                      (np.mean(prob[:, :, 1], axis = 0).reshape(180, -1)),
                      'b-',linewidth=3.5, alpha = .5, label = 'ensemble avobsge')
            plt.title('Classifier Output by Ensemble using Training Data')
            plt.xlabel('year')
            plt.yticks((0, 1), ['Pre-Baseline', 'Post-Baseline'])
            plt.legend()
            plt.show()
        
            tprob = test_class_prob[0].reshape(Xtest_shape)
            
            plt.figure(figsize=(14, 6))
            plt.plot(((np.arange(Xtest_shape[1]) + yearsall[sis].min())),tprob[:,:,1].T,'-',
                      alpha = .7)
            plt.plot((np.arange(Xtest_shape[1]) + yearsall[sis].min()), 
                      (np.mean(tprob[:, :, 1], axis = 0).reshape(180, -1)),
                      'r-',linewidth=4,alpha = .5,label = 'ensemble avobsge')
            plt.title('Classifier Output by Ensemble using Test Data')
            plt.xlabel('year')
            plt.yticks((0, 1), ['Pre-Baseline', 'Post-Baseline'])
            plt.legend()
            plt.show()
            
        def beginFinalPlot(YpredTrain,YpredTest,Ytrain,Ytest,testIndices,years,yearsObs,YpredObs):
            """
            Plot prediction of year
            """
            
            plt.rc('text',usetex=True)
            plt.rc('font',**{'family':'sans-serif','sans-serif':['Avant Garde']}) 
                    
            fig = plt.figure()
            ax = plt.subplot(111)
            
            adjust_spines(ax, ['left', 'bottom'])
            ax.spines['top'].set_color('none')
            ax.spines['right'].set_color('none')
            ax.spines['left'].set_color('dimgrey')
            ax.spines['bottom'].set_color('dimgrey')
            ax.spines['left'].set_linewidth(2)
            ax.spines['bottom'].set_linewidth(2)
            ax.tick_params('both',length=4,width=2,which='major',color='dimgrey')
        
            train_output_rs = YpredTrain.reshape(len(trainIndices),
                                              len(years))
            test_output_rs = YpredTest.reshape(len(testIndices),
                                          len(years))
        
            xs_test = (np.arange(np.shape(test_output_rs)[1]) + yearsall[sis].min())
        
            for i in range(0,train_output_rs.shape[0]):
                if i == train_output_rs.shape[0]-1:
                    plt.plot(xs_test,train_output_rs[i,:],'o',
                                markersize=4,color='lightgray',clip_on=False,
                                alpha=0.4,markeredgecolor='k',markeredgewidth=0.4,
                                label=r'\textbf{%s - Training Data}' % singlesimulation)
                else:
                    plt.plot(xs_test,train_output_rs[i,:],'o',
                                markersize=4,color='lightgray',clip_on=False,
                                alpha=0.4,markeredgecolor='k',markeredgewidth=0.4)
            for i in range(0,test_output_rs.shape[0]):
                if i == test_output_rs.shape[0]-1:
                    plt.plot(xs_test,test_output_rs[i,:],'o',
                            markersize=4,color='crimson',clip_on=False,alpha=0.3,
                            markeredgecolor='crimson',markeredgewidth=0.4,
                            label=r'\textbf{%s - Testing Data}' % singlesimulation)
                else:
                    plt.plot(xs_test,test_output_rs[i,:],'o',
                            markersize=4,color='crimson',clip_on=False,alpha=0.3,
                            markeredgecolor='crimson',markeredgewidth=0.4)
            
            if rm_ensemble_mean == False:
                iy = np.where(yearsObs>=obsyearstart)[0]
                plt.plot(yearsObs[iy],YpredObs[iy],'x',color='deepskyblue',
                          label=r'\textbf{%s}' % dataset_obs,clip_on=False)
            
            plt.xlabel(r'\textbf{ACTUAL YEAR - %s - %s}' % (monthlychoice,reg_name),fontsize=10,color='dimgrey')
            plt.ylabel(r'\textbf{PREDICTED YEAR}',fontsize=10,color='dimgrey')
            plt.plot(np.arange(yearsall[sis].min(),yearsall[sis].max()+1,1),np.arange(yearsall[sis].min(),yearsall[sis].max()+1,1),'-',
                      color='black',linewidth=2,clip_on=False)
            
            plt.xticks(np.arange(yearsall[sis].min(),2101,5),map(str,np.arange(yearsall[sis].min(),2101,5)),size=6)
            plt.yticks(np.arange(yearsall[sis].min(),2101,5),map(str,np.arange(yearsall[sis].min(),2101,5)),size=6)
            plt.xlim([yearsall[sis].min(),yearsall[sis].max()+1])   
            plt.ylim([yearsall[sis].min(),yearsall[sis].max()+1])
            
            plt.title(r'\textbf{[ %s ] $\bf{\longrightarrow}$ RMSE Train = %s; RMSE Test = %s}' % (variq,np.round(dSS.rmse(YpredTrain[:,],
                                                                            Ytrain[:,0]),1),np.round(dSS.rmse(YpredTest[:,],
                                                                                                                  Ytest[:,0]),
                                                                                                                  decimals=1)),
                                                                                                              color='k',
                                                                                                              fontsize=15)
            
            iyears = np.where(Ytest<2035)[0]
            plt.text(yearsall[sis].max()+1,yearsall[sis].min()+1.5, r'\textbf{Test RMSE before 2035 = %s}' % (np.round(dSS.rmse(YpredTest[iyears,],
                                                                                Ytest[iyears,0]),
                                                                          decimals=1)),
                      fontsize=5,ha='right')
            
            iyears = np.where(Ytest>=2035)[0]
            plt.text(yearsall[sis].max()+1,yearsall[sis].min(), r'\textbf{Test RMSE after 2035 = %s}' % (np.round(dSS.rmse(YpredTest[iyears,],
                                                                                  Ytest[iyears,0]),
                                                                              decimals=1)),
                      fontsize=5,ha='right')
            
            plt.legend(shadow=False,fontsize=7,loc='upper left',
                          bbox_to_anchor=(-0.01,1),fancybox=True,ncol=1,frameon=False,
                          handlelength=1,handletextpad=0.5)
            savefigName = modelType+'_'+variq+'_scatterPred_'+savename 

            plt.savefig(directoryfigure+savefigName+'_%s_land%s_ocean%s_20ens.png' % (monthlychoice,land_only,ocean_only),
                        dpi=300)      
            print(np.round(np.corrcoef(yearsObs,YpredObs)[0,1],2))
            return 
        
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
            if hidden[0]==0:
                ### Model is linear
                model.add(Dense(1,input_shape=(input_shape,),
                                activation='linear',use_bias=True,
                                kernel_regularizer=regularizers.l1_l2(l1=0.00,l2=ridgePenalty),
                                bias_initializer=keras.initializers.RandomNormal(seed=random_network_seed),
                        kernel_initializer=keras.initializers.RandomNormal(seed=random_network_seed)))
                print('\nTHIS IS A LINEAR NN!\n')
            else:
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
                                    kernel_regularizer=regularizers.l1_l2(l1=0.00, l2=0.00),
                                    bias_initializer=keras.initializers.RandomNormal(seed=random_network_seed),
                        kernel_initializer=keras.initializers.RandomNormal(seed=random_network_seed)))
                    
                print('\nTHIS IS A ANN!\n')
        
            #### Initialize output layer
            model.add(Dense(output_shape,activation=None,use_bias=True,
                            kernel_regularizer=regularizers.l1_l2(l1=0.00, l2=0.0),
                            bias_initializer=keras.initializers.RandomNormal(seed=random_network_seed),
                        kernel_initializer=keras.initializers.RandomNormal(seed=random_network_seed)))
        
            ### Add softmax layer at the end
            model.add(Activation('softmax'))
            
            return model
        
        def trainNN(model, Xtrain, Ytrain, niter=500, verbose=True):
          
            global lr_here, batch_size
            lr_here = .01
            model.compile(optimizer=optimizers.SGD(lr=lr_here,
                                                    momentum=0.9,nesterov=True), 
                          loss = 'binary_crossentropy',
                          metrics=[metrics.categorical_accuracy],)
        
            ### Declare the relevant model parameters
            batch_size = 32 
        
            print('----ANN Training: learning rate = '+str(lr_here)+'; activation = '+actFun+'; batch = '+str(batch_size) + '----')    
            time_callback = TimeHistory()
            history = model.fit(Xtrain,Ytrain,batch_size=batch_size,epochs=niter,
                                shuffle=True,verbose=verbose,
                                callbacks=[time_callback],
                                validation_split=0.)
            print('******** done training ***********')
        
            return model, history
        
        def test_train_loopClass(Xtrain,Ytrain,Xtest,Ytest,iterations,ridge_penalty,hiddens,plot_in_train=True):
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
        
                        ### Standardize the data
                        Xtrain,Xtest,stdVals = dSS.standardize_data(Xtrain,Xtest)
                        Xmean,Xstd = stdVals
                        
                        ### Define the model
                        model = defineNN(hidden,
                                          input_shape=np.shape(Xtrain)[1],
                                          output_shape=np.shape(Ytrain)[1],
                                          ridgePenalty=penalty)  
                       
                        ### Train the net
                        model, history = trainNN(model,Xtrain,
                                                  Ytrain,niter=niter,verbose=1)
        
                        ### After training, use the network with training data to 
                        ### check that we don't have any errors and output RMSE
                        rmse_train = dSS.rmse(convert_fuzzyDecade_toYear(Ytrain,startYear,
                                                                      classChunk),
                                          convert_fuzzyDecade_toYear(model.predict(Xtrain),
                                                                      startYear,
                                                                      classChunk))
                        if type(Ytest) != bool:
                            rmse_test = 0.
                            rmse_test = dSS.rmse(convert_fuzzyDecade_toYear(Ytest,
                                                                        startYear,classChunk),
                                                 convert_fuzzyDecade_toYear(model.predict(Xtest),
                                                                        startYear,
                                                                        classChunk))
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
        
                        #if True to plot each iter's graphs.
                        if plot_in_train == True:
                            plt.figure(figsize = (16,6))
        
                            plt.subplot(1,2,1)
                            plt.plot(history.history['loss'],label = 'training')
                            plt.title(history.history['loss'][-1])
                            plt.xlabel('epoch')
                            plt.xlim(2,len(history.history['loss'])-1)
                            plt.legend()
        
                            plt.subplot(1,2,2)
                            
                            plt.plot(convert_fuzzyDecade_toYear(Ytrain,startYear,
                                                                classChunk),
                                      convert_fuzzyDecade_toYear(model.predict(Xtrain),
                                                                startYear,
                                                                classChunk),'o',
                                                                  color='gray')
                            plt.plot(convert_fuzzyDecade_toYear(Ytest,startYear,
                                                                classChunk),
                                      convert_fuzzyDecade_toYear(model.predict(Xtest),
                                                                startYear,
                                                                classChunk),'x', 
                                                                color='red')
                            plt.plot([startYear,yearsall[sis].max()],[startYear,yearsall[sis].max()],'--k')
                            plt.yticks(np.arange(yearsall[sis].min(),yearsall[sis].max(),10))
                            plt.xticks(np.arange(yearsall[sis].min(),yearsall[sis].max(),10))
                            
                            plt.grid(True)
                            plt.show()
        
                        #'unlock' the random seed
                        np.random.seed(None)
                        random.seed(None)
                        tf.set_random_seed(None)
          
            return experiment_result, model
        
        def convert_fuzzyDecade(data,startYear,classChunk):
            years = np.arange(startYear-classChunk*2,yearsall[sis].max()+classChunk*2)
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
            years = np.arange(startYear-classChunk*2,yearsall[sis].max()+classChunk*2)
            chunks = years[::int(classChunk)] + classChunk/2
            
            return np.sum(label*chunks,axis=1)
        
        def invert_year_output(ypred,startYear):
            if(option4):
                inverted_years = convert_fuzzyDecade_toYear(ypred,startYear,classChunk)
            else:
                inverted_years = invert_year_outputChunk(ypred,startYear)
            
            return inverted_years
        
        def invert_year_outputChunk(ypred,startYear):
            
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
        debug = True
        NNType = 'ANN'
        classChunkHalf = 5
        classChunk = 10
        option4 = True
        biasBool = False
        
        if variq == 'TREFHT':
            if reg_name == 'Arctic':
                if NNType == 'ANN':
                    hiddensList = [[20,20]]
                    ridge_penalty = [0.4]
                    if land_only == True:
                        ridge_penalty = [0.02]
                    actFun = 'relu'
                    iterations = [300]
            elif reg_name == 'wideTropics':
                if NNType == 'ANN':
                    hiddensList = [[20,20]]
                    ridge_penalty = [0.4]
                    if land_only == True:
                        ridge_penalty = [0.16]
                    actFun = 'relu'
                    iterations = [400]
                    ############################
                    if rm_standard_dev == True:
                        hiddensList = [[20,20]]
                        ridge_penalty = [0.4]
                        if land_only == True:
                            ridge_penalty = [0.1]
                        actFun = 'relu'
                        iterations = [900]
            else:
                if NNType == 'ANN':
                    hiddensList = [[20,20]]
                    ridge_penalty = [0.4]
                    if land_only == True:
                        ridge_penalty = [0.65]
                    actFun = 'relu'
                    iterations = [700]
                elif NNType == 'linear':
                    hiddensList = [[0]]
                    ridge_penalty = [0.]
                    actFun = 'linear'
                    iterations = [700]
        elif variq == 'PRECT':
            if reg_name == 'Arctic':
                if NNType == 'ANN':
                    hiddensList = [[20,20]]
                    ridge_penalty = [0.4]
                    if land_only == True:
                        ridge_penalty = [0.75]
                    actFun = 'relu'
                    iterations = [700]
            elif reg_name == 'wideTropics':
                if NNType == 'ANN':
                    hiddensList = [[20,20]]
                    ridge_penalty = [0.4]
                    if land_only == True:
                        ridge_penalty = [0.2]
                    actFun = 'relu'
                    iterations = [600]
            else:
                ### Type of network
                if NNType == 'ANN':
                    hiddensList = [[20,20]]
                    ridge_penalty = [0.4]
                    if land_only == True:
                        ridge_penalty = [0.5]
                    actFun = 'relu'
                    iterations = [500]
                elif NNType == 'linear':
                    hiddensList = [[0]]
                    ridge_penalty = [0.]
                    actFun = 'linear'
                    iterations = [700]
        
        expList = [(0)]
        expN = np.size(expList)
        random_segment = True
        foldsN = 1
        
        session_conf = tf.ConfigProto(intra_op_parallelism_threads=1,
                                      inter_op_parallelism_threads=1)
        sess = tf.Session(graph=tf.get_default_graph(), config=session_conf)
        K.set_session(sess)
        K.clear_session()
        
        for loop in ([0]):
            # get info about the region
            lat_bounds,lon_bounds = UT.regions(reg_name)
            data_all,lats,lons = read_primary_dataset(variq,dataset,monthlychoice,numOfEns,
                                                                           lensalso,randomalso,ravelyearsbinary,
                                                                           ravelbinary,shuffletype,timeper,
                                                                           lat_bounds,lon_bounds)
            data_obs_all,lats_obs,lons_obs = read_primary_dataset(variq,dataset_obs,monthlychoice,numOfEns,
                                                                           lensalso,randomalso,ravelyearsbinary,
                                                                           ravelbinary,shuffletype,timeper,
                                                                           lat_bounds,lon_bounds)
            ### Pick number of ensemble members for ARISE
            ensARISE = -1
            data_obs_all = data_obs_all[ensARISE,:,:,:]
            if dataset_obs == 'WACCM':
                data_obs_all = data_obs_all[20:,:,:]
            
            for exp in expList:  
                # get the data together
                data, data_obs, = data_all, data_obs_all,
                if rm_annual_mean == True:        
                    data, data_obs = dSS.remove_annual_mean(data,data_obs,lats,lons,lats_obs,lons_obs)
                    print('*Removed annual mean*')
                if rm_merid_mean == True:
                    data, data_obs = dSS.remove_merid_mean(data,data_obs,lats,lons,lats_obs,lons_obs)
                    print('*Removed meridian mean*')  
                if rm_ensemble_mean == True:
                    data = dSS.remove_ensemble_mean(data,ravel_modelens,ravelmodeltime,rm_standard_dev,numOfEns)
                    print('*Removed ensemble mean*')
                if rm_standard_dev == True:
                    data = dSS.rm_standard_dev(data,window,ravelmodeltime,numOfEns)
                    print('*Removed standard deviation*')
                if land_only == True:
                    data, data_obs = dSS.remove_ocean(data,data_obs,lat_bounds,lon_bounds) 
                    print('*Removed ocean*')
                if ocean_only == True:
                    data, data_obs = dSS.remove_land(data,data_obs,lat_bounds,lon_bounds) 
                    print('*Removed land*')
                    
                for ih in np.arange(0,len(hiddensList)):
                    hiddens = [hiddensList[ih]]
                    if hiddens[0][0]==0:
                        annType = 'linear'
                    elif hiddens[0][0]==1 and len(hiddens)==1:
                        annType = 'layers1'
                    else:
                        annType = 'layers10x10'
    
                 ### Loop over folds
                for loop in np.arange(0,foldsN): 
    
                    K.clear_session()
                    #---------------------------
                    random_segment_seed = int(np.genfromtxt('/Users/zlabe/Documents/Research/SolarIntervention/Data/SelectedSegmentSeed.txt',unpack=True))
                    #---------------------------
                    Xtrain,Ytrain,Xtest,Ytest,Xtest_shape,Xtrain_shape,data_train_shape,data_test_shape,testIndices = segment_data(data,segment_data_factor)
    
                    ### Convert year into decadal class
                    startYear = Ytrain[0] 
                    YtrainClassMulti, decadeChunks = convert_fuzzyDecade(Ytrain,
                                                                          startYear,
                                                                          classChunk)  
                    YtestClassMulti, __ = convert_fuzzyDecade(Ytest,
                                                              startYear,
                                                              classChunk)  
    
                    ### For use later
                    XtrainS,XtestS,stdVals = dSS.standardize_data(Xtrain,Xtest)
                    Xmean, Xstd = stdVals      
    
                    #---------------------------
                    random_network_seed = 87750
                    #---------------------------
    
                    # Create and train network
                    exp_result,model = test_train_loopClass(Xtrain,
                                                            YtrainClassMulti,
                                                            Xtest,
                                                            YtestClassMulti,
                                                            iterations=iterations,
                                                            ridge_penalty=ridge_penalty,
                                                            hiddens=hiddensList,
                                                            plot_in_train = True)
                    model.summary()  
                    
                    ################################################################################################################################################                
                    # save the model
                    dirname = '/Users/zlabe/Documents/Research/SolarIntervention/savedModels/'
                    savename = modelType+'_'+variq+'_' + reg_name + '_' + monthlychoice + '_' + str(classChunk)+'yrChunks' + '_L2'+ str(ridge_penalty[0])+ '_LR' + str(lr_here)+ '_Batch'+ str(batch_size)+ '_Iters' + str(iterations[0]) + '_' + NNType + str(hiddensList[0][0]) + 'x' + str(hiddensList[0][-1]) + '_SegSeed' + str(random_segment_seed) + '_NetSeed'+ str(random_network_seed) 
                    savenameModelTestTrain = modelType+'_'+variq+'_modelTrainTest_SegSeed'+str(random_segment_seed)+'_NetSeed'+str(random_network_seed)
                    
                    if rm_annual_mean == True:
                        savename = savename + '_AnnualMeanRemoved' 
                        savenameModelTestTrain = savenameModelTestTrain + '_AnnualMeanRemoved'
                    if rm_merid_mean == True:
                        savename = savename + '_MeridionalMeanRemoved' 
                        savenameModelTestTrain = savenameModelTestTrain + '_MeridionalMeanRemoved'
                    if rm_ensemble_mean == True:
                        savename = savename + '_EnsembleMeanRemoved' 
                        savenameModelTestTrain = savenameModelTestTrain + '_EnsembleMeanRemoved'
                    if rm_standard_dev == True:
                        savename = savename + '_STDCALC'
                        savenameModelTestTrain = savenameModelTestTrain + '_STDCALC'
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
    
                    annType = 'class'
                    if monthlychoice == 'DJF':
                        startYear = yearsall[sis].min()+1
                        endYear = yearsall[sis].max()
                    else:
                        startYear = yearsall[sis].min()
                        endYear = yearsall[sis].max()
                    years = np.arange(startYear,endYear+1,1)    
                    
                    ### Standardize testing 'obs'
                    XobsS = (Xobs-Xmean)/Xstd
                    XobsS[np.isnan(XobsS)] = 0
                    
                    ### Standardize training
                    XtrainS = (Xtrain-Xmean)/Xstd
                    XtrainS[np.isnan(XtrainS)] = 0
                    
                    ### Standardize testing
                    XtestS = (Xtest-Xmean)/Xstd
                    XtestS[np.isnan(XtestS)] = 0                                                                                          
                    
                    if(annType=='class'):
                        ### Chunk by individual year
                        YpredObs = convert_fuzzyDecade_toYear(model.predict(XobsS),
                                                              startYear,
                                                              classChunk)
                        YpredTrain = convert_fuzzyDecade_toYear(model.predict(XtrainS),
                                                                startYear,
                                                                classChunk)
                        YpredTest = convert_fuzzyDecade_toYear(model.predict(XtestS),
                                                                startYear,
                                                                classChunk)
                        
                    ### Create final plot
                    beginFinalPlot(YpredTrain,YpredTest,Ytrain,Ytest,
                                    testIndices,years,
                                    yearsObs,YpredObs)
        
        ##############################################################################
        ##############################################################################
        ##############################################################################
        ### Visualizing through LRP
        numLats = lats.shape[0]
        numLons = lons.shape[0]  
        numDim = 3
        num_of_class = len(yearsall)
        infoSave = savename

        
        lrpallz = LRP.calc_LRPModel(model,np.append(XtrainS,XtestS,axis=0),
                                                np.append(Ytrain,Ytest,axis=0),
                                                biasBool,annType,num_of_class,
                                                yearsall,lrpRule1,normLRP,
                                                numLats,numLons,numDim,
                                                classChunk,startYear)
        meanlrpz = np.nanmean(lrpallz,axis=0)
        fig=plt.figure()
        plt.contourf(meanlrpz,np.arange(0,0.21,0.01),cmap=cmocean.cm.thermal)
        
        # ### For training data only
        # lrptrainz = LRP.calc_LRPModel(model,XtrainS,Ytrain,biasBool,
        #                                         annType,num_of_class,
        #                                         yearsall,lrpRule1,normLRP,
        #                                         numLats,numLons,numDim,
        #                                         classChunk,startYear)
        
        ### For training data only
        lrptestz = LRP.calc_LRPModel(model,XtestS,Ytest,biasBool,
                                                annType,num_of_class,
                                                yearsall,lrpRule1,normLRP,
                                                numLats,numLons,numDim,
                                                classChunk,startYear)
        
        
        ### For observations data only
        lrpobservationsz = LRP.calc_LRPObs(model,XobsS,biasBool,annType,
                                            num_of_class,yearsall,lrpRule1,
                                            normLRP,numLats,numLons,numDim)
  
############################################################################### 
############################################################################### 
############################################################################### 
### LRP epsilon rule
        lrpalle = LRP.calc_LRPModel(model,np.append(XtrainS,XtestS,axis=0),
                                                np.append(Ytrain,Ytest,axis=0),
                                                biasBool,annType,num_of_class,
                                                yearsall,lrpRule2,normLRP,
                                                numLats,numLons,numDim,
                                                classChunk,startYear)
        meanlrpe = np.nanmean(lrpalle,axis=0)
        fig=plt.figure()
        plt.contourf(meanlrpe,np.arange(0,0.21,0.01),cmap=cmocean.cm.thermal)
        
        # ### For training data only
        # lrptraine = LRP.calc_LRPModel(model,XtrainS,Ytrain,biasBool,
        #                                         annType,num_of_class,
        #                                         yearsall,lrpRule2,normLRP,
        #                                         numLats,numLons,numDim,
        #                                         classChunk,startYear)
        
        ### For training data only
        lrpteste = LRP.calc_LRPModel(model,XtestS,Ytest,biasBool,
                                                annType,num_of_class,
                                                yearsall,lrpRule2,normLRP,
                                                numLats,numLons,numDim,
                                                classChunk,startYear)
        
        
        ### For observations data only
        lrpobservationse = LRP.calc_LRPObs(model,XobsS,biasBool,annType,
                                            num_of_class,yearsall,lrpRule2,
                                            normLRP,numLats,numLons,numDim)
        
############################################################################### 
############################################################################### 
############################################################################### 
###  Integrated Gradient
        lrpallig = LRP.calc_LRPModel(model,np.append(XtrainS,XtestS,axis=0),
                                                np.append(Ytrain,Ytest,axis=0),
                                                biasBool,annType,num_of_class,
                                                yearsall,lrpRule3,normLRP,
                                                numLats,numLons,numDim,
                                                classChunk,startYear)
        meanlrpig = np.nanmean(lrpallig,axis=0)
        fig=plt.figure()
        plt.contourf(meanlrpig,np.arange(0,0.21,0.01),cmap=cmocean.cm.thermal)
        
        # ### For training data only
        # lrptrainig = LRP.calc_LRPModel(model,XtrainS,Ytrain,biasBool,
        #                                         annType,num_of_class,
        #                                         yearsall,lrpRule3,normLRP,
        #                                         numLats,numLons,numDim,
        #                                         classChunk,startYear)
        
        ### For training data only
        lrptestig = LRP.calc_LRPModel(model,XtestS,Ytest,biasBool,
                                                annType,num_of_class,
                                                yearsall,lrpRule3,normLRP,
                                                numLats,numLons,numDim,
                                                classChunk,startYear)
        
        
        ### For observations data only
        lrpobservationsig = LRP.calc_LRPObs(model,XobsS,biasBool,annType,
                                            num_of_class,yearsall,lrpRule3,
                                            normLRP,numLats,numLons,numDim)
        
        ##############################################################################
        ##############################################################################
        ##############################################################################
        def netcdfLRPz(lats,lons,var,directory,typemodel,trainingdata,variq,infoSave):
            print('\n>>> Using netcdfLRP-z function!')
            
            from netCDF4 import Dataset
            import numpy as np
            
            name = 'LRPMap_Z_' + typemodel + '_' + variq + '_' + infoSave + '.nc'
            filename = directory + name
            ncfile = Dataset(filename,'w',format='NETCDF4')
            ncfile.description = 'LRP maps for using selected seed' 
            
            ### Dimensions
            ncfile.createDimension('years',var.shape[0])
            ncfile.createDimension('lat',var.shape[1])
            ncfile.createDimension('lon',var.shape[2])
            
            ### Variables
            years = ncfile.createVariable('years','f4',('years'))
            latitude = ncfile.createVariable('lat','f4',('lat'))
            longitude = ncfile.createVariable('lon','f4',('lon'))
            varns = ncfile.createVariable('LRP','f4',('years','lat','lon'))
            
            ### Units
            varns.units = 'unitless relevance'
            ncfile.title = 'LRP relevance'
            ncfile.instituion = 'Colorado State University'
            ncfile.references = 'Barnes et al. [2020]'
            
            ### Data
            years[:] = np.arange(var.shape[0])
            latitude[:] = lats
            longitude[:] = lons
            varns[:] = var
            
            ncfile.close()
            print('*Completed: Created netCDF4 File!')
        
        directoryoutput = '/Users/zlabe/Documents/Research/SolarIntervention/Data/'
        # netcdfLRPz(lats,lons,lrpallz,directoryoutput,'AllData',dataset,variq,infoSave)
        # netcdfLRPz(lats,lons,lrptrainz,directoryoutput,'Training',dataset,variq,infoSave)
        netcdfLRPz(lats,lons,lrptestz,directoryoutput,'Testing',dataset,variq,infoSave)
        netcdfLRPz(lats,lons,lrpobservationsz,directoryoutput,'Obs',dataset,variq,infoSave)
        
        ##############################################################################
        ##############################################################################
        ##############################################################################
        def netcdfLRPe(lats,lons,var,directory,typemodel,trainingdata,variq,infoSave):
            print('\n>>> Using netcdfLRP-e function!')
            
            from netCDF4 import Dataset
            import numpy as np
            
            name = 'LRPMap_E_' + typemodel + '_' + variq + '_' + infoSave + '.nc'
            filename = directory + name
            ncfile = Dataset(filename,'w',format='NETCDF4')
            ncfile.description = 'LRP maps for using selected seed' 
            
            ### Dimensions
            ncfile.createDimension('years',var.shape[0])
            ncfile.createDimension('lat',var.shape[1])
            ncfile.createDimension('lon',var.shape[2])
            
            ### Variables
            years = ncfile.createVariable('years','f4',('years'))
            latitude = ncfile.createVariable('lat','f4',('lat'))
            longitude = ncfile.createVariable('lon','f4',('lon'))
            varns = ncfile.createVariable('LRP','f4',('years','lat','lon'))
            
            ### Units
            varns.units = 'unitless relevance'
            ncfile.title = 'LRP relevance'
            ncfile.instituion = 'Colorado State University'
            ncfile.references = 'Barnes et al. [2020]'
            
            ### Data
            years[:] = np.arange(var.shape[0])
            latitude[:] = lats
            longitude[:] = lons
            varns[:] = var
            
            ncfile.close()
            print('*Completed: Created netCDF4 File!')
        
        directoryoutput = '/Users/zlabe/Documents/Research/SolarIntervention/Data/'
        # netcdfLRPe(lats,lons,lrpalle,directoryoutput,'AllData',dataset,variq,infoSave)
        # netcdfLRPe(lats,lons,lrptraine,directoryoutput,'Training',dataset,variq,infoSave)
        netcdfLRPe(lats,lons,lrpteste,directoryoutput,'Testing',dataset,variq,infoSave)
        netcdfLRPe(lats,lons,lrpobservationse,directoryoutput,'Obs',dataset,variq,infoSave)
        
        ##############################################################################
        ##############################################################################
        ##############################################################################
        def netcdfLRPig(lats,lons,var,directory,typemodel,trainingdata,variq,infoSave):
            print('\n>>> Using netcdfLRP-e function!')
            
            from netCDF4 import Dataset
            import numpy as np
            
            name = 'LRPMap_IG_' + typemodel + '_' + variq + '_' + infoSave + '.nc'
            filename = directory + name
            ncfile = Dataset(filename,'w',format='NETCDF4')
            ncfile.description = 'LRP maps for using selected seed' 
            
            ### Dimensions
            ncfile.createDimension('years',var.shape[0])
            ncfile.createDimension('lat',var.shape[1])
            ncfile.createDimension('lon',var.shape[2])
            
            ### Variables
            years = ncfile.createVariable('years','f4',('years'))
            latitude = ncfile.createVariable('lat','f4',('lat'))
            longitude = ncfile.createVariable('lon','f4',('lon'))
            varns = ncfile.createVariable('LRP','f4',('years','lat','lon'))
            
            ### Units
            varns.units = 'unitless relevance'
            ncfile.title = 'LRP relevance'
            ncfile.instituion = 'Colorado State University'
            ncfile.references = 'Barnes et al. [2020]'
            
            ### Data
            years[:] = np.arange(var.shape[0])
            latitude[:] = lats
            longitude[:] = lons
            varns[:] = var
            
            ncfile.close()
            print('*Completed: Created netCDF4 File!')
        
        directoryoutput = '/Users/zlabe/Documents/Research/SolarIntervention/Data/'
        # netcdfLRPe(lats,lons,lrpalle,directoryoutput,'AllData',dataset,variq,infoSave)
        # netcdfLRPe(lats,lons,lrptraine,directoryoutput,'Training',dataset,variq,infoSave)
        netcdfLRPig(lats,lons,lrptestig,directoryoutput,'Testing',dataset,variq,infoSave)
        netcdfLRPig(lats,lons,lrpobservationsig,directoryoutput,'Obs',dataset,variq,infoSave)
 
############################################################################### 
############################################################################### 
###############################################################################        
        ### Define variable for analysis
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
        modeldatamean = np.nanmean(modeldata,axis=0)
        
        spatialmean_obs = UT.calc_weightedAve(observations,lats2)
        spatialmean_mod = UT.calc_weightedAve(modeldata,lats2)
        spatialmean_modmean = np.nanmean(spatialmean_mod,axis=0)
        plt.figure()
        plt.plot(spatialmean_modmean.transpose())
        plt.plot(spatialmean_obs,color='k',linewidth=2)