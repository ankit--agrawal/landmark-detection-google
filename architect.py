#!usr/bin/env python3
import os
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID" 
os.environ["CUDA_VISIBLE_DEVICES"] = "2"
# Importing the Keras libraries and packages
import numpy as np
import pandas as pd
import h5py
from keras import regularizers
from keras.models import Sequential, load_model, Model
from keras import optimizers, applications
from keras.layers import Convolution2D, MaxPooling2D, Flatten, Dense
from keras.layers import Dropout, BatchNormalization, Input
from keras.callbacks import ModelCheckpoint, EarlyStopping
from keras_preprocessing.image import ImageDataGenerator
from PIL import Image
from keras.utils import to_categorical

def bin_to_dec(x):
    binary = ''.join(x)
    return int(binary,2)

class cnn_architecture():
    def __init__(self, learn_rate, mode='binary', output_neurons=1):
        self.h, self.w = 224, 224 #image height, width
        self.mode = mode
        self.batch, self.epoc = 1, 1
        self.lr = learn_rate
        self.last = output_neurons

    def create_model(self):
        # SET ALL THE PARAMETERS
        # LOAD VGG16
        input_tensor = Input(shape=(self.h,self.w,3))
        model = applications.VGG16(weights='imagenet', 
                                   include_top=False,
                                   input_tensor=input_tensor)
       
        # CREATE AN "REAL" MODEL FROM VGG16 BY COPYING ALL THE LAYERS OF VGG16
        new_model = Sequential()
        for l in model.layers:
            new_model.add(l)
        
        # CREATE A TOP MODEL
        top_model = Sequential()
        top_model.add(Flatten(input_shape=model.output_shape[1:]))
        top_model.add(Dropout(0.5))
        top_model.add(Dense(1000, activation='relu'))
        top_model.add(Dropout(0.3))
        top_model.add(Dense(18, activation='relu'))
        top_model.add(Dense(self.last, activation='sigmoid'))
        
        # CONCATENATE THE TWO MODELS
        new_model.add(top_model)
        
        # LOCK THE TOP CONV LAYERS        
        for layer in new_model.layers[:15]:
            layer.trainable = False
        
        return new_model

    def image_gen(self, train, test=None, train_dir=None, test_dir=None,mode='binary'):
        #rescale, split in train/ validation set, create batches
        # Part 2 - Fitting the CNN to the images

        labels = train.columns.values
        print('number of labels/ columns for y_col: ',len(labels))
        print('begin data pre-processing')
        train_datagen = ImageDataGenerator(rescale = 1./255., rotation_range=30,
                                           width_shift_range=0.2,
                                           height_shift_range=0.2,
                                           zoom_range=0.5,
                                           validation_split = 0.2)
        test_datagen = ImageDataGenerator(rescale = 1./255.)

        train_gen = train_datagen.flow_from_dataframe(dataframe = train,
                                                 directory = train_dir,
                                                 x_col = 'path',
                                                 y_col = labels[3:],
                                                 subset = "training",
                                                 class_mode = self.mode,
                                                 target_size = (self.h,self.w),
                                                 batch_size = self.batch,
                                                 shuffle = True)

        val_gen = train_datagen.flow_from_dataframe(dataframe = train,
                                                 directory = train_dir,
                                                 x_col = 'path',
                                                 y_col = labels[3:],
                                                 subset = "validation",
                                                 class_mode = self.mode,
                                                 target_size = (self.h,self.w),
                                                 batch_size = self.batch,
                                                 shuffle = True)

        print('ended data pre-processing')
        #return train_gen, val_gen, test_gen
        return train_gen, val_gen

    def run(self, train_df,num, l='categorical_crossentropy'):
        #stage = 1 or 2
        val_accuracy = 100000
        try:
            classifier = load_model('final_submission.h5')
            print('---loading old model')
        except:
            classifier = self.create_model()
            print('--new model detected')
            print(classifier.summary())

        opt = optimizers.SGD(lr=self.lr)
        classifier.compile(loss = l, metrics =['accuracy'], optimizer=opt)
        

        epochs = 5; limit = 700;
        iterations = (len(train_df)//limit)+1 

        for i in range(epochs):
            for j in range(2):
                train_df = train_df.sample(frac=1).reset_index(drop=True) #if not reset_index, has issues with concatenation below
        
            train_df = train_df.drop_duplicates('landmark_id').reset_index(drop=True)
            print('----length of new dataframe for 1 example per label=',len(train_df))

            for m,k in enumerate(range(0, len(train_df), limit)):
            
                try:
                    new_train_df = train_df.iloc[k:k+limit,:].reset_index(drop=True)
                except:
                    new_train_df = train_df.iloc[k:,:].reset_index(drop=True)
            

                print('Range of input {0} to {1}'.format(k, k+limit))
                #the following section creates new columns for one hot encoding
                encode = to_categorical(new_train_df['landmark_id'].tolist(),num).tolist()
                onehot_df = pd.DataFrame(encode)
                #print(encode[0].index(1), onehot_df[encode[0].index(1)].iloc[0])
                #print(onehot_df.shape)
           
                #merge the columns together into 1 dataframe
                f_train_df = pd.concat([new_train_df, onehot_df],axis=1, sort=False)
                #print(new_train_df.columns.values) 
                #print(f_train_df.shape)
                #print(f_train_df.isnull().sum())
                print('------Running epoch {0} iteration number {1} out of {2} iterations'.format(i+1,m+1,iterations))
                #train_set, val_set, test_set = self.image_gen(train, test)
                train_set, val_set = self.image_gen(f_train_df)
                 
                model_checkpoint = ModelCheckpoint('final_submission.h5',monitor = 'val_acc',verbose=1,
                                           save_best_only= True,mode='max')

                #se ting step size
                TRAIN_STEPS_SIZE = train_set.n//train_set.batch_size
                VAL_STEPS_SIZE = val_set.n//val_set.batch_size
        
                classifier.fit_generator(generator = train_set,
                         steps_per_epoch = TRAIN_STEPS_SIZE,
                         epochs =self.epoc, callbacks=[model_checkpoint],
                         validation_data = val_set,
                         validation_steps = VAL_STEPS_SIZE)
                
                #hist = classifier.history()
                #print(hist)
                '''
                if hist['val_acc'] < val_accuracy:
                    model_checkpoint = ModelCheckpoint('final_submission.h5',monitor = 'val_acc',verbose=1,
                                           save_best_only= True,mode='max')
                '''
                classifier.evaluate_generator(generator=val_set, steps=VAL_STEPS_SIZE)

if __name__ == '__main__':

    train_df = pd.read_csv('remaining_binary.csv') #train set with binary labels
    train_df = train_df[['id','path','landmark_id']]

    num = train_df['landmark_id'].nunique()
    arch_2 = cnn_architecture(1e-3,mode='other', output_neurons=num) #using the same object for every mini-batch ensures that the same model trains
    arch_2.run(train_df, num)
