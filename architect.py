#!usr/bin/env python3

# Importing the Keras libraries and packages
import numpy as np
import pandas as pd
from keras import regularizers
from keras.models import Sequential, load_model
from keras import optimizers, applications
from keras.layers import Convolution2D, MaxPooling2D, Flatten, Dense
from keras.layers import Dropout, BatchNormalization, Input
from keras.callbacks import ModelCheckpoint, EarlyStopping
from keras_preprocessing.image import ImageDataGenerator
from PIL import Image

class cnn_architecture():
    def __init__(self, learn_rate, mode='binary', output_neurons=1):
        self.h, self.w = 224, 224 #image height, width
        self.mode = mode
        self.batch, self.epoc = 100, 250
        self.lr = learn_rate
        self.last = output_neurons

    def create_model(self):
        # SET ALL THE PARAMETERS
        #weights_path = 'models/vgg16.h5'
        top_model_weights_path = 'bottleneck_fc_model.h5'
        
        # LOAD VGG16
        input_tensor = Input(shape=(self.h,self.w,3))
        model = applications.VGG16(weights='imagenet', 
                                   include_top=False,
                                   input_tensor=input_tensor)
       
        
        # CREATE AN "REAL" MODEL FROM VGG16
        # BY COPYING ALL THE LAYERS OF VGG16
        new_model = Sequential()
        for l in model.layers:
            new_model.add(l)
        
        # CREATE A TOP MODEL
        top_model = Sequential()
        top_model.add(Flatten(input_shape=model.output_shape[1:]))
        top_model.add(Dense(1000, activation='relu'))
        top_model.add(Dropout(0.5))
        top_model.add(Dense(150, activation='relu'))
        top_model.add(Dense(self.last, activation='sigmoid'))
        
        try:
            top_model.load_weights(top_model_weights_path)
            print('loading old weights')
        except:
            print('saving weights for the \' top_model\' ')
            top_model.save_weights(top_model_weights_path)
        
        # CONCATENATE THE TWO MODELS
        new_model.add(top_model)
        
        # LOCK THE TOP CONV LAYERS
        for layer in new_model.layers[:15]:
            layer.trainable = False
    
        return new_model

    def image_gen(self, train, test, train_dir, test_dir, mode='binary'):
        #rescale, split in train/ validation set, create batches
        # Part 2 - Fitting the CNN to the images

        train_datagen = ImageDataGenerator(rescale = 1./255., rotation_range=30,
                                           width_shift_range=0.2,
                                           height_shift_range=0.2,
                                           zoom_range=0.5,
                                           validation_split = 0.2)
        test_datagen = ImageDataGenerator(rescale = 1./255)

        train_gen = train_datagen.flow_from_dataframe(dataframe = train,
                                                 directory = train_dir,
                                                 x_col = 'path',
                                                 y_col = 'classifier_label',
                                                 subset = "training",
                                                 class_mode = self.mode,
                                                 target_size = (self.h,self.w),
                                                 batch_size = self.batch,
                                                 shuffle = True)

        val_gen = train_datagen.flow_from_dataframe(dataframe = train,
                                                 directory = train_dir,
                                                 x_col = 'path',
                                                 y_col = 'classifier_label',
                                                 subset = "validation",
                                                 class_mode = self.mode,
                                                 target_size = (self.h,self.w),
                                                 batch_size = self.batch,
                                                 shuffle = True)

        test_gen = test_datagen.flow_from_dataframe(dataframe = test,
                                           directory = test_dir,
                                           x_col = 'path',
                                           target_size = (self.h, self.w),
                                           batch_size = 1,
                                           class_mode = None, shuffle = False)

        return train_gen, val_gen, test_gen

    def run(self, train, test, train_dir=None, test_dir=None, l='binary_crossentropy'):
        
        try:
            classifier = keras.models.load_model('classify.h5')
            print('---loading old model')
        except:
            classifier = self.create_model()
            print('--new model detected')

        #callbacks
        #early_stop = EarlyStopping(monitor='val_acc', patience=4, verbose=1)
        model_checkpoint = ModelCheckpoint('classify.h5',monitor = 'val_acc',verbose=1,
                                           save_best_only= True,mode='max',
                                           save_weights_only=False)
        
        opt = optimizers.SGD(lr=self.lr)
        classifier.compile(loss = l, metrics =['accuracy'], optimizer=opt)
        print(classifier.summary())

        #load data generators
        train_set, val_set, test_set = self.image_gen(train, test, train_dir,
                                                 test_dir)
        
        #setting step size
        TRAIN_STEPS_SIZE = train_set.n//train_set.batch_size
        VAL_STEPS_SIZE = val_set.n//val_set.batch_size
        TEST_STEPS_SIZE = test_set.n//test_set.batch_size
        
        classifier.fit_generator(generator = train_set,
                         steps_per_epoch = TRAIN_STEPS_SIZE,
                         epochs =self.epoc, callbacks=[model_checkpoint],
                         validation_data = val_set,
                         validation_steps = VAL_STEPS_SIZE)

        classifier.evaluate_generator(generator=val_set, steps=VAL_STEPS_SIZE)

        test_set.reset()
        loaded_model = create_model()
        loaded_model.load_weights('classify.h5')

        pred=classifier.predict_generator(test_set,verbose=1, 
                                          steps=TEST_STEPS_SIZE)
        loaded_model.load_weights('speed_car_cifar.h5')
        
        #predicted_class_indices=np.argmax(pred,axis=1)
        out = test
        out['predictions'] = pred
        #print(pred)
        #writing to a file
        out.to_csv('output.csv')
