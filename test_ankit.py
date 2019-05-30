#!usr/bin/env python3
import os
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID" 
os.environ["CUDA_VISIBLE_DEVICES"] = "1"
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
#from architect import cnn_architecture


def new_run(test_df):
    try:
        classifier = load_model('final_submission.h5')
        print('---successfully loaded saved model')
    except:
        print('---Couldn\'t find saved model. Debug and try again')

    test_datagen = ImageDataGenerator(rescale = 1./255.)

    test_gen = test_datagen.flow_from_dataframe(dataframe = test_df,
            directory = None, x_col = 'path', class_mode = None,
            target_size = (224,224), batch_size = 700, shuffle=False)

    test_gen.reset()
    Test_STEPS = test_gen.n//test_gen.batch_size

    pred = classifier.predict_generator(test_gen, verbose=1, steps=Test_STEPS)
    predicted_class_indices = np.argmax(pred, axis=1)
    predicted_class_confidence = np.amax(pred, axis=1)
    
    print(predicted_class_indices[0], predicted_class_confidence[0])
    output_df = test_df
    output_df['class'] = predicted_class_indices
    output_df['confidence'] = predicted_class_confidence

    output_df.to_csv('submit.csv', index = False, header=False)




if __name__ == '__main__':
    #arch = cnn_architecture(0.5)
    test = pd.read_csv('output.csv')
    t = 0.97
    test['final_predictions'] = np.where(test['predictions'] >=t, 1, 0)
    test_df = test.loc[test['final_predictions']==1] #images predicted as landmarks in stage 1
    print(test_df['final_predictions'].value_counts())
    print(len(test)-len(test_df))
    print(test_df.columns.values)

    new_run(test_df)
