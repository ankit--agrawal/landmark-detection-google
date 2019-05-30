import os
import pandas as pd
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID" 
os.environ["CUDA_VISIBLE_DEVICES"] = "1"
from architect import cnn_architecture
import tensorflow as tf
import numpy as np
from tensorflow.keras.utils import to_categorical


def cate(x,num):
    encode = to_categorical(y=x-1,num_classes=num,dtype='int32')
    return encode.tolist()

    for i in range(len(train_df)):
        m,n = train_df['binary_vals'][i].shape
        if m != 18: print(i, m);
    for i in range(len(train_df)):
        m,n = train_df['binary_vals'][i].shape
        if m != 18: print(i, m);

if __name__=='__main__':
    
    #----stage 1: Classify whether an image is landmark or not
    pos_df = pd.read_csv('pos_examples.csv') #landmark image sample
    neg_df = pd.read_csv('neg_examples.csv') #non-landmark image sample
    test_df = pd.read_csv('test_examples.csv') #real test dataset
    
    pos_df = pos_df[['path','landmark_id','classifier_label','id']]
    pos_df = pos_df.sample(n=8000)
    neg_df = neg_df.sample(n=9000)
    
    final_df = pd.concat([pos_df,neg_df],sort=False, ignore_index=True)
    #concat puts all label=0 and then label=1 so training can be skewed
    for i in range(150):
        final_df = final_df.sample(frac=1)

    arch_1 = cnn_architecture(1e-2)
    arch_1.run(final_df, test_df)
