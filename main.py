import os
import pandas as pd
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID" 
os.environ["CUDA_VISIBLE_DEVICES"] = "2"
from architect import cnn_architecture
import tensorflow as tf
import numpy as np
from tensorflow.keras.utils import to_categorical
from sklearn.preprocessing import MultiLabelBinarizer, OneHotEncoder


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
    '''
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
    '''
    #----stage 2: TAke the landmark images from stage 1 and predict the landmark category
    
    train_df = pd.read_csv('remaining_binary.csv') #train set with binary labels
    train_df = train_df[['id','path','landmark_id']]
    
    num = train_df['landmark_id'].nunique()
    
    test = pd.read_csv('output.csv')
    
    t = 0.98
    test['final_predictions'] = np.where(test['predictions']>=t, 1, 0)
    test_df = test.loc[test['final_predictions']==1] #images predicted as landmarks in stage 1
    #print(test_df['final_predictions'].value_counts())
    print(len(test)-len(test_df))
    #print(test_df.columns.values)

    arch_2 = cnn_architecture(1e-3,mode='other', output_neurons=num) #using the same object for every mini-batch ensures that the same model trains
    epochs = 2; limit = 50;

    for i in range(epochs):
        for j in range(2):
            train_df = train_df.sample(frac=1).reset_index(drop=True)
        for k in range(0, len(train_df), limit):
            try:
                new_train_df = train_df.iloc[k:k+limit,:]
            except:
                new_train_df = train_df.iloc[k:,:]
            #print(type(new_train_df['landmark_id'])
            new_train_df['path'].astype(str, copy=False)
            #the following section creates new columns for one hot encoding
            s = []
            '''
            for j in range(len(new_train_df)):
                encode = to_categorical(new_train_df['landmark_id'].iloc[j], num)
                s.append(encode.tolist())
            '''
            encode = to_categorical(new_train_df['landmark_id'].tolist(),num).tolist()
            onehot_df = pd.DataFrame(encode)
            print(onehot_df.shape)
           
            #merge the columns together into 1 dataframe
            #f_train_df = pd.concat([new_train_df, onehot_df],axis=1, sort=False)
            #print(new_train_df.columns.values) 

            #arch_2.run(new_train_df, test_df)
