import os
import pandas as pd
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID" 
os.environ["CUDA_VISIBLE_DEVICES"] = "2"
from architect import cnn_architecture


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

    arch_1 = cnn_architecture(1e-4)
    arch_1.run(final_df, test_df)

    #----stage 2: TAke the landmark images from stage 1 and predict the landmark category
    '''
    train_df = pd.read_csv('remaining_binary.csv') #train set with binary labels
    #Use binary labels instead of keras.utils.to_categircal <one_hot encoding>

    test_df = pd.read_csv('output.csv[predictions==1]') #images predicted as landmarks in stage 1
    

    arch_2 = cnn_architecture(1e-3,mode='categorical', output_neurons=8)
    arch2_run(train_df, test_df,l='categorical_crossentropy')

    '''
