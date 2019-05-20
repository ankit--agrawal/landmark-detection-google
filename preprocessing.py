import glob2, os, csv
import numpy as np
import pandas as pd

#---------one time process: collect all images from all folders in a single DataFrame.
#----------We also collect the filename (without extension) in the DataFrame to perform the join for training data

def formatting(x): #extracting the filename without the jpg <file_code>
    tmp = x.split('/')
    path = '/'
    for i in range(len(tmp)-1):
        path += tmp[i]+'/'
    filename = tmp[-1]
    file_code = filename.split('.')
    file_code = file_code[0]
    return file_code

def walk_dir(full_path, folder, extensions): #collect all files of <file_type> in a directory
    #all_images = glob2.glob(full_path + folder + '/**/*.' + extensions) #use glob for small dataset

    print("Walking through the Directory: ",full_path + folder)
    matches = []
    for root, dirnames, filenames in os.walk(folder): #use when number of files is huge
        for filename in filenames:
            if filename.endswith(extensions):
                matches.append(os.path.join(full_path, root, filename))

    print(len(matches))
    return matches

def creating_final_files(main_dir_path):
    #dataframe of negative examples
    df_neg = pd.DataFrame(walk_dir(main_dir_path,'not_landmark','jpg'), columns = ['path']) 
    df_neg['landmark_id'] = 0
    df_neg['classifier_label'] = 'no'
    df_neg['id'] = df_neg['path'].apply(lambda x: formatting(x))
    df_neg.to_csv('neg_examples.csv',index=False) #writing to a csv for later use
    #print(df_neg.head())
    
    #dataframe of positive examples <Google dataset>
    df_pos = pd.DataFrame(walk_dir(main_dir_path,'google-landmark-master','jpg'), columns = ['path'])
    df_pos['id'] = df_pos['path'].apply(lambda x: formatting(x))
    df_pos['classifier_label'] = 'yes'
    df_true_train = pd.read_csv('train.csv', header='infer')
    print('train.csv shape: ',df_true_train.shape)

    #--------merging the 2 DataFrames
    df_train = df_true_train.merge(df_pos, on='id', how='inner')
    print('shape after merging: ',df_train.shape) #verify that len(match) = len(df_true_train) = len(df_train)
    df_train.to_csv('pos_examples.csv',index=False)
    #print(df_train.head())
    df_test = pd.DataFrame(walk_dir(main_dir_path,'test_dataset','jpg'), columns = ['path'])
    df_test['id'] = df_test['path'].apply(lambda x: formatting(x))
    df_test.to_csv('test_examples.csv', index=False)
    
if __name__ == '__main__':
    main_dir_path = '/home/sci/amanpreet/Documents/Kaggle/Google_Landmarks/'
    creating_final_files(main_dir_path)
    
    pos_df = pd.read_csv('pos_examples.csv')
    neg_df = pd.read_csv('neg_examples.csv')

    pos_df = pos_df[['path','landmark_id','classifier_label','id']]
    #print(pos_df.columns.values) #match the column names of both dataframes
    #print(neg_df.columns.values
