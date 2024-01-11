import torch
import json
import os
import pandas as pd
from collections import Counter


def balance_dataset(df):
    df_no_change = df[df['comparison'] == 'no change']
    num_no_change = len(df_no_change)
    df_improved = df[df['comparison'] == 'improved']
    num_improved = len(df_improved)
    df_worsened = df[df['comparison'] == 'worsened']
    num_worsened = len(df_worsened)
    num_samples = min(num_no_change, num_improved, num_worsened)
    new_df = pd.concat([ df_no_change.sample(num_samples), df_improved.sample(num_samples), df_worsened.sample(num_samples)], axis=0)
    return new_df

#The final_comparison_dataset.txt and bbox_objects_tabular224.txt are derived from chest imagenome dataset
#The train.csv, valid.csv and test.csv are from the chest imagenome dataset
file_path = "~/CXR/data/chest_imagenome/1.0.0/gk_parser_code/final_comparison_dataset.txt"
bbox_file_path = "~/CXR/data/chest_imagenome/1.0.0/silver_dataset/scene_tabular/bbox_objects_tabular224.txt"
splits=  ["~/CXR/code/splits/train.csv",
                        "~/CXR/code/splits/valid.csv",
                        "~/CXR/code/splits/test.csv"]



def split_dataset(file_path,bbox_file_path, splits):
    bbox_df = pd.read_csv(bbox_file_path, sep='\t').convert_dtypes()
    comp_df = pd.read_csv(file_path, sep='\t').convert_dtypes()
   
    bbox_df.dropna(inplace=True)
    comp_df.drop_duplicates(subset='current_image_id', keep="last", inplace=True)
    
    bbox_pid = set(bbox_df['image_id'])
    comp_pid = set(comp_df['current_image_id']).union(set(comp_df['previous_image_id']))
    bbox_pid = bbox_pid.intersection(comp_pid)
    comp_df = comp_df[comp_df['current_image_id'].isin(bbox_pid) & comp_df['previous_image_id'].isin(bbox_pid)]


    train_split = pd.read_csv(splits[0])
    valid_split = pd.read_csv(splits[1])
    test_split = pd.read_csv(splits[2])

    pid = set(list(train_split['dicom_id'].unique()))
    train = balance_dataset(comp_df[comp_df['current_image_id'].isin(pid)])

    pid = set(list(valid_split['dicom_id'].unique()))
    dev = balance_dataset(comp_df[comp_df['current_image_id'].isin(pid)])

    pid = set(list(test_split['dicom_id'].unique()))
    test = balance_dataset(comp_df[comp_df['current_image_id'].isin(pid)])
    
    
    print(Counter(train['comparison']))
    print(Counter(dev['comparison']))
    print(Counter(test['comparison']))
    print(len(train))
    print(len(dev))
    print(len(test))


    # train.to_csv('~/CheXRelFormer/data_extraction/train.csv', index=False)
    # dev.to_csv('~/CheXRelFormer/data_extraction/val.csv', index=False)
    # test.to_csv('~/CheXRelFormer/data_extraction/test.csv', index=False)
    return train, dev, test



split_dataset(file_path,bbox_file_path, splits)