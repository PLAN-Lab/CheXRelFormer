import pandas as pd
import shutil
import numpy as np

#The splited dataset, train, test, and val are used to create folders A, B, label and the train, val, test text files.
df = pd.read_csv('val.csv',header=0)
# df = pd.read_csv('train.csv',header=0)
# df = pd.read_csv('test.csv',header=0)
disease_labels = {
                        'lung opacity',
                        'pleural effusion',
                        'atelectasis',
                        'enlarged cardiac silhouette',
                        'pulmonary edema/hazy opacity',
                        'pneumothorax', 
                        'consolidation',
                        'fluid overload/heart failure', 
                        'pneumonia'
                    }


filtered_df = df[df['label_name'].isin(disease_labels)]
df = filtered_df[['current_image_id', 'previous_image_id','comparison']]

#Only keep one label for one pair of data, reduce the effect of bbox and disease
df = df.drop_duplicates(
  subset = ['current_image_id', 'previous_image_id'],
  keep = 'first').reset_index(drop = True)

#for finding the path for pair of images
df1 = pd.read_csv('cxr-record-all-jpg.csv', sep=",",header=0)
df1 = df1[['dicom_id','path']]


for idx, row in df.iterrows():
    current_img_id = row['current_image_id']
    previous_img_id = row['previous_image_id']
    comparsion = row['comparison']
    current_img_path = df1.loc[df1['dicom_id']==current_img_id]['path'].values.tolist()[0].strip()
    previous_img_path = df1.loc[df1['dicom_id']==previous_img_id]['path'].values.tolist()[0].strip()
  

    current_img_name = current_img_path.split("/")[-1]
    previous_img_name = previous_img_path.split("/")[-1]

    new_img_name = current_img_name.split(".")[0] + "_" + previous_img_name.split(".")[0] + ".jpg"
    
    new_img_names = []

    new_img_names.append(new_img_name)

    # change the names of the files as train.txt, test.txt and val.txt based on the data read in.
    with open("val.txt", "a") as file:
        for name in new_img_names:
            file.write(name + "\n")


    base_path = '~/physionet.org/files/mimic-cxr-jpg/2.0.0/'

    shutil.copy2(str(base_path + current_img_path), 'A/' + new_img_name) #copy the current image to A
    shutil.copy2(str(base_path + previous_img_path), 'B/' + new_img_name) #copy the previous image to B
    label_path = 'Labels/' + new_img_name.split(".")[0] + '.txt'

    with open(label_path, "w") as f: #write the comparison result to Labels folder
        f.write(comparsion)




