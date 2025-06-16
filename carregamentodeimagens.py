"""
Code to organize data from
https://www.kaggle.com/datasets/kmader/skin-cancer-mnist-ham10000
into three distinct CSV files, with train, test and validation sets.

Assuming this script is executed at folder CODE_FOLDER, put the
data  in a parent parant folder called ../../data_ham1000.

Assume that all JPG images were manually copied to the same folder.
For instance, I moved the files in the original folder
HAM10000_images_part_2 into
HAM10000_images_part_1, such that all JPG files are into a unique folder 
(../../data_ham1000/HAM10000_images_part_1)
"""


import numpy as np
import pandas as pd


all_ham_data_df = pd.read_csv("classificador_medico/ISIC2018_Task3_Training_GroundTruth/ISIC2018_Task3_Training_GroundTruth.csv")

print(all_ham_data_df.info())
print(all_ham_data_df.head())

mel = all_ham_data_df["MEL"]
print(np.sum(mel))

# rename columns
all_ham_data_df.rename(columns={"MEL": "label"}, inplace=True)
all_ham_data_df.rename(columns={"image": "image_name"}, inplace=True)

# Add the extension '.jpg' to all items in the 'image_name' column
all_ham_data_df["image_name"] = all_ham_data_df["image_name"].apply(
    lambda x: x + ".jpg"
)

all_ham_data_df = all_ham_data_df.drop(['NV','BCC','AKIEC','BKL','DF','VASC'], axis=1)

print(all_ham_data_df.info())
print(all_ham_data_df.head())

# https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.to_csv.html
output_file_name = "classificador_medico/binary_HAM10000_metadata.csv"
all_ham_data_df.to_csv(output_file_name)
print("Wrote", output_file_name)