import os
import re
import torch
import numpy as np
import pandas as pd
from tqdm import tqdm

# Navigate Folders
def navigate_folders(path:str, lst:list):
    if os.path.isfile(path) and ".nii" in path: # Base case: I've reached the file itself
        id_img = re.split("/", os.path.abspath(path))[-2]
        id_patient = re.split("/", os.path.abspath(path))[-5]
        visit_date = re.findall(r"([0-9]+-[0-9]+-[0-9]+)", re.split("/", os.path.abspath(path))[-3])[0]

        # Looks if a patient has two MRI referred to the same visit (scaled/scaled_2)
        already_in_lst = False
        for tup in lst:
            if tup[:2] == (id_patient, visit_date): already_in_lst=True; break
        if already_in_lst==False: lst.append((id_patient, id_img, path)) # We take into account just scaled .nii files
        # else: print(f"- Already inside the list: {path}")

        return lst
    else:
        sub_dir=sorted(os.listdir(path)) #; print(sub_dir)
        if ".DS_Store" in sub_dir: sub_dir.remove(".DS_Store") ## Avoiding hidden files in MAC OS
        for dir in sub_dir:
            # print(dir)
            if ".csv" in dir: continue
            lst = navigate_folders(os.path.join(path, dir), lst)
    return lst

# Patiens Data Inspection
def inspect_patiens(data_path:str=None, img_dir:str=None, patient_dict:dict=None, process_desc:str=None):
    df = pd.read_csv(data_path)
    dict_row = {} if patient_dict == None else patient_dict
    
    list_path = navigate_folders(path=img_dir, lst=[])
    with tqdm(range(len(df)), desc=process_desc) as pbar:
        for i, row in zip(pbar, df.index):
            # if i < 30:
            row = df.loc[row] # <class 'pandas.core.series.Series'>
            subject = row["Subject"]
            last_age = []
            for path_row in list_path:
                if path_row[0] == subject:
                    if path_row[1] == row["Image Data ID"]:
                        img_path = path_row[2]
                        if not "Scaled_2" in row["Description"]: # If the image is not "Scaled_2"
                            value = {"sex": row["Sex"],
                                    "age": str(row["Age"]),
                                    "visit": row["Visit"],
                                    "group": row["Group"],
                                    "date": row["Acq Date"],
                                    "img_id": row["Image Data ID"],
                                    "img_path": img_path}
                            if subject in dict_row.keys():
                                # print("Paziente già presente")
                                if value in dict_row[subject]: continue; print(dict_row[subject])
                                for elem in dict_row[subject]:
                                    if value["age"] in elem["age"]: break; # print(elem); break
                                else: dict_row.setdefault(subject,[]).append(value); last_age.append(value["age"])
                            else:
                                # if value["age"] in last_age: continue
                                # else:
                                dict_row.setdefault(subject,[]).append(value); last_age.append(value["age"])
                            pbar.set_postfix({"ROW":i, "PAT": subject, "TOT_PATIENT": len(dict_row)})
                            pbar.update()
                    else: continue
                else: continue

    return dict_row

## Dataset Analysis - Weighted Unbalanced Data
def classes_weigths_creation(length_data:int, class_freq:list):
    classes_weights = np.empty(shape=(0),dtype=np.float32)
    for freq in class_freq:
        epsilon = 1.0 / (length_data * np.sqrt(freq*length_data))
        weight = 1.0 / (freq + epsilon)
        classes_weights = np.append(classes_weights, [weight], axis=0)
        print(f"{freq} -> weigth = {weight}")
    classes_weights = torch.from_numpy(classes_weights).type(torch.FloatTensor)
    
    return classes_weights


