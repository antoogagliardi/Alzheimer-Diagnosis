# Data Creation
from pprint import pprint
import yaml
import os
import json
import torch
import pandas as pd
import matplotlib.pyplot as plt

from src.utils.utils import navigate_folders, inspect_patiens
from src.data.utils_data import MRIDataset


with open("configs/config.yaml") as f:
    cfg = yaml.safe_load(f)
pprint(cfg)


## Open MRI Data
data_path = os.path.join(os.getcwd(), cfg["paths"]["data"])
mri_path = os.path.join(os.getcwd(), cfg["paths"]["adni_complete2"])
print(mri_path)
mri_csv = os.path.join(mri_path, "ADNI1_Complete_2Yr_1.5T_12_13_2022.csv")
print(mri_csv)

mri_imgs = os.path.join(mri_path, "ADNI")
print(navigate_folders(mri_imgs,[]))



## Save Entire Patients Dictionary
patients = inspect_patiens(data_path=mri_csv, img_dir=mri_imgs, process_desc="Screening Patients")
# patients = inspect_patiens(data_path=adni_Complete_1yr_csv, img_dir=adni_Complete_1yr_img,
#                                      patient_dict=patients, process_desc="One Year Patients")
# patients = inspect_patiens(data_path=adni_Complete_2yr_csv, img_dir=adni_Complete_2yr_img,
#                                      patient_dict=patients, process_desc="Two Year Patients")
# patients = inspect_patiens(data_path=adni_Complete_3yr_csv, img_dir=adni_Complete_3yr_img,
#                                      patient_dict=patients, process_desc="Three Year Patients")
with open(os.path.join(data_path, "patients.json"), "w") as outfile: 
    json.dump(patients, outfile, indent=2)
print("-- Entire Patients JSON FIle Saved --")



## Screening and Complete division and save
screening_dictionary = {}
complete_dictionary = {}

for key, val in zip(patients.keys(), patients.values()):
    if len(val) == 0: print("Error")
    if len(val) == 1: screening_dictionary[key] = val
    else: complete_dictionary[key] = val

print("Length of entire Patients Dictionary: ", len(patients))
print("Length Screening Dictionary: ", len(screening_dictionary))
with open(os.path.join(data_path, "screening.json"), "w") as outfile: 
   json.dump(screening_dictionary, outfile, indent=2)
print(" - Screening Dictionary Saved")
print("Length Complete Dictionary: ", len(complete_dictionary))
with open(os.path.join(data_path, "complete.json"), "w") as outfile: 
   json.dump(complete_dictionary, outfile, indent=2)
print(" - Complete Dictionary Saved")

print("\n")

pprint(screening_dictionary)



## MRI Dataset Creation from JSON
data_screening = MRIDataset(metadata_json=f"{data_path}/complete.json", img_zize=[80,100,80])
img_prova_idx = 85
img = data_screening[0]["Image"]
print(img.shape)

plt.imshow(img[0][:, :, img.shape[3]//2], cmap='gray')
plt.show()
plt.imshow(img[0][:, img.shape[2]//2, :], cmap='gray')
plt.show()
plt.imshow(img[0][img.shape[1]//2, :, :], cmap='gray')
plt.show()



## Save the JSON into .db File
torch.save(data_screening, os.path.join(data_path, f"complete.db"))
print("Dataset Correctly Saved")