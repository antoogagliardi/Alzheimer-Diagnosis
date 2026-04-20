import pytorch_lightning as pl
import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader, random_split

from tqdm import tqdm
import os
import json

import SimpleITK as sitk
from dltk.io import preprocessing
from skimage import filters

from src.data.mri_processing import resample_img, resize_image_with_crop_or_pad
from src.utils.plots import compute_class_frequency, plot_class_frequency
from src.utils.utils import classes_weigths_creation




class MRIDataset(Dataset):
    def __init__(self, metadata_json:str=None, img_zize:list=[181,217,181]):
        if metadata_json == None: print("You're not providing a path to JSON Metadata File")
        else:
            self.dataset = []
            self.levelDisease = {'CN':0, 'MCI':1, 'AD':2}
            self.whitening = False

            self.create(metadata_json, img_zize)
            print("-- Dataset Creation Completed --")

    def create(self, metadata_json, img_size):
        with open(file=metadata_json, mode="r") as infile: content = json.load(infile)
        with tqdm(range(len(content))) as pbar:
            for i, key, val in zip(pbar, content.keys(), content.values()):
                for j in range(len(val)):
                    if os.path.exists(val[j]["img_path"]) == False: print(f"La directory {val[j]['img_path']} non esiste"); continue
                    else:
                        # Load Image in SIKT format
                        t1_img = sitk.ReadImage(val[j]["img_path"])
                        # Transform Image into a Numpy Array
                        t1_img = resample_img(t1_img)
                        img = sitk.GetArrayFromImage(t1_img)

                        # Normalise the image to fit [-1, 1] range:
                        # norm_img = preprocessing.normalise_one_one(img)

                        # Normalise the image to fit [0, 1] range:
                        norm_img = preprocessing.normalise_zero_one(img)

                        # Crop or Pad (with optionally thresholding to black and white)
                        if self.whitening == True:
                            otsu = filters.threshold_otsu(norm_img)
                            otsu_img = norm_img > otsu
                            img = resize_image_with_crop_or_pad(image=otsu_img, img_size=img_size, mode='edge')     # Crop or Pad
                        else: img = resize_image_with_crop_or_pad(image=norm_img, img_size=img_size, mode='edge')   # Crop or Pad

                        if img.size == 0: print("Error: There's an empty image container")
                        
                        # Add dimention corresponding to "Channel"
                        # channel dimension (then accordingly with Conv3D input to NN would be ([batch],channels,depth,height,width)
                        img = np.expand_dims(img, axis=0)

                        try:
                            item = {}
                            item['Subject'] = key
                            item['idImage'] = val[j]["img_id"]
                            item['Age'] = val[j]["age"]
                            item['Sex'] = [0] if val[j]["sex"] == 'M' else [1]
                            item['Visit'] = val[j]["date"]
                            item['ADType'] = list([0,0,0]); item['ADType'][self.levelDisease[val[j]['group']]] = 1
                            item['Image'] = img

                            self.dataset.append(item)
                            pbar.set_postfix(FILE= key)
                            pbar.update()
                        except: print("Error: It has occurred an error while creating the dataset")
           
    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        return self.dataset[idx]
    

class MRIDataModule(pl.LightningDataModule):
    def __init__(self, dataset:Dataset, test_set:Dataset, saved_db_folder:str, batch_size:int=2, reload_data:bool=False, drop_last:bool=True):
        super(MRIDataModule, self).__init__()
        self.batch_size = batch_size
        self.drop_last_batch = drop_last
        
        if reload_data == True:
            if f'trainDataset.db' in os.listdir(saved_db_folder) and f'validDataset.db' in os.listdir(saved_db_folder):
                print('-- Loading existing dataset --')
                self.train_dataset = torch.load(os.path.join(saved_db_folder, f'trainDataset.db'), weights_only=False)
                self.valid_dataset = torch.load(os.path.join(saved_db_folder, f'validDataset.db'), weights_only=False)
                print("  - Data Loaded")
                
                ## Inspect Labels Frequency after Splitting
                print("Training Dataset Classes Distribution")
                train_samples, train_freq, scaled_train_freq = compute_class_frequency(self.train_dataset)
                plot_class_frequency(train_freq)
                self.training_class_weigths = classes_weigths_creation(length_data=train_samples,
                                                                    class_freq=scaled_train_freq)
                print("Validation Dataset Classes Distribution")
                val_samples, val_freq, scaled_val_freq = compute_class_frequency(self.valid_dataset)
                plot_class_frequency(val_freq)
                self.validation_class_weigths = classes_weigths_creation(length_data=val_samples,
                                                                        class_freq=scaled_val_freq)
            else: print("No Valid Dataset found")

            if f'testDataset.db' in os.listdir(saved_db_folder):
                self.test_dataset = torch.load(os.path.join(saved_db_folder, f"testDataset.db"), weights_only=False)
                print("Test Data Loaded")  
        else:
            if dataset != None:
                # At this step we may have to perform a dataset split
                print("-- Splitting the entire dataset into Training and Validation Sets --")
                seed = torch.Generator(device="cpu").seed()
                seed = torch.Generator(device="cpu").manual_seed(seed)
                print("  - Random Torch Seed: ", seed.initial_seed())
                initial_dataset_length = len(dataset)
                print("Lenght dataset: ", initial_dataset_length)
                training_length = int((2/3)*len(dataset))
                print("  - Training Dataset Length: ", training_length)
                validation_length = int(len(dataset) - training_length)
                print("  - Validation Dataset Length: ", validation_length)
                
                self.train_dataset, self.valid_dataset = random_split(dataset,
                                                                    [training_length, validation_length], generator=seed)
            
                ## Inspect Labels Frequency after Splitting
                print("Training Dataset Classes Distribution")
                train_samples, train_freq, scaled_train_freq = compute_class_frequency(self.train_dataset)
                plot_class_frequency(train_freq)
                self.training_class_weigths = classes_weigths_creation(length_data=train_samples,
                                                                    class_freq=scaled_train_freq)
                print("Validation Dataset Classes Distribution")
                val_samples, val_freq, scaled_val_freq = compute_class_frequency(self.valid_dataset)
                plot_class_frequency(val_freq)
                self.validation_class_weigths = classes_weigths_creation(length_data=val_samples,
                                                                        class_freq=scaled_val_freq)
                ###########################################
                torch.save(self.train_dataset, os.path.join(saved_db_folder, f'trainDataset.db'))
                torch.save(self.valid_dataset, os.path.join(saved_db_folder, f'validDataset.db'))
                if (len(self.train_dataset) + len(self.valid_dataset)) == initial_dataset_length: print("-- The split of the dataset has been performed correctly --")
                else: print("-- Something went wrong during the split --")
            else: print("No Dataset Valid Path has been provided")
            if test_set != None: self.test_dataset = test_set; torch.save(self.test_dataset, os.path.join(saved_db_folder, f"testDataset.db"))
            print("  - Training Dataset Length: ", len(self.test_dataset))
    
    def collate_data(self, samples):
        mri_image = torch.FloatTensor([sample["Image"] for sample in samples])
        sex = torch.LongTensor([sample["Sex"] for sample in samples])
        age = torch.LongTensor([int(sample["Age"]) for sample in samples])
        disease_status = torch.LongTensor([sample["ADType"] for sample in samples])
        
        return {"Image": mri_image, "Sex": sex, "Age": age, "Disease": disease_status}
    
    def train_dataloader(self):
        print("-- Preparing DataLoader for the training set --")
        train_dataloader = DataLoader(self.train_dataset, batch_size=self.batch_size,
                                      shuffle=True, collate_fn=self.collate_data,
                                      drop_last=True if self.drop_last_batch == True else False)
        print(" - Training DataLoader Ready")
        return train_dataloader

    def val_dataloader(self):
        print("-- Preparing DataLoader for the validation set --")
        valid_dataloader = DataLoader(self.valid_dataset, batch_size=self.batch_size,
                                      shuffle=False, collate_fn=self.collate_data,
                                      drop_last=True if self.drop_last_batch == True else False)
        print(" - Validation DataLoader Ready")
        return valid_dataloader
    
    def test_dataloader(self):
        print("-- Preparing DataLoader for the test set --")
        test_dataloader = DataLoader(self.test_dataset, batch_size=self.batch_size,
                                      shuffle=True, collate_fn=self.collate_data,
                                      drop_last=True if self.drop_last_batch == True else False)
        print(" - Test DataLoader Ready")
        return test_dataloader