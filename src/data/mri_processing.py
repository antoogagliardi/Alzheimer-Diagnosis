import numpy as np
import torch
import re
import os
from sh import gunzip   # Save the MRI Images after Skull-Removal Pre-Processing

###### Other Usefull Experimental Libraries #########################################################################################
import SimpleITK as sitk # Similar to NiBabel API, is a Python package for reading and writing Neuroimaging Data (files NIfTIs)

# from dltk.io import preprocessing
# from dltk.io import augmentation

# from skimage import filters

from nipype.interfaces import fsl # Skull-Stripping
#####################################################################################################################################


# Pre-Processing - Resample image
def resample_img(itk_image, out_spacing=[2.0, 2.0, 2.0]):
    ''' This function resamples images to 2-mm isotropic voxels. 
        Parameters:
            itk_image -- Image in simpleitk format, not a numpy array
            out_spacing -- Space representation of each voxel        
        Returns: 
            Resulting image in simpleitk format, not a numpy array
    '''
    
    # Resample images to 2mm spacing with SimpleITK
    original_spacing = itk_image.GetSpacing()
    original_size = itk_image.GetSize()

    out_size = [int(np.round(original_size[0] * (original_spacing[0] / out_spacing[0]))),
                int(np.round(original_size[1] * (original_spacing[1] / out_spacing[1]))),
                int(np.round(original_size[2] * (original_spacing[2] / out_spacing[2])))]

    resample = sitk.ResampleImageFilter()
    resample.SetOutputSpacing(out_spacing)
    resample.SetSize(out_size)
    resample.SetOutputDirection(itk_image.GetDirection())
    resample.SetOutputOrigin(itk_image.GetOrigin())
    resample.SetTransform(sitk.Transform())
    resample.SetDefaultPixelValue(itk_image.GetPixelIDValue())

    resample.SetInterpolator(sitk.sitkBSpline)

    return resample.Execute(itk_image)

# Pre-Processing - Remove Skull (useless for the analysis)
def skull_strip_nii(original_img, destination_img, frac=0.3):
    ''' Practice skull stripping on the given image, and save
        the result to a new .nii image.
        Uses FSL-BET 
        (https://fsl.fmrib.ox.ac.uk/fsl/fslwiki/BET/UserGuide#Main_bet2_options:)
        
        Parameters:
            original_img -- Original nii image
            destination_img -- The new skull-stripped image
            frac -- Fractional intensity threshold for BET
    '''
    
    btr = fsl.BET()
    btr.inputs.in_file = original_img
    btr.inputs.frac = frac
    btr.inputs.out_file = destination_img
    btr.cmdline
    res = btr.run()

# Image Resize
def resize_image_with_crop_or_pad(image, img_size=(64, 64, 64), **kwargs):
    """Image resizing. Resizes image by cropping or padding dimension
     to fit specified size.

    Args:
        image (np.ndarray): image to be resized
        img_size (list or tuple): new image size
        kwargs (): additional arguments to be passed to np.pad

    Returns:
        np.ndarray: resized image
    """

    assert isinstance(image, (np.ndarray, np.generic))
    assert (image.ndim - 1 == len(img_size) or image.ndim == len(img_size)), \
        'Example size doesnt fit image size'

    # Convert image to Tensor to avoid problem with np.pad() function (ValueError on slicing procedure)
    image = torch.tensor(image)
    
    # Get the image dimensionality
    rank = len(img_size)

    # Create placeholders for the new shape
    from_indices = [[0, image.shape[dim]] for dim in range(rank)]
    to_padding = [[0, 0] for dim in range(rank)]

    slicer = [slice(None)] * rank

    # For each dimensions find whether it is supposed to be cropped or padded
    for i in range(rank):
        if image.shape[i] < img_size[i]:
            to_padding[i][0] = (img_size[i] - image.shape[i]) // 2
            to_padding[i][1] = img_size[i] - image.shape[i] - to_padding[i][0]
        else:
            from_indices[i][0] = int(np.floor((image.shape[i] - img_size[i]) / 2.))
            from_indices[i][1] = from_indices[i][0] + img_size[i]

        # Create slicer object to crop or leave each dimension
        slicer[i] = slice(from_indices[i][0], from_indices[i][1])

    # Pad the cropped image to extend the missing dimension
    return np.pad(image[slicer], to_padding, **kwargs)


# Main Functions
def process_nii_image(nii_img):
    new_img_path = re.split(r".nii", nii_img)[0]
    new_img_path = f"{new_img_path}_PROCESSED.nii"
    print("NEW PATH: ", new_img_path)
    skull_strip_nii(nii_img, new_img_path, frac=0.5)

    filename = f"{new_img_path}.gz"
    gunzip(filename)                # Extract File
    os.remove(nii_img)


def image_PreProcessing(path:str, lst:list):
    if os.path.isfile(path) and ".nii" in path:                 # Base case: I've reached the file itself
        id_img = re.split("/", os.path.abspath(path))[-2]
        id_patient = re.split("/", os.path.abspath(path))[-5]
        visit_date = re.findall(r"([0-9]+-[0-9]+-[0-9]+)", re.split("/", os.path.abspath(path))[-3])[0]

        # Looks if a patient has two MRI referred to the same visit (scaled/scaled_2)
        already_in_lst = False
        for tup in lst:
            if tup[:2] == (id_patient, visit_date): already_in_lst=True; break
        if already_in_lst==False:
            lst.append((id_patient, visit_date, id_img, path)); print(path) # Take into account just scaled .nii files

            # Perform the Skull-Removal Procedure on the Image
            process_nii_image(path)
        else: print(f"- Already inside the list: {path}")

        return lst
    else:
        sub_dir=sorted(os.listdir(path)) #; print(sub_dir)
        if ".DS_Store" in sub_dir: sub_dir.remove(".DS_Store")          ## Avoiding hidden files in MAC OS
        for dir in sub_dir:
            # print(dir)
            if ".csv" in dir: continue
            lst = image_PreProcessing(os.path.join(path, dir), lst)
    return lst


""" 
    Eventually perform Skull Removal on MRI Images
        ----  PRE-PROCESSED ADNI MRI DATASET  ----
    (X) ./datasets/ADNI1-Screening_1.5T
    (X) ./datasets/ADNI1-Complete_1Yr_1.5T
    (X) ./datasets/ADNI1-Complete_2Yr_1.5T
    (X) ./datasets/ADNI1-Complete_3Yr_1.5T
"""
# print(image_PreProcessing("./datasets/ADNI1-Complete_3Yr_1.5T",[]))