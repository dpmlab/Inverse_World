# import warnings
# import sys
# if not sys.warnoptions:
    # warnings.simplefilter("ignore")
import os
import glob
# from collections import defaultdict

# import matplotlib.pyplot as plt
import numpy as np
# from scipy import stats
from sklearn.linear_model import Ridge
# from sklearn.model_selection import GridSearchCV
# from sklearn.metrics import make_scorer
# from sklearn.model_selection import train_test_split
# from sklearn.metrics import r2_score
import nibabel as nib
# import re
import pickle

import matplotlib.pyplot as plt

import torch
# import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as transforms
from torch.autograd import Variable
from PIL import Image

from pathlib import Path
import argparse

import pandas as pd
from nipype.interfaces import afni
from nipype.interfaces import fsl

num_images = 0

def main():
    args = get_args()
    temp_dirs = ['temp/activations', 'temp/subj_space', 
                 'temp/temp', 'temp/mni']
    for directory in temp_dirs:
        os.makedirs(directory, exist_ok=True)

    num_images = len(glob.glob(f"{args.input[0]}/*"))
    generate_activations(args.input[0])
    generate_brains()
    transform_to_MNI()
    smooth_brains(args.output, args.sigma)
    


def generate_activations(input_dir):
    output = f"temp/activations/"

    # Default input image transformations for ImageNet
    scaler = transforms.Resize((224, 224))
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])
    to_tensor = transforms.ToTensor()

    # Load the pretrained model, set to eval mode
    model = models.resnet18(pretrained=True)
    model.eval()

    for filename in glob.glob(f"{input_dir}/*"):
        img = Image.open(filename)
        t_img = Variable(normalize(to_tensor(scaler(img))).unsqueeze(0))

        # Create network up to last layer, push image through, flatten
        layer_extractor = torch.nn.Sequential(*list(model.children())[:-1])
        feature_vec = layer_extractor(t_img).data.numpy().squeeze()
        feature_vec = feature_vec.flatten()
        
        # Save image
        image_name = Path(filename).stem
        np.save(f"{output}{image_name}.npy", feature_vec)
        img.close()

    num_images = len(glob.glob(f"{input_dir}/*"))
    print(f"Saved: CNN activations of {num_images} images")



# TODO: simplify models, better way to do order files so fewer loads?
def generate_brains():
    roi_list = ["EarlyVis","OPA", "LOC", "RSC", "PPA"]
    ridge_p_grid = {'alpha': np.logspace(1, 5, 10)}
    # print(f"Param Grid: {[i for i in ridge_p_grid['alpha']]}")
    # bold = np.load('bold5000_subj_data.pkl', allow_pickle=True)

    model_dir = f"models/"
    shape_array = np.load('derivatives/shape_array.npy', allow_pickle=True)
    random_state = 3


    # subj_brains = np.zeros((num_images,), dtype=object)
    for subj in range(0,3):
        for filename in glob.glob('temp/activations/*'):
            actv  = np.load(open(filename, 'rb'), allow_pickle=True)
            for roi_idx, roi in enumerate(range(0,5)): # All ROIs
                model = pickle.load(open(f'models/subj{subj+1}_{roi_list[roi]}_model.pkl', 'rb'))
                y_pred_brain = model.predict([actv])
                brain = y_pred_brain[0]

                # Save subj specific predicted brain
                # Get left hemisphere
                T1_mask_nib = nib.load(f"derivatives/bool_masks/derivatives_spm_sub-CSI{subj+1}" \
                                       f"_sub-CSI{subj+1}_mask-LH{roi_list[roi]}.nii.gz")
                T1_mask_shape = T1_mask_nib.header.get_data_shape()[0:3]
                LH_T1_mask = T1_mask_nib.get_data() > 0  # 3D boolean array

                # Get right hemisphere
                T1_mask_nib = nib.load(f"derivatives/bool_masks/derivatives_spm_sub-CSI{subj+1}" \
                                       f"_sub-CSI{subj+1}_mask-RH{roi_list[roi]}.nii.gz")                            
                RH_T1_mask = T1_mask_nib.get_data() > 0  # 3D boolean array

                # Initialize subj's 3d volume if first time through
                if (roi_idx == 0):
                    subj_brain = np.empty(T1_mask_shape)
                    subj_brain[:, :, :] = np.NaN  

                # LH
                a = np.array([subj_brain[LH_T1_mask], 
                              brain[:int(shape_array[subj][roi][0])]]) # nanmean of new with existing
                a = np.nanmean(a, axis=0)
                subj_brain[LH_T1_mask] = a # Hopefully the length of your vector = T1_mask.sum()

                # RH
                a = np.array([subj_brain[RH_T1_mask],
                              brain[int(shape_array[subj][roi][0]):]])
                a = np.nanmean(a, axis=0)
                subj_brain[RH_T1_mask] = a

            nib.save(nib.Nifti1Image(subj_brain, affine=T1_mask_nib.affine),
                     f'temp/subj_space/sub{subj+1}_{Path(filename).stem}.nii.gz')
    print(f"Saved: Predictions into subjects' brains")
    # Probably should save these for the future in a specific folder to this run, so don't do everything everytime



def transform_to_MNI():
    for subj in range(0,3):
        filename = f'temp/subj_space/sub{subj+1}*'
        for file in glob.glob(filename):
            stem = Path(file).stem
            resample = afni.Resample()
            resample.inputs.in_file = file
            resample.inputs.master = f'derivatives/T1/sub-CSI{subj+1}_ses-16_anat_sub-CSI{subj+1}_ses-16_T1w.nii.gz'
            resample.inputs.out_file = f'temp/temp/{stem}'
            # print(resample.cmdline)
            resample.run()

            aw = fsl.ApplyWarp()
            aw.inputs.in_file = f'temp/temp/{stem}'
            aw.inputs.ref_file = f'derivatives/sub{subj+1}.anat/T1_to_MNI_nonlin.nii.gz'
            aw.inputs.field_file = f'derivatives/sub{subj+1}.anat/T1_to_MNI_nonlin_coeff.nii.gz'
            aw.inputs.premat = f'derivatives/sub{subj+1}.anat/T1_nonroi2roi.mat'
            aw.inputs.interp = 'nn'
            aw.inputs.out_file = f'temp/mni/{stem}.gz' # Note: stem here contains '*.nii'
            # print(aw.cmdline)
            aw.run()
            os.remove(f'temp/temp/{stem}')
    print("Saved: Brains in MNI space")



def smooth_brains(output_dir, sig):

    filename = f"temp/mni/*"
    for file in glob.glob(filename):
        stem = Path(file).stem
        out = f"{output_dir}/{stem}.gz"
        smooth = fsl.IsotropicSmooth()
        smooth.inputs.in_file = file
        smooth.inputs.sigma = sig
        smooth.inputs.out_file = out
        smooth.run()
#         ! fslmaths $bm -kernel gauss $sigval -fmean $new
    print("Saved: Smoothed MNI brains")   



def get_args():
    parser = argparse.ArgumentParser(description='Convert images into MNI brains.')
    parser.add_argument('--input', metavar='input_dir', type=dir_path, required=True, nargs=1,
                        help='input directory which contains all images to be processed.')
    parser.add_argument('--output', metavar='output_dir', type=dir_path, default='.',
                        help='output directory where MNI NIFTI files will be saved.')
    parser.add_argument('--sigma', metavar='sigma', type=float, default=1.0,
                        help='sigma for smoothing MNI brains')

    return parser.parse_args()   


def dir_path(string):
    if os.path.isdir(string):
        return Path(string).resolve()
    else:
        raise NotADirectoryError(string)

if __name__ == '__main__':
    main()