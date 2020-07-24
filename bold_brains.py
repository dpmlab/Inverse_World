import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.linear_model import Ridge
import pickle
from scipy import stats

import torch
import torchvision.models as models
import torchvision.transforms as transforms
from torch.autograd import Variable
from PIL import Image

from pathlib import Path
import argparse
import os
import glob

import nibabel as nib
from nipype.interfaces import afni
from nipype.interfaces import fsl

from natsort import natsorted


num_images = 0
input_dir = ''
output_dir = ''

def main():
    args = get_args()
    temp_dirs = ['temp/activations', 'temp/subj_space', 
                 'temp/temp', 'temp/mni', 'temp/mni_s']
    for directory in temp_dirs:
        os.makedirs(directory, exist_ok=True)


    global num_images, input_dir, output_dir
    num_images = len(glob.glob(f"{args.input[0]}/*"))
    input_dir = args.input[0]
    output_dir = args.output

    generate_activations(args.input[0])
    generate_brains()
    transform_to_MNI()
    smooth_brains(args.sigma)
    average_subj_brains(args.input[0], args.output)

    if (args.true != None):
        compute_correlations(args.true)
        compute_ranking()
        # frame_by_frame_correlation(args.input[0], args.output)
    


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

    model_dir = f"models/"
    shape_array = np.load('derivatives/shape_array.npy', allow_pickle=True)

    # For each subject, for each input file, predict all ROIs for that subject and save prediction
    for subj in range(0,3):
        for filename in glob.glob('temp/activations/*'):
            actv  = np.load(open(filename, 'rb'), allow_pickle=True)
            for roi_idx, roi in enumerate(range(0,5)): # Only generating for LOC, RSC, PPA

                model = pickle.load(open(f'models/subj{subj+1}_{roi_list[roi]}_model.pkl', 'rb'))
                pickle.dump(model, open(f'models/subj{subj+1}_{roi_list[roi]}_model.pkl', 'wb'))
                y_pred_brain = model.predict([actv])
                brain = y_pred_brain[0]

                # Get left hemisphere
                T1_mask_nib = nib.load(f"derivatives/bool_masks/derivatives_spm_sub-CSI{subj+1}" \
                                       f"_sub-CSI{subj+1}_mask-LH{roi_list[roi]}.nii.gz")
                T1_mask_shape = T1_mask_nib.header.get_data_shape()[0:3]
                LH_T1_mask = T1_mask_nib.get_fdata() > 0  # 3D boolean array

                # Get right hemisphere
                T1_mask_nib = nib.load(f"derivatives/bool_masks/derivatives_spm_sub-CSI{subj+1}" \
                                       f"_sub-CSI{subj+1}_mask-RH{roi_list[roi]}.nii.gz")                            
                RH_T1_mask = T1_mask_nib.get_fdata() > 0  # 3D boolean array

                # Initialize subj's 3d volume if first time through
                if (roi_idx == 0):
                    subj_brain = np.empty(T1_mask_shape)
                    subj_brain[:, :, :] = np.NaN  

                # LH Nanmean
                a = np.array([subj_brain[LH_T1_mask], 
                              brain[:int(shape_array[subj][roi][0])]]) # nanmean of new w existing
                a = np.nanmean(a, axis=0)
                subj_brain[LH_T1_mask] = a # Hopefully the length of your vector = T1_mask.sum()

                # RH Nanmean
                a = np.array([subj_brain[RH_T1_mask],
                              brain[int(shape_array[subj][roi][0]):]])
                a = np.nanmean(a, axis=0)
                subj_brain[RH_T1_mask] = a

            nib.save(nib.Nifti1Image(subj_brain, affine=T1_mask_nib.affine),
                     f'temp/subj_space/sub{subj+1}_{Path(filename).stem}.nii.gz')
    print(f"Saved: Predictions into subjects' brains")
    # Probably should save these for the future in a specific folder to this run, so don't do 
    # everything everytime



def transform_to_MNI():
    for subj in range(1,4):
        filename = f'temp/subj_space/sub{subj}*'
        for file in glob.glob(filename):
            stem = Path(file).stem

            resample = afni.Resample()
            resample.inputs.in_file = file
            resample.inputs.master = f"derivatives/T1/sub-CSI{subj}_ses-16_anat_sub-CSI{subj}" \
                                     f"_ses-16_T1w.nii.gz"
            resample.inputs.out_file = f'temp/temp/{stem}'
            # print(resample.cmdline)
            resample.run()

            aw = fsl.ApplyWarp()
            aw.inputs.in_file = f'temp/temp/{stem}'
            aw.inputs.ref_file = f'derivatives/sub{subj}.anat/T1_to_MNI_nonlin.nii.gz'
            aw.inputs.field_file = f'derivatives/sub{subj}.anat/T1_to_MNI_nonlin_coeff.nii.gz'
            aw.inputs.premat = f'derivatives/sub{subj}.anat/T1_nonroi2roi.mat'
            aw.inputs.interp = 'nn'
            aw.inputs.out_file = f'temp/mni/{stem}.gz' # Note: stem here contains '*.nii'
            # print(aw.cmdline)
            aw.run()
            os.remove(f'temp/temp/{stem}')
    print("Saved: Brains in MNI space")



def smooth_brains(sig):
    filename = f"temp/mni/*"
    for file in glob.glob(filename):
        stem = Path(file).stem

        out = f"temp/mni_s/{stem}.gz"
        smooth = fsl.IsotropicSmooth()
        smooth.inputs.in_file = file
        smooth.inputs.sigma = sig
        smooth.inputs.out_file = out
        smooth.run()
    print("Saved: Smoothed MNI brains")



def average_subj_brains(input_dir, output_dir):
    # TODO Change to be relative to text file or sorting criteria
    for file in glob.glob(f'{input_dir}/*'):
        stem = Path(file).stem
        subj_mask_nib = nib.load(f'temp/mni_s/sub1_{stem}.nii.gz')
        subj_brain = np.empty((subj_mask_nib.shape))
        subj_brain[:,:,:] = np.NaN

        for subj in range(1,4):
            filename = f'temp/mni_s/sub{subj}_{stem}.nii.gz'
            brain = nib.load(filename).get_fdata()
            subj_brain = np.nanmean([subj_brain, brain], axis=0)
                
        nib.save(nib.Nifti1Image(subj_brain, affine=subj_mask_nib.affine),
                                 f'{output_dir}/{stem}.nii.gz')
    print("Saved: Averaged MNI brains")
        


def compute_correlations(true_dir):
    sorted_output = [Path(i).stem for i in natsorted(glob.glob(f'{output_dir}/*'))]
    sorted_true   = [Path(i).stem for i in natsorted(glob.glob(f'{true_dir}/*'))]
    # print(sorted_output[:10], sorted_true[:10])
    assert(len(sorted_output) == len(sorted_true))

    threshold = 0.15 # threshold for smoothed bool mask
    corr = np.empty((num_images, num_images,))
    test_indices = range(0, num_images)
    rois = ['LOC', 'PPA', 'RSC']
    overlap = get_subj_overlap(rois)

    for i_pred, image_pred in enumerate(sorted_output):
        # Load averaged brain output
        pred_pattern = nib.load(f'{output_dir}/{image_pred}.gz').get_fdata()

        for i_test, image_test in enumerate(sorted_true):
            true_pattern = nib.load(f'{true_dir}/{image_test}.gz').get_fdata()
            pred_overlap = pred_pattern[overlap] 
            true_overlap = true_pattern[overlap]

            corr[i_pred, i_test] = stats.pearsonr(pred_overlap, true_overlap)[0] #just get r, not p val

    pkl_filename = f"temp/corr_matrix.pkl"
    with open(pkl_filename, 'wb') as file:
        pickle.dump(corr, file)

    print(f"Saved: Correlation matrix with true brains")



def compute_ranking():
    corr = np.load('temp/corr_matrix.pkl', allow_pickle = True)
    (num_images, _) = corr.shape

    ranked = np.empty((num_images))

    for i in range(num_images):
        sort_r = {'r':corr[i, :], 'col': [z for z in range(156)]}

        sort_r = pd.DataFrame(sort_r).sort_values(by='r', ascending=False)
        sort_r['default_rank'] = sort_r['r'].rank(ascending=False)

        ranked[i] = sort_r.loc[i, 'default_rank']

    print(f"Average ranking of predicted to true brain: " +
          "{:.3f}".format(np.mean(ranked, axis=0)*100/num_images) + " / 50.5")



def frame_by_frame_correlation(input_dir, output_dir):
    directory = glob.glob(f'{input_dir}/*')
    num_images = len(directory)
    sorted_input = [Path(i).stem for i in natsorted(directory)]

    overlap = get_subj_overlap(['LOC', 'PPA', 'RSC'])
    corr = np.zeros((num_images - 1))
    for i in range(num_images - 1):
        first  = nib.load(f'{output_dir}/{sorted_input[i]}.nii.gz').get_fdata()
        second = nib.load(f'{output_dir}/{sorted_input[i + 1]}.nii.gz').get_fdata()

        corr[i] = stats.pearsonr(first[overlap], second[overlap])[0] #just get r, not p val

    plt.figure(figsize=(15, 2))
    plt.title('frame by frame correlation for Partly Cloudy')
    plt.xlabel('TR')
    plt.ylabel('')
    plt.imshow([corr], cmap='viridis', aspect='auto')
    plt.colorbar(orientation="horizontal",)
    plt.show()



def get_subj_overlap(rois):
    threshold = 0.15
    for roi_idx, roi in enumerate(rois):
        for subj in range(0,3):
            f_p_lh = nib.load(f'derivatives/s_bool_masks/s_sub{subj+1}_LH{roi}_MNI.nii.gz').get_fdata()
            f_p_rh = nib.load(f'derivatives/s_bool_masks/s_sub{subj+1}_RH{roi}_MNI.nii.gz').get_fdata()
            first_pred_LH_mask = f_p_lh > np.max(f_p_lh) * threshold
            first_pred_RH_mask = f_p_rh > np.max(f_p_rh) * threshold

            if (roi_idx == 0) and (subj == 0):
                overlap = first_pred_LH_mask | first_pred_RH_mask
            else: 
                overlap = overlap | first_pred_LH_mask | first_pred_RH_mask
    return overlap



def get_args():
    parser = argparse.ArgumentParser(description='Convert images into MNI brains.')
    parser.add_argument('--input', metavar='input_dir', type=dir_path, required=True, nargs=1,
                        help='input directory which contains all images to be processed.')
    parser.add_argument('--output', metavar='output_dir', type=dir_path, default='.',
                        help='output directory where MNI NIFTI files will be saved.')
    parser.add_argument('--sigma', metavar='sigma', type=float, default=1.0,
                        help='sigma for smoothing MNI brains')
    parser.add_argument('--true', metavar='true_brain_dir', type=dir_path, default=None,
                        help='directory with true brains to compare predicted with')

    return parser.parse_args()   



def dir_path(string):
    if os.path.isdir(string):
        return Path(string).resolve()
    else:
        raise NotADirectoryError(string)


if __name__ == '__main__':
    main()