import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as transforms
from torch.autograd import Variable
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
import scipy.spatial
import re
from scipy import stats
from collections import defaultdict
# For file locating and viewing
import glob
import os


# save_location = "/Volumes/The Drive/Research/"
save_location = "./"

generate_activations = True
save_activations = True
load_activations = True


bdata_raw = np.load('./BOLD5000_data_dpml_good/bold5000_data.npy', allow_pickle=True)

# ROI order: 0 LHPPA | 1 RHLOC | 2 LHLOC | 3 RHEarlyVis | 4 RHRSC | 5 RHOPA 
#            6 RHPPA | 7 LHEarlyVis | 8 LHRSC | 9 LHOPA}
# Pairs: LH, RH: PPA 0,6 | LOC 2,1 | EarlyVis 7,3 | RSC 8,4 | OPA 9,5

# Shape    : 4 x 10 x 4916 (subj x ROI x image)
bdata = np.zeros((4, 5, 4916), dtype = object)
for subj in range(len(bdata_raw)):
    for i in range(4916):
        bdata[subj][0][i] = np.concatenate((bdata_raw[subj][0][i],bdata_raw[subj][6][i]))
        bdata[subj][1][i] = np.concatenate((bdata_raw[subj][2][i],bdata_raw[subj][1][i]))
        bdata[subj][2][i] = np.concatenate((bdata_raw[subj][7][i],bdata_raw[subj][3][i]))
        bdata[subj][3][i] = np.concatenate((bdata_raw[subj][8][i],bdata_raw[subj][4][i]))
        bdata[subj][4][i] = np.concatenate((bdata_raw[subj][9][i],bdata_raw[subj][5][i]))


# zscore first three participants
for subj in range(len(bdata) - 1):
    for roi in range(len(bdata[subj])):
        mean = np.mean(bdata[subj][roi])
        std  = np.std(bdata[subj][roi])
        for vect in range(len(bdata[subj][roi])):
            bdata[subj][roi][vect] = (bdata[subj][roi][vect] - mean) / std

# Separately zscore and combine final participant's vectors (was getting scipy err)
last_part = np.zeros((5, 2904), dtype = object)
for roi in range(len(last_part)):
    for idx, vect in enumerate(bdata[3][roi]):
        if vect.size != 0:
            last_part[roi][idx] = vect
# combine last participant back into bdata 
for roi in range(len(last_part)):
    mean = np.mean(last_part[roi])
    std  = np.std(last_part[roi])
    for vect in range(len(last_part[roi])):
        bdata[3][roi][vect] = (last_part[roi][vect] - mean) / std


# Read ordered stimulus files, creating the following:
# Create Dict in form of: { filename: [subj0_img_idx, ... , subj3_img_idx] }
# if last subj didn't view the image, array length will be 3
# Create ordered list of first subject's viewing (same order as saved activations) 
subj_stim_dict = defaultdict()
subj0_stim_list = []
for subj in range(4):
    fo = open("./BOLD5000_data_dpml_good/subj" + str(subj+1) + "_stimlist.txt")
    print("Reading: ", fo.name)
    
    stim = fo.readline()
    counter = 0 
    while stim != "":
        stim = stim.rstrip('\n')

        if stim in subj_stim_dict:
            subj_stim_dict[stim].append(counter)
        else: 
            subj_stim_dict[stim] = [counter]
        
        if subj == 0: subj0_stim_list.append(stim)

        stim = fo.readline()
        counter += 1 

    fo.close()


# Re-order and concatenate the brain vectors  
ordered_bdata = np.zeros((5, 4916), dtype = object)
for image_idx, image in enumerate(subj0_stim_list):
    for roi in range(5):
        roi_per_image = []
        for subj, img_idx in enumerate(subj_stim_dict[image]):
            roi_per_image.append(bdata[subj][roi][img_idx])
        
        ordered_bdata[roi][image_idx] = [item for sublist in roi_per_image for item in sublist]


# Load the pretrained model
model = models.resnet18(pretrained=True)
model.eval()

# Requisite transforms
scaler = transforms.Scale((224, 224))
normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
to_tensor = transforms.ToTensor()

# Set model to evaluation mode
model.eval()


# Generate Activations and save activations of each layer to files in form of
# [('image name', [activation array]), ... ] and saved in file named: 
# img_actv_[#from]_upto_[#to].npy
counter = 0
chunked_tuples = list()

if generate_activations: 
    for image_name in subj0_stim_list:
        for filename in glob.glob('./Presented_Stimuli/*/'+ image_name): 

            img = Image.open(filename)            
            t_img = Variable(normalize(to_tensor(scaler(img))).unsqueeze(0))
            
            feature_all=np.array([])

            for i in range(1,10):
                # define which layer you want to extract features from
                layer=i
                # set feature extractor 
                feature_extractor = torch.nn.Sequential(*list(model.children())[:(layer*-1)])
                
                # get feature
                feature_vec = feature_extractor(t_img).data.numpy().squeeze()

                # flatten layers and concatenate
                feature_vec=feature_vec.flatten()                
                feature_all=np.concatenate((feature_all,feature_vec))


            img.close()
            chunked_tuples.append((image_name, feature_all))

            counter += 1
            if save_activations:
                if counter > 0 and not counter % 20: 
                    print ("Processed ", counter, " images")
                    np.save(save_location + 'img_actv_out_'  
                             + str(counter - len(chunked_tuples)) + "_upto_" + str(counter) + ".npy", chunked_tuples)
                    print ("Completed saving") 
                    chunked_tuples = list()
                    
    # Save final batch of images (the final 16 in 4916 imgs)
    if save_activations:
        print("Saving final batch of activations")
        np.save(save_location + 'img_actv_out_' + str(counter-len(chunked_tuples)) 
                + "_upto_" + str(counter) + ".npy", chunked_tuples)


# Loads files one by one by the date they were created
if load_activations:
    img_actv_tuples = list()
    files = glob.glob(save_location + "*")
    files.sort(key=os.path.getmtime)
    for file in files:
        print("Reading: ", file)
        tuple_chunk = np.load(file, allow_pickle = True)
        for img, actv in tuple_chunk:
            img_actv_tuples.append((img, actv))

