# Inverse World -- BOLD5000 Brain Prediction

Requires: Python3, AFNI installed, FSL installed

After cloning, run: 

$ pip install -r requirements.txt

After AFNI installation, to reduce output:

$ echo "AFNI_NIFTI_TYPE_WARN = NO" >> ~/.afnirc

usage: python bold_brains.py [-h] --input input_dir [--output output_dir]
                      		 [--true true_brain_dir] [--sigma sigma] 
