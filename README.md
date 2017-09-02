# Keras_SegmentationDataGenerator

The segmentation data generator for Keras. It supports multi input and multi output plus a faster implementation of hdf5 datasets handling.

The original implementation of SegDataGen was the work of https://github.com/ahundt. I was in a hurry so I had to develop it independently from the original developer.

This code performs data augmentation for segmentation tasks. In other words, it performs the same augmentation on the images and the masks.

Two main difference between my code and ahundt's are:
1, I made a small change so that the code would perform faster when the input is a HDF5matrix dataset. For a batch of 100, this code takes 0.05s to load the data (loads the entire batch at once) while loading one by one takes 1.5s
and 2, It also suppports multi input and multi output. when instantiating the SegmentationDataGeneratr, specify the number of inputs and outputs.

In case of multi-input/output, the input datasets have to be a multi channel format. For example, in case of a multi-input structure with 4 inputs, if each of the inputs has 3 channels (e.g. shape (None,128,128,3) for each input) then the input to the flow method should be (None,128,128,4*3). The code then automatically separates channels 0:3, 3:6, ..., 9:12 and considers each of them as inputs.

TO DO:
1, Add more augmentation methods, such as:
    a, deformations
    b, adding additive noise to inputs
    c, adding error to the edges of the ground truth segmentations
    d, blurring inputs
    e, adding noises specific to medical imaging modalities like MRI (spatial gaussian), Ultrasound (speckle, shadowing), OCT (speckle,             shadowing), and CT (simulation of a metal in the body). However, speed is an issue for these noises.
    f, histogram normalization
2, Adding the flow_from_directory for .tif, .npy, .npz, .nii.gz and maybe even .mhd

Gradually, I will provide more details on everything.
