#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Mon Oct  2 19:32:40 2017

@author: mheybpoosh
"""
from glob import glob
import matplotlib.pyplot as plt
from pandas import DataFrame as df
import numpy as np
from tqdm import tqdm
import SimpleITK as sitk
import dicom
import scipy.ndimage
from dicompylercore import dicomparser
import h5py as h5
import math
import os
from sklearn.utils import shuffle
from scipy.misc import imresize
import nibabel as nb
import cv2

#%%
def separate_Volume(volume,gt=False,nb_channels=3,nb_dist=1,ch_index=0):
    """
    Separate the input slices to prepare it for the multi input format. Also separates the outputs.
    
    Arguments:
        volume: np.ndarray
            the input 3D images.
        gt: np.ndarray
            the output slices
        nb_channels: int
            number of channels for each input stack, i.e. number of channels for each input sample
        nb_dist: int
            number of slices between the auxiliary slices to skip.
        ch_index: int
            the dimension where the channels are
    Outputs:
        two numpy arrays, with the required shape
        
    """
    # Move the channel axis to first dimension
    volume = np.moveaxis( volume , ch_index , 0 )
    if isinstance(gt,np.ndarray):
        gt = np.moveaxis( gt , ch_index , 0 )
        # Find nonzero coordinates in the GT to start from there
        nonz_z = np.nonzero(gt)[0]
    else:
        nonz_z = np.asarray([0,volume.shape[0]-1])
    # Determine the number of channel auxiliary channels
    nb_aux = nb_channels // 2
    
    # Store the data
    slices = []
    outs = []
    
    for ind in range(np.min(nonz_z),np.max(nonz_z)+1):
        # Get the input slices
        if ind-(nb_dist*nb_aux) < 0 or (ind+(nb_dist*nb_aux)) > volume.shape[0]-1:
            continue
        slices.append(volume[ind-(nb_dist*nb_aux):ind+(nb_dist*nb_aux)+1:nb_dist])
        
        # Get the outputs
        if isinstance(gt,np.ndarray):
            outs.append(np.expand_dims(gt[ind],0))
        
    slices = np.moveaxis(np.asarray(slices),1,ch_index+1)
    if isinstance(gt,np.ndarray):
        outs = np.moveaxis(np.asarray(outs),1,ch_index+1)
    return slices, outs

def hist_match(source, template):
    """
    Adjust the pixel values of a grayscale image such that its histogram
    matches that of a target image

    Arguments:
    -----------
        source: np.ndarray
            Image to transform; the histogram is computed over the flattened
            array
        template: np.ndarray
            Template image; can have different dimensions to source
    Returns:
    -----------
        matched: np.ndarray
            The transformed output image
    """

    oldshape = source.shape
    source = source.ravel()
    template = template.ravel()

    # get the set of unique pixel values and their corresponding indices and
    # counts
    s_values, bin_idx, s_counts = np.unique(source,
                                            return_inverse=True,
                                            return_counts=True)
    t_values, t_counts = np.unique(template,
                                   return_counts=True)

    # take the cumsum of the counts and normalize by the number of pixels to
    # get the empirical cumulative distribution functions for the source and
    # template images (maps pixel value --> quantile)
    s_quantiles = np.cumsum(s_counts).astype(np.float64)
    s_quantiles /= s_quantiles[-1]
    t_quantiles = np.cumsum(t_counts).astype(np.float64)
    t_quantiles /= t_quantiles[-1]

    # interpolate linearly to find the pixel values in the template image
    # that correspond most closely to the quantiles in the source image
    interp_t_values = np.interp(s_quantiles, t_quantiles, t_values)

    return interp_t_values[bin_idx].reshape(oldshape)

def exclude_Zeros(imgs,value):
    """
    Excludes (crops) the frame around volume.
    Arguments:
        imgs: np.ndarray
            volume
        value: int
            the value that is considered in the frame.
    """
    non_z = np.asarray(np.nonzero(imgs[1]-value)).astype('int')
    start = np.min(non_z,axis=1)
    stop = np.max(non_z,axis=1)
    imgs = imgs[start[0]:stop[0]+1,start[1]:stop[1]+1,start[2]:stop[2]+1].astype('float32')
    return imgs

def cropVolume_Nx( imgs, N, ch_index ):
    """
    Crops the input 3D volumes into a size that is the multiple of N. Useful for segmentation using
    U-Net like structures to make sure that the input image size is the multiple of 2^(the number of
    times there are max poolings in the structure).
    Arguments:
        imgs: np.ndarray
            the input image volume
        N: int
            the value (should be 2^(nb_maxpoolings))
        ch_index: int
            index of the channels
    """
    # Put the channels first
    imgs = np.moveaxis( imgs , ch_index , 0 )
    
    # Determine the output size:
    shape = np.asarray(imgs.shape[ 1 : ])
    out_sz_Nx = ( shape // N ) * N
    
    # Start and Stop index
    start = ( shape - out_sz_Nx ) // 2
    stop = start + out_sz_Nx
    
    # Crop
    imgs = imgs[:,start[0]:stop[0],start[1]:stop[1]].astype('float32')
    
    return np.moveaxis( imgs , 0 , ch_index )

def weighted_avg_and_std(values, weights):
    """
    Return the weighted average and standard deviation.

    values, weights -- Numpy ndarrays with the same shape.
    """
    average = np.average(values, weights=weights)
    variance = np.average((values-average)**2, weights=weights)  # Fast and numerically precise
    return average, math.sqrt(variance)

def normalize(imgs,frame_value=0,ignore_frame=True,clip_vals=[-1,6]):
    if ignore_frame:
        a = np.reshape( imgs , -1 )
        _,madvar = weighted_avg_and_std(a,a!=frame_value)
        
        med = np.median( a )
        imgs = (imgs - med).astype('float64')
        
        imgs /= madvar
        imgs = np.clip(imgs,clip_vals[0],clip_vals[1])
    
    imgs -= imgs.min()
    imgs /= imgs.max()
    
    return imgs

def sepSlices_process(path, out_path, phrase, nb_channels, nb_dist, ch_index, gts=True):
    """
    Separates the slices according to the specified circumstances and processes the data.
    Stores the output slices in the location specified in out_path.
    Arguments:
        path: path of the inputs
        out_path: path to save the outputs
        phrase: the phrase to look for in path
        nb_channels: number of output channels for each slice
        nb_dist: number of 2D frames to put between
        ch_index: the dimension that corresponds to the height or channels
        gts: if True, the input npz files contain a variable named 'gts' that contains ground truth outputs
    """
    # Get list of files
    file_list = glob(path+'*'+phrase)
    for ind , fl in enumerate(tqdm(file_list)):
        # Check to see if it has been a bad CT scan
        name = fl.replace(path,'').replace(phrase,'')

        # Load The Data and crop
        volume = np.load(fl)['imgs'][:, 90:-90, 60:-60]
        volume = cropVolume_Nx( volume, N, ch_index )
        if gts:
            gts = np.load(fl)['gts'][:, 90:-90, 60:-60]
            gts = cropVolume_Nx( gts, N, ch_index )
        
        # Separate Slices
        slc_ls , gt_ls = separate_Volume(volume, gts,nb_channels, nb_dist, ch_index)
        
        # Normalize
        slc_ls = normalize(slc_ls,frame_value)
        
        # Get shapes
        shapes = slc_ls.shape
        
        # Save Patient Slices
        if isinstance(gts,np.ndarray):
            np.savez_compressed(out_path+name,slices=slc_ls,gts=gt_ls,shapes=np.asarray(shapes,dtype='int32'))
            gts=True
        else:
            np.savez_compressed(out_path+name,slices=slc_ls,shapes=np.asarray(shapes,dtype='int32'))
            gts=False

def get_CropInds(slc, b, slope):
    # Get nonzeros
    nonz = np.nonzero(slc)
    # None zero center
    center = np.round(np.asarray([np.median(nonz[0]),np.median(nonz[1])])).astype('int')
    # Get area
    area = slc.sum()
    print(area)
    print(slope)
    print(b)
    # get new ratio
    rat = slope * area + b
    print(rat)
    new_sz = int(np.round(np.sqrt(area/rat)))
    new_sz = np.asarray([new_sz,new_sz])
    
    old_size = slc.shape
    
    # get start and stop indices
    start = center - (new_sz // 2)
    for ind , a in enumerate(start):
        if a < 0:
           start[ind] = 0 
    stop = center + (new_sz // 2)
    for ind , a in enumerate(stop):
        if a > old_size[ind]:
           stop[ind] = old_size[ind]
    
    return start, stop
    
def crop_ratio(slices, segs, out_sz, ch_index, seg_info_Channel, b, slope):
    # Bring channels to first dimension
    slices = np.moveaxis(slices,ch_index,0)
    segs = np.moveaxis(segs,ch_index,0)
    # Get new size
    start, stop = get_CropInds(segs[seg_info_Channel], b, slope)
    # Crop images
    c_seg = segs[:,start[0]:stop[0],start[1]:stop[1]]
    c_slc = slices[:,start[0]:stop[0],start[1]:stop[1]]
    # Resize
    out_slc = np.zeros((slices.shape[0],)+out_sz)
    out_seg = np.zeros((segs.shape[0],)+out_sz)
    for ind , slc in enumerate(c_slc):
        out_slc[ind] = imresize(slc,out_sz)
    for ind , seg in enumerate(c_seg):
        out_seg[ind] = imresize(seg,out_sz)
    return out_slc , out_seg
    
def get_NB_Slices(list_of_files):
    shapes = []
    for fl in list_of_files:
        shapes.append(np.load(fl)['shapes'])
    # Get the number of training Data
    shapes = np.asarray(shapes)
    nb_slcs = shapes[:,0].sum()
    return nb_slcs,shapes[0][1:]

#TODO
def store_SliceInHDF5_Lung1(file_name,in_path,n_out = 1,val_percent=0.85,crop_ratio=False,):
    with h5.File(file_name,'w') as f:
        all_list = shuffle(glob(in_path+'*'))
        nb_train_slices,shapes = get_NB_Slices(all_list[:int(np.round(len(all_list)*val_percent))])
        nb_valid_slices,shapes = get_NB_Slices(all_list[int(np.round(len(all_list)*val_percent)):])
        # Train Data
        train_X_dset = f.create_dataset('train_x',(nb_train_slices,) + tuple(shapes),dtype='float32')
        #TODO: Make it channel-index-aware
        train_Y_dset = f.create_dataset('train_y',(nb_train_slices,) + (n_out,) + tuple(shapes[1:]),dtype='uint8')
        # Validation Data
        val_X_dset = f.create_dataset('valid_x',(nb_valid_slices,) + tuple(shapes),dtype='float32')
        #TODO: Make it channel-index-aware
        val_Y_dset = f.create_dataset('valid_y',(nb_valid_slices,) + (n_out,) + tuple(shapes[1:]),dtype='uint8')
        
        # Get random indices
        train_inds = shuffle(range(nb_train_slices))
        valid_inds = shuffle(range(nb_valid_slices))
        
        # Train Dataset
        count = 0
        for fl in tqdm(all_list[:int(np.round(len(all_list)*val_percent))]):
            # Load data
            slices = np.load(fl)['slices']
            gts = np.load(fl)['gts']
            
            for slc , gt in zip(slices,gts):
                if gt.sum()==0:
                    continue
                if crop_ratio:
                    new_x[count], new_y[count] = crop_ratio(slc, gt, out_sz,
                                                             ch_index,
                                                             seg_info_Channel,
                                                             b, slope)
                train_X_dset[train_inds[count]] = slc
                if n_out == 2:
                    train_Y_dset[train_inds[count]] = np.squeeze(np.asarray([gt==0,gt!=0],dtype='uint8'))
                else:
                    train_Y_dset[train_inds[count]] = np.asarray(gt==0,dtype='uint8')
                count += 1
            
        # Validation Dataset
        count = 0
        for fl in tqdm(all_list[int(np.round(len(all_list)*val_percent)):]):
            slices = np.load(fl)['slices']
            gts = np.load(fl)['gts']
            
            for slc , gt in zip(slices,gts):
                if gt.sum()==0:
                    continue
                val_X_dset[valid_inds[count]] = slc
                if n_out == 2:
                    val_Y_dset[valid_inds[count]] = np.squeeze(np.asarray([gt==0,gt!=0],dtype='uint8'))
                else:
                    val_Y_dset[valid_inds[count]] = np.asarray(gt==0,dtype='uint8')
                count += 1

def resample(image, SliceThickness, PixelSpacing, new_spacing=[1,1,1], interp_order=3):
    # Determine current pixel spacing
    spacing = map(float, ([SliceThickness] + PixelSpacing))
    spacing = np.array(list(spacing))

    resize_factor = spacing / new_spacing
    new_real_shape = image.shape * resize_factor
    new_shape = np.round(new_real_shape)
    real_resize_factor = new_shape / image.shape
    new_spacing = spacing / real_resize_factor
    image = scipy.ndimage.interpolation.zoom(image, real_resize_factor, order=interp_order, mode='nearest')
    
    return image, new_spacing, spacing

def onehot_Volume(vol,lbls):
    # Predefine
    onehot = np.zeros((len(lbls),)+vol.shape,dtype='uint8')
    for ind,lbl in enumerate(lbls):
        onehot[ind] = vol==lbl
    return np.moveaxis(onehot,0,-1)
    
def check_CropIndices(shape, list_of_indices):
    # TODO: give multiple options, on how to treat different situations, like should the size remain constant?
    for ind, dim_inds in enumerate(list_of_indices):
        if dim_inds[0] < 0:
            dim_inds[1] -= dim_inds[0]
            dim_inds[0] -= dim_inds[0]
        elif dim_inds[1] > shape[ind]:
            dim_inds[0] -= (dim_inds[1] - shape[ind])
            dim_inds[1] = shape[ind]
        list_of_indices[ind] = dim_inds
    return list_of_indices

def get_ProcessedVolumes_BraTS17(fl,lbls,bout_shape,h_bout_shape,tout_shape,h_tout_shape,contains_segmentation=False):
    # Get individual files of interest
    all_fls = sorted(glob(fl+'/*.nii.gz'))
    # Load the data
    if contains_segmentation:
        # Get file
        seg_fl = glob(fl+'/*_seg.nii.gz')[0]
        # Load images accordingly
        images = [ nb.load(f).get_data() for f in all_fls if f != seg_fl ]
        # load the segmentation
        seg = nb.load(seg_fl).get_data()
        # Onehot the segmentation labels 
        onehot_seg = onehot_Volume(seg,lbls)
        # Find out where the tumor is
        tumor_nonz = np.nonzero(seg)
        # Find the center of the tumor
        tumor_center = [np.median(a) for a in tumor_nonz]
        # Find the bounding box around tumor
        t_start_stop = [[int(b_cent-half),int(b_cent+half)] for b_cent,half in zip(tumor_center,h_tout_shape)]
        t_start_stop = check_CropIndices(seg.shape,t_start_stop)
        # Crop the Tumor 
        tumor_img = [normalize(img[t_start_stop[0][0]:t_start_stop[0][1],
                           t_start_stop[1][0]:t_start_stop[1][1],
                           t_start_stop[2][0]:t_start_stop[2][1]]) for img in images]
        tumor_seg = onehot_seg[t_start_stop[0][0]:t_start_stop[0][1],
                           t_start_stop[1][0]:t_start_stop[1][1],
                           t_start_stop[2][0]:t_start_stop[2][1]]
    else:
        images = [ nb.load(f).get_data() for f in all_fls ]
    # Get nonezero (croped brain) area size
    try:
        brain_nonz = np.nonzero(images[0])
    except:
        print(fl)
        print(all_fls)
    # Get center
    brain_center = [np.median(a) for a in brain_nonz]
    # Start and Stop
    b_start_stop = [[int(b_cent-half),int(b_cent+half)] for b_cent,half in zip(brain_center,h_bout_shape)]
    # Get/Check Indices 
    b_start_stop = check_CropIndices(images[0].shape,b_start_stop)
    # Crop the Brain
    brain_img = [normalize(img[b_start_stop[0][0]:b_start_stop[0][1],
                       b_start_stop[1][0]:b_start_stop[1][1],
                       b_start_stop[2][0]:b_start_stop[2][1]]) for img in images]
    if contains_segmentation:
        brain_seg = onehot_seg[b_start_stop[0][0]:b_start_stop[0][1],
                           b_start_stop[1][0]:b_start_stop[1][1],
                           b_start_stop[2][0]:b_start_stop[2][1]]
    else:
        tumor_img = 0
        tumor_seg = 0
        brain_seg = 0
        
    return tumor_img, tumor_seg, brain_img, brain_seg

def get_ImageAndGroundTruth( dcms , seg_fls_ls='none' ):
    """
    Given a list of paths to DCM files, and a RTSTRCT DCM file, it returns the 3D volumes.
    
    Arguments:
        dcms: list of str
            list of paths to DCM files containing the scan
        seg_fls_ls: str
            path to the RTSTRCT file
    """
    # Get the 3D CT scan
    image = np.stack([s.pixel_array for s in dcms])
    gts = np.zeros_like( image )
    # Get pixel spacings
    pix_spc = [float(a) for a in dcms[0].PixelSpacing]
    # Get Planes
    if seg_fls_ls != 'none':
        planes = get_Planes( seg_fls_ls )
        if not isinstance(planes,str):
            # X-Y Origin
            xy_cent = [float(a) for a in dcms[0].ImagePositionPatient[:2]]
            
            # Fill in the points in GTS
            keys=[float(a) for a in planes.keys()]
            z_pos = [float(x.ImagePositionPatient[2]) for x in dcms]
            for ind , z in enumerate( z_pos ):
                if z in keys:
                    points = [[int(np.round(np.abs(float(a[0])-xy_cent[0])/pix_spc[0])),int(np.round(np.abs(float(a[1])-xy_cent[1])/pix_spc[0]))] for a in planes['%.2f' % z]]
                    gts[ind] = cv2.fillPoly(gts[ind],[np.asarray(points,dtype='int32')],color=(255))
        else:
            gts = 'none'
    # Add slice spacing to the pix_spc:
    try:
        slice_thickness = np.abs(dcms[0].ImagePositionPatient[2] - 
                                 dcms[1].ImagePositionPatient[2])
    except:
        slice_thickness = np.abs(dcms[0].SliceLocation -
                                 dcms[1].SliceLocation)
    pix_spc += [slice_thickness]
    
    return image , gts , pix_spc

def get_ContourPoints(array):
    """Parses an array of xyz points & returns a array of point dicts."""

    n = 3
    return [array[i:i+n] for i in range(0, len(array), n)]

def get_Planes(seg_fls_ls):
    """
    Takes in the path to RTSTRCT. Returns a dict of contours in different z locations (heights).
    the keys of the output dict are strings corresponding to the z location.
    """
    seg = dicom.read_file(seg_fls_ls)
    # Get the contour sequence for all slices
    seg_data = seg.ROIContourSequence    
    if len(seg_data):
        planes = {}
        for c in seg_data[0].ContourSequence:
            # For each plane, initialize a new plane dict
            plane = {}
        
            # Determine all the plane properties
            plane['type'] = c.ContourGeometricType
            plane['num_points'] = int(c.NumberOfContourPoints)
            plane['data'] = get_ContourPoints(c.ContourData)
        
            # Each plane which coincides with an image slice
            # will have a unique ID
            if 'ContourImageSequence' in c:
                # Add each plane to the planes dict
                # of the current ROI
                z = str(round(plane['data'][0][2], 2)) + '0'
                if z not in planes:
                    planes[z] = []
                planes[z] = plane['data']
    else:
        planes = 'none'
    return planes


def get_ProcessedVolumes_LUNG1(fl,h_shape,clip_vals=[-1,1]):
    # Get volumes
    imgs = np.load(fl)['imgs']
    gts = np.load(fl)['gts']
    # Find the center of data
    center = [np.median(a) for a in np.nonzero(gts)]
    # Find the bounding box around tumor
    t_start_stop = [[int(b_cent-half),int(b_cent+half)] for b_cent,half in zip(center,h_shape)]
    t_start_stop = check_CropIndices(gts.shape,t_start_stop)
    # Crop
    cimgs = imgs[t_start_stop[0][0]:t_start_stop[0][1],
                 t_start_stop[1][0]:t_start_stop[1][1],
                 t_start_stop[2][0]:t_start_stop[2][1]]
    cgts = gts[t_start_stop[0][0]:t_start_stop[0][1],
              t_start_stop[1][0]:t_start_stop[1][1],
              t_start_stop[2][0]:t_start_stop[2][1]]
    cimgs = normalize(cimgs.astype('float64'),frame_value=0,ignore_frame=False,clip_vals=clip_vals)
    return cimgs.astype('float32'), cgts



































































