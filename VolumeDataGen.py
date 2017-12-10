from __future__ import absolute_import
from __future__ import print_function

import numpy as np
from scipy.ndimage.interpolation import map_coordinates
from scipy.ndimage.filters import gaussian_filter
from scipy import linalg
import scipy.ndimage as ndi
# from six.moves import range
import os
import threading

from keras import backend as K

def deform(x, y, out_augmentStatus, alpha, sigma, random_state=10):
    if random_state is None:
        random_state = np.random.RandomState(None)
    
    shape = x.shape[1:] # first dim is the channels
    random_state = np.random.RandomState(random_state)
    
    ds = [gaussian_filter((np.random.rand(*shape)*2-1),sigma,mode='constant',cval=0)*alpha for _ in range(len(shape))]

    xis = tuple([np.arange(val) for val in shape])
    
    grid = np.meshgrid(*xis, indexing='ij')
    
    indices = [np.reshape(grd+d,(-1,1)) for grd,d in zip(grid,ds)]
    for ind, x_channel in enumerate(x):
        x[ind] = map_coordinates(x_channel,indices,order=1).reshape(shape)
    
    if isinstance(y, list):
        for ind, y_ in enumerate(y):
            if out_augmentStatus[ind]:
                for ind_ch, y_channel in enumerate(y_):
                    y[ind][ind_ch] = map_coordinates(y_channel,indices,order=1).reshape(shape)

    else:
        for ind, y_channel in enumerate(y):
            y[ind] = map_coordinates(y_channel,indices,order=1).reshape(shape)
    return x,y

def get_rotMatrix(theta,vector=[0,0,1]):
    # TODO make it invariant to 3D and 2D
    l=vector[0]
    m=vector[1]
    n=vector[2]
    return np.array([[(l*l)*(1-np.cos(theta))+1*np.cos(theta), (m*l)*(1-np.cos(theta))-n*np.sin(theta), (n*l)*(1-np.cos(theta))+m*np.sin(theta), 0],
                     [(l*m)*(1-np.cos(theta))+n*np.sin(theta), (m*m)*(1-np.cos(theta))+1*np.cos(theta), (n*m)*(1-np.cos(theta))-l*np.sin(theta), 0],
                     [(l*n)*(1-np.cos(theta))-m*np.sin(theta), (m*n)*(1-np.cos(theta))+l*np.sin(theta), (n*n)*(1-np.cos(theta))+1*np.cos(theta), 0],
                     [0, 0, 0, 1]])
    
#%%
def random_channel_shift(x, intensity, channel_index=0):
    x = np.rollaxis(x, channel_index, 0)
    min_x, max_x = np.min(x), np.max(x)
    channel_images = [np.clip(x_channel + np.random.uniform(-intensity, intensity), min_x, max_x)
                      for x_channel in x]
    x = np.stack(channel_images, axis=0)
    x = np.rollaxis(x, 0, channel_index+1)
    return x


def transform_matrix_offset_center(matrix, shape):
    center = [float(x) / 2 + 0.5 for x in shape]
    offset_matrix = np.eye(len(shape)+1)
    reset_matrix = np.eye(len(shape)+1)
    for ax, cnt in enumerate(center):
        offset_matrix[ax,-1] = cnt
        reset_matrix[ax,-1] = -cnt
    transform_matrix = np.dot(np.dot(offset_matrix, matrix), reset_matrix)
    return transform_matrix


def apply_transform(x, transform_matrix, channel_index=0, fill_mode='nearest', cval=0.):
#    print(x.shape)
    x = np.rollaxis(x, channel_index, 0)
    final_affine_matrix = transform_matrix[:-1, :-1]
    final_offset = transform_matrix[:-1, -1]
    channel_images = [ndi.interpolation.affine_transform(x_channel, final_affine_matrix,
                      final_offset, order=1, mode=fill_mode, cval=cval) for x_channel in x ]
    x = np.stack(channel_images, axis=0)
    x = np.rollaxis(x, 0, channel_index+1)
    return x


def flip_axis(x, axis):
    x = np.asarray(x).swapaxes(axis, 0)
    x = x[::-1, ...]
    x = x.swapaxes(0, axis)
    return x


def array_to_img(x, scale=True):
    from PIL import Image
    if scale:
        x += max(-np.min(x), 0)
        x /= np.max(x)
        x *= 255
    if x.shape[2] == 3:
        # RGB
        return Image.fromarray(x.astype('uint8'), 'RGB')
    elif x.shape[2] == 1:
        # grayscale
        return Image.fromarray(x[:, :, 0].astype('uint8'), 'L')
    else:
        raise Exception('Unsupported channel number: ', x.shape[2])


def img_to_array(img):
    # image has dim ordering (height, width, channel)
    x = np.asarray(img, dtype='float32')
    if len(x.shape) == 3:
        x = x.transpose(2, 0, 1)
    elif len(x.shape) == 2:
        x = x.reshape((x.shape[0], x.shape[1], 1))
    else:
        raise Exception('Unsupported image shape: ', x.shape)
    return x


class VolumeDataGenerator(object):
    '''Generate minibatches with real-time data augmentation.
    Assume X is train img, Y is train label (same size as X with only 0 and 255 for values)
    # Arguments
        nb_inputs: int, determines whether you have multiple inputs or not:
            0: single input
            n: n inputs
        nb_outputs: boolean, determines whether you have multiple outputs or not:
            0: single output
            n: n outputs
        featurewise_center: set input mean to 0 over the dataset. Only to X
        samplewise_center: set each sample mean to 0. Only to X
        featurewise_std_normalization: divide inputs by std of the dataset. Only to X
        samplewise_std_normalization: divide each input by its std. Only to X
        zca_whitening: apply ZCA whitening. Only to X
        rotation_range: degrees (0 to 180). To X and Y
        dims_shift_range: fraction of total lenght. To X and Y
        shear_range: shear intensity (shear angle in radians). To X and Y
        zoom_range: amount of zoom. if scalar z, zoom will be randomly picked
            in the range [1-z, 1+z]. A sequence of two can be passed instead
            to select this range. To X and Y
        channel_shift_range: shift range for each channels. Only to X
        fill_mode: points outside the boundaries are filled according to the
            given mode ('constant', 'nearest', 'reflect' or 'wrap'). Default
            is 'nearest'. For Y, always fill with constant 0
        cval: value used for points outside the boundaries when fill_mode is
            'constant'. Default is 0.
        horizontal_flip: whether to randomly flip images horizontally. To X and Y
        vertical_flip: whether to randomly flip images vertically. To X and Y
        rescale: rescaling factor. If None or 0, no rescaling is applied,
            otherwise we multiply the data by the value provided (before applying
            any other transformation). Only to X
        volumetric_outs: a list of bool determining which outputs are volumetric
            and need to be augmented
    '''
    def __init__(self,
                 nb_inputs=0,
                 nb_outputs=0,
                 featurewise_center=False,
                 samplewise_center=False,
                 featurewise_std_normalization=False,
                 samplewise_std_normalization=False,
                 zca_whitening=False,
                 dims_rotation_range=[0.,0.,0.],
                 dims_shift_range=[0.,0.,0.],
                 shear_range=0.,
                 zoom_range=0.,
                 channel_shift_range=0.,
                 fill_mode='nearest',
                 cval=0.,
                 flips=[0,0,0],
                 deformation=[0.,0.],
                 rescale=None,
                 view='axial',
                 out_augmentStatus=[True,True,False],
                 out_isInteger=[False,True,False],
                 is_autoencoder=True):
        self.__dict__.update(locals())
        self.mean = None
        self.std = None
        self.principal_components = None
        self.rescale = rescale
        self.nb_inputs = nb_inputs
        self.nb_outputs = nb_outputs
        self.out_augmentStatus = out_augmentStatus
        self.out_isInteger = out_isInteger
        self.is_autoencoder = is_autoencoder
        self.deformation = deformation
        if len(dims_rotation_range) not in [1,3]:
            raise Exception('dims_rotation_range should be a list of 3 elements (for 3D input)'
                            'or 1 element (for 2D input). Received arg: ', dims_rotation_range)
        if len(dims_shift_range) not in [2,3]:
            raise Exception('dims_shift_range should be a list of 3 elements (for 3D input)'
                            'or 2 element (for 2D input). Received arg: ', dims_rotation_range)

        self.channel_index = len(dims_shift_range)+1

        if np.isscalar(zoom_range):
            self.zoom_range = [1 - zoom_range, 1 + zoom_range]
        elif len(zoom_range) == 2:
            self.zoom_range = [zoom_range[0], zoom_range[1]]
        else:
            raise Exception('zoom_range should be a float or '
                            'a tuple or list of two floats. '
                            'Received arg: ', zoom_range)

    def flow(self, X, y=None, batch_size=32, shuffle=False,
             seed=None, save_to_dir=None, save_prefix='', save_format='jpeg'):
        return NumpyArrayIterator(
            X, y, self,
            batch_size=batch_size, shuffle=shuffle, seed=seed,
            save_to_dir=save_to_dir, save_prefix=save_prefix, save_format=save_format)

    def standardize(self, x):
        # Only applied to X
        if self.rescale:
            x *= self.rescale
        # x is a single image, so it doesn't have image number at index 0
        img_channel_index = self.channel_index - 1
        if self.samplewise_center:
            x -= np.mean(x, axis=img_channel_index, keepdims=True)
        if self.samplewise_std_normalization:
            x /= (np.std(x, axis=img_channel_index, keepdims=True) + 1e-7)

        if self.featurewise_center:
            x -= self.mean
        if self.featurewise_std_normalization:
            x /= (self.std + 1e-7)

        if self.zca_whitening:
            flatx = np.reshape(x, (x.size))
            whitex = np.dot(flatx, self.principal_components)
            x = np.reshape(whitex, (x.shape[0], x.shape[1], x.shape[2]))

        return x

    def random_transform(self, x, y):
        # Need to modify to transform both X and Y ---- to do
        # x is a single image, so it doesn't have image number at index 0
        img_channel_index = self.channel_index - 1 # Because the batch is gone hence '-1'
        shape = [x.shape[ind] for ind in range(len(x.shape)) if ind!=img_channel_index]
        # use composition of homographies to generate final transform that needs to be applied
        
        ############################################################################
        ################################# Rotation #################################
        ############################################################################
        rotation_matrix = []
        rotation_matrix.append(np.eye(len(shape)+1))
        for axis , rot_range in enumerate(self.dims_rotation_range):
            vector=[0,0,0]
            if rot_range:
                theta = np.pi / 180 * np.random.uniform(-rot_range, rot_range)
            else:
                theta = 0
            vector[-axis-1]=1
            rotation_matrix.append(get_rotMatrix(theta,vector))
        
        for mat in rotation_matrix[1:]:
            rotation_matrix[0] = np.dot(rotation_matrix[0],mat)
        rotation_matrix = rotation_matrix[0]
        
        ############################################################################
        ############################### Translation ################################
        ############################################################################

        t = []
        for dim , shift_range in enumerate(self.dims_shift_range):
            if shift_range:
                t.append(np.random.uniform(-shift_range, shift_range) * shape[dim])
            else:
                t.append(0)
        translation_matrix = np.eye(len(shape)+1)
        
        for ind, t_x in enumerate(t):
            translation_matrix[ind,-1] = t_x
        
#        print('Translation',translation_matrix)
#        print('Rotation',rotation_matrix)
        
        transform_matrix = np.dot(rotation_matrix, translation_matrix)
        
#        print('Final Transform',transform_matrix)
        transform_matrix = transform_matrix_offset_center(transform_matrix, shape)
        
#        print('Final',transform_matrix)
        x = apply_transform(x, transform_matrix, img_channel_index,
                            fill_mode=self.fill_mode, cval=self.cval )
        
        ###########################################################################
        ########################## Apply Transformation ###########################
        ###########################################################################
        # For y, mask data, fill mode constant, cval = 0
        if isinstance(y,list):
            for ind, y_ in enumerate(y):
                if self.out_augmentStatus[ind]:
                    y[ind] = apply_transform(y_.astype('float32'), transform_matrix, img_channel_index,
                                             fill_mode="constant", cval=0 )
        else:
            y = apply_transform(y.astype('float32'), transform_matrix, img_channel_index,
                                fill_mode="constant", cval=0 )
        
        if self.channel_shift_range != 0:
            x = random_channel_shift(x, self.channel_shift_range, img_channel_index)
        
        
        
        
        # SwapAxes so that channels are first
        x = np.rollaxis(x, img_channel_index, 0)
        if isinstance(y,list):
            for ind, y_ in enumerate(y):
                if self.out_augmentStatus[ind]:
                    y[ind] = np.rollaxis(y_, img_channel_index, 0)
        else:
            y = np.rollaxis(y, img_channel_index, 0)
        
        ############################################################################
        ################################### Flip ###################################
        ############################################################################
        for dim, dim_flp in enumerate(self.flips):
            if dim_flp and np.random.random() < 0.5:
                x = flip_axis(x, dim+1)
                if isinstance(y,list):
                    for ind, y_ in enumerate(y):
                        if self.out_augmentStatus[ind]:
                            y[ind] = flip_axis(y_, dim+1)
                else:
                    y = flip_axis(y, dim+1)
                    
        ############################################################################
        ################################## Deform ##################################
        ############################################################################
        if self.deformation[0] and self.deformation[1] and np.random.random() < 0.5:
            x, y = deform(x, y, self.out_augmentStatus, self.deformation[0], self.deformation[1], random_state=10)

                    
        ############################################################################
        ################################### VIEW ###################################
        ############################################################################
        # TODO make it a true sagittal
        if self.view=='sagital': # TODO make an exception for 2D cases
            x = np.rollaxis(x, 2,4)
            if isinstance(y,list):
                for ind, y_ in enumerate(y):
                    if self.out_augmentStatus[ind]:
                        y[ind] = np.rollaxis(y_, 2, 4)
            else:
                y = np.rollaxis(y, 2,4)
        elif self.view=='coronal':
            x = np.rollaxis(x, 1, 4)
            if isinstance(y,list):
                for ind, y_ in enumerate(y):
                    if self.out_augmentStatus[ind]:
                        y[ind] = np.rollaxis(y_, 1, 4)
            else:
                y = np.rollaxis(y, 1, 4)
        else:
            pass # default view is assumed to be axial
        
        # Axis back to normal
        x = np.rollaxis(x, 0, img_channel_index+1)
        if isinstance(y,list):
            for ind, y_ in enumerate(y):
                if self.out_augmentStatus[ind]:
                    if self.out_isInteger[ind]:
                        y[ind] = np.round(np.rollaxis(y_, 0, img_channel_index+1)).astype('uint8')
        else:
            y = np.rollaxis(y, 0, img_channel_index+1)

        # TODO: channel-wise normalization
        # barrel/fisheye
        return x, y

    def fit(self, X,
            augment=False,
            rounds=1,
            seed=None):
        '''Required for featurewise_center, featurewise_std_normalization
        and zca_whitening.
        # Arguments
            X: Numpy array, the data to fit on.
            augment: whether to fit on randomly augmented samples
            rounds: if `augment`,
                how many augmentation passes to do over the data
            seed: random seed.
        # Only applied to X
        '''
        X = np.copy(X)
        if augment:
            aX = np.zeros(tuple([rounds * X.shape[0]] + list(X.shape)[1:]))
            for r in range(rounds):
                for i in range(X.shape[0]):
                    aX[i + r * X.shape[0]] = self.random_transform(X[i])
            X = aX

        if self.featurewise_center:
            self.mean = np.mean(X, axis=0)
            X -= self.mean

        if self.featurewise_std_normalization:
            self.std = np.std(X, axis=0)
            X /= (self.std + 1e-7)

        if self.zca_whitening:
            flatX = np.reshape(X, (X.shape[0], X.shape[1] * X.shape[2] * X.shape[3]))
            sigma = np.dot(flatX.T, flatX) / flatX.shape[1]
            U, S, V = linalg.svd(sigma)
            self.principal_components = np.dot(np.dot(U, np.diag(1. / np.sqrt(S + 10e-7))), U.T)


class Iterator(object):

    def __init__(self, N, batch_size, shuffle, seed):
        self.N = N
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.batch_index = 0
        self.total_batches_seen = 0
        self.lock = threading.Lock()
        self.index_generator = self._flow_index(N, batch_size, shuffle, seed)

    def reset(self):
        self.batch_index = 0

    def _flow_index(self, N, batch_size=32, shuffle=False, seed=None):
        # ensure self.batch_index is 0
        self.reset()
        while 1:
            if self.batch_index == 0:
                index_array = np.arange(N)
                if shuffle:
                    if seed is not None:
                        np.random.seed(seed + self.total_batches_seen)
                    index_array = np.random.permutation(N)

            current_index = (self.batch_index * batch_size) % N
            if N >= current_index + batch_size:
                current_batch_size = batch_size
                self.batch_index += 1
            else:
                current_batch_size = N - current_index
                self.batch_index = 0
            self.total_batches_seen += 1
            yield (index_array[current_index: current_index + current_batch_size],
                   current_index, current_batch_size)

    def __iter__(self):
        # needed if we want to do something like:
        # for x, y in data_gen.flow(...):
        return self

    def __next__(self, *args, **kwargs):
        # ?
        return self.next(*args, **kwargs)


class NumpyArrayIterator(Iterator):

    def __init__(self, X, y, image_data_generator,
                 batch_size=32, shuffle=False, seed=None,
                 save_to_dir=None, save_prefix='', save_format='jpeg'):
        if isinstance(y, list):
            for y_ in y:
                if len(X) != len(y_):
                    raise Exception('X (images tensor) and y (labels) '
                                    'should have the same length. '
                                    'Found: X.shape = %s, y.shape = %s' % (np.asarray(X).shape, np.asarray(y_).shape))
        elif len(X) != len(y):
            raise Exception('X (images tensor) and y (labels) '
                            'should have the same length. '
                            'Found: X.shape = %s, y.shape = %s' % (np.asarray(X).shape, np.asarray(y).shape))
        
        self.X = X
        self.y = y
        self.image_data_generator = image_data_generator
        self.save_to_dir = save_to_dir
        self.save_prefix = save_prefix
        self.save_format = save_format
        super(NumpyArrayIterator, self).__init__(X.shape[0], batch_size, shuffle, seed)

    def next(self):
        # for python 2.x.
        # Keeps under lock only the mechanism which advances
        # the indexing of each batch
        # see http://anandology.com/blog/using-iterators-and-generators/
        with self.lock:
            index_array, current_index, current_batch_size = next(self.index_generator)
        # The transformation of images is not under thread lock so it can be done in parallel
        batch_x = []#TODO np.zeros(tuple([current_batch_size] + list(self.X.shape)[1:]))
        if isinstance(self.y,list):
            batch_y = []
            for y_ in self.y:
                batch_y.append(list())#TODO np.zeros(tuple([current_batch_size] + list(y_.shape)[1:])))
        else:
            batch_y = [] #TODO np.zeros(tuple([current_batch_size] + list(self.y.shape)[1:]))
        # Load the data data Outside the for loop. It's much faster when dealing with HDF5
        self_X = self.X[index_array]
        if isinstance(self.y,list):
            self_y = []
            for y_ in self.y:
                self_y.append(y_[index_array])
        else:
            self_y = self.y[index_array]
        
        # Itterate over the batch
        for i, j in enumerate(index_array):
            x = self_X[i]
            if isinstance(self_y,list):
                label = []
                for y_ in self_y:
                    label.append(y_[i])
            else:
                label = self_y[i]
            x, label = self.image_data_generator.random_transform(x.astype('float32'),
                                                                  label)
            x = self.image_data_generator.standardize(x)
            batch_x.append(x)
#            batch_x[i] = x
            if isinstance(label,list):
                for ind, lbl in enumerate(label):
                    batch_y[ind].append(lbl)
#                    batch_y[ind][i] = lbl #TODO d
            else:
#                batch_y[i] = label # TODO d
                batch_y.append(label)
        batch_x = np.asarray(batch_x)
        if isinstance(label,list):
            for ind, stuf in enumerate(batch_y):
                batch_y[ind] = np.asarray(stuf)
        else:
            batch_y = np.asarray(batch_y)
        
        # List the outputs
        if self.image_data_generator.nb_inputs>1:
            inputs = []
            if batch_x.shape[self.image_data_generator.channel_index] % self.image_data_generator.nb_inputs:
                raise Exception("Number of channels doesn't match the number of inputs (different number of channels for individual inputs is not supported yet)")
            else:
                nb_channels_4inputs = batch_x.shape[self.image_data_generator.channel_index] / self.image_data_generator.nb_inputs
                input_channels = []
                for i in xrange(batch_x.shape[self.image_data_generator.channel_index]):
                    if nb_channels_4inputs == 1:
                        inputs.append(np.expand_dims(np.rollaxis(batch_x,self.image_data_generator.channel_index,0)[i],-1))
                    else:
                        input_channels.append(np.rollaxis(batch_x,self.image_data_generator.channel_index,0)[i])
                        if (i+1) % nb_channels_4inputs == 0:
                            inputs.append(np.rollaxis(np.asarray(input_channels),0,self.image_data_generator.channel_index+1))
                            input_channels = []
        else:
            inputs = batch_x
            
        if self.image_data_generator.nb_outputs>1:
            outputs = []
            if batch_y.shape[self.image_data_generator.channel_index] % self.image_data_generator.nb_outputs:
                raise Exception("Number of channels doesn't match the number of outputs (different number of channels for individual outputs is not supported yet)")
            else:
                nb_channels_4outputs = batch_y.shape[self.image_data_generator.channel_index] / self.image_data_generator.nb_outputs
                output_channels = []
                for i in xrange(batch_y.shape[self.image_data_generator.channel_index]):
                    if nb_channels_4outputs == 1:
                        outputs.append(np.expand_dims(np.rollaxis(batch_y,self.image_data_generator.channel_index,0)[i],-1))
                    else:
                        output_channels.append(np.rollaxis(batch_y,self.image_data_generator.channel_index,0)[i])
                        if (i+1) % nb_channels_4outputs == 0:
                            outputs.append(np.rollaxis(np.asarray(output_channels),0,self.image_data_generator.channel_index+1))
                            output_channels = []
        else:
            outputs = batch_y
        
        if self.image_data_generator.is_autoencoder:
            if isinstance(batch_y,list): # TODO check for batch_x to see if it is a list
                batch_y.append(batch_x)
            else:
                batch_y = [batch_y, batch_x]
        return inputs, outputs
    
    
    
    
    
    
    
    
    
    
