import numpy as np
import random
from math import ceil


def destruct(image, mask, imagesz = (210, 238, 200), inputsz = (1, 32, 32), outputsz = (2,14,14)):
    """The input is an image and a mask of shape (modalities, imagesz) and (imagesz).
    The method takes an arbitrary slice of the first dimension (imagesz[1], imagesz[2])
    and decompses it into patches of size (inputsz) and (outputsz).
    Finally it return a numpy array of shape (number of patches, inputsz) and 
    (number of patches, outputsz)"""
    
    cut_mask = np.zeros(imagesz)
    while not np.any(cut_mask):
        index = random.randrange(imagesz[0])
        cut_img = image[:,index]
        cut_mask = mask[index]

    n_height = ceil(imagesz[1]/outputsz[1])
    n_width = ceil(imagesz[2]/outputsz[2])
    n_patches = n_height * n_width
    patches_mask = np.zeros((n_patches, outputsz[1],outputsz[2]))
    patches_image = np.zeros((n_patches, inputsz[0], inputsz[1], inputsz[2]))

    big_cut_mask = np.zeros((imagesz[1] + inputsz[1]*2, imagesz[2] + inputsz[2]*2))
    big_cut_mask[inputsz[1]:inputsz[1]+imagesz[1], inputsz[2]:inputsz[2]+imagesz[2]] = cut_mask
    big_cut_img = np.zeros((inputsz[0], imagesz[1] + inputsz[1]*2, imagesz[2] + inputsz[2]*2))
    big_cut_img[:, inputsz[1]:inputsz[1]+imagesz[1], inputsz[2]:inputsz[2]+imagesz[2]] = cut_img

    for i in range(n_height):
        for j in range(n_width):
            k = i*outputsz[1]
            l = j*outputsz[2]
            shift_k = int((inputsz[1]-outputsz[1])/2)
            shift_l = int((inputsz[2]-outputsz[2])/2)
            patches_mask[i*n_width+j] = big_cut_mask[inputsz[1]+k:inputsz[1]+k+outputsz[1],
                                                     inputsz[2]+l:inputsz[2]+l+outputsz[2]]
            patches_image[i*n_width+j] = big_cut_img[:,inputsz[1]+k-shift_k:inputsz[1]+k+inputsz[1]-shift_k,
                                                    inputsz[2]+l-shift_l:inputsz[2]+l+inputsz[2]-shift_l]
    
    patches_mask = np.reshape(patches_mask,(n_patches, 1, outputsz[1],outputsz[2]))
    return patches_image, patches_mask


def reconstruct(patches, imagesz = (210, 238, 200), inputsz = (32, 32), outputsz = (14, 14)):
    """Input is an numpy array of size (number of patches, outputsz), which
    is a number of patches of an image. The method composes the patches to an image.
    The shape of the image is (imagesz[1], imagesz[2])"""
    
    n_height = ceil(imagesz[1]/outputsz[0])
    n_width = ceil(imagesz[2]/outputsz[1])
    reconstr = np.zeros((n_height*outputsz[0], n_width*outputsz[1]))
    for i in range(n_height):
        for j in range(n_width):
            k = i*outputsz[0]
            l = j*outputsz[1]
            reconstr[k:k+outputsz[0], l:l+outputsz[1]] = patches[i*n_width+j]
    reconstr = reconstr[:imagesz[1],:imagesz[2]]
    return reconstr


