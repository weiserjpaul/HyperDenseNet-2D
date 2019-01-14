from keras.utils import to_categorical

import numpy as np
import random


def generator(images, masks, batchsz=10, input_shape=(1,32,32), output_shape=(2,14,14),
              img_shape=(210,238,200)):
    while True:
        data = np.zeros((batchsz,input_shape[0],input_shape[1],input_shape[2]))
        labels = np.zeros((batchsz,output_shape[1],output_shape[2]))

        for i in range(batchsz):
            index_patient = random.randrange(len(masks))
            
            index_cut = random.randrange(img_shape[0])
            count = 0
            while not np.any(masks[index_patient, index_cut,:,:]):
                index_cut = random.randrange(img_shape[0])

            dim1, dim2 = return_coords_all(masks, input_shape=input_shape,
                                       output_shape=output_shape, img_shape=img_shape,
                                      index_patient=index_patient, index_cut=index_cut)
            
            data[i] = images[index_patient,:,index_cut,
                             dim1:dim1+input_shape[1], dim2:dim2+input_shape[2]]
            shift1 = int((input_shape[1]-output_shape[1])/2)
            shift2 = int((input_shape[2]-output_shape[2])/2)
            labels[i] = masks[index_patient,index_cut,
                              dim1+shift1:dim1+shift1+output_shape[1], 
                              dim2+shift2:dim2+shift2+output_shape[2]]
        labels = to_categorical(labels, num_classes=output_shape[0])
        labels = np.moveaxis(labels,-1,1)
        yield tuple([data, labels])
        

    
def return_coords_brain_and_non_brain(masks, index_patient, index_cut,  
                                      input_shape=(1,32,32), output_shape=(2,14,14), 
                                      img_shape=(210,238,200)):
    """Returns coordinates such that the patch contains brain
    and non-brain areas"""
    
    dim1 = random.randrange(img_shape[1] - input_shape[1])
    dim2 = random.randrange(img_shape[2] - input_shape[2])
    shift1 = int((input_shape[1]-output_shape[1])/2)
    shift2 = int((input_shape[2]-output_shape[2])/2)
    tmp_mask = masks[index_patient,index_cut, 
                     dim1+shift1:dim1+shift1+output_shape[1], 
                     dim2+shift2:dim2+shift2+output_shape[2]]
    
    while (not np.any(tmp_mask)) or np.all(tmp_mask):
        dim1 = random.randrange(img_shape[1] - input_shape[1])
        dim2 = random.randrange(img_shape[2] - input_shape[2])
        tmp_mask = masks[index_patient,index_cut, 
                         dim1+shift1:dim1+shift1+output_shape[1], 
                         dim2+shift2:dim2+shift2+output_shape[2]]
    
    return dim1, dim2



def return_coords_all(masks, index_patient, index_cut,
                      input_shape=(1,32,32), output_shape=(2,14,14), img_shape=(210,238,200)):
    """Returns random coordinates"""
    dim1 = random.randrange(img_shape[1] - input_shape[1])
    dim2 = random.randrange(img_shape[2] - input_shape[2])
    return dim1, dim2


