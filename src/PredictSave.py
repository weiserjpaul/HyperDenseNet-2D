import numpy as np
import scipy.misc
from src.Decompose import destruct, reconstruct


def predict(model, images, masks, image_shape=(210,238,200), input_shape=(2,32,32)):
    
    output_shape = (int(model.output.shape[1]), int(model.output.shape[2]), int(model.output.shape[3]))
    predictions_0 = np.zeros((len(images), image_shape[1], image_shape[2]))
    predictions_1 = np.zeros((len(images), image_shape[1], image_shape[2]))
    constr_masks = np.zeros((len(images), image_shape[1], image_shape[2]))
    
    for i in range(len(images)):
        patches_image, patches_mask = destruct(images[i], masks[i], 
                                               inputsz=input_shape, outputsz=output_shape)
        pred = model.predict(patches_image)
        predictions_0[i] = reconstruct(pred[:,0,:,:])
        predictions_1[i] = reconstruct(pred[:,1,:,:])
        constr_masks[i] = reconstruct(patches_mask)
    return predictions_0, predictions_1, constr_masks


def save_images(predictions, constr_masks):
    for n in range(len(constr_masks)):
        file_pred = "predictions/pred_"+str(n)+".jpg"
        file_mask = "predictions/mask_"+str(n)+".jpg"
        scipy.misc.toimage(predictions[n]).save(file_pred)
        scipy.misc.toimage(constr_masks[n]).save(file_mask)
