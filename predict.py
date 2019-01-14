from keras.models import load_model
from NetworkConfigurations import config
from src.load_data import load_data, get_files
from src.PredictSave import predict, save_images
from src.loss_functions import weighted_dice_coefficient_loss2d


model = load_model(config["path_to_model"], 
                   custom_objects={'weighted_dice_coefficient_loss2d': weighted_dice_coefficient_loss2d})

paths, masks_paths = get_files(path_to_data = config["path_to_data"])

images, masks = load_data(paths = paths, masks_paths = masks_paths, 
                          train_modalities = config["train_modalities"], 
                          image_shape = config["image_shape"], 
                          train_validate_rate = config["train_validate_rate"])

predictions_0, predictions_1, constr_masks = predict(model=model, images=images, masks=masks,
                                                     image_shape = config["image_shape"], 
                                                     input_shape = config["input_shape"])

save_images(predictions=predictions_1, constr_masks=constr_masks)

