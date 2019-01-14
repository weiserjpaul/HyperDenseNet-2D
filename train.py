from keras.optimizers import Adam
from src.load_data import load_data, get_files
from src.HyperDenseModel import HyperDenseModel
from src.training import train

from NetworkConfigurations import config


paths, masks_paths = get_files(path_to_data = config["path_to_data"])

images, masks = load_data(paths = paths, masks_paths = masks_paths,
                          train_modalities = config["train_modalities"],
                          image_shape = config["image_shape"],
                          train_validate_rate=0)

model = HyperDenseModel(kernelshapes2d = config["kernelshapes"],
                          numkernelsperlayer2d = config["numkernelsperlayer"],
                          input_shape2d = config["input_shape"],
                          n_labels = config["n_labels"],
                          activation_name = config["activation_name"],
                          dropout_rate = config["dropout_rate"],
                          initial_learning_rate = config["initial_learning_rate"],
                          loss_function = config["loss_function"],
                          optimizer = config["optimizer"])

model = train(model2d=model, images=images, masks=masks,
                image_shape = config["image_shape"],
                input_shape = config["input_shape"],
                train_validate_rate = config["train_validate_rate"],
                patience = config["patience"],
                steps_per_epoch = config["steps_per_epoch"],
                validation_steps = config["validation_steps"],
                epochs = config["epochs"])

model.save(config["path_to_model"])

