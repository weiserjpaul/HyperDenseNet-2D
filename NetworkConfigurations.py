from keras.optimizers import Adam

config = {}

config["image_shape"] = (210,238,200)
config["activation_name"] = "sigmoid"
config["dropout_rate"] = 0.3
config["n_labels"] = 2
config["optimizer"] = Adam
config["initial_learning_rate"] = 5e-4
config["train_validate_rate"] = 0.85
config["modalities"] = ["images", "radial_dist"]
config["train_modalities"] = config["modalities"]
config["path_to_data"] = "/home/cir/pweiser/neuralnetworks/Data/"
config["path_to_model"] = "models/jupyter_model_7.h5"

config["input_shape"] = (len(config["train_modalities"]), 32, 32)
config["kernelshapes"] = [[3, 3], [3, 3], [3, 3],[3, 3],[3, 3], [3, 3], [3, 3],[3, 3],[3, 3],[1],[1],[1]]
config["numkernelsperlayer"] = [25,25,25,50,50,50,75,75,75,400,200,100]
#config["kernelshapes2d"] = [[3, 3], [3, 3], [3, 3], [3, 3], [3, 3],[3, 3],[1],[1]]
#config["numkernelsperlayer2d"] = [25,25,50,50,75,75,200,100]
#config["loss_function"] = weighted_dice_coefficient_loss2d
config["loss_function"] = "categorical_crossentropy"

config["patience"] = 50
config["steps_per_epoch"] = 300
config["validation_steps"] = 50
config["epochs"] = 500
config["activation_name"] = "sigmoid"
config["dropout_rate"] = 0.3
config["n_labels"] = 2
config["optimizer"] = Adam
config["initial_learning_rate"] = 5e-4
config["train_validate_rate"] = 0.85
