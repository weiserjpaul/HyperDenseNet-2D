from keras.layers import Input, LeakyReLU, Add, UpSampling3D, Activation, SpatialDropout3D
from keras.layers import Conv3D, Softmax
from keras.layers import Concatenate, UpSampling2D, SpatialDropout2D, Conv2D, Cropping2D
from keras.engine import Model
from keras.optimizers import Adam



def HyperDenseModel(kernelshapes2d, numkernelsperlayer2d, input_shape2d = (2,32,32),
                    n_labels = 2, activation_name = "sigmoid", dropout_rate = 0.3, 
                    initial_learning_rate = 5e-4, loss_function = "categorical_crossentropy",
                    optimizer = Adam):
    n_conv_layer = 0
    for kernel in kernelshapes2d:
        if len(kernel) == 2:
            n_conv_layer += 1
    layers = []

    inputs = Input(input_shape2d)
    current_layer = inputs
    layers.append(current_layer)

    for i in range(n_conv_layer):
        current_layer = Conv2D(numkernelsperlayer2d[i], kernelshapes2d[i], strides=(1, 1),
                               padding='valid', activation=activation_name, data_format='channels_first')(current_layer)
        layers.append(current_layer)
        cropped_layers = []
        n_layers = len(layers)
        for count, layer in enumerate(layers):
            cropped_layer = Cropping2D(cropping=(n_layers-1-count), data_format="channels_first")(layer)
            cropped_layers.append(cropped_layer)
        current_layer = Concatenate(axis = 1)(cropped_layers)


    for i in range(n_conv_layer, len(kernelshapes2d)):
        current_layer = Conv2D(numkernelsperlayer2d[i], [1,1], strides=(1, 1), padding='valid',
                               activation=activation_name, data_format='channels_first')(current_layer)
        current_layer = SpatialDropout2D(rate=dropout_rate, data_format='channels_first')(current_layer)

    current_layer = Conv2D(n_labels, [1,1], strides=(1, 1), padding='valid', activation=None,
                           data_format='channels_first')(current_layer)
    current_layer = Softmax(axis=1)(current_layer)

    model2d = Model(inputs = inputs, outputs = current_layer)
    model2d.compile(optimizer=optimizer(lr=initial_learning_rate), loss=loss_function)
    return model2d
