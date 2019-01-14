from keras.callbacks import EarlyStopping
from src.Generator import generator
from math import floor


def train(model2d, images, masks, image_shape=(210,238,200), input_shape=(2,32,32), 
          train_validate_rate=0.85, patience=50, 
          steps_per_epoch=300, validation_steps=50, epochs=500):
    
    output_shape = (int(model2d.output.shape[1]), int(model2d.output.shape[2]), int(model2d.output.shape[3]))
    train_val_index = floor(train_validate_rate * len(images))

    training_generator = generator(images[:train_val_index], masks[:train_val_index],
                                   batchsz=100, input_shape=input_shape,
                                   img_shape=image_shape, output_shape=output_shape)
    validation_generator = generator(images[train_val_index:], masks[train_val_index:], 
                                     batchsz=50, input_shape=input_shape,
                                     img_shape=image_shape, output_shape=output_shape)

    es = EarlyStopping(patience = patience)
    model2d.fit_generator(generator=training_generator,
                        validation_data=validation_generator,
                         steps_per_epoch=steps_per_epoch,
                         validation_steps=validation_steps,
                         epochs=epochs,
                         callbacks=[es])
    return model2d
