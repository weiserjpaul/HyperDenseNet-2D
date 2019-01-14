from keras import backend as K



def weighted_dice_coefficient2d(y_true, y_pred, axis=(-2, -1), smooth=0.00001):
    return K.mean(2. * (K.sum(y_true * y_pred,
                              axis=axis) + smooth/2)/(K.sum(y_true,
                                                            axis=axis) + K.sum(y_pred,
                                                                               axis=axis) + smooth))


def weighted_dice_coefficient_loss2d(y_true, y_pred):
    return -weighted_dice_coefficient2d(y_true, y_pred)
