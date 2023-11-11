import tensorflow as tf
from keras import backend as K

def identify_axis(shape):
    # Three dimensional
    if len(shape) == 5 : return [1,2,3]
    # Two dimensional
    elif len(shape) == 4 : return [1,2]
    # Exception - Unknown
    else : raise ValueError('Metric: Shape of tensor is neither 2D or 3D.')

def tversky(y_true, y_pred, smooth = 1E-5):
    alpha = 0.3
    beta = 0.7
    
    axis = identify_axis(y_true.get_shape())
    tp = K.sum(y_true * y_pred, axis=axis)
    fn = K.sum(y_true * (1-y_pred), axis=axis)
    fp = K.sum((1-y_true) * y_pred, axis=axis)
    tversky_class = (tp + smooth)/(tp + alpha*fn + beta*fp + smooth)
    return tversky_class

def tversky_loss(y_true, y_pred):
    n = K.cast(K.shape(y_true)[-1], 'float32')
    tver = K.sum(tversky(y_true, y_pred, smooth = 1E-5), axis = [-1])
    return n - tver

def tversky_crossentropy(y_true, y_pred):
    tver = tversky_loss(y_true, y_pred)
    cross_entropy = K.mean(tf.keras.losses.binary_crossentropy(y_true, y_pred))
    return tver + cross_entropy