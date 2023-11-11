from keras import backend as K

def dice_coef(y_true, y_pred):
    smoothing_factor = 1
    flat_y_true = K.flatten(y_true)
    flat_y_pred = K.flatten(y_pred)
    return (2. * K.sum(flat_y_true * flat_y_pred) + smoothing_factor) / (K.sum(flat_y_true) + K.sum(flat_y_pred) + smoothing_factor)

# def POD(y_true, y_pred):
#     y_true_pos = K.flatten(y_true)
#     y_pred_pos = K.flatten(y_pred)
#     true_pos = K.sum(y_true_pos * y_pred_pos)
#     false_neg = K.sum(y_true_pos * (1 - y_pred_pos))
#     return true_pos / (true_pos + false_neg)

metrics = [dice_coef]

