# Import Libraries -------------------------------------------------------------------------------------------------------------------------------------------------------------
import os
import h5py
import keras
import loss
import Helper
import allMetrics
import numpy as np
import tensorflow as tf
import UNetModel_3D
import matplotlib.pyplot as plt
from keras import backend as K
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint, CSVLogger

os.environ["CUDA_VISIBLE_DEVICES"]="3"

physical_devices = tf.config.list_physical_devices('GPU')
try:
  for device in physical_devices:
    tf.config.experimental.set_memory_growth(device, True)
except:
  # Invalid device or cannot modify virtual devices once initialized.
  pass

# Retrieving Data
train_fileName = "/home/tester/jianhoong/jh_fyp_work/3D_UNet/trials/numpyDatasets/numPyArrays/train_Scans/train_DS2.hdf5"
train_maskfileName = "/home/tester/jianhoong/jh_fyp_work/3D_UNet/trials/numpyDatasets/numPyArrays/train_Mask_Scans/train_maskDS2.hdf5"

train_DatasetName = "trainScans_DataSet2"
train_maskDatasetName = "trainMaskScans_DataSet2"

# Train Dataset
with h5py.File(train_fileName, 'r') as hf: # File Dir
    train_array = hf[train_DatasetName][:675]
    
with h5py.File(train_maskfileName,'r') as hf:
    train_mask_array = hf[train_maskDatasetName][:675]
    
train_array = np.expand_dims(train_array, axis=4)
train_mask_array = np.expand_dims(train_mask_array, axis=4)

print(train_array.shape)

# Valid Dataset
with h5py.File(train_fileName, 'r') as hf: # File Dir
    valid_array = hf[train_DatasetName][675:900]
    
with h5py.File(train_maskfileName,'r') as hf:
    valid_mask_array = hf[train_maskDatasetName][675:900]
    
valid_array = np.expand_dims(valid_array, axis=4)
valid_mask_array = np.expand_dims(valid_mask_array, axis=4)

print(valid_array.shape)
# Loss Function and coefficients to be used during training:

LR = 0.001
opt = tf.keras.optimizers.Nadam(LR)

input_shape = (64,64,96,1)
num_class = 1

metrics = [allMetrics.dice_coef]

model = UNetModel_3D.build_unet(input_shape, n_classes = num_class)
model.compile(optimizer=opt, loss=loss.tversky_crossentropy, metrics=metrics)
print(model.summary())

class CustomCallBack(keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs = None):
        lr = self.model.optimizer.lr
        decay = self.model.optimizer.decay
        iterations = self.model.optimizer.iterations
        lr_with_decay = lr / (1. + decay * K.cast(iterations, K.dtype(decay)))

        Helper.telegram_bot_sendtext(f'''
                            End of Epoch {epoch}
                            Train Dice: {logs.get("dice_coef")}
                            Valid Dice: {logs.get("val_dice_coef")}
                            Train Loss: {logs.get("loss")}
                            Val Loss: {logs.get('val_loss')}
                            LR: {lr_with_decay}
                              ''')
        
csv_path = '/home/tester/jianhoong/jh_fyp_work/3D_UNet/Final/CSVLogs/Model7.csv'
model_checkpoint_path = '/home/tester/jianhoong/jh_fyp_work/3D_UNet/Final/SavedModels/Model7.hdf5'

my_callbacks = [
    ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=20, mode = 'auto'),
    EarlyStopping(monitor='val_loss', patience=100, restore_best_weights=True, mode = 'auto'),
    CSVLogger(csv_path, separator=',', append=True),
    ModelCheckpoint(filepath=model_checkpoint_path,
                    monitor='val_loss',
                    mode='auto',
                    verbose=1,
                    save_best_only= True),
    CustomCallBack()
]

model_name = 'Model7 (500 Epochs)'
Helper.telegram_bot_sendtext(f'Model {model_name} started training')

history = model.fit(train_array,
                    train_mask_array,
                    batch_size=5,
                    epochs=500,
                    verbose=1,
                    shuffle = True,
                    validation_data=(valid_array, valid_mask_array),
                    callbacks=my_callbacks)


# Send Results -------------------------------------------------------------------------------------------------------------------------------------------------------------
Helper.telegram_bot_sendtext(f'''
Model has completed training and saved successfully.
Train Dice Score : {history.history['dice_coef']}
Valid Dice Score : {history.history['val_dice_coef']}
Train Loss : {history.history['loss']}
Val Loss : {history.history['val_loss']}
''')

# Plotting Model Performance ----------------------------------------------------------
plt.figure(figsize=(10, 10))
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.plot(history.history['dice_coef'])
plt.plot(history.history['val_dice_coef'])
plt.title('Metrics vs Num_Epochs')
plt.ylabel('Coefficients')
plt.xlabel('Epochs')
plt.legend(['loss', 'val_loss', 'dice_coef', 'val_dice_coef'], loc='best')
img_filepath = '/home/tester/jianhoong/jh_fyp_work/3D_UNet/trials/ModelPerformanceImages/Model7.png'
plt.savefig(img_filepath)
Helper.send_photo(open(img_filepath, 'rb'))
# -------------------------------------------------------------------------------------------------------------------------------------------------------------