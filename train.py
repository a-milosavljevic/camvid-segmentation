"""
Execute this script to train a model specified by "settings.py" script.
In case of an out of memory problem adjust batch_size in "settings.py".
Be sure to download dataset from https://www.kaggle.com/carlolepelaars/camvid
and unpack it to "data" subfolder.
"""
from model import *
import matplotlib.pyplot as plt


print("Tensorflow version: " + tf.__version__)
print("Keras version: " + tf.keras.__version__)


########################################################################################################################
# CREATE MODEL
########################################################################################################################

model = create_model()

model.compile(optimizer=tf.keras.optimizers.RMSprop(lr=init_lr),
              loss=tf.keras.losses.categorical_crossentropy,
              metrics=[tf.keras.metrics.categorical_accuracy])

model.summary()


########################################################################################################################
# CALLBACKS
########################################################################################################################

path = os.path.join(tmp_folder, 'trained_model.h5')
save_model = tf.keras.callbacks.ModelCheckpoint(path, monitor=cb_monitor, mode=cb_mode, verbose=1,
                                                save_best_only=True)

csv_logger = tf.keras.callbacks.CSVLogger(os.path.join(tmp_folder, 'training.csv'))

early_stopping = tf.keras.callbacks.EarlyStopping(monitor=cb_monitor, mode=cb_mode, verbose=1,
                                                  patience=early_stopping_patience, restore_best_weights=True)

reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(monitor=cb_monitor, mode=cb_mode, verbose=1,
                                                 factor=reduce_lr_factor, patience=reduce_lr_patience)


########################################################################################################################
# TRAINING MODEL
########################################################################################################################

data_gen_train = DataProvider(batch_size, is_validation=False, process_input=preprocessing)
data_gen_valid = DataProvider(batch_size, is_validation=True, process_input=preprocessing)

hist = model.fit(data_gen_train,
                 epochs=1000,
                 validation_data=data_gen_valid,
                 shuffle=True,
                 callbacks=[save_model, csv_logger, early_stopping, reduce_lr],
                 verbose=2)

model.save(path, include_optimizer=False)

plt.clf()
plt.plot(hist.history['loss'])
plt.plot(hist.history['val_loss'])
plt.savefig(os.path.join(tmp_folder, 'training_loss.png'))

plt.clf()
plt.plot(hist.history['categorical_accuracy'])
plt.plot(hist.history['val_categorical_accuracy'])
plt.savefig(os.path.join(tmp_folder, 'training_accuracy.png'))


########################################################################################################################
# EVALUATE MODEL
########################################################################################################################

res_train = model.evaluate(data_gen_train)
res_test = model.evaluate(data_gen_valid)
print('[train_loss, train_accuracy] =', res_train)
print('[val_loss, val_accuracy] =', res_test)
