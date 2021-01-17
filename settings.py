"""
INCLUDE ONLY, DO NOT EXECUTE

Check and adjust settings specified in this file.
"""
import os


########################################################################################################################
# TRAINING SETTINGS
########################################################################################################################

image_width = 768   # original 960
image_height = 512  # original 720

batch_size = 4

init_lr = 1e-3
early_stopping_patience = 30
reduce_lr_patience = 10
reduce_lr_factor = 0.1
cb_monitor = 'val_categorical_accuracy'
cb_mode = 'max'


########################################################################################################################
# MODEL SETTINGS
########################################################################################################################

# model_type = 'myresunet'

model_type = 'unet'
# model_type = 'fpn'
# model_type = 'linknet'

# backbone = 'vgg16'
# backbone = 'vgg19'
# backbone = 'resnet18'
# backbone = 'resnet34'
# backbone = 'resnet50'
# backbone = 'resnet101'
# backbone = 'resnet152'
# backbone = 'resnext50'
# backbone = 'resnext101'
# backbone = 'inceptionv3'
# backbone = 'inceptionresnetv2'
# backbone = 'densenet121'
# backbone = 'densenet169'
# backbone = 'densenet201'
# backbone = 'seresnet18'
# backbone = 'seresnet34'
# backbone = 'seresnet50'
# backbone = 'seresnet101'
# backbone = 'seresnet152'
# backbone = 'seresnext50'
# backbone = 'seresnext101'
# backbone = 'senet154'
# backbone = 'mobilenet'
# backbone = 'mobilenetv2'
# backbone = 'efficientnetb0'
# backbone = 'efficientnetb1'
backbone = 'efficientnetb2'
# backbone = 'efficientnetb3'
# backbone = 'efficientnetb4'
# backbone = 'efficientnetb5'
# backbone = 'efficientnetb6'
# backbone = 'efficientnetb7'

decoder_scaler = 1


########################################################################################################################
# FOLDER SETTINGS
########################################################################################################################

root_folder = os.getcwd()
data_folder = os.path.join(root_folder, 'data')
tmp_folder = os.path.join(root_folder, 'tmp')
if not os.path.exists(tmp_folder):
    os.makedirs(tmp_folder)

train_folder = os.path.join(data_folder, 'train')
train_labels_folder = os.path.join(data_folder, 'train_labels')

val_folder = os.path.join(data_folder, 'val')
val_labels_folder = os.path.join(data_folder, 'val_labels')

test_folder = os.path.join(data_folder, 'test')
test_labels_folder = os.path.join(data_folder, 'test_labels')

train_images = [fname[:-4] for fname in os.listdir(train_folder)]
val_images = [fname[:-4] for fname in os.listdir(val_folder)]
test_images = [fname[:-4] for fname in os.listdir(test_folder)]
