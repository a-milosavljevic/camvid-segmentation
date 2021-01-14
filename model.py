"""
INCLUDE ONLY, DO NOT EXECUTE
"""
from data import *
import myresunet
import segmentation_models as sm


"""
Initialize preprocessing function
"""
if model_type == 'myresunet':
    preprocessing = myresunet.preprocessing
else:
    preprocessing = sm.get_preprocessing(backbone)


def create_model():
    """
    Function used to create model based on parameters specified in settings.py
    """
    if model_type == 'myresunet':
        model = myresunet.create_model()
    elif model_type == 'unet':
        model = sm.Unet(backbone_name=backbone,
                        input_shape=(image_height, image_width, 3),
                        classes=num_classes,
                        activation='softmax',
                        encoder_weights='imagenet',
                        encoder_freeze=False,
                        encoder_features='default',
                        decoder_block_type='upsampling',
                        decoder_filters=(decoder_scaler * 256, decoder_scaler * 128, decoder_scaler * 64,
                                         decoder_scaler * 32, decoder_scaler * 16),
                        decoder_use_batchnorm=True)
    elif model_type == 'fpn':
        model = sm.FPN(backbone_name=backbone,
                       input_shape=(image_height, image_width, 3),
                       classes=num_classes,
                       activation='softmax',
                       encoder_weights='imagenet',
                       encoder_freeze=False,
                       encoder_features='default',
                       pyramid_block_filters=decoder_scaler * 256,
                       pyramid_use_batchnorm=True,
                       pyramid_aggregation='concat',
                       pyramid_dropout=None)
    elif model_type == 'linknet':
        model = sm.Linknet(backbone_name=backbone,
                           input_shape=(image_height, image_width, 3),
                           classes=num_classes,
                           activation='softmax',
                           encoder_weights='imagenet',
                           encoder_freeze=False,
                           encoder_features='default',
                           decoder_block_type='upsampling',
                           decoder_filters=(None, None, None, None, decoder_scaler * 16),
                           decoder_use_batchnorm=True)
    elif model_type == 'pspnet':
        model = sm.PSPNet(backbone_name=backbone,
                          input_shape=(image_height, image_width, 3),
                          classes=num_classes,
                          activation='softmax',
                          encoder_weights='imagenet',
                          encoder_freeze=False,
                          downsample_factor=8,
                          psp_conv_filters=decoder_scaler * 512,
                          psp_pooling_type='avg',
                          psp_use_batchnorm=True,
                          psp_dropout=None)
    else:
        print('Invalid segmentation model type')
        exit(0)
    return model
