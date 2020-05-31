import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Conv2D, MaxPooling2D, BatchNormalization
from keras.layers import GlobalAveragePooling2D
from keras import regularizers
from keras.applications import vgg16

def get_weights_and_bias(img_size=(224, 224, 3)):
    pretrained_model = vgg16.VGG16(include_top=False, weights='imagenet', input_shape=img_size)
    weights_and_bias = [x.get_weights() for x in pretrained_model.layers if type(x) == Conv2D]
    return weights_and_bias

def set_weights_and_bias(model, weights_and_bias):
    conv2d_layers = [x for x in model.layers if type(x) == Conv2D]
    weights_and_bias = weights_and_bias[:min(len(conv2d_layers), len(weights_and_bias))]
    for iter_conv2d_init, conv2d_init in enumerate(weights_and_bias):
        conv2d_layer = conv2d_layers[iter_conv2d_init]
        is_weight_shape_equal = conv2d_init[0].shape == conv2d_layer.get_weights()[0].shape
        is_bias_shape_equal = conv2d_init[1].shape == conv2d_layer.get_weights()[1].shape
        assert (is_weight_shape_equal and is_bias_shape_equal)
        conv2d_layer.set_weights(conv2d_init)
    print('Initialized weights/biases for first %d of %d Conv2D layers with pre-trained values' %
          (len(weights_and_bias), len(conv2d_layers)))
    return model

def build_fg(input_shape=[224, 224, 3], num_classes=20):
    weight_decay = 5e-4
    model = Sequential()
    # Block 1
    model.add(Conv2D(64, (3, 3), input_shape=input_shape,
                     padding='same', kernel_regularizer=regularizers.l2(weight_decay)))
    model.add(Activation('relu'))
    model.add(Conv2D(64, (3, 3), padding='same', kernel_regularizer=regularizers.l2(weight_decay)))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    # Block 2
    model.add(Conv2D(128, (3, 3), padding='same', kernel_regularizer=regularizers.l2(weight_decay)))
    model.add(Activation('relu'))
    model.add(Conv2D(128, (3, 3), padding='same', kernel_regularizer=regularizers.l2(weight_decay)))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    # Block 3
    model.add(Conv2D(256, (3, 3), padding='same', kernel_regularizer=regularizers.l2(weight_decay)))
    model.add(Activation('relu'))
    model.add(Conv2D(256, (3, 3), padding='same', kernel_regularizer=regularizers.l2(weight_decay)))
    model.add(Activation('relu'))
    model.add(Conv2D(256, (3, 3), padding='same', kernel_regularizer=regularizers.l2(weight_decay)))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    # Block 4
    model.add(Conv2D(512, (3, 3), padding='same', kernel_regularizer=regularizers.l2(weight_decay)))
    model.add(Activation('relu'))
    model.add(Conv2D(512, (3, 3), padding='same', kernel_regularizer=regularizers.l2(weight_decay)))
    model.add(Activation('relu'))
    model.add(Conv2D(512, (3, 3), padding='same', kernel_regularizer=regularizers.l2(weight_decay)))
    model.add(Activation('relu'))
    # Block 5
    model.add(Conv2D(512, (3, 3), padding='same', kernel_regularizer=regularizers.l2(weight_decay)))
    model.add(Activation('relu'))
    model.add(Conv2D(512, (3, 3), padding='same', kernel_regularizer=regularizers.l2(weight_decay)))
    model.add(Activation('relu'))
    model.add(Conv2D(512, (3, 3), padding='same', kernel_regularizer=regularizers.l2(weight_decay)))
    model.add(Activation('relu'))
    # Block 6 (orig stddev=0.005, ADP stddev=0.05)
    model.add(Conv2D(1024, (3, 3), padding='same', kernel_regularizer=regularizers.l2(weight_decay)))
    # model.add(Conv2D(1024, (3, 3), padding='same', kernel_regularizer=regularizers.l2(weight_decay),
    #           kernel_initializer=keras.initializers.RandomNormal(mean=0.0, stddev=0.005),
    #           bias_initializer=keras.initializers.Zeros()))
    model.add(Activation('relu'))
    model.add(Dropout(0.5))
    # Block 7 (orig stddev=0.005, ADP stddev=0.05)
    model.add(Conv2D(1024, (3, 3), padding='same', kernel_regularizer=regularizers.l2(weight_decay)))
    # model.add(Conv2D(1024, (3, 3), padding='same', kernel_regularizer=regularizers.l2(weight_decay),
    #           kernel_initializer=keras.initializers.RandomNormal(mean=0.0, stddev=0.005),
    #           bias_initializer=keras.initializers.Zeros()))
    # model.add(Activation('relu'))
    model.add(Dropout(0.5))
    model.add(GlobalAveragePooling2D())
    # Block 8 (orig stddev=0.005, ADP stddev=0.05)
    model.add(Dense(num_classes))
    # model.add(Dense(num_classes, kernel_regularizer=regularizers.l2(weight_decay), use_bias=False))
    # model.add(Dense(num_classes, kernel_regularizer=regularizers.l2(weight_decay),
    #           kernel_initializer=keras.initializers.RandomNormal(mean=0.0, stddev=0.005), use_bias=False))
    model.add(Activation('sigmoid'))

    return model

def build_VGG16fg_bn(input_shape=[224, 224, 3], num_classes=20):
    weight_decay = 5e-4
    model = Sequential()
    # Block 1
    model.add(Conv2D(64, (3, 3), input_shape=input_shape,
                     padding='same', kernel_regularizer=regularizers.l2(weight_decay)))
    model.add(Activation('relu'))
    model.add(BatchNormalization())
    model.add(Conv2D(64, (3, 3), padding='same', kernel_regularizer=regularizers.l2(weight_decay)))
    model.add(Activation('relu'))
    model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size=(2, 2)))
    # Block 2
    model.add(Conv2D(128, (3, 3), padding='same', kernel_regularizer=regularizers.l2(weight_decay)))
    model.add(Activation('relu'))
    model.add(BatchNormalization())
    model.add(Conv2D(128, (3, 3), padding='same', kernel_regularizer=regularizers.l2(weight_decay)))
    model.add(Activation('relu'))
    model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size=(2, 2)))
    # Block 3
    model.add(Conv2D(256, (3, 3), padding='same', kernel_regularizer=regularizers.l2(weight_decay)))
    model.add(Activation('relu'))
    model.add(BatchNormalization())
    model.add(Conv2D(256, (3, 3), padding='same', kernel_regularizer=regularizers.l2(weight_decay)))
    model.add(Activation('relu'))
    model.add(BatchNormalization())
    model.add(Conv2D(256, (3, 3), padding='same', kernel_regularizer=regularizers.l2(weight_decay)))
    model.add(Activation('relu'))
    model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size=(2, 2)))
    # Block 4
    model.add(Conv2D(512, (3, 3), padding='same', kernel_regularizer=regularizers.l2(weight_decay)))
    model.add(Activation('relu'))
    model.add(BatchNormalization())
    model.add(Conv2D(512, (3, 3), padding='same', kernel_regularizer=regularizers.l2(weight_decay)))
    model.add(Activation('relu'))
    model.add(BatchNormalization())
    model.add(Conv2D(512, (3, 3), padding='same', kernel_regularizer=regularizers.l2(weight_decay)))
    model.add(Activation('relu'))
    model.add(BatchNormalization())
    # Block 5
    model.add(Conv2D(512, (3, 3), padding='same', kernel_regularizer=regularizers.l2(weight_decay)))
    model.add(Activation('relu'))
    model.add(BatchNormalization())
    model.add(Conv2D(512, (3, 3), padding='same', kernel_regularizer=regularizers.l2(weight_decay)))
    model.add(Activation('relu'))
    model.add(BatchNormalization())
    model.add(Conv2D(512, (3, 3), padding='same', kernel_regularizer=regularizers.l2(weight_decay)))
    model.add(Activation('relu'))
    model.add(BatchNormalization())
    # Block 6 (orig stddev=0.005, ADP stddev=0.05)
    model.add(Conv2D(1024, (3, 3), padding='same', kernel_regularizer=regularizers.l2(weight_decay),
              kernel_initializer=keras.initializers.RandomNormal(mean=0.0, stddev=0.05),
              bias_initializer=keras.initializers.Zeros()))
    model.add(Activation('relu'))
    model.add(BatchNormalization())
    model.add(Dropout(0.5))
    # Block 7 (orig stddev=0.005, ADP stddev=0.05)
    model.add(Conv2D(1024, (3, 3), padding='same', kernel_regularizer=regularizers.l2(weight_decay),
              kernel_initializer=keras.initializers.RandomNormal(mean=0.0, stddev=0.05),
              bias_initializer=keras.initializers.Zeros()))
    model.add(Activation('relu'))
    model.add(BatchNormalization())
    model.add(Dropout(0.5))
    model.add(GlobalAveragePooling2D())
    # Block 8 (orig stddev=0.005, ADP stddev=0.05)
    model.add(Dense(num_classes, kernel_regularizer=regularizers.l2(weight_decay),
              kernel_initializer=keras.initializers.RandomNormal(mean=0.0, stddev=0.05), use_bias=False))
    model.add(Activation('sigmoid'))

    return model

def build_X1p7():
    weight_decay = 5e-4
    model = Sequential()
    # Block 1
    model.add(Conv2D(64, (3, 3), input_shape=[224, 224, 3], padding='same',
                     kernel_regularizer=regularizers.l2(weight_decay)))
    model.add(Activation('relu'))
    model.add(BatchNormalization())
    model.add(Dropout(0.3))
    model.add(Conv2D(64, (3, 3), padding='same', kernel_regularizer=regularizers.l2(weight_decay)))
    model.add(Activation('relu'))
    model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size=(2, 2)))
    # Block 2
    model.add(Conv2D(128, (3, 3), padding='same', kernel_regularizer=regularizers.l2(weight_decay)))
    model.add(Activation('relu'))
    model.add(BatchNormalization())
    model.add(Dropout(0.4))
    model.add(Conv2D(128, (3, 3), padding='same', kernel_regularizer=regularizers.l2(weight_decay)))
    model.add(Activation('relu'))
    model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size=(2, 2)))
    # Block 3
    model.add(Conv2D(256, (3, 3), padding='same', kernel_regularizer=regularizers.l2(weight_decay)))
    model.add(Activation('relu'))
    model.add(BatchNormalization())
    model.add(Dropout(0.4))
    model.add(Conv2D(256, (3, 3), padding='same', kernel_regularizer=regularizers.l2(weight_decay)))
    model.add(Activation('relu'))
    model.add(BatchNormalization())
    model.add(Dropout(0.4))
    model.add(Conv2D(256, (3, 3), padding='same', kernel_regularizer=regularizers.l2(weight_decay)))
    model.add(Activation('relu'))
    model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(GlobalAveragePooling2D())
    # Classifier
    model.add(Dense(51, kernel_regularizer=regularizers.l2(weight_decay)))
    model.add(Activation('sigmoid'))

    return model

def build_vgg16_experimental(input_shape=[224, 224, 3], num_classes=31,
                             var_code='M1'):
    assert(var_code in ['M1', 'M2', 'M3', 'M4', 'M5', 'M6', 'M7', 'M7bg'])
    conv_depths = {}
    vec_ops = {}
    fc_depths = {}
    conv_depths['M1'] = [64, 64, 128, 128, 256, 256, 256, 512, 512, 512, 512, 512, 512]
    vec_ops['M1'] = 'GAP'
    fc_depths['M1'] = [num_classes]
    conv_depths['M2'] = [64, 64, 128, 128, 256, 256, 256, 512, 512, 512]
    vec_ops['M2'] = 'GAP'
    fc_depths['M2'] = [num_classes]
    conv_depths['M3'] = [64, 64, 128, 128, 256, 256, 256]
    vec_ops['M3'] = 'GAP'
    fc_depths['M3'] = [num_classes]
    conv_depths['M4'] = [64, 64, 128, 128]
    vec_ops['M4'] = 'GAP'
    fc_depths['M4'] = [num_classes]
    conv_depths['M5'] = [64, 64, 128, 128, 256, 256, 256]
    vec_ops['M5'] = 'GAP'
    fc_depths['M5'] = [256, num_classes]
    conv_depths['M6'] = [64, 64, 128, 128, 256, 256, 256]
    vec_ops['M6'] = 'Flatten'
    fc_depths['M6'] = [num_classes]
    conv_depths['M7'] = [64, 64, 128, 128, 256, 256, 256]
    vec_ops['M7'] = 'GMP'
    fc_depths['M7'] = [num_classes]
    conv_depths['M7bg'] = [64, 64, 128, 128, 256]
    vec_ops['M7bg'] = 'Flatten'
    fc_depths['M7bg'] = [256, 256, num_classes]

    weight_decay = 5e-4
    model = Sequential()
    # Block 1
    if len(conv_depths[var_code]) >= 2:
        model.add(Conv2D(conv_depths[var_code].pop(0), (3, 3), padding='same', input_shape=input_shape, kernel_regularizer=regularizers.l2(weight_decay)))
        model.add(Activation('relu'))
        model.add(BatchNormalization())
        # model.add(Dropout(0.3))
        model.add(Conv2D(conv_depths[var_code].pop(0), (3, 3), padding='same', kernel_regularizer=regularizers.l2(weight_decay)))
        model.add(Activation('relu'))
        model.add(BatchNormalization())
        model.add(MaxPooling2D(pool_size=(2, 2)))
    # Block 2
    if len(conv_depths[var_code]) >= 2:
        model.add(Conv2D(conv_depths[var_code].pop(0), (3, 3), padding='same', kernel_regularizer=regularizers.l2(weight_decay)))
        model.add(Activation('relu'))
        model.add(BatchNormalization())
        # model.add(Dropout(0.4))
        model.add(Conv2D(conv_depths[var_code].pop(0), (3, 3), padding='same', kernel_regularizer=regularizers.l2(weight_decay)))
        model.add(Activation('relu'))
        model.add(BatchNormalization())
        model.add(MaxPooling2D(pool_size=(2, 2)))
    # Block 3
    if len(conv_depths[var_code]) >= 3:
        model.add(Conv2D(conv_depths[var_code].pop(0), (3, 3), padding='same', kernel_regularizer=regularizers.l2(weight_decay)))
        model.add(Activation('relu'))
        model.add(BatchNormalization())
        # model.add(Dropout(0.4))
        model.add(Conv2D(conv_depths[var_code].pop(0), (3, 3), padding='same', kernel_regularizer=regularizers.l2(weight_decay)))
        model.add(Activation('relu'))
        model.add(BatchNormalization())
        # model.add(Dropout(0.4))
        model.add(Conv2D(conv_depths[var_code].pop(0), (3, 3), padding='same', kernel_regularizer=regularizers.l2(weight_decay)))
        model.add(Activation('relu'))
        model.add(BatchNormalization())
        model.add(MaxPooling2D(pool_size=(2, 2)))
    # Block 4
    if len(conv_depths[var_code]) >= 3:
        model.add(Conv2D(conv_depths[var_code].pop(0), (3, 3), padding='same', kernel_regularizer=regularizers.l2(weight_decay)))
        model.add(Activation('relu'))
        model.add(BatchNormalization())
        # model.add(Dropout(0.4))
        model.add(Conv2D(conv_depths[var_code].pop(0), (3, 3), padding='same', kernel_regularizer=regularizers.l2(weight_decay)))
        model.add(Activation('relu'))
        model.add(BatchNormalization())
        # model.add(Dropout(0.4))
        model.add(Conv2D(conv_depths[var_code].pop(0), (3, 3), padding='same', kernel_regularizer=regularizers.l2(weight_decay)))
        model.add(Activation('relu'))
        model.add(BatchNormalization())
        model.add(MaxPooling2D(pool_size=(2, 2)))
    # Block 5
    if len(conv_depths[var_code]) >= 3:
        model.add(Conv2D(conv_depths[var_code].pop(0), (3, 3), padding='same', kernel_regularizer=regularizers.l2(weight_decay)))
        model.add(Activation('relu'))
        model.add(BatchNormalization())
        # model.add(Dropout(0.4))
        model.add(Conv2D(conv_depths[var_code].pop(0), (3, 3), padding='same', kernel_regularizer=regularizers.l2(weight_decay)))
        model.add(Activation('relu'))
        model.add(BatchNormalization())
        # model.add(Dropout(0.4))
        model.add(Conv2D(conv_depths[var_code].pop(0), (3, 3), padding='same', kernel_regularizer=regularizers.l2(weight_decay)))
        model.add(Activation('relu'))
        model.add(BatchNormalization())
        model.add(MaxPooling2D(pool_size=(2, 2)))
    # Vectorization operation
    model.add(Dropout(0.5))
    if vec_ops[var_code] == 'Flatten':
        model.add(Flatten())
    elif vec_ops[var_code] == 'GAP':
        model.add(GlobalAveragePooling2D())
    elif vec_ops[var_code] == 'GMP':
        model.add(MaxPooling2D(pool_size=tuple(model.layers[-1].output_shape[1:3]))) # Make pool size adaptive
        model.add(Flatten())
    # FC
    while len(fc_depths[var_code]) >= 1:
        # Original: stddev = 0.5
        model.add(Dense(fc_depths[var_code].pop(0), kernel_initializer=keras.initializers.RandomNormal(mean=0.0, stddev=0.05),
                        bias_initializer=keras.initializers.Zeros(),
                        kernel_regularizer=regularizers.l2(weight_decay)))
        if len(fc_depths[var_code]) > 0:
            model.add(Activation('relu'))
            model.add(BatchNormalization())
            model.add(Dropout(0.5))
        else:
            model.add(Activation('sigmoid'))

    return model