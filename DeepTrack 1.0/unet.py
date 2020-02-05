from keras.layers import *

def create_unet(pretrained_weights=None, input_size=(51, 51, 1)):
    """Creates a unet
    Inputs:
    pretrained_weights: if not None, loads the pretrained weights into the network
    input_size: the size of the input image (px,px,color channels)

    Outputs:
    network: the created network
    """

    from keras import layers, optimizers, models
    import keras

    #amount of downsizes
    n = 4

    input_unpadded = Input(input_size)
    padding = get_padding(input_size, n)
    input_padded = ZeroPadding2D(padding=padding)(input_unpadded)

    conv1 = layers.Conv2D(8, 3, activation='relu', padding='same', kernel_initializer='he_normal')(input_padded)
    conv1 = Conv2D(8, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv1)
    pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)
    conv2 = Conv2D(16, 3, activation='relu', padding='same', kernel_initializer='he_normal')(pool1)
    conv2 = Conv2D(16, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv2)
    pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)
    conv3 = Conv2D(32, 3, activation='relu', padding='same', kernel_initializer='he_normal')(pool2)
    conv3 = Conv2D(32, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv3)
    pool3 = MaxPooling2D(pool_size=(2, 2))(conv3)
    conv4 = Conv2D(64, 3, activation='relu', padding='same', kernel_initializer='he_normal')(pool3)
    conv4 = Conv2D(64, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv4)
    drop4 = Dropout(0.5)(conv4)
    pool4 = MaxPooling2D(pool_size=(2, 2))(drop4)

    conv5 = Conv2D(128, 3, activation='relu', padding='same', kernel_initializer='he_normal')(pool4)
    conv5 = Conv2D(128, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv5)
    drop5 = Dropout(0.5)(conv5)

    up6 = Conv2D(64, 2, activation='relu', padding='same', kernel_initializer='he_normal')(
        UpSampling2D(size=(2, 2))(drop5))
    merge6 = concatenate([drop4, up6], axis=3)
    conv6 = Conv2D(64, 3, activation='relu', padding='same', kernel_initializer='he_normal')(merge6)
    conv6 = Conv2D(64, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv6)

    up7 = Conv2D(32, 2, activation='relu', padding='same', kernel_initializer='he_normal')(
        UpSampling2D(size=(2, 2))(conv6))
    merge7 = concatenate([conv3, up7], axis=3)
    conv7 = Conv2D(32, 3, activation='relu', padding='same', kernel_initializer='he_normal')(merge7)
    conv7 = Conv2D(32, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv7)

    up8 = Conv2D(16, 2, activation='relu', padding='same', kernel_initializer='he_normal')(
        UpSampling2D(size=(2, 2))(conv7))
    merge8 = concatenate([conv2, up8], axis=3)
    conv8 = Conv2D(16, 3, activation='relu', padding='same', kernel_initializer='he_normal')(merge8)
    conv8 = Conv2D(16, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv8)

    up9 = Conv2D(8, 2, activation='relu', padding='same', kernel_initializer='he_normal')(
        UpSampling2D(size=(2, 2))(conv8))
    merge9 = concatenate([conv1, up9], axis=3)
    conv9 = Conv2D(8, 3, activation='relu', padding='same', kernel_initializer='he_normal')(merge9)
    conv9 = Conv2D(8, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv9)
    conv9 = Conv2D(2, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv9)
    conv10 = Conv2D(1, 1, activation='sigmoid')(conv9)

    output = Cropping2D(padding)(conv10)

    model = models.Model(input=input_unpadded, output=output)

    model.compile(optimizer=optimizers.Adam(lr=1e-4), loss='binary_crossentropy', metrics=['accuracy'])

    # model.summary()

    if (pretrained_weights):
        model.load_weights(pretrained_weights)

    return model

def balanced_cross_entropy(beta):
    # see https://lars76.github.io/neural-networks/object-detection/losses-for-segmentation/#references
    import tensorflow as tf
    def loss(y_true, y_pred):
        #normalize it to [0;1]
        y_true = tf.math.divide(y_true, 255)
        y_pred = tf.math.divide(y_pred, 255)
        pos_weight = beta / (1 - beta)
        loss = tf.nn.weighted_cross_entropy_with_logits(logits=y_pred, labels=y_true, pos_weight=pos_weight)

        # or reduce_sum and/or axis=-1
        return tf.reduce_mean(loss * (1 - beta))

    return loss
   
def get_padding(input_size, n):
    """Adds padding to the input image
    Inputs:
    input: the input image
    input_size: the size of the input image
    n: the input image dimensions are changed to be divisible by 2**n

    Outputs:
    padding: the padding that was used
    """
    C0 = 2 ** (n - 1)
    C1 = 2 ** (n - 1)
    if (input_size[0] % 8 != 0):
        top_pad = (input_size[0] % (2 * n) // 2);
        bottom_pad = (input_size[0] % (2 * n) - top_pad)
    else:
        top_pad = 0;
        bottom_pad = 0;
        C0 = 0
    if input_size[1] % 8 != 0:
        left_pad = (input_size[1] % (2 * n) // 2);
        right_pad = (input_size[1] % (2 * n) - left_pad)
    else:
        left_pad = 0;
        right_pad = 0;
        C1 = 0
    padding = ((C0 - top_pad, C0 - bottom_pad), (C1 - left_pad, C1 - right_pad))

    return (padding)

def train_deep_learning_network(
        network,
        image_generator,
        sample_sizes=(32, 128, 512, 2048),
        iteration_numbers=(3001, 2001, 1001, 101),
        verbose=True):
    """Train a deep learning network.

    Input:
    network: deep learning network
    image_generator: image generator
    sample_sizes: sizes of the batches of images used in the training [tuple of positive integers]
    iteration_numbers: numbers of batches used in the training [tuple of positive integers]
    verbose: frequency of the update messages [number between 0 and 1]

    Output:
    training_history: dictionary with training history

    Note: The MSE is in px^2 and the MAE in px
    """

    import numpy as np
    from time import time
    import deeptrack as dt

    # number_of_outputs = 3

    training_history = {}
    training_history['Sample Size'] = []
    training_history['Iteration Number'] = []
    training_history['Iteration Time'] = []

    for sample_size, iteration_number in zip(sample_sizes, iteration_numbers):
        for iteration in range(iteration_number):

            # meaure initial time for iteration
            initial_time = time()

            # generate images and targets
            image_shape = network.get_layer(index=0).get_config()['batch_input_shape'][1:]

            input_shape = (sample_size, image_shape[0], image_shape[1], image_shape[2])
            images = np.zeros(input_shape)

            # output_shape = (sample_size, number_of_outputs)
            targets = np.zeros(input_shape)

            for image_number, image, image_parameters, target in image_generator():
                if image_number >= sample_size:
                    break

                resized_image = dt.resize_image(image, (image_shape[0], image_shape[1]))
                images[image_number] = resized_image.reshape(image_shape)
                resized_target = dt.resize_image(target, (image_shape[0], image_shape[1]))
                targets[image_number] = resized_target.reshape(image_shape)

            # training
            history = network.fit(images,
                                  targets,
                                  epochs=1,
                                  batch_size=sample_size,
                                  verbose=False)

            # measure elapsed time during iteration
            iteration_time = time() - initial_time

            # record training history
            # mse = history.history['mean_squared_error'][0] * half_image_size**2
            # mae = history.history['mean_absolute_error'][0] * half_image_size

            training_history['Sample Size'].append(sample_size)
            training_history['Iteration Number'].append(iteration)
            training_history['Iteration Time'].append(iteration_time)
            # training_history['MSE'].append(mse)
            # training_history['MAE'].append(mae)

            if not (iteration % int(verbose ** -1)):
                print('Sample size %6d   iteration number %6d Time %10.2f ms' % (
                sample_size, iteration + 1, iteration_time * 1000))

    return training_history
