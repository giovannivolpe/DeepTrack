def create_unet(pretrained_weights = None, input_size = (51,51,1)):
    """Creates a unet
    Inputs:
    pretrained_weights: if not None, loads the pretrained weights into the network
    input_size: the size of the input image (px,px,color channels)

    Outputs:
    network: the created network
    """

    from keras import layers,optimizers, models

    n = 4

    input_unpadded = Input(input_size)
    padding = get_padding(input_unpadded, input_size, n)
    input_padded = ZeroPadding2D(padding=padding)(input_unpadded)


    conv1 = layers.Conv2D(64, 3, activation='relu', padding='same', kernel_initializer='he_normal')(input_padded)
    conv1 = Conv2D(64, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv1)
    pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)
    conv2 = Conv2D(128, 3, activation='relu', padding='same', kernel_initializer='he_normal')(pool1)
    conv2 = Conv2D(128, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv2)
    pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)
    conv3 = Conv2D(256, 3, activation='relu', padding='same', kernel_initializer='he_normal')(pool2)
    conv3 = Conv2D(256, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv3)
    pool3 = MaxPooling2D(pool_size=(2, 2))(conv3)
    conv4 = Conv2D(512, 3, activation='relu', padding='same', kernel_initializer='he_normal')(pool3)
    conv4 = Conv2D(512, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv4)
    drop4 = Dropout(0.5)(conv4)
    pool4 = MaxPooling2D(pool_size=(2, 2))(drop4)

    conv5 = Conv2D(1024, 3, activation='relu', padding='same', kernel_initializer='he_normal')(pool4)
    conv5 = Conv2D(1024, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv5)
    drop5 = Dropout(0.5)(conv5)

    up6 = Conv2D(512, 2, activation='relu', padding='same', kernel_initializer='he_normal')(
        UpSampling2D(size=(2, 2))(drop5))
    merge6 = concatenate([drop4, up6], axis=3)
    conv6 = Conv2D(512, 3, activation='relu', padding='same', kernel_initializer='he_normal')(merge6)
    conv6 = Conv2D(512, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv6)

    up7 = Conv2D(256, 2, activation='relu', padding='same', kernel_initializer='he_normal')(
        UpSampling2D(size=(2, 2))(conv6))
    merge7 = concatenate([conv3, up7], axis=3)
    conv7 = Conv2D(256, 3, activation='relu', padding='same', kernel_initializer='he_normal')(merge7)
    conv7 = Conv2D(256, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv7)

    up8 = Conv2D(128, 2, activation='relu', padding='same', kernel_initializer='he_normal')(
        UpSampling2D(size=(2, 2))(conv7))
    merge8 = concatenate([conv2, up8], axis=3)
    conv8 = Conv2D(128, 3, activation='relu', padding='same', kernel_initializer='he_normal')(merge8)
    conv8 = Conv2D(128, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv8)

    up9 = Conv2D(64, 2, activation='relu', padding='same', kernel_initializer='he_normal')(
        UpSampling2D(size=(2, 2))(conv8))
    merge9 = concatenate([conv1, up9], axis=3)
    conv9 = Conv2D(64, 3, activation='relu', padding='same', kernel_initializer='he_normal')(merge9)
    conv9 = Conv2D(64, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv9)
    conv9 = Conv2D(2, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv9)
    conv10 = Conv2D(1, 1, activation='sigmoid')(conv9)

    output = Cropping2D(padding)(conv10)

    model = models.Model(input=input_unpadded, output=output)

    model.compile(optimizer=optimizers.Adam(lr=1e-4), loss='binary_crossentropy', metrics=['accuracy'])

    # model.summary()

    if (pretrained_weights):
        model.load_weights(pretrained_weights)

    return model

def get_target_binary_image(image_parameters):
    """Create and return binary target image given image parameters
    Input: Image parameters
    Output: Binary image of the input image size where pixels containing particles are marked as ones, while rest are zeros
    """

    import numpy as np

    import matplotlib.pyplot as plt

    particle_center_x_list = image_parameters['Particle Center X List']
    particle_center_y_list = image_parameters['Particle Center Y List']
    particle_radius_list = image_parameters['Particle Radius List']
    image_half_size = image_parameters['Image Half-Size']

    targetBinaryImage = np.zeros((2*image_half_size+1, 2*image_half_size+1))

    for particle_index in range(0, len(particle_center_x_list)):
        center_x = particle_center_x_list[particle_index]
        center_y = particle_center_y_list[particle_index]
        radius = particle_radius_list[particle_index]

        """Loops over all pixels with center in coordinates = [ceil(center - radius): floor(center + radius)]. Adds the ones with
        center within radius.
        """
        for pixel_x in range(int(np.floor(center_x-radius)), int(np.ceil(center_x+radius)+1)):
            for pixel_y in range(int(np.floor(center_y - radius)), int(np.ceil(center_y + radius) + 1)):
                if((pixel_x - center_x)**2 + (pixel_y - center_y)**2 <= radius**2):
                    targetBinaryImage[pixel_x+image_half_size, pixel_y+image_half_size] = 1

    targetBinaryImage[image_half_size, image_half_size] = 0.5

    targetBinaryImage[10, 10] = 0.5

    plt.imshow(targetBinaryImage, cmap='Greys', interpolation='nearest', origin = 'lower',
                   extent=(-image_half_size, image_half_size, -image_half_size, image_half_size))
    plt.colorbar()
    plt.show()

    return targetBinaryImage

def get_image_parameters_optimized():

    import deeptrack
    from numpy.random import randint, uniform, normal, choice


    from math import pi, floor
    
    
    particle_number = floor(uniform(2,25))
    first_particle_range = 50
    other_particle_range = 90
    particle_distance = 10
    
    (particles_center_x, particles_center_y) = deeptrack.particle_positions(particle_number, first_particle_range, other_particle_range, particle_distance)
    image_parameters = {}

    image_parameters['Particle Center X List'] = particles_center_x
    image_parameters['Particle Center Y List'] = particles_center_y
    image_parameters['Particle Radius List'] = uniform(2.5, 4, particle_number)
    
    mylist = []
    for i in range(particle_number):
        mylist.append([1, ])
    
    image_parameters['Particle Bessel Orders List'] = mylist
    
    mylist2 = []
    for i in range(particle_number):
        mylist2.append([uniform(0.5,0.7,1), ])
    
    image_parameters['Particle Intensities List'] = mylist2
    
    image_parameters['Image Half-Size'] = 100
    image_parameters['Image Background Level'] = uniform(.2, .4)
    image_parameters['Signal to Noise Ratio'] = uniform(3,6)
    image_parameters['Gradient Intensity'] = uniform(0, 0.2)
    image_parameters['Gradient Direction'] = uniform(-pi, pi)
    image_parameters['Ellipsoid Orientation'] = uniform(-pi, pi, particle_number)
    image_parameters['Ellipticity'] = 1
    
    
    return image_parameters

def get_padding(input, input_size, n):
    """Adds padding to the input image
    Inputs:
    input: the input image
    input_size: the size of the input image
    n: the input image dimensions are changed to be divisible by 2**n

    Outputs:
    padding: the padding that was used
    """
    n = 4
    C0 = 2 ** (n - 1)
    C1 = 2 ** (n - 1)
    if (input_size[0] % 8 != 0):
        top_pad = (input_size[0] % (2 * n) // 2); bottom_pad = (input_size[0] % (2 * n) - top_pad)
    else:
        top_pad = 0;bottom_pad = 0;C0 = 0
    if input_size[1] % 8 != 0:
        left_pad = (input_size[1] % (2 * n) // 2); right_pad = (input_size[1] % (2 * n) - left_pad)
    else:
        left_pad = 0; right_pad = 0; C1 = 0
    padding = ((C0 - top_pad, C0 - bottom_pad), (C1 - left_pad, C1 - right_pad))
    
    return(padding)


def get_image_generator(image_parameters_function=lambda : get_image_parameters(), max_number_of_images=1e+9):
    """Generator of particle images.
    
    Inputs:
    image_parameters_function: lambda function to generate the image parameters (this is typically get_image_parameters())
    max_number_of_images: maximum number of images to be generated (positive integer)
        
    Outputs:
    image_number: image number in the current generation cycle
    image: image of the particles [2D numpy array of real numebrs betwen 0 and 1]
    image_parameters: list with the values of the image parameters in a dictionary:
        image_parameters['Particle Center X List']
        image_parameters['Particle Center Y List']
        image_parameters['Particle Radius List']
        image_parameters['Particle Bessel Orders List']
        image_parameters['Particle Intensities List']
        image_parameters['Image Half-Size']
        image_parameters['Image Background Level']
        image_parameters['Signal to Noise Ratio']
        image_parameters['Gradient Intensity']
        image_parameters['Gradient Direction']
        image_parameters['Ellipsoid Orientation']
        image_parameters['Ellipticity']
    """    
    import deeptrack as dt

    image_number = 0
    while image_number<max_number_of_images:
        
        image_parameters = image_parameters_function()
        image = dt.generate_image(image_parameters)
        target = get_target_binary_image(image_parameters)

        yield image_number, image, image_parameters, target
        image_number += 1

def generate_image_with_gpu(image_parameters):
    
    """Generate image with particles.
    
    Input:
    image_parameters: list with the values of the image parameters in a dictionary:
        image_parameters['Particle Center X List']
        image_parameters['Particle Center Y List']
        image_parameters['Particle Radius List']
        image_parameters['Particle Bessel Orders List']
        image_parameters['Particle Intensities List']
        image_parameters['Image Half-Size']
        image_parameters['Image Background Level']
        image_parameters['Signal to Noise Ratio']
        image_parameters['Gradient Intensity']
        image_parameters['Gradient Direction']
        image_parameters['Ellipsoid Orientation']
        image_parameters['Ellipticity']
        
    Note: image_parameters is typically obained from the function get_image_parameters()
        
    Output:
    image: image of the particle [2D numpy array of real numbers betwen 0 and 1]
    """


    from numpy import meshgrid, arange, ones, zeros, sin, cos, sqrt, clip, array
    from scipy.special import jv as bessel
    from numpy.random import poisson as poisson

    from numba import jit, cuda

    @jit(target = "cuda")
    def part_prof(bessel_order,intensity):
        image_particle = 4 * particle_bessel_order**2.5 * (bessel(particle_bessel_order, elliptical_distance_from_particle) / elliptical_distance_from_particle)**2
        image_particles = image_particles + particle_intensity * image_particle


    
    particle_center_x_list = image_parameters['Particle Center X List']
    particle_center_y_list = image_parameters['Particle Center Y List']
    particle_radius_list = image_parameters['Particle Radius List']
    particle_bessel_orders_list = image_parameters['Particle Bessel Orders List']
    particle_intensities_list = image_parameters['Particle Intensities List']
    image_half_size = image_parameters['Image Half-Size'] 
    image_background_level = image_parameters['Image Background Level']
    signal_to_noise_ratio = image_parameters['Signal to Noise Ratio']
    gradient_intensity = image_parameters['Gradient Intensity']
    gradient_direction = image_parameters['Gradient Direction']
    ellipsoidal_orientation_list = image_parameters['Ellipsoid Orientation']
    ellipticity = image_parameters['Ellipticity']
    

    ### CALCULATE IMAGE PARAMETERS
    # calculate image full size
    image_size = image_half_size * 2 + 1

    # calculate matrix coordinates from the center of the image
    image_coordinate_x, image_coordinate_y = meshgrid(arange(-image_half_size, image_half_size + 1), 
                                                      arange(-image_half_size, image_half_size + 1), 
                                                      sparse=False, 
                                                      indexing='ij')

    


    ### CALCULATE BACKGROUND
    # initialize the image at the background level
    image_background = ones((image_size, image_size)) * image_background_level
    
    # add gradient to image background
    if gradient_intensity!=0:
        image_background = image_background + gradient_intensity * (image_coordinate_x * sin(gradient_direction) + 
                                                                    image_coordinate_y * cos(gradient_direction) ) / (sqrt(2) * image_size)

    

    ### CALCULATE IMAGE PARTICLES
    image_particles = zeros((image_size, image_size))
    for particle_center_x, particle_center_y, particle_radius, particle_bessel_orders, particle_intensities, ellipsoidal_orientation in zip(particle_center_x_list, particle_center_y_list, particle_radius_list, particle_bessel_orders_list, particle_intensities_list, ellipsoidal_orientation_list):
    
    
        # calculate the radial distance from the center of the particle 
        # normalized by the particle radius
        radial_distance_from_particle = sqrt((image_coordinate_x - particle_center_x)**2 
                                         + (image_coordinate_y - particle_center_y)**2 
                                         + .001**2) / particle_radius
        

        # for elliptical particles
        rotated_distance_x = (image_coordinate_x - particle_center_x)*cos(ellipsoidal_orientation) + (image_coordinate_y - particle_center_y)*sin(ellipsoidal_orientation)
        rotated_distance_y = -(image_coordinate_x - particle_center_x)*sin(ellipsoidal_orientation) + (image_coordinate_y - particle_center_y)*cos(ellipsoidal_orientation)
        
        
        elliptical_distance_from_particle = sqrt((rotated_distance_x)**2 
                                         + (rotated_distance_y / ellipticity)**2 
                                         + .001**2) / particle_radius


        # calculate particle profile
        for particle_bessel_order, particle_intensity in zip(particle_bessel_orders, particle_intensities):
            part_prof(particle_bessel_order, particle_intensity)

    
    

    # calculate image without noise as background image plus particle image
    image_particles_without_noise = clip(image_background + image_particles, 0, 1)

    ### ADD NOISE
    image_particles_with_noise = poisson(image_particles_without_noise * signal_to_noise_ratio**2) / signal_to_noise_ratio**2
    

    return image_particles_with_noise

def train_deep_learning_network(
    network,
    image_generator,
    sample_sizes = (32, 128, 512, 2048),
    iteration_numbers = (3001, 2001, 1001, 101),
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

    #number_of_outputs = 3
    
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
            
            #output_shape = (sample_size, number_of_outputs)
            targets = np.zeros(input_shape)
            
            for image_number, image, image_parameters, target in image_generator():
                if image_number>=sample_size:
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

            if not(iteration%int(verbose**-1)):
                print('Sample size %6d   iteration number %6d Time %10.2f ms' % (sample_size, iteration + 1, mse, mae, iteration_time * 1000))
                
    return training_history
