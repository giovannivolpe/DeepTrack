def get_target_binary_image(image_parameters):
    """Create and return binary target image given image parameters
    Input: Image parameters
    Output: Binary image of the input image size where pixels containing particles are marked as ones, while rest are zeros
    """

    import numpy as np


    particle_center_x_list = image_parameters['Particle Center X List']
    particle_center_y_list = image_parameters['Particle Center Y List']
    particle_radius_list = image_parameters['Particle Radius List']
    image_half_size = image_parameters['Image Half-Size']

    targetBinaryImage = np.zeros((2 * image_half_size + 1, 2 * image_half_size + 1))

    for particle_index in range(0, len(particle_center_x_list)):
        center_x = particle_center_x_list[particle_index]
        center_y = particle_center_y_list[particle_index]
        radius = particle_radius_list[particle_index]

        """Loops over all pixels with center in coordinates = [ceil(center - radius): floor(center + radius)]. Adds the ones with
        center within radius.
        """
        for pixel_x in range(int(np.floor(center_x - radius)), int(np.ceil(center_x + radius) + 1)):
            for pixel_y in range(int(np.floor(center_y - radius)), int(np.ceil(center_y + radius) + 1)):
                if ((pixel_x - center_x) ** 2 + (pixel_y - center_y) ** 2 <= radius ** 2):
                    targetBinaryImage[pixel_x + image_half_size, pixel_y + image_half_size] = 1

    #    plt.imshow(targetBinaryImage, cmap='Greys', interpolation='nearest', origin = 'lower',
    #                  extent=(-image_half_size, image_half_size, -image_half_size, image_half_size))
    #  plt.colorbar()
    # plt.show()

    return targetBinaryImage

def get_image_parameters_optimized():
    import deeptrack
    from numpy.random import randint, uniform, normal, choice

    from math import pi, floor

    particle_number = floor(uniform(10, 30))
    first_particle_range = 10
    other_particle_range = 200
    particle_distance = 30

    (particles_center_x, particles_center_y) = deeptrack.particle_positions(particle_number, first_particle_range,
                                                                            other_particle_range, particle_distance)
    image_parameters = {}

    image_parameters['Particle Center X List'] = particles_center_x
    image_parameters['Particle Center Y List'] = particles_center_y
    image_parameters['Particle Radius List'] = uniform(2, 5, particle_number)

    mylist = []
    for i in range(particle_number):
        mylist.append([1, ])

    image_parameters['Particle Bessel Orders List'] = mylist

    mylist2 = []
    for i in range(particle_number):
        mylist2.append([uniform(0.5, 0.7, 1), ])

    image_parameters['Particle Intensities List'] = mylist2

    image_parameters['Image Half-Size'] = 256
    image_parameters['Image Background Level'] = uniform(.3, .5)
    image_parameters['Signal to Noise Ratio'] = uniform(3, 5)
    image_parameters['Gradient Intensity'] = uniform(0.25, 0.75)
    image_parameters['Gradient Direction'] = uniform(-pi, pi)
    image_parameters['Ellipsoid Orientation'] = uniform(-pi, pi, particle_number)
    image_parameters['Ellipticity'] = 1

    return image_parameters

def get_image_generator(image_parameters_function=lambda: get_image_parameters(), max_number_of_images=1e+9):
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
    while image_number < max_number_of_images:
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

    @jit(target="cuda")
    def part_prof(bessel_order, intensity):
        image_particle = 4 * particle_bessel_order ** 2.5 * (bessel(particle_bessel_order,
                                                                    elliptical_distance_from_particle) / elliptical_distance_from_particle) ** 2
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
    if gradient_intensity != 0:
        image_background = image_background + gradient_intensity * (image_coordinate_x * sin(gradient_direction) +
                                                                    image_coordinate_y * cos(gradient_direction)) / (
                                       sqrt(2) * image_size)

    ### CALCULATE IMAGE PARTICLES
    image_particles = zeros((image_size, image_size))
    for particle_center_x, particle_center_y, particle_radius, particle_bessel_orders, particle_intensities, ellipsoidal_orientation in zip(
            particle_center_x_list, particle_center_y_list, particle_radius_list, particle_bessel_orders_list,
            particle_intensities_list, ellipsoidal_orientation_list):

        # calculate the radial distance from the center of the particle
        # normalized by the particle radius
        radial_distance_from_particle = sqrt((image_coordinate_x - particle_center_x) ** 2
                                             + (image_coordinate_y - particle_center_y) ** 2
                                             + .001 ** 2) / particle_radius

        # for elliptical particles
        rotated_distance_x = (image_coordinate_x - particle_center_x) * cos(ellipsoidal_orientation) + (
                    image_coordinate_y - particle_center_y) * sin(ellipsoidal_orientation)
        rotated_distance_y = -(image_coordinate_x - particle_center_x) * sin(ellipsoidal_orientation) + (
                    image_coordinate_y - particle_center_y) * cos(ellipsoidal_orientation)

        elliptical_distance_from_particle = sqrt((rotated_distance_x) ** 2
                                                 + (rotated_distance_y / ellipticity) ** 2
                                                 + .001 ** 2) / particle_radius

        # calculate particle profile
        for particle_bessel_order, particle_intensity in zip(particle_bessel_orders, particle_intensities):
            part_prof(particle_bessel_order, particle_intensity)

    # calculate image without noise as background image plus particle image
    image_particles_without_noise = clip(image_background + image_particles, 0, 1)

    ### ADD NOISE
    image_particles_with_noise = poisson(
        image_particles_without_noise * signal_to_noise_ratio ** 2) / signal_to_noise_ratio ** 2

    return image_particles_with_noise

def plot_sample_image(image, image_parameters, figsize=(15, 5)):
    """Plot a sample image.

    Inputs:
    image: image of the particles
    image_parameters: list with the values of the image parameters
    figsize: figure size [list of two positive numbers]


    Output: none
    """

    import matplotlib.pyplot as plt

    image_half_size = image_parameters['Image Half-Size']

    plt.figure(figsize=figsize)

    plt.subplot(1, 2, 1)
    plt.imshow(image, cmap='gray', vmin=0, vmax=1, origin='lower', aspect='equal',
               extent=(-image_half_size, image_half_size, -image_half_size, image_half_size))
    plt.xlabel('y (px)', fontsize=16)
    plt.ylabel('x (px)', fontsize=16)

    plt.subplot(1, 2, 2)

    binary_image = get_target_binary_image(image_parameters)
    plt.imshow(binary_image, cmap='gray', vmin=0, vmax=1, origin='lower', aspect='equal',
               extent=(-image_half_size, image_half_size, -image_half_size, image_half_size))
    plt.xlabel('y (px)', fontsize=16)
    plt.ylabel('x (px)', fontsize=16)

    plt.show()
    
def save_image_and_target(image_number,image,image_parameters, image_path, target_path):
    """Creates an image and target from image parameters, saves them to the specified paths"""
    import cv2
    cv2.imwrite(image_path + "/" + str(image_number) + '.png', image*255)
    target = get_target_binary_image(image_parameters)
    cv2.imwrite(target_path + "/" + str(image_number) + '.png', target*255)

def adjustData(img,mask,flag_multi_class,num_class):
    import numpy as np

    if(flag_multi_class):
        img = img / 255
        mask = mask[:,:,:,0] if(len(mask.shape) == 4) else mask[:,:,0]
        new_mask = np.zeros(mask.shape + (num_class,))
        for i in range(num_class):
            #for one pixel in the image, find the class in mask and convert it into one-hot vector
            #index = np.where(mask == i)
            #index_mask = (index[0],index[1],index[2],np.zeros(len(index[0]),dtype = np.int64) + i) if (len(mask.shape) == 4) else (index[0],index[1],np.zeros(len(index[0]),dtype = np.int64) + i)
            #new_mask[index_mask] = 1
            new_mask[mask == i,i] = 1
        new_mask = np.reshape(new_mask,(new_mask.shape[0],new_mask.shape[1]*new_mask.shape[2],new_mask.shape[3])) if flag_multi_class else np.reshape(new_mask,(new_mask.shape[0]*new_mask.shape[1],new_mask.shape[2]))
        mask = new_mask
    elif(np.max(img) > 1):
        img = img / 255
        mask = mask /255
        mask[mask > 0.5] = 1
        mask[mask <= 0.5] = 0
    return (img,mask)

def trainGenerator(batch_size,train_path,image_folder,mask_folder,aug_dict,image_color_mode = "grayscale",
                    mask_color_mode = "grayscale",image_save_prefix  = "image",mask_save_prefix  = "mask",
                    flag_multi_class = False,num_class = 2,save_to_dir = None,target_size = (256,256),seed = 1):
    '''
    can generate image and mask at the same time
    use the same seed for image_datagen and mask_datagen to ensure the transformation for image and mask is the same
    if you want to visualize the results of generator, set save_to_dir = "your path"
    '''
    from keras.preprocessing.image import ImageDataGenerator

    import cv2
    image_datagen = ImageDataGenerator(**aug_dict)
    mask_datagen = ImageDataGenerator(**aug_dict)
    image_generator = image_datagen.flow_from_directory(
        train_path,
        classes = [image_folder],
        class_mode = None,
        color_mode = image_color_mode,
        target_size = target_size,
        batch_size = batch_size,
        save_to_dir = save_to_dir,
        save_prefix  = image_save_prefix,
        seed = seed)
    mask_generator = mask_datagen.flow_from_directory(
        train_path,
        classes = [mask_folder],
        class_mode = None,
        color_mode = mask_color_mode,
        target_size = target_size,
        batch_size = batch_size,
        save_to_dir = save_to_dir,
        save_prefix  = mask_save_prefix,
        seed = seed)
    train_generator = zip(image_generator, mask_generator)
    for (img,mask) in train_generator:
        img,mask = adjustData(img,mask,flag_multi_class,num_class)
        yield (img,mask)

def testGenerator(test_path,num_image = 30,target_size = (256,256),flag_multi_class = False,as_gray = True):
    import skimage.transform as trans
    import skimage.io as io
    import os
    import numpy as np

    for i in range(num_image):
        img = io.imread(os.path.join(test_path,"%d.png"%i),as_gray = as_gray)
        img = img / 255
        img = trans.resize(img,target_size)
        img = np.reshape(img,img.shape+(1,)) if (not flag_multi_class) else img
        img = np.reshape(img,(1,)+img.shape)
        yield img

def geneTrainNpy(image_path,mask_path,flag_multi_class = False,num_class = 2,image_prefix = "image",mask_prefix = "mask",image_as_gray = True,mask_as_gray = True):
    import skimage.io as io
    import glob
    import os
    import numpy as np

    image_name_arr = glob.glob(os.path.join(image_path,"%s*.png"%image_prefix))
    image_arr = []
    mask_arr = []
    for index,item in enumerate(image_name_arr):
        img = io.imread(item,as_gray = image_as_gray)
        img = np.reshape(img,img.shape + (1,)) if image_as_gray else img
        mask = io.imread(item.replace(image_path,mask_path).replace(image_prefix,mask_prefix),as_gray = mask_as_gray)
        mask = np.reshape(mask,mask.shape + (1,)) if mask_as_gray else mask
        img,mask = adjustData(img,mask,flag_multi_class,num_class)
        image_arr.append(img)
        mask_arr.append(mask)
    image_arr = np.array(image_arr)
    mask_arr = np.array(mask_arr)
    return image_arr,mask_arr

def labelVisualize(num_class,color_dict,img):
    import numpy as np

    img = img[:,:,0] if len(img.shape) == 3 else img
    img_out = np.zeros(img.shape + (3,))
    for i in range(num_class):
        img_out[img == i,:] = color_dict[i]
    return img_out / 255

def saveResult(save_path,npyfile,flag_multi_class = False,num_class = 2):
    import skimage.io as io
    import os

    for i,item in enumerate(npyfile):
        img = labelVisualize(num_class,COLOR_DICT,item) if flag_multi_class else item[:,:,0]
        io.imsave(os.path.join('data/particles/train/images',"particle_%d.png"%image_number),image


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

