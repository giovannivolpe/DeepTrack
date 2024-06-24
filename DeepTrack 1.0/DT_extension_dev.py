

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
    
    image_number = 0
    while image_number<max_number_of_images:
        
        image_parameters = image_parameters_function()

        import time
        t0 = time.time()
        
        image = generate_image_with_gpu(image_parameters)

        t1 = time.time()
        print("calctime with gpu", t1-t0)

        image2 = generate_image(image_parameters)

        t2 = time.time()
        print("calctime without gpu", t2-t1)

        yield image_number, image,image2, image_parameters
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
    from numpy.random import poisson as poisson
    
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
    
    ### CALCULATE BACKGROUND
    # initialize the image at the background level
    image_background = ones((image_size, image_size)) * image_background_level

    # calculate matrix coordinates from the center of the image
    image_coordinate_x, image_coordinate_y = meshgrid(arange(-image_half_size, image_half_size + 1), 
                                                      arange(-image_half_size, image_half_size + 1), 
                                                      sparse=False, 
                                                      indexing='ij')
    
    # add gradient to image background
    if gradient_intensity!=0:
        image_background = image_background + gradient_intensity * (image_coordinate_x * sin(gradient_direction) + 
                                                                    image_coordinate_y * cos(gradient_direction) ) / (sqrt(2) * image_size)

    ### CALCULATE IMAGE PARTICLES
    image_particles = zeros((image_size, image_size))


    # shift particle centers so that the origin is in top left corner of image instead of the center. 
    #TODO: this solution is not optimal, work in same coordinate system from the beginning instead.
    particle_center_x_list = particle_center_x_list + image_half_size
    particle_center_y_list = particle_center_y_list + image_half_size

    # calculate the particle profiles of all particles and add them to image_particles
    calc_particle_profile(particle_center_x_list, particle_center_y_list,particle_radius_list, image_particles,particle_intensities_list)
    

    # calculate image without noise as background image plus particle image
    image_particles_without_noise = clip(image_background + image_particles, 0, 1)

    ### ADD NOISE
    image_particles_with_noise = poisson(image_particles_without_noise * signal_to_noise_ratio**2) / signal_to_noise_ratio**2
    

    return image_particles_with_noise

def calc_particle_profile(particle_center_x_list, particle_center_y_list,particle_radius_list, image_particles,particle_intensities_list):
    from numba import cuda
    from math import ceil,exp


    # the cuda kernel calculating the value of the Gauss function for each pixel in out image
    @cuda.jit
    def part_prof(d_dist_x,d_dist_y,d_radiuses,d_img_part,d_particle_intensities):

        x, y = cuda.grid(2)

        if x >= d_img_part.shape[0] and y >= d_img_part.shape[1]:
            # Quit if (x, y) is outside of valid C boundary
            return

        for i in range(d_dist_x.shape[0]):

            tmp = d_particle_intensities[i][0][0]*exp(-((x-d_dist_x[i])**2/(2*d_radiuses[i]**2) + (y-d_dist_y[i])**2/(2*d_radiuses[i]**2)))

            d_img_part[x, y] = d_img_part[x,y] + tmp

    # define threads per block and blocks per grid. This dictates how our cuda kernel devides tasks.
    TPB = 32
    threadsperblock = (TPB, TPB)
    blockspergrid_x = int(ceil(image_particles.shape[0] / threadsperblock[0]))
    blockspergrid_y = int(ceil(image_particles.shape[1] / threadsperblock[1]))
    blockspergrid = (blockspergrid_x, blockspergrid_y)
    
    # introduce stream dictating the order of data being sent to GPU
    # create an cuda object of each object we wish to have handled by the GPU. This is because data transfer to and from GPU is costly.
    stream = cuda.stream()
    d_pos_x = cuda.to_device(particle_center_x_list,stream = stream)
    d_pos_y = cuda.to_device(particle_center_y_list,stream = stream)
    d_radiuses = cuda.to_device(particle_radius_list,stream = stream)
    d_img_part = cuda.to_device(image_particles,stream = stream)
    d_particle_intensities = cuda.to_device(particle_intensities_list,stream = stream)

    # call the cuda kernel
    part_prof[blockspergrid, threadsperblock](d_pos_x,d_pos_y,d_radiuses, d_img_part,d_particle_intensities)

    # retrieve our image particle matrix from GPU
    d_img_part.copy_to_host(image_particles, stream = stream)

def generate_image(image_parameters):
    
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
            image_particle = 4 * particle_bessel_order**2.5 * (bessel(particle_bessel_order, elliptical_distance_from_particle) / elliptical_distance_from_particle)**2
            image_particles = image_particles + particle_intensity * image_particle

        

    # calculate image without noise as background image plus particle image
    image_particles_without_noise = clip(image_background + image_particles, 0, 1)

    ### ADD NOISE
    image_particles_with_noise = poisson(image_particles_without_noise * signal_to_noise_ratio**2) / signal_to_noise_ratio**2
    

    return image_particles_with_noise

    
   