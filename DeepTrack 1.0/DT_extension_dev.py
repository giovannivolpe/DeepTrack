

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

