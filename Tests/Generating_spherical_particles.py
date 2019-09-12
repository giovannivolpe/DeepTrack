


import sys
sys.path.append("../DeepTrack")

from DeepTrack.Generators import Generator
from DeepTrack.Particles import SphericalParticle
from DeepTrack.Backend.Distributions import uniform_random
import numpy as np
import matplotlib.pyplot as plt

from timeit import default_timer as timer

'''
    Simple example showcassing current possiblites.
    
    A generator is created with certai optical properties. To the generator two particles are added.
 '''




G = Generator(
    shape = (64,64),    # Desired output shape of the generator.
    NA = 0.7,           # The NA of the optical system.
    pixel_size=0.1,     # The pixel_size of the optical system (mu^-1).
    wavelength=0.68     # The wavelength of the illuminating source (mu).
)


G.add_particle(SphericalParticle(
    radius = 0.5,                                   # Radius of the generated particles
    intensity = 0.5,                                # Peak intensity of the generated particle
    position_distribution=uniform_random((64,64))   # The distrbution from which to draw the position of the particle
))
G.add_particle(SphericalParticle(
    radius = 0.2,       
    intensity = 0.5,    
    position_distribution=uniform_random((64,64))
))

# Time the average generation time for 100 particles
start = timer()

for i in range(100):
    image, position = G.get()

end = timer()

print("Generates (128,128) particles at {0}s per image".format((end - start)/100))
# Show one typical particle

plt.gray()
for i in range(1):
    image, position = G.get()
    plt.imshow(image)
    plt.scatter(position[0], position[1], 2)
    plt.show()
