#all of these are lower/upper bounds for a uniform distribution
from numpy import pi
import random

warp_ax = (-2,2)
warp_ay = (-2,2)
warp_perx = (70,200)
warp_pery = (70,200)
warp_phax = (-100,100)
warp_phay = (-100,100)

wave_ax = (-2,2)
wave_ay = (-2,2)
wave_perx = (70,200)
wave_pery = (70,200)
wave_phax = (-100,100)
wave_phay = (-100,100)

rot_theta = (-8*pi/180.,8*pi/180.)
rot_offset_x = [-30,30]
rot_offset_y = [-30,30]

scale_x = [0.9,1.1]
scale_y = [0.9,1.1]
scale_x_offset = [-15,15]
scale_y_offset = [-15,15]

x_offset = [-8,8]
y_offset = [-8,8]

#if tinting is used
rgb_shift = [-12,12]

#determines priority distribution for mappings

wave = (0,1)
warp = (0,1)
affine = (0,1)

#used for processing these intervals
def urand(tup):
    return random.uniform(tup[0],tup[1])
