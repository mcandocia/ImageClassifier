from distort import distortion
import cv2
import constants as co
from constants import urand 
import random
import numpy as np

class distortion_wheel:
    def __init__(self,xdim,ydim):
        """remember xdim and ydim are flipped >_<"""
        self.warp_distort = distortion(xdim,ydim)
        self.wave_distort = distortion(xdim,ydim)
        self.scale_distort = distortion(xdim,ydim)
        self.rot_distort = distortion(xdim,ydim)
        self.displace = distortion(xdim,ydim)
        self.make_priorities()
        self.initialize_distortions()
    def set_warp(self):
        self.warp_distort.create_sinusoidal_warp(
            urand(co.warp_ax),
            urand(co.warp_ay),
            urand(co.warp_perx),
            urand(co.warp_pery),
            urand(co.warp_phax),
            urand(co.warp_phay)
            )
    def set_wave(self):
        self.wave_distort.create_sinusoidal_wave(
            urand(co.wave_ax),
            urand(co.wave_ay),
            urand(co.wave_perx),
            urand(co.wave_pery),
            urand(co.wave_phax),
            urand(co.wave_phay)
            )
    def set_tint(self):
        self.tint = [urand(co.rgb_shift) for _ in range(3)]
    def set_scale(self):
        self.scx = urand(co.scale_x)
        self.scy = urand(co.scale_y)
        self.scale_distort.calculate_scale(
            (self.scx,self.scy),
            offset=(
                urand(co.scale_x_offset),
                urand(co.scale_y_offset)
                )
            )

    def set_rotation(self):
        self.rot_distort.calculate_rotation(
            urand(co.rot_theta),
            offset = (
                urand(co.rot_offset_x),
                urand(co.rot_offset_y)
                )
            )

    def set_offset(self):
        self.displace.create_affine(
            1.,
            1.,
            0,
            0,
            urand(co.x_offset),
            urand(co.y_offset)
            )
    def initialize_distortions(self):
        self.set_warp()
        self.set_wave()
        self.set_scale()
        self.set_rotation()
        self.set_offset()
        self.set_tint()
        self.make_priorities()
    def make_priorities(self):
        self.wav_priority = urand(co.wave)
        self.warp_priority = urand(co.warp)
        self.affine_priority = urand(co.affine)
        self.maxv = max(self.wav_priority,self.warp_priority,self.affine_priority)
        self.minv = min(self.wav_priority,self.warp_priority,self.affine_priority)

    def rotate_values(self, num_distorts=1):
        #not particularly safe to use exec(), but should be fine
        distort_list = ['self.set_' + x + '()' for x in \
                            ['scale','wave','warp','offset','rotation','tint']]
        funcs = random.sample(distort_list,num_distorts)
        for func in funcs:
            exec(func)
        self.make_priorities()
    def process_image(self,image):
        #handle tint
        for j, tint in enumerate(self.tint):
            image[:,:,j] = np.uint8(
                np.maximum(0,np.minimum(255,
                                        np.uint(image[:,:,j]) + tint
                                        )))
        #now do distortions
        if self.wav_priority == self.maxv:
            image = self.wave_distort.process_image(image)
            if self.warp_priority == self.minv:
                if (self.scx + self.scy)/2 > 1:
                    image = self.scale_distort.process_image(image)
                    image = self.rot_distort.process_image(image)
                else:
                    image = self.rot_distort.process_image(image)
                    image = self.scale_distort.process_image(image)
                image = self.warp_distort.process_image(image)
            else:
                image = self.warp_distort.process_image(image)            
                if (self.scx + self.scy)/2 > 1:
                    image = self.scale_distort.process_image(image)
                    image = self.rot_distort.process_image(image)
                else:
                    image = self.rot_distort.process_image(image)
                    image = self.scale_distort.process_image(image)
        elif self.warp_priority == self.maxv:
            image = self.warp_distort.process_image(image)
            if self.wav_priority == self.minv:
                if (self.scx + self.scy)/2 > 1:
                    image = self.scale_distort.process_image(image)
                    image = self.rot_distort.process_image(image)
                else:
                    image = self.rot_distort.process_image(image)
                    image = self.scale_distort.process_image(image)
                image = self.wave_distort.process_image(image)
            else:
                image = self.wave_distort.process_image(image)            
                if (self.scx + self.scy)/2 > 1:
                    image = self.scale_distort.process_image(image)
                    image = self.rot_distort.process_image(image)
                else:
                    image = self.rot_distort.process_image(image)
                    image = self.scale_distort.process_image(image)
        else:
            if (self.scx + self.scy)/2 > 1:
                image = self.scale_distort.process_image(image)
                image = self.rot_distort.process_image(image)
            else:
                image = self.rot_distort.process_image(image)
                image = self.scale_distort.process_image(image)
            if self.wav_priority == self.minv:
                image = self.warp_distort.process_image(image)
                image = self.wave_distort.process_image(image)
            else:
                image = self.wave_distort.process_image(image)
                image = self.warp_distort.process_image(image)
        #displacement
        self.displace.process_image(image)
        return image
