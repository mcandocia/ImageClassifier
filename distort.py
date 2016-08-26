import cv2
import numpy as np
from scipy.interpolate import griddata


#class that contains basic grid info and then can generate wacky 
#distortions that can be used on images arbitrarily
#one distortion can be saved at a time but used indefinitely
#note that x is vertical and y is horizontal
class distortion:
    #supports linear and cubic interpolation
    def __init__(self,xdim,ydim,method = 'linear'):
        xdm = xdim - 1
        ydm = ydim - 1
        self.xdim = xdim
        self.method = method
        self.ydim = ydim
        self.grid_x, self.grid_y = np.mgrid[0:xdm:(xdim*1j),0:ydm:(ydim*1j)]
        self.source = np.array([[i,j] for i,j in\
                                    zip(self.grid_x.flat,self.grid_y.flat)])
    def create_sinusoidal_warp(self,ax,ay,perx,pery,phax,phay):
        """strange warping that crosses between dimensions"""
        sinfunc = lambda x, y: (
            x + ax * np.sin(np.pi*2/pery*(y-phay)),
            y + ay * np.sin(np.pi*2/perx*(x-phax))
            )
        self.destination = [sinfunc(x,y) for x,y in self.source]
        self.calculate_transformation()
    def create_sinusoidal_wave(self,ax,ay,perx,pery,phax,phay):
        """warping that creates wave-like compressions and rarefactions"""
        sinfunc = lambda x, y: (
            x + ax * np.sin(np.pi*2/perx*(x-phax)),
            y + ay * np.sin(np.pi*2/pery*(y-phay))
            )
        self.destination = [sinfunc(x,y) for x,y in self.source]
        self.calculate_transformation()
    def create_affine(self,mx,my,mxy,myx,dx,dy):
        tfunc = lambda x, y: (mx * x + mxy * y + dx, my * y + myx * x + dy)
        self.destination = [tfunc(x,y) for x,y in self.source]
        self.calculate_transformation()
    def calculate_rotation(self,theta,center=None,offset = None):
        """rotates image; more useful than other rotations since most
        calculations are done initially rather than for each image"""
        if center==None:
            if offset == None:
                offset = (0,0)
            center = (self.xdim/2 + offset[0],self.ydim/2 + offset[1])
        xtrans = (np.cos(theta),np.sin(theta))
        ytrans = (-np.sin(theta),np.cos(theta))
        dx = center[0] - (xtrans[0] * center[0] + xtrans[1] * center[1])
        dy = center[1] - (ytrans[0] * center[0] + ytrans[1] * center[1])
        self.create_affine(xtrans[0],ytrans[1],xtrans[1],ytrans[0],dx,dy)
    def calculate_scale(self,scales,center=None,offset=None):
        if center==None:
            if offset == None:
                offset = (0,0)
            center = (self.xdim/2 + offset[0],self.ydim/2 + offset[1])
        dx = center[0] * (1-scales[0])
        dy = center[1] * (1-scales[1])
        self.create_affine(scales[0],scales[1],0,0,dx,dy)
    def calculate_transformation(self):
        self.grid_z = griddata(
            self.destination,
            self.source,
            (self.grid_x,self.grid_y),
            method=self.method
            )
        map_x = np.append([],[ar[:,1] for ar in self.grid_z]).\
            reshape(self.xdim,self.ydim)
        map_y = np.append([],[ar[:,0] for ar in self.grid_z]).\
            reshape(self.xdim,self.ydim)
        self.mx32 = map_x.astype('float32')
        self.my32 = map_y.astype('float32')
    def process_image(self,img):
        if self.method=='linear':
            method = cv2.INTER_LINEAR
        else:
            method = cv2.INTER_CUBIC
        return cv2.remap(img,self.mx32,self.my32,method)
