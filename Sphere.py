import numpy as np
from mayavi import mlab
from pdb import set_trace

#------------------------------------------------------------------------------
#
#------------------------------------------------------------------------------
class Sphere(object):
    """
    Rigid sphere used for contact
    """
    #------------------------------------------------------------------------------
    #
    #------------------------------------------------------------------------------
    def __init__(self, r, C):
        assert r >  0
        assert C.shape == (3,)
        self.r = r
        self.C = C.astype(np.float)
        self.initC = np.copy(self.C)

    #------------------------------------------------------------------------------
    #
    #------------------------------------------------------------------------------
    def Move(self, u):
        assert u.shape == (3,)
        self.C += u.ravel()

    #------------------------------------------------------------------------------
    #
    #------------------------------------------------------------------------------
    def setCenterPosition(self, C):
        assert C.shape == (3,)
        self.C = np.copy(C)
    #------------------------------------------------------------------------------
    #
    #------------------------------------------------------------------------------
    def plot(self, color = (0.,0.,1.), opacity = 0.5):
        xc, yc, zc = self.C
        [phi,theta] = np.mgrid[0:2*np.pi:20j,0:np.pi:20j]  #increase the numbers before j for better resolution but more time
        r = self.r
        x = r*np.cos(phi)*np.sin(theta) + xc
        y = r*np.sin(phi)*np.sin(theta) + yc
        z = r*np.cos(theta) + zc
        try:
            sphere = self.mayaSphere
            Sc = sphere.mlab_source
            Sc.set(x=x, y=y, z=z)
        except AttributeError:
            self.mayaSphere = mlab.mesh(x, y, z, color = color,
                    opacity = opacity )

    #------------------------------------------------------------------------------
    #
    #------------------------------------------------------------------------------
    def getPointOnSurface(self, phi, theta):
        x = self.r*np.cos(phi)*np.sin(theta)
        y = self.r*np.sin(phi)*np.sin(theta)
        z = self.r*np.cos(theta)
        return self.C + np.array([x, y, z])

    #------------------------------------------------------------------------------
    #
    #------------------------------------------------------------------------------
    def SamplePointsOnSurface(self, beta1, beta2):
        assert beta1.ndim == 1
        assert beta2.ndim == 1
        nbeta1 = len(beta1)
        nbeta2 = len(beta2)
        SPt = np.zeros((nbeta1, nbeta2, 3))
        for ii, beta1ii in enumerate(beta1):
            for jj, beta2jj in enumerate(beta2):
                SPt[ii,jj] = self.getPointOnSurface(beta1ii, beta2jj)
        return SPt
