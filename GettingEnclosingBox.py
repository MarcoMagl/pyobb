import numpy as np
from numpy import array, zeros_like
from Beams.GeometricallyExact import SR3DBeams
from mayavi import mlab


try:
    mlab.close()
except AttributeError:
    pass
# coordinates nodes beam 1
nnY0 = 2
X = np.zeros((nnY0, 3), dtype = float)
X[:, 0] = np.linspace(-10,10,nnY0)
con = array([[0,1]]).astype(int)
E = 1.
# we will get a circular cross section of radius = 0.5
b = 0.6
h = 0.3
# 6 dofs per node
ndofsPern = 6
# we just set the x coordinates
# dofs per node
dperN = np.arange(ndofsPern * 2).reshape(-1, ndofsPern)
# Construct beam elements
nEl = len(con)
el = np.zeros(nEl, dtype = np.object)

t1 = array([0,1,0])
t2 = array([0,0,1] )
for ii in range(nEl):
    el[ii] = SR3DBeams(X = X[con[ii],], E = E,
                        nu = 0.3, b = b, h = h,
                        nID = con[ii], dofs = dperN[con[ii],],
                        E1= t1,E2 = t2,
                       rot_vars_storage= 1,
                       shapeCrossSec='Elliptical')

from tvtk.tools import visual
u_I = zeros_like(X)
v_I = zeros_like(X)
package_plot = 'mayavi'
nS = 10
ntheta = 8
el[0].plotSurface(X, u_I, v_I, package_plot, nS, ntheta, opacity = 1.,
        color = (1., 0., 0.))

meshobj = el[0].mesh
src = meshobj.mlab_source.points
bounding_lim_min = np.min(meshobj.mlab_source.points, axis = 0)
bounding_lim_max = np.max(meshobj.mlab_source.points, axis = 0)
blim= np.vstack((bounding_lim_min, bounding_lim_max))
from itertools import product
vertices = np.asarray([i for i in product(blim[:,0], blim[:,1], blim[:,2])])
x,y,z = vertices.T[(0,1,2),]
glyph= mlab.points3d(x,y,z, color = (0.,1.,0.), scale_factor = 0.1)

try:
    engine = mayavi.engine
except NameError:
    from mayavi.api import Engine
    engine = Engine()
    engine.start()
if len(engine.scenes) == 0:
    engine.new_scene()
from mayavi.modules.outline import Outline
outline = Outline()
# a glyph is a module !!
engine.add_filter(outline, glyph)

mlab.show()


def check_Collision(blim1, blim2):
    """
    Check Collision of 2 Bounding Boxes
    #  ________
    # |\       |\
    # |_\______|_\
    # \ |      \ |
    #  \|_______\|
    #
    #
    """
    for blim in [blim1, blim2]:
       assert blim.shape== (2,3)
       assert np.all(blim[0] < blim[1]), 'first line should be\
       minimums while second line should correspond to maximums'

    xmin_1, ymin_1, zmin_1, xmax_1, ymax_1, zmax_1 = blim1.flatten()
    xmin_2, ymin_2, zmin_2, xmax_2, ymax_2, zmax_2 = blim2.flatten()

    # true if collision
    return ((x_max_1 >= x_min_2 and x_max_1 <= x_max_2) \
                    or (x_min_1 <= x_max_2 and x_min_1 >= x_min_2)) \
                    and ((y_max_1 >= y_min_2 and y_max_1 <= y_max_2) \
                    or (y_min_1 <= y_max_2 and y_min_1 >= y_min_2)) \
                    and ((z_max_1 >= z_min_2 and z_max_1 <= z_max_2) \
                    or (z_min_1 <= z_max_2 and z_min_1 >= z_min_2))



