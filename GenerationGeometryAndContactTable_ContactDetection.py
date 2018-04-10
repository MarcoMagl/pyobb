import numpy as np
from numpy import array, zeros
import matplotlib.pylab as plt
import FEM
from FEM import Model
from scipy import spatial
from numpy.linalg import norm
from pdb import set_trace
import matplotlib
from mayavi import mlab
from Beams.GeometricallyExact import SR3DBeams
from numpy.linalg import norm
from Utilities import collision_bounding_boxes, plot_sphere
from mayavi.modules.outline import Outline
import mayavi
from itertools import product

a = 0.1
b = 0.5 * a
xinterval= array([ 0, 1])
yinterval= array([ 0, 1])

# the horizontal yarns are the master yarns
XYM, XYS, conYM, conYS, ndYM, ndYS, nnYM, nnYS, X,\
con, nel, which_yarn_is_el, which_fam_of_yarn_is_el =\
FEM.generate_regular_lattice(4, 4, 12, 12,xinterval, yinterval)

nn = int(np.sum((nnYM, nnYS)))
dperN = np.arange(nn * 6).reshape(nn,6)
el = np.zeros(nel, dtype = np.object)
for ii in range(nel):
    nIDsi = con[ii]
    # id of the nodes of the el
    t3 = (X[con[ii, 1]] - X[con[ii, 0]]) / norm((X[con[ii, 1]] -\
        X[con[ii, 0]]))
    if which_fam_of_yarn_is_el[ii] == 0:
        t1 = array([0,1,0])
    else:
        t1 = array([-1,0,0])
    t2 = np.cross( t3, t1)
    el[ii] = SR3DBeams(X = X[con[ii]], E = 1.,
                            nu = 0.3, b = a, h = b,
                            nID = nIDsi, dofs = dperN[nIDsi],
                            E1= t1,E2 = t2,
                           rot_vars_storage= 1,
                           shapeCrossSec='Elliptical')

# Construct FE assembly
BeamMod = Model(el, nn, 6, X,\
                HasConstraints = 1, dperN = dperN,
                ConstantContactTable = 0)

BeamMod.create_unset_Contact_Table(0,
'curve_to_curve', alpha = 0.8, smooth_cross_section_vectors = True)
BeamMod.set_plotting_package('mayavi')

# construct connectiviy of the curves IN TERMS OF PAIR OF ELEMENT IDs
# TODO: add this to the function in FEM.py
elYMs = np.arange(np.product(conYM.shape[:2])).reshape(conYM.shape[:2])
elYSs= elYMs[-1,-1]+ 1+ np.arange(np.product(conYS.shape[:2])).reshape(conYS.shape[:2])
conCurveYM = np.zeros((conYM.shape[0], conYM.shape[1] - 1,2), dtype=int)
conCurveYS = np.zeros((conYS.shape[0], conYS.shape[1] - 1,2), dtype=int)

for ii, elYM in enumerate(elYMs):
    conCurveYM[ii,:,0] = elYMs[ii][:-1]
    conCurveYM[ii,:,1] = elYMs[ii][1:]

for ii, elYS in enumerate(elYSs):
    conCurveYS[ii,:,0] = elYSs[ii][:-1]
    conCurveYS[ii,:,1] = elYSs[ii][1:]

# store AABB limits
bYM = zeros( conCurveYM.shape[:2] + (2,3), dtype = float)
bYS = zeros( conCurveYS.shape[:2] + (2,3), dtype = float)
# center of the enclosing spheres enclosing the curves
ctrYM = zeros( bYM.shape[:2]  + (3,), dtype = float)
ctrYS = zeros( bYS.shape[:2] + (3,), dtype = float)
radYM = zeros( bYM.shape[:2], dtype = float)
radYS = zeros( bYS.shape[:2], dtype = float)

for i in range(bYM.shape[0]):
    for j in range(bYM.shape[1]):
         bYM[i, j] =\
                 BeamMod.get_bounding_box_one_curve(\
                         conCurveYM[i][j], nxi=3,\
                         ntheta = 4)
         ctrYM[i,j] = 0.5 * (bYM[i, j, 0] + bYM[i, j, 1])
         radYM[i,j] = norm(ctrYM[i,j]-bYM[i, j, 0])

for i in range(bYS.shape[0]):
    for j in range(bYS.shape[1]):
         bYS[i, j] =\
                 BeamMod.get_bounding_box_one_curve(\
                         conCurveYS[i][j], nxi=3,\
                         ntheta = 4)

         ctrYS[i,j] = 0.5 * (bYS[i, j, 0] + bYS[i, j, 1])
         radYS[i,j] = norm(ctrYS[i,j]-bYS[i, j, 0])


# brute force here, we do not put any a priori info
conCurveYM= conCurveYM.reshape(-1,2)
conCurveYS= conCurveYS.reshape(-1,2)

from mayavi.api import Engine
try:
    # mlab.get_engine().stop()
    engine =  mlab.get_engine()
except NameError:
    engine = Engine()
    engine.start()
if len(engine.scenes) == 0:
    engine.new_scene()
else:
    mlab.clf()



close_mat = np.zeros( ( np.product(ctrYM.shape[:2]),
    np.product(ctrYS.shape[:2])), dtype = np.bool)
ctrYM = ctrYM.reshape(-1,3)
ctrYS = ctrYS.reshape(-1,3)
radYM = radYM.ravel()
radYS = radYS.ravel()

"""
for ii, ctrYMi in enumerate(ctrYM):
    plot_sphere(ctrYMi, radYM[ii], color = (0.,1.,0.))
for ii, ctrYSi in enumerate(ctrYS):
    plot_sphere(ctrYSi, radYS[ii], color = (1.,0.,0.))
"""

# check intersection of the bounding boxes
for ii, bYMii in enumerate(bYM.reshape(-1, 2 , 3)):
    #BeamMod.plot_one_smoothed_surface(conCurveYM[ii],color= (0.,0.,1.))
    """
    vertices = np.asarray([i for i in product(bYMii[:,0], bYMii[:,1], bYMii[:,2])])
    x,y,z = vertices.T[(0,1,2),]
    glyph= mlab.points3d(x,y,z, color = (0.,1.,0.), scale_factor = 0.1)
    engine.add_filter(Outline(), glyph)
    """
    for jj, bYSjj in enumerate(bYS.reshape(-1, 2 , 3)):
        if (norm(ctrYM[ii] - ctrYS[jj]) -\
                (radYM[ii] + radYS[jj])) < 1e-10:
            # the enclosing spheres are close enough
            close_mat[ii,jj] = collision_bounding_boxes(bYMii, bYSjj)
            if close_mat[ii,jj]:
                color = (1.,0.,0.)
                BeamMod.plot_one_smoothed_surface(conCurveYM[ii],color=
                        color)
                BeamMod.plot_one_smoothed_surface(conCurveYS[jj], color=color)
"""
vertices = np.asarray([i for i in product(bYSjj[:,0], bYSjj[:,1], bYSjj[:,2])])
x,y,z = vertices.T[(0,1,2),]
glyph= mlab.points3d(x,y,z, color = (0.,1.,0.), scale_factor = 0.1)
engine.add_filter(Outline(), glyph)
"""

