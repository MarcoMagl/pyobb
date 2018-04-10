import numpy as np
from numpy import array
import matplotlib.pylab as plt
import FEM
from FEM import Model
from scipy import spatial
from numpy.linalg import norm
from pdb import set_trace
import matplotlib
from mayavi import mlab
from Beams.GeometricallyExact import SR3DBeams

xinterval= array([ 0, 10])
yinterval= array([ 0, 10])

XYH, XYV, conYH, conYV, ndYH, ndYV, nnYH, nnYV, X,\
con, nel, which_yarn_is_el, which_fam_of_yarn_is_el =\
FEM.generate_regular_lattice(10, 10, 20, 20 ,xinterval, yinterval)

# generate center of enclosing sphere as middle of the centroid line
# (straightforward for a linearly interpolated element)
mid =  X[con[:,0]]+ 0.5 * (X[con[:,1]] - X[con[:,0]])
len_els = norm(X[con[:,1]] - X[con[:,0]], axis = 1).reshape(-1, 1)
a = 0.5
b= 0.5 * a
dims_cross_sec = np.multiply( [a , b], np.ones((nel,2)))
radius_to_check = np.max(np.concatenate((0.5 * len_els,
    dims_cross_sec), axis =1), axis = 1)

"""
plt.close('all')
ax = plt.gca()
ax.cla()
x,y = (X[:,(i),] for i in range(2))
ax.scatter(x,y,color = 'k', marker ='.')
x,y = (mid[:,(i),] for i in range(2))
ax.scatter(x,y,color = 'b', marker ='.')
plt.pause(1e-5)
"""

tree = spatial.KDTree(mid)
# ATTENTION:
# we choose to check distances from the mid point of the family of the
# yarn 0
# this avoids the doublons in distances d(A->B) = d(B->A)
el_on_Y_fam0 = np.where(which_fam_of_yarn_is_el == 0)[0]
closest_element = np.zeros(el_on_Y_fam0.shape[0], dtype = int)
for ii, elii in enumerate(el_on_Y_fam0):
    candidates = tree.query_ball_point( mid[elii], radius_to_check[elii])
    entry_to_del = list()
    for jj, eljj in enumerate(candidates):
        # remove the elements of the same yarns and same family of
        # yarn
        # ATTENTION: the point itself is returned by the query
        if (which_yarn_is_el[elii] ==  which_yarn_is_el[eljj]) or \
           (which_fam_of_yarn_is_el[elii] ==
                   which_fam_of_yarn_is_el[eljj])\
                           or (ii == jj):
            entry_to_del.append(jj)

    candidates = np.delete(candidates, entry_to_del)
    if candidates.shape[0] > 0:
        # argmin is life saving here !
        closest_element[ii] = candidates[ array([norm(mid[elii] - mid[candj]) for candj in
            candidates]).argmin() ].astype(int)
    else:
        # we pick an invalid value, and not a Nan because it is float
        # only
        closest_element[ii] = -999

pairs_to_check = np.vstack((el_on_Y_fam0, closest_element)).T
# remove invalid entries, ie element that do not have a close
# neighbour
pairs_to_check=\
np.delete(pairs_to_check,  np.where(pairs_to_check == -999)[0], axis =
        0)

"""
# PLOT THE IDENTIFIED PAIRS
ax = plt.gca()
x,y = (X[:,(i),] for i in range(2))
empty_circle =matplotlib.markers.MarkerStyle(marker='o', fillstyle='none')
for pair in pairs_to_check:
    if not (np.isnan(pair[0])) and not np.isnan(pair[1]):
        color = np.random.rand(3)

        #ax.scatter(mid[pair[0]][0] ,mid[pair[0]][1],color = color,
                #marker =empty_circle, facecolors='none')
        #ax.scatter(mid[pair[1]][0] ,mid[pair[1]][1],color = color, marker ='x')

        ax.plot([mid[pair[0]][0], mid[pair[1]][0]] ,
                [mid[pair[0]][1], mid[pair[1]][1]],
                    color = color,
                marker = 'o')
plt.pause(1e-15)
"""

#function to move the nodes vertically
def u_vert():
    c1 = b
    c2 = (x1 + x2)/2
    c3 = -((-x1 + x2)/ np.pi)
    return c1 * np.sin( (x - c2)/c3)

# select the nodes of ony yarn
XYHi = X[ndYH][0]
x = XYHi[:,0]
x1 = XYV[0,0,0]
x2 = XYV[1,0,0]
uy = u_vert()
"""
ax.plot(x, uy)
plt.pause(1)
"""
try:
    mlab.close()
except AttributeError:
    pass

# access the nodes of the horizontal yarns
# alternate weaving
nn = X.shape[0]
dperN = np.arange(nn * 6,).reshape(nn,6)

# ATTENTION. CES DEUX LIGNES SONT ULTRAS IMOPRTNATES AU NIVEAU DU
# CODE. LES PARENTHESES SONT FONDAMENTALES POUR PERMETTRE LE
# BROADCASTING
X[(ndYH[0::2],), 2] += uy
X[(ndYH[1::2],), 2] -= uy

"""
# plotting the nodes !
x,y,z = (X[ndYH].reshape(-1,3)[:,(i),] for i in range(3))
mlab.points3d(x,y,z, color = (1.,0.,0.))
x,y,z = (X[ndYV].reshape(-1,3)[:,(i),] for i in range(3))
mlab.points3d(x,y,z, color = (0.,1.,0.))
"""


nEl = con.shape[0]
el = np.zeros(nel, dtype = np.object)
for ii in range(nEl):
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
                HasConstraints = 1, dperN = dperN)
BeamMod.set_plotting_package('mayavi')


# choose which element will be used to build curves
els_for_curves = np.zeros(pairs_to_check.shape + (2,), dtype = int)
for ii in range(pairs_to_check.shape[0]) :
    for jj in range(2):
        if pairs_to_check[ii,jj] == 0:
            els_for_curves[ii,jj] = [0,1]
        elif pairs_to_check[ii,jj] == nel -1:
             els_for_curves[ii,jj] = [pairs_to_check[ii,jj]-1,
                     pairs_to_check[ii,jj]]
        elif which_yarn_is_el[pairs_to_check[ii,jj]] !=\
        which_yarn_is_el[pairs_to_check[ii,jj] + 1]:
            # we cannot take the next element because it corresponds to
            # another yarn so we take the previous element instead
             els_for_curves[ii,jj] = [pairs_to_check[ii,jj],
                     pairs_to_check[ii,jj]-1]
        else:
             els_for_curves[ii,jj] = [pairs_to_check[ii,jj],
                     pairs_to_check[ii,jj]+ 1]


#BeamMod.plot( opacity = 1.)

value = 100 * np.ones(els_for_curves.shape[0], dtype = float)
BeamMod.create_unset_Contact_Table(0,els_for_curves, value,
'curve_to_curve', alpha = 0.8, smooth_cross_section_vectors = True)
BeamMod.set_penalty_options(1, -1e-2 * b, 10)

#BeamMod.plot_smoothed_geometries()
BeamMod.plot()
