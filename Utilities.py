from __future__ import division
import numpy as np
from numpy import array, zeros, eye, dot
from math import sin, cos
import math
from numpy.linalg import norm
from pdb import set_trace
from pyquaternion import Quaternion
from mayavi import mlab
import cProfile, pstats
from interval import Interval
import itertools


#------------------------------------------------------------------------------

#

#-----------------------------------------------------------------------------
def isOrthogonal(A):
    assert A.shape == (3,3), "Incorrect shape of the input array"
    #assert np.allclose(A.dot(A.T), eye(3), dtype = float), "The array is not orthogonal!"
    return np.allclose(A.dot(A.T), eye(3))

#------------------------------------------------------------------------------

#

#-----------------------------------------------------------------------------

def issym(A):
    assert A.shape == (3,3), "Incorrect shape of the input array"
    #assert np.allclose(A - A.T, zeros((3,3), dtype = float)), "The array is not symmetric !"
    return np.allclose(A - A.T, zeros((3,3), dtype = float))
#------------------------------------------------------------------------------

#

#-----------------------------------------------------------------------------

def isskew(A):
    assert A.shape == (3,3), "Incorrect shape of the input array"
    #assert np.allclose(A + A.T, zeros((3,3), dtype = float)), "The array is not skew !"
    return np.allclose(A + A.T, zeros((3,3), dtype = float)), "The array is not skew !"

#------------------------------------------------------------------------------

#

#-----------------------------------------------------------------------------
def getQuaternions(Q):
    #Spurrier's algorithm
    #the corresponding algorithm is described p.113 of Simo's paper
    #Check my implementation notes to understand the cyclic permutations
    assert Q.shape == (3,3)
    # prealloc for the quaternion parameters
    q = zeros(4, dtype = float)
    M= np.max(np.concatenate((Q[np.diag_indices(3)], array([np.trace(Q)]))))
    #TODO : as we iterate over the permutations, it might worth the case to do something more
    # efficient
    ijk_perm = array([[1,2,3], [2,3,1], [3,1,2]])

    if M == np.trace(Q):
        # one has to be careful when dealing with the formula given in B.$. i is kept fiexd, and
        # j and k are given from the permutations. There is no summation to make
        q[0] = 0.5 * np.sqrt(1. + np.trace(Q))
        q[1] = 0.25 * (Q[3-1,2-1] - Q[2-1,3-1]) / q[0]
        q[2] = 0.25 * (Q[1-1,3-1] - Q[3-1,1-1]) / q[0]
        q[3] = 0.25 * (Q[2-1,1-1] - Q[1-1,2-1]) / q[0]
    else:
        # identify diagonal entry having the highest value
        for m in range(3):
            if M == Q[m,m]:
                # + 1 for the same reason as the one mentionned above ie that we start counting from
                # 0 when dealing wt indices
                i = np.copy(m) + 1
                break
        # now that we have identified i, j and k will be given thanks to the cyclic permutation
        # of (1 ,2 ,3)
        j,k = ijk_perm[i - 1, (1,2)]
        assert isinstance(j, int)
        assert isinstance(k, int)
        q[i] = (0.5 * M + 0.25 * (1 - np.trace(Q)))**(0.5)
        q[0] += 0.25 * ( Q[k-1, j-1] - Q[j - 1 , k - 1]) * q[i]**(-1)
        # same indicial structure as in the paper
        for l in [j, k]:
            q[l] = 0.25 * q[i]**(-1) * (Q[l-1, i-1 ]+ Q[i-1, l-1])

    # check that we get the same orthogonal amtrix as in input
    assert np.allclose(Q, getOrthoMatFromQuaternions(q))
    # We can see that we get the same results as from the package
    q_package = Quaternion(matrix=Q).elements
    assert np.allclose(q, q_package)
    """
    try:
    except AssertionError:
        if np.allclose(np.abs(q) - np.abs(q_package), zeros(4)):
            print('quaternions differ by their sign')
        else:
            raise AssertionError('quaternions are different')
    """

    return q_package
    """
    # extract quaternions from SO(3) element
    return Quaternion(matrix=Q).elements
    """


#------------------------------------------------------------------------------
#
#-----------------------------------------------------------------------------
def getOrthoMatFromQuaternions(q):
    assert q.shape == (4,)
    assert np.allclose(sum(q[i]**2 for i in range(4)) , 1), "Not unit quaternions"
    q0, q1, q2, q3 = q
    # Double checked in the paper of Simo
    Q = 2 *  np.array([[q0**2 + q1**2 - 0.5, q1*q2- q3*q0, q1*q3 + q2*q0],\
                     [q2*q1+q3*q0, q0**2+q2**2-0.5, q2*q3-q1*q0],\
                     [q3*q1-q2*q0, q3*q2 + q1*q0, q0**2 + q3**2 - 0.5] ])

    # create quaternion object from and rotation matrix on the fly
    R = Quaternion(q).rotation_matrix
    assert np.allclose(Q,R)
    assert isOrthogonal(Q), "Not orthogonal"
    return R

#------------------------------------------------------------------------------
#
#-----------------------------------------------------------------------------
def getAngleFromQuaternions(q):
    # ATTENTION: in the doc of the decorator get angle, warning is given about jump in the angle
    # extracted from the quaternions.
    return Quaternion(q).degrees
#------------------------------------------------------------------------------
#
#-----------------------------------------------------------------------------
def quaternionsFromRotationVect(v):
    """
    https://en.wikipedia.org/wiki/Conversion_between_quaternions_and_Euler_angles
    what we do here is getting a Euler rotation vector from the quaternions
    """
    # appendix B - [Simo] dynamics
    q0 = cos(norm(v)/2)
    if norm(v) > 1e-13:
        q1 = (v[0]/norm(v)) * sin(norm(v)/2)
        q2 = (v[1]/norm(v)) * sin(norm(v)/2)
        q3 = (v[2]/norm(v)) * sin(norm(v)/2)
        q = array([q0, q1,q2,q3])
        qout= Quaternion(axis=v/norm(v), radians=norm(v)).elements
        assert np.allclose(qout,q)
    else:
        q1 = 0
        q2 = 0
        q3 = 0
        q = array([q0, q1,q2,q3])
    assert np.allclose(sum(q[i]**2 for i in range(4)) , 1), "Not unit quaternions"
    assert not np.any(np.isnan(q))
    # the axis is given by the unit vector in the same direction as v
    # the value of the rotation is given by the norm of v
    return q

    """
    if norm(v) > 1e-10:
        R = Quaternion(axis=v/norm(v), radians=norm(v)).rotation_matrix
        assert np.allclose(R, RodriguesFormula(v))

    if norm(v) < 1e-10:
        # null rotation
        return Quaternion(axis=[1,0,0], radians=0).elements
    else:
        return Quaternion(axis=v/norm(v), radians=norm(v)).elements
    """
#------------------------------------------------------------------------------
#
#-----------------------------------------------------------------------------
def rotationVectFromQuaternions(q):
    #appendix B, p.154, [Simo dynamics]
    #return Quaternion(q).angle * Quaternion(q).get_axis()
    if np.allclose(q[1:], zeros(3)): return zeros(3)
    if np.allclose( np.sqrt(q[1]**2 + q[2]**2 + q[3]**2), 1):
        normv =  2 * math.asin(1)
        # no need to take care of -1 because sqrt > 0
    else:
        normv =  2 * math.asin(np.sqrt(q[1]**2 + q[2]**2 + q[3]**2))

    assert not np.any(np.isnan(normv))
    v1 = normv * q[1:] * (np.sqrt(q[1]**2 + q[2]**2 + q[3]**2))**(-1)
    return v1
    qp = Quaternion(q)
    # definition of the Euler rotation vector
    v = qp.angle * qp.get_axis()
    assert np.allclose(v, v1)
    return v
#------------------------------------------------------------------------------
#
#-----------------------------------------------------------------------------
def skew(x):
    # return the skew symmetric form of a vector
    assert x.shape == (3,)
    SK = np.array([[0, -x[2], x[1]],
                     [x[2], 0, -x[0]],
                     [-x[1], x[0], 0]])
    assert isskew(SK)
    return SK

#------------------------------------------------------------------------------
#
#-----------------------------------------------------------------------------
def getAxialVector(Xsk):
    """
    get axial vector from skew symmetric matrix
    """
    assert isskew(Xsk), "Asking for the axial vector of a\
    non-ss matrix"
    #extract the axial vector that is our strain measure
    return array([Xsk[2,1], Xsk[0,2], Xsk[1,0]])

#------------------------------------------------------------------------------
#
#-----------------------------------------------------------------------------

def RodriguesFormula(r):
    """
    Simo in his paper on dynamics is right when he says that numerically, the RodriguesFormula that
    uses the sin is NOT ACCURATE. In this code. I use (1 - cos(the)) instead of sin(the)
    We finally get an agreement between the exponential map and the rotations coming from the the
    quaternions
    The code comes from:
        https://github.com/robEllenberg/comps-plugins/blob/master/python/rodrigues.py
    """
    nrmr = norm(r)
    if nrmr > 1e-30:
        # make the rotation axis a unit vector
        n = r/nrmr
        Sn = skew(n)
        # in perfect agreement with http://mathworld.wolfram.com/RodriguesRotationFormula.html
        R = eye(3) + sin(nrmr)*Sn + (1-cos(nrmr))*np.dot(Sn,Sn)
        # ATTENTION, Sn is hiding n that divide by the norm of r
        RSimo = eye(3) + (sin(nrmr)/nrmr)*skew(r) + 0.5 *  (sin(0.5 * nrmr) / (0.5 * nrmr))**2 * skew(r).dot(skew(r))

        assert np.allclose(R, RSimo)
    else:
        Sr = skew(r)
        nrmr2 = nrmr**2
        R = eye(3) + (1-nrmr2/6.)*Sr + (.5-nrmr2/24.)*np.dot(Sr,Sr)
    return R



#------------------------------------------------------------------------------
#
#-----------------------------------------------------------------------------
def TransformQuantityBetweenBasis(v, Tr):
    """
    v denotes a tensorial quantity that can be ndim =1 or nidm =2
    Te is the transformatin matrix whose columns are the components of the new basis vector
    expressed in the old basis
    """
    # equation 5.4.17 Hugues
    if v.ndim == 1:
        return Tr.T.dot(v)
    else:
        assert v.shape == (3, 3)
        # http://www.continuummechanics.org/coordxforms.html
        # but careful to the definition of Q in this link. It is the transpose of what I am used to
        # use
        return Tr.T.dot(v).dot(Tr)


#------------------------------------------------------------------------------
#
#-----------------------------------------------------------------------------

def get_linear_transfo_between_tgt_spaces_of_SO3(Psi) :
    # evaluate T operator at Psi to obtain the current increment of rotation that
    # belongs currently to the tangent space at R (from previous iterative
    # configuration) with into the tangent space at the identity tensor ! The formula is
    # given in [Cardona] (38)
    assert Psi.shape == (3,)
    if norm(Psi) > np.pi: raise ValueError('the construction of the linear operator is incorrect\
                                            for a norm of a rotation vector over pi')
    if norm(Psi) < 1e-12:
        return eye(3)
    else:
        normP = norm(Psi)
        e = Psi * (norm(Psi))**(-1)
        T =  (sin(normP)/normP)*eye(3) + (1 - (sin(normP) / (normP))) * (np.outer(e,e)) -\
                0.5 * (sin(0.5 * normP) / (0.5 * normP))**2 * skew(Psi)
        assert not np.any(np.isnan(T))
        return T


#------------------------------------------------------------------------------
#
#-----------------------------------------------------------------------------
def collision_AABB(blim1, blim2, eps = 0):
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

    """
    from mayavi import mlab
    mlab.points3d([xmin_1, xmin_2],[ymin_1, ymin_2], [zmin_1, zmin_2],
            color = (1.,0.,0.), scale_factor = 0.1)
    mlab.points3d([xmax_1, xmax_2],[ymax_1, ymax_2], [zmax_1, zmax_2],
            color = (0.,1.,0.), scale_factor = 0.1)
    """
    # www.miguelcasillas.com/?p=30
    return (xmax_1 >= xmin_2 - eps) and\
            (xmin_1 <= xmax_2 + eps) and\
            (ymax_1 >= ymin_2 -eps) and\
            (ymin_1 <= ymax_2+eps) and\
            (zmax_1 >= zmin_2-eps) and\
            (zmin_1 <= zmax_2+eps)


#------------------------------------------------------------------------------
#
#-----------------------------------------------------------------------------
def plot_AABB(blim):
   assert blim.shape== (2,3)
   coord = np.array([ci for ci in itertools.product(blim[:,0], blim[:,1], blim[:,2]) ])
   mlab.points3d(coord[:, 0],
           coord[:,1],
           coord[:, 2],
        color = (0.,1.,0.),
        scale_factor = 0.02)


#------------------------------------------------------------------------------
#
#-----------------------------------------------------------------------------
def is_point_in_AABB(blim, Pt):
    xmin, ymin, zmin, xmax, ymax, zmax = blim.flatten()
    return (xmin <= Pt[0] <= xmax) and\
    (ymin <= Pt[1] <= ymax) and\
    (zmin <= Pt[2] <= zmax)


#------------------------------------------------------------------------------
#
#-----------------------------------------------------------------------------
def get_surface_points_sphere(r,c):
    assert c.shape==(3,)
    assert isinstance(r, float)
    [phi,theta] = np.mgrid[0:2*np.pi:12j,0:np.pi:12j]  #increase the numbers before j for better resolution but more time
    return r*np.cos(phi)*np.sin(theta) + c[0],\
           r*np.sin(phi)*np.sin(theta) + c[1],\
           r*np.cos(theta) + c[2]
#------------------------------------------------------------------------------
#
#-----------------------------------------------------------------------------
def plot_sphere(c,\
        r,\
        color = (1.,0.,0.),\
        opacity = 0.5):
    x,y,z = get_surface_points_sphere(r,c)
    return mlab.mesh(x, y, z, color = color, opacity = opacity )

#------------------------------------------------------------------------------
#
#-----------------------------------------------------------------------------
def nn_from_nel(nn_per_el, nel):
    # checked
    assert isinstance(nn_per_el, int)
    assert isinstance(nel, int)
    return nel * (nn_per_el - 1) + 1

#------------------------------------------------------------------------------
#
#-----------------------------------------------------------------------------
def nel_from_nn(nn_per_el, nn):
    # checked
    assert isinstance(nn_per_el, int)
    assert isinstance(nn, int)
    return int( (nn-1) / (nn_per_el - 1))

#------------------------------------------------------------------------------
#
#-----------------------------------------------------------------------------
def TransformationQuadraturePointsAndWeigths1D(\
        Interval, xiGQP, wiGQP):
    """
    xi and wi are the standard coordinates and weigths of the
    quadrature points
    TransformationQuadraturePointsAndWeigths1D(np.array([0,1]), 0., 2.)
    (0.5, 1.0)
    TransformationQuadraturePointsAndWeigths1D(np.array([0,np.pi]), 0., 2.)
    (1.5707963267948966, 3.1415926535897931)
    """
    assert Interval.shape == (2,)
    assert Interval[0] < Interval[1]
    assert -1 < xiGQP < 1
    #return  the coordinate of the GQP in the current 1D interval
    # and associated weigth
    return 0.5 * (Interval[0] + Interval[1] + xiGQP * (Interval[1] -
        Interval[0])), 0.5 * (Interval[1] - Interval[0])



#------------------------------------------------------------------------------
#
#-----------------------------------------------------------------------------
def scatter_3d(coords, color = (1.,0.,0.),\
        scale_factor = 0.1):
    assert (coords.ndim == 2 ) and (coords.shape[1] == 3)
    return mlab.points3d(
            coords[:,0],\
            coords[:,1],\
            coords[:,2],\
            color = color,\
            scale_factor = scale_factor)
    return mlab.mesh(x, y, z, color = color )



#------------------------------------------------------------------------------
#
#-----------------------------------------------------------------------------
def getAABBLim(X):
    assert X.ndim == 2 and X.shape[1] == 3
    return array([ np.min(X , axis = 0),\
            np.max(X, axis = 0)] )




#------------------------------------------------------------------------------
#
#-----------------------------------------------------------------------------
def do_cprofile(func):
    def profiled_func(*args, **kwargs):
        profile = cProfile.Profile()
        try:
            profile.enable()
            result = func(*args, **kwargs)
            profile.disable()
            return result
        finally:
            # M : I sort by cumulative time and only output the 10
            # most expensive ones
            #profile.sort_stats('cumulative').print_stats(10)
            #profile.print_stats()
            #s = StringIO.StringIO()
            #sortby = 'cumulative'
            #ps = pstats.Stats(profile, stream=s).sort_stats(sortby)
            #ps.print_stats()
            profile.print_stats(sort='time')
    return profiled_func


#------------------------------------------------------------------------------
#
#-----------------------------------------------------------------------------
def overlaps(int1, int2):
    assert len(int1) == 2 and len(int2) == 2
    int1 = Interval(int1[0], int1[1])
    int2 = Interval(int2[0], int2[1])
    return int1.overlaps(int2)


#------------------------------------------------------------------------------
#
#-----------------------------------------------------------------------------
def collision_btw_obb(obb0_obj, obb1_obj, eps = 0 ):
    # determine the normals of all the facets of the obb
    # order the vertices of the 3D geometry
    assert eps >= 0
    obb0 = np.concatenate(obb0_obj.points).reshape(-1,3)
    obb1 = np.concatenate(obb1_obj.points).reshape(-1,3)
    assert obb0.shape == (8,3)
    assert obb1.shape == (8,3)
    # from the doc of OBB, https://github.com/pboechat/pyobb/blob/master/pyobb/obb.py
    # I got the following connectivity for each facet
    con = array([
        [0,1,2,3],
        [4,5,6,7],
        [1,2,7,4],
        [0,3,6,5],
        [0,1,4,5],
        [2,3,6,7]])

    # coordinate center of the box
    c0 = obb0[0] + 0.5 * (obb0[7] - obb0[0])
    c1 = obb1[0] + 0.5 * (obb1[7] - obb1[0])
    assert np.allclose(c0, obb0_obj.centroid)
    assert np.allclose(c1, obb1_obj.centroid)
    # center of each facet
    cf_0 = zeros((6,3))
    cf_1 = zeros((6,3))
    # normal of each facet
    n_0 = zeros((6,3))
    n_1 = zeros((6,3))

    for i in range(6):
        cf_0[i] = 0.5 * ( obb0[con[i,0]] + obb0[con[i,2]] )
        cf_1[i] = 0.5 * (obb1[con[i,0]] + obb1[con[i,2]] )
        n_0[i] = cf_0[i] - c0
        n_1[i] = cf_1[i] - c1

    n = np.vstack((n_0, n_1))

    # test projection along all the directions of the normal to the facets
    for ii, nii  in enumerate(n):
        # project all the vertices of the first obb
        p0 = np.dot(obb0, nii)
        p1 = np.dot(obb1, nii)
        # check if interval of projection overlap
        # this is where we take into account a possible error
        if not overlaps([np.min(p0)-eps , np.max(p0) + eps] , [np.min(p1) - eps, np.max(p1) + eps]):
            # use SAT theorem: if we found a direction and an axis of porjection where
            # the intervals are not overlapping, it means the 2 convex hull are not colliding
            return False


    """
    mlab.points3d(cf_0[:,0], cf_0[:,1], cf_0[:,2])
    mlab.points3d(cf_1[:,0], cf_1[:,1], cf_1[:,2])
    plot_obb_vertices(obb0_obj, color =  (1.,1.,1.) )
    plot_obb_vertices(obb1_obj, color =  (1.,1.,1.) )
    mlab.quiver3d(cf_0[:,0], cf_0[:,1], cf_0[:,2],
            n_0[:,0], n_0[:,1], n_0[:,2])
    mlab.quiver3d(cf_1[:,0], cf_1[:,1], cf_1[:,2],
            n_1[:,0], n_1[:,1], n_1[:,2])
    set_trace()
    """
    # did not find a separating axis
    return True


#------------------------------------------------------------------------------
#
#-----------------------------------------------------------------------------
def plot_obb_vertices(obb, color =  (1.,1.,1.) ):
    vertices = np.concatenate(obb.points).reshape(-1,3)
    # the order of the vertices is given in one of the method of the class file
    con = [[0,1], [1,4],
            [4,5], [5,0],
            [2,3], [3,6],
            [6,7], [7,2],
            [0,3], [1,2],
            [4,7], [5,6]]
    for coni in con:
        vi = vertices[coni,]
        mlab.plot3d(vi[:,0],
                vi[:,1],
                vi[:,2], color = color)
    """
    mlab.points3d(vertices[:,0],
        vertices[:,1],
        vertices[:,2],
        color = (0.,1.,0.))
    """



#------------------------------------------------------------------------------
#
#-----------------------------------------------------------------------------
def generation_grid_quadri_and_connectivity(X, Y, nx, ny):
    """
    X : lim of the grid in the first direction
    Y : lim of the grid in the second direction
    nx : number of points in the first direction
    ny : number of points in the secind direction
    """
    assert len(X) == 2
    assert len(Y) == 2
    x = np.linspace(X[0], X[1], nx)
    y = np.linspace(Y[0], Y[1], ny)
    grid = np.meshgrid(x,y)
    C = array([grid[0].ravel(), grid[1].ravel()]).T
    nID = np.arange(np.product(grid[0].shape)).reshape(grid[0].shape)
    Nx = nx - 1
    Ny = ny - 1

    Con = zeros((Nx * Ny, 4), dtype = int)
    Con[:,0] = nID[:-1, :-1].ravel()
    Con[:,1] = nID[:-1, 1:].ravel()
    Con[:,2] = nID[1:, 1:].ravel()
    Con[:,3] = nID[1:, :-1].ravel()

    return C, Con


#------------------------------------------------------------------------------
#
#-----------------------------------------------------------------------------
def plot_cells_regular_grid(X, con, In2D = False):
    """
    designed to work with generation_grid_quadri_and_connectivity
    """
    if In2D:
        import matplotlib.pylab as plt
    plt.close('all')
    ncel = con.shape[0]
    for i in range(ncel):
        conCell = con[i]
        xCell = X[(con[i][0],
                con[i][1],
                con[i][2],
                con[i][3],
                con[i][0]),]
        xCenter = 0.5 * (X[con[i][0]] + X[con[i][2]])
        if In2D:
            plt.plot(xCell[:,0],
                    xCell[:,1])
            plt.scatter(xCenter[0], xCenter[1])
        else:
            mlab.plot3d(xCell[:,0],
                    xCell[:,1],
                    xCell[:,2])
            mlab.points3d(xCenter[0], xCenter[1],xCenter[1])
    if In2D: plt.pause(1e-5)

#------------------------------------------------------------------------------
#
#-----------------------------------------------------------------------------







