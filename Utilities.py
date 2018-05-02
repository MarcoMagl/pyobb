from __future__ import division
import numpy as np
from numpy import array, zeros, ones, eye, dot
from math import sin, cos
import math
from numpy.linalg import norm
from pdb import set_trace
from pyquaternion import Quaternion
from mayavi import mlab
import cProfile, pstats
from interval import Interval
import itertools
from pyobb.obb import OBB
from scipy.spatial import ConvexHull
from sklearn.preprocessing import normalize

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
#2-----------------------------------------------------------------------------
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
            mode = 'point')




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
def build_obb(pts, use_convex_hull = False):
    if not pts.ndim == 2:
        pts = pts.reshape(-1,3)
    assert pts.shape[-1] == 3, '3D coordinates are expected'

    if not use_convex_hull:
        return OBB.build_from_points(pts)
    else:
        # construct the convex hull first and determine the obb from its vertices
        return OBB.build_from_points(pts[ConvexHull(pts).vertices])


#------------------------------------------------------------------------------
#
#-----------------------------------------------------------------------------
def build_obb_from_control_points(bi, exp = 0.):
    if not bi.shape == (4,3):
        raise ValueError('incorrect shape')
    obbi = build_obb(bi, use_convex_hull = False)
    if exp < 0:
        raise ValueError('extension of the obb must be positive')
    #plot_obb_vertices(obbi, color = (1., 0., 0.))
    if exp > 0. :
        # expand the obb in all the directions by exp
        # one only has to expand the limits of the obb measured in the local frame
        obbi.max += exp
        obbi.min -= exp
        #plot_obb_vertices(obbi, color = (1., 1., 0.))
    return obbi




#------------------------------------------------------------------------------
#
#-----------------------------------------------------------------------------
def build_obb_along_triad(pts, triad):
    if not pts.ndim == 2:
        pts = pts.reshape(-1,3)
    assert pts.shape[-1] == 3, '3D coordinates are expected'
    # the OBB is constructed along the triad provided
    return CustomOBB(triad, points).points



#------------------------------------------------------------------------------
#
#-----------------------------------------------------------------------------
def get_center_facet_and_normal_vec_obb(obb,):
    obb_vrtx = array(obb.points)
    assert obb_vrtx.shape == (8,3)
    con = array([
        [0,1,2,3],
        [4,5,6,7],
        [1,2,7,4],
        [0,3,6,5],
        [0,1,4,5],
        [2,3,6,7]])
    # center of gravity of the obb
    ctr = obb_vrtx[0] + 0.5 * (obb_vrtx[7] - obb_vrtx[0])
    assert np.allclose(ctr, obb.centroid)
    cf = 0.5 * ( obb_vrtx[con[:,0]] + obb_vrtx[con[:,2]] )
    # directly return the matrix containing the normed normal vectors to the faced
    return cf, normalize(cf - ctr , axis=1, norm='l1')



#------------------------------------------------------------------------------
#
#-----------------------------------------------------------------------------
def collision_btw_obb(obb0_obj, obb1_obj, eps = 0 ):
    # determine the normals of all the facets of the obb
    # order the vertices of the 3D geometry
    cf_0, n_0 = get_center_facet_and_normal_vec_obb(obb0_obj)
    cf_1, n_1 = get_center_facet_and_normal_vec_obb(obb0_obj)
    obb0 =array( obb0_obj.points)
    obb1 = array(obb1_obj.points)
    assert eps >= 0
    # test projection along all the directions of the normal to the facets
    for ii, nii  in enumerate(np.vstack((n_0, n_1)) ):
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
    mlab.points3d(cf_0[:,0], cf_0[:,1], cf_0[:,2], mode = 'point')
    mlab.points3d(cf_1[:,0], cf_1[:,1], cf_1[:,2], mode = 'point')
    plot_obb_vertices(obb0_obj, color =  (1.,1.,1.) )
    plot_obb_vertices(obb1_obj, color =  (1.,0.,0.) )
    """

    #mlab.quiver3d(cf_0[:,0], cf_0[:,1], cf_0[:,2],
    #        n_0[:,0], n_0[:,1], n_0[:,2])
    #mlab.quiver3d(cf_1[:,0], cf_1[:,1], cf_1[:,2],
    #        n_1[:,0], n_1[:,1], n_1[:,2])
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
                vi[:,2], color = color,
                tube_radius = None)
    """
    mlab.points3d(vertices[:,0],
        vertices[:,1],
        vertices[:,2],
        color = (0.,1.,0.))
    """



#------------------------------------------------------------------------------
#
#-----------------------------------------------------------------------------
def generation_grid_quadri_and_connectivity(X, Y, Nx, Ny):
    """
    X : lim of the grid in the first direction
    Y : lim of the grid in the second direction
    Nx : number of cells wanted along first axis
    Ny : number of cells wanted along second axis
    """
    assert Nx > 0
    assert Ny > 0
    nx = Nx + 1
    ny = Ny + 1
    x = np.linspace(X[0], X[1], nx)
    y = np.linspace(Y[0], Y[1], ny)
    # ATTENTION : the x coordinates are read along columns and the y along axis 0 !
    coord = zeros((nx , ny, 2), dtype = float)
    for i in range(coord.shape[0]):
        for j in range(coord.shape[1]):
            coord[i,j] = [x[i], y[j]]

    nID = np.arange(nx * ny).reshape((ny,nx))
    # ATTENTION: check the grid_cell array to understand why Ny and Nx are inverted
    grid_cell = np.arange(Nx * Ny).reshape(Ny, Nx)

    # connectivity per cell
    Con = zeros((Nx * Ny, 4), dtype = int)
    Con[:,0] = nID[:-1, :-1].ravel()
    Con[:,1] = nID[:-1, 1:].ravel()
    Con[:,2] = nID[1:, 1:].ravel()
    Con[:,3] = nID[1:, :-1].ravel()
    Con = Con.reshape( (Ny, Nx, 4) )
    # the connectivity is at this point given in terms of nID. We would like to have it directly in
    # terms of indiced in the grid of nodes
    #ConUnrav = array(np.unravel_index(Con, grid[0].shape ))
    #plot_cells_regular_grid(grid , ConUnrav, In2D = True)

    return coord, Con, grid_cell


#------------------------------------------------------------------------------
#
#-----------------------------------------------------------------------------
def plot_cells_regular_grid(grid, Con, In2D = False):
    """
    designed to work with generation_grid_quadri_and_Connectivity
    """
    if In2D:
        import matplotlib.pylab as plt
    plt.close('all')
    ncel = Con[0].shape[0]

    for i in range(ncel):
        xCell = getCellVertices(i, grid, Con)
        # to close the polygon
        xCell = np.vstack([xCell, xCell[0]])
        xCenter = 0.5 * (xCell[0] + xCell[2])
        if In2D:
            plt.plot(xCell[:,0],
                    xCell[:,1])
            plt.scatter(xCenter[0], xCenter[1])
            plt.text(xCenter[0], xCenter[1], str(i))
        else:
            mlab.plot3d(xCell[:,0],
                    xCell[:,1],
                    xCell[:,2])
            mlab.points3d(xCenter[0], xCenter[1],xCenter[1])
    if In2D: plt.pause(1e-5)


#------------------------------------------------------------------------------
#
#-----------------------------------------------------------------------------
def getCellVertices(cell, grid, Con):
    ii, jj = np.unravel_index(cell, Con.shape[:-1])
    nIDs = Con[ii][jj]
    nIDs = np.unravel_index(nIDs, grid.shape[:-1])
    return grid[nIDs ]

#------------------------------------------------------------------------------
#
#-----------------------------------------------------------------------------
def recursive_AABB(\
        ncells,
        xvertex0,
        xvertex1,
        connectivity,
        grid_cell):

    assert ncells.shape == (2,)
    nvrtx0, nvrtx1 = [ncells[0] + 1, ncells[1] + 1]
    for vrtx in (xvertex0, xvertex1,):
        assert vrtx.shape == (nvrtx0 , nvrtx1, 3)

    pairs_chunks_to_check = ones((1,1,1,1), dtype = bool)
    collision_chunks = np.zeros(pairs_chunks_to_check.shape, dtype = bool)
    curr_chunk_cells = array([[grid_cell]])
    nchunk = array([1, 1])
    ct = 0

    while True:
        for (ii,jj,kk,ll) in np.argwhere( pairs_chunks_to_check ):
            # get the coordinates of the vertices from the chunk of cells
            chunk_vrtx_coord_0 = get_vertices_from_chunk_cells(\
                    xvertex0, connectivity, curr_chunk_cells[ii,jj])
            chunk_vrtx_coord_1 = get_vertices_from_chunk_cells(\
                    xvertex1, connectivity, curr_chunk_cells[kk,ll])
            if not chunk_vrtx_coord_0.shape[0] == 0 and\
                    not chunk_vrtx_coord_1.shape[0] == 0:
                # create AABBs
                aabb0 = getAABBLim(chunk_vrtx_coord_0.reshape(-1,3))
                aabb1 = getAABBLim(chunk_vrtx_coord_1.reshape(-1,3))
                collision_chunks[ii,jj, kk, ll] = collision_AABB(aabb0, aabb1)
                """
                if ct > 0:
                    aabb0 = getAABBLim(chunk_vrtx_coord_0.reshape(-1,3))
                    aabb1 = getAABBLim(chunk_vrtx_coord_1.reshape(-1,3))
                    plot_AABB(aabb0 , s_in = 1/(ct + 1) )
                    plot_AABB(aabb1 , s_in=  1/(ct + 1) )
                    set_trace()
                """


        if np.argwhere(collision_chunks).shape[0] == 0:
            # no intersection between aabb has been found. there is not contact possible
            print('No Collision detected')
            return 0, None
        else:
            # TODO: if there is an entire line of 0 ir an entire row of 0, means the chunk is not
            # intersecting with anything

            if ct > 0 and np.all(el_per_chunk_xi<=1) and np.all(el_per_chunk_theta<=1):
                # graphical checking
                """
                for (ii,jj,kk,ll) in np.argwhere(collision_chunks):
                    chunk_vrtx_coord_0 = get_vertices_from_chunk_cells(\
                            xvertex0, connectivity, curr_chunk_cells[ii,jj])
                    chunk_vrtx_coord_1 = get_vertices_from_chunk_cells(\
                            xvertex1, connectivity, curr_chunk_cells[kk,ll])
                    assert chunk_vrtx_coord_0.shape == (4,3)
                    assert chunk_vrtx_coord_1.shape == (4,3)
                    # create AABBs
                    aabb0 = getAABBLim(chunk_vrtx_coord_0.reshape(-1,3))
                    aabb1 = getAABBLim(chunk_vrtx_coord_1.reshape(-1,3))
                    plot_AABB(aabb0 , s_in = 0. )
                    plot_AABB(aabb1 , s_in=  1. )
                """

                return 1, np.argwhere(collision_chunks)

            nchunk += 1
            # generate chunks in xi and theta dir
            chunk_xi = np.array_split(np.arange(ncells[0]), nchunk[0])
            chunk_theta = np.array_split(np.arange(ncells[1]), nchunk[1])
            # el per chunk
            el_per_chunk_xi = array([len(chunk_xii) for chunk_xii in chunk_xi])
            el_per_chunk_theta = array( [len(chunk_thetai) for chunk_thetai in chunk_theta])
            if np.any(el_per_chunk_xi ==0):
                el_per_chunk_xi = np.delete(el_per_chunk_xi, np.argwhere(el_per_chunk_xi == 0))
            if np.any(el_per_chunk_theta ==0):
                el_per_chunk_theta = np.delete(el_per_chunk_theta, np.argwhere(el_per_chunk_theta == 0))

            # get rows and colums needed to split the cell table to get chunks
            limits_chunks_xi = np.hstack((0, np.cumsum(el_per_chunk_xi)))
            limits_chunks_theta = np.hstack((0, np.cumsum(el_per_chunk_theta)))
            curr_chunk_cells=zeros((len(el_per_chunk_xi), len(el_per_chunk_theta)), dtype =
                    np.object)
            for i in range(len(el_per_chunk_xi)):
                for j in range(len(el_per_chunk_theta)):
                    curr_chunk_cells[i,j] = grid_cell[limits_chunks_xi[i]: limits_chunks_xi[i+1], \
                            limits_chunks_theta[j] : limits_chunks_theta[j+1]]
            # pairs of chunk cells to check
            pairs_chunks_to_check = np.ones(curr_chunk_cells.shape + curr_chunk_cells.shape, dtype = bool)
            collision_chunks = np.zeros(pairs_chunks_to_check.shape, dtype = bool)

        ct += 1
        if ct > 100:
            raise ValueError('max number of truncation attained')

#------------------------------------------------------------------------------
#
#-----------------------------------------------------------------------------
def get_vertices_from_chunk_cells(grid, Con, cells):
    Xvert = zeros((cells.flatten().shape + (4, 3) ) )
    for ii, cell in enumerate(cells.flatten()):
        Xvert[ii] =  getCellVertices( cell, grid, Con)
    return Xvert


#------------------------------------------------------------------------------
#
#-----------------------------------------------------------------------------
def getAABBLim(X):
    assert X.shape[0] >0 and X.shape[1] == 3

    return array([ np.min(X , axis = 0),\
            np.max(X, axis = 0)] )



#------------------------------------------------------------------------------
#
#-----------------------------------------------------------------------------
def collision_AABB(aabb1, aabb2, eps = 0, sanity_check = 1):
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
    if sanity_check:
        for aabb in [aabb1, aabb2]:
           assert aabb.shape== (2,3)
           assert np.all(aabb[0] < aabb[1]), 'first line should be\
           minimums while second line should correspond to maximums'

    xmin_1, ymin_1, zmin_1, xmax_1, ymax_1, zmax_1 = aabb1.flatten()
    xmin_2, ymin_2, zmin_2, xmax_2, ymax_2, zmax_2 = aabb2.flatten()
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
def plot_AABB(lim, s_in = 0. ):
   assert lim.shape== (2,3)

   Pts = zeros((8,3), dtype = float)
   xmin, ymin, zmin, xmax, ymax, zmax = lim.flatten()
   Pts[:4, 2] = zmin
   Pts[4:, 2] = zmax
   Pts[0, :2] = [xmax, ymin]
   Pts[1, :2] = [xmax, ymax]
   Pts[2, :2] = [xmin, ymax]
   Pts[3, :2] = [xmin, ymin]
   Pts[4, :2] = [xmax, ymin]
   Pts[5, :2] = [xmax, ymax]
   Pts[6, :2] = [xmin, ymax]
   Pts[7, :2] = [xmin, ymin]


   #s = np.vstack( (np.zeros(Pts[:,0].shape), np.zeros(Pts[:,0].shape))).T.flatten()
   s = s_in * np.ones(Pts[:,0].shape, dtype=float)
   src = mlab.pipeline.scalar_scatter(Pts[:,0] , Pts[:,1] , Pts[:,2])
   connections = array([
       [0,1], [1,2], [2,3], [3,0],
       [4,5], [5,6], [6,7], [7,4],
       [0,4], [1,5], [2,6], [3,7]
       ])

   # Connect them
   src.mlab_source.dataset.lines = connections
   # The stripper filter cleans up connected lines
   lines = mlab.pipeline.stripper(src)
   mlab.pipeline.surface(lines, line_width=1, opacity=.4)

#------------------------------------------------------------------------------
#
#-----------------------------------------------------------------------------
def is_point_in_AABB(aabb, Pt):
    xmin, ymin, zmin, xmax, ymax, zmax = aabb.flatten()
    return (xmin <= Pt[0] <= xmax) and\
    (ymin <= Pt[1] <= ymax) and\
    (zmin <= Pt[2] <= zmax)



