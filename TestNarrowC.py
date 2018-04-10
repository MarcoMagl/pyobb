from Beams.BeamElements import KirchhoffBeam, TimoshenkoBeam, TimoshenkoBeam3D
from Beams.GeometricallyExact import SR3DBeams
from Common import BoundaryConditions, Solver, FEM
from Common.FEM import Model
from Common.Utilities import nel_from_nn
import numpy as np
from numpy import zeros, ones_like,  zeros_like, asarray, array, ones, ix_, allclose, arange
from numpy.linalg import norm, solve
from pdb import set_trace



E = 100.
# we will get a circular cross section of radius = 0.5
b = array([0.6, 0.6])
h = array([0.2, 0.2])

# Y1 beam suppose to represent a rigid obstacle
nnY1 = 5
XY1 = zeros((nnY1, 3), dtype = float)
XY1[:,1] = 0.501 * (b[0] + b[1])
XY1[:,2] = np.linspace( -2, 2, nnY1)
nel_Y1 = nel_from_nn(2, nnY1)
nIDs_Y1 = zeros((nel_Y1,2), dtype = int)
nIDs_Y1[:,0] = np.arange(nnY1 -1)
nIDs_Y1[:,1] = np.arange(1, nnY1)
el_Y1 = np.arange(nel_Y1)

nnY2 = 5
XY2 = zeros((nnY2, 3), dtype = float)
XY2[:,0] = np.linspace(-4, 4, nnY2)
nel_Y2 = nel_from_nn(2, nnY2)
nIDs_Y2 = zeros((nel_Y2,2), dtype = int)
nIDs_Y2[:,0] = nnY1 + np.arange(nnY2-1)
nIDs_Y2[:,1] = nnY1 + np.arange(1, nnY2)
el_Y2= nel_Y1 + np.arange(nel_Y2)


# 6 dofs per node
ndofsPern = 6
nN = nnY1 + nnY2
nEl = nel_Y1 + nel_Y2
# we just set the x coordinates
X = np.vstack((XY1, XY2))
# dofs per node
dperN = np.arange(ndofsPern * nN).reshape(-1, ndofsPern)
# Construct beam elements
el = np.zeros(nEl, dtype = np.object)
nIDs = np.vstack((nIDs_Y1, nIDs_Y2))

for ii in range(nEl):
    nIDsi = array(nIDs[ii] )
    t3 =  X[nIDsi][1] - X[nIDsi][0]
    t3 /= norm(t3)

    if ii in range(nel_Y1):
        bii = b[0]
        hii = h[0]
        t1 = array([1,0,0])
        t2 = np.cross(t3, t1)
    else:
        bii = b[1]
        hii = h[1]
        t1 = array([0,0,-1])
        t2 = np.cross(t3, t1)

    el[ii] = SR3DBeams( X = X[nIDsi], E = E,
                        nu = 0.3, b = bii, h = hii,
                        nID = nIDsi, dofs = dperN[nIDsi],
                        E1= t1,E2 = t2,
                        rot_vars_storage= 1,
                        shapeCrossSec='Elliptical')

# Construct FE assembly
# 3 rotations and translations per node
BeamMod = Model(el, X.shape[0], ndofsPern, X,\
                HasConstraints = 1,\
                dperN = dperN)

BeamMod.set_plotting_package('mayavi')

# the first elements are the one associated with the Y1 yarn !
el_curves_YY1 = np.vstack((np.arange(nel_Y1)[:-1],
    np.arange(nel_Y1)[1:])).T
el_curves_YY2 = nel_Y1 + np.vstack((np.arange(nel_Y2)[:-1],
    np.arange(nel_Y2)[1:])).T
elements_per_curve = np.concatenate((el_curves_YY1,
    el_curves_YY2))

# Be extremely careful when setting which curves are master and which
# ones are slaves. Be careful not to mix the element per curves

ID_master_curves = np.arange( el_curves_YY1.shape[0])
ID_slave_curves = ID_master_curves[-1] + 1 +  np.arange( el_curves_YY2.shape[0])

BeamMod.create_unset_Contact_Table(\
            enforcement = 1,\
            method = 'curve_to_curve',\
            isConstant = False,\
            alpha = 0.9,\
            smooth_cross_section_vectors = True,\
            elements_per_curve = elements_per_curve,
            master_curves_ids= ID_master_curves,
            slave_curves_ids= ID_slave_curves,\
            nintegrationIntervals = array([2,4]),\
            nxiGQP = 6,\
            nthetaGQP = 6)

BeamMod.ContactTable.set_default_value_LM(0)


conYY1 = zeros((nel_Y1, 2), dtype = int)
conYY2 = zeros((nel_Y2, 2), dtype = int)
conYY1[:,0] = np.arange(nnY1-1)
conYY1[:,1] = np.arange(1, nnY1)
conYY2[:,0] = np.arange(nnY2-1)
conYY2[:,1] = np.arange(1, nnY2)
conYY2 += nnY1


con = np.vstack((conYY1, conYY2))
nYY1 = np.arange(nnY1)
nYY2 = nnY1+ np.arange(nnY2)

Load = 0
if Load == 0:
    nBCs = 5
    dofs = np.zeros((nBCs,), dtype = np.object)
    values = np.zeros((nBCs,), dtype = np.object)
    t0s = zeros(nBCs,)
    tmaxs = zeros(nBCs,)
    tMAX = 4

# the nodes of the Y1 yarn are completely blocked
    dofs[0] = dperN[nYY1[(0,-1),]].ravel()
    values[0] = np.zeros_like(dofs[0])
    t0s[0] = 0
    tmaxs[0]=tMAX

    dofs[1] = dperN[nYY2[(0,-1),]]
    values[1] = np.zeros_like(dofs[1], dtype = float)
    values[1][:,1] = 0.4
    t0s[1] = 0
    tmaxs[1]= 1


    dofs[2] = dperN[nYY2[(0,-1),]]
    values[2] = np.zeros_like(dofs[1], dtype = float)
    values[2][:,0] = 0.6
    t0s[2] = 1+1e-10
    tmaxs[2]=2


    dofs[3] = dperN[nYY2[(0,-1),]]
    values[3] = - values[2]
    t0s[3] = 2+1e-10
    tmaxs[3]=3


# move node along the direction of the beam
    dofs[4] = dperN[nYY2[(0,-1),]]
    values[4] = -2 * values[1]
    t0s[4] = 3 + 1e-10
    tmaxs[4]=tMAX


    """
    Tab = BeamMod.ContactTable
    BeamMod.plot(opacity = 1 )
    BeamMod.choose_active_set()
    BeamMod.get_all_GQP_and_CPP_coupleII()
    """

# all the Bcs made are Dirichlet
    types = ones(nBCs, dtype = int)
    Intervals = array([0,tMAX])
    timestepping = array([50])
elif Load == 1:
    # only up and down
    nBCs = 3
    dofs = np.zeros((nBCs,), dtype = np.object)
    values = np.zeros((nBCs,), dtype = np.object)
    t0s = zeros(nBCs,)
    tmaxs = zeros(nBCs,)
    tMAX = 4

# the nodes of the Y1 yarn are completely blocked
    dofs[0] = dperN[nYY1[(0,-1),]].ravel()
    values[0] = np.zeros_like(dofs[0])
    t0s[0] = 0
    tmaxs[0]=tMAX

    dofs[1] = dperN[nYY2[(0,-1),]]
    values[1] = np.zeros_like(dofs[1], dtype = float)
    values[1][:,1] = 0.4
    t0s[1] = 0
    tmaxs[1]= 1


    dofs[2] = dperN[nYY2[(0,-1),]]
    values[2] = np.zeros_like(dofs[1], dtype = float)
    values[2][:,0] = 0.6
    t0s[2] = 1+1e-10
    tmaxs[2]=2

    # all the Bcs made are Dirichlet
    types = ones(nBCs, dtype = int)
    Intervals = array([0,tMAX])
    timestepping = array([50])

Loading = BoundaryConditions.Static_Loading(t0s, tmaxs, types ,dofs, values, Intervals, timestepping)
Solv = Solver.Solver(Loading,\
                    BeamMod,\
                    100, 1e-4,\
                    wtConstraints = 1,\
                    store_contact_history = False )

Solv.set_ignore_previous_results(False)
Solv.Solve()

BeamMod.FinalAnimation(Solv.pathfile)


