###########
# simple tests with the different beam elements I implemented (2D nonlinear or 3D linear)
###
from __future__ import division
import numpy as np
from numpy import zeros, ones, dot, asarray, array, eye, outer, mat, empty, sqrt, zeros_like, ix_, inner,\
allclose, where, ones_like, linspace
from scipy.linalg import norm
from math import atan2, sin, cos, tan
from pdb import set_trace
# essential to catch the error thrown by mayavi
import logging
from tables import open_file
from scipy.spatial.distance import cdist
# import itertools
# Custom Wrappers
from Extension.RoutinesGeoExactBeams.TwoNodesGEB import GEB2Nodes
from Extension.RoutinesGeoExactBeams.ThreeNodesGEB import GEB3Nodes
from Extension.RoutinesGeoExactBeams.SmoothingGEB2Nodes import Smooth2Nodes
from Extension.RoutinesGeoExactBeams.Smoothing2NVectInterp.ctcAtCPP import ctcAtCPP
import\
Extension.RoutinesGeoExactBeams.Smoothing2NVectInterp.Common.SmoothedGeo as GeoSmooth
from Extension.RoutinesGeoExactBeams.Smoothing2NVectInterp.weakIneqConstraint.RigidSphere import ctcVolumeSphere
from Extension.RoutinesGeoExactBeams.Smoothing2NVectInterp.weakIneqConstraint.BeamToBeam import ctcVolumeBtoB
from Common.Utilities import *
import matplotlib.pylab as plt
from Common.ContactTable import ContactTable
from mayavi import mlab
from itertools import product
# from sympy import Line3D, Point, Point3D, Plane
from collections import deque
from Common.UserDefErrors import *
from joblib import Parallel, delayed
from joblib.pool import has_shareable_memory
import multiprocessing
from Common.cython_ext import narrow_phase

#------------------------------------------------------------------------------
#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
                            # MODEL CLASS
#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
#------------------------------------------------------------------------------
class Model():

    ################################################################################
                            # CONSTRUCTOR
    ################################################################################
    #------------------------------------------------------------------------------
    def __init__(self, el, nn, ndofsPern, X,\
                 HasConstraints, dperN, Enforcement = 0,\
                  **kwargs):
        """
        - el is an iterable containing the finite element objects
        - nn number of nodes in total
        - ndofsPern the number of dofs per node
        - X the coordinates of the nodes in the reference config
        """
        assert isinstance(el, list) or (isinstance(el, np.ndarray) and el.dtype==np.object)
        self.el = el
        self.nel = len(el)
        self.ndofsPern = ndofsPern
        # nn is the number of nodes
        assert isinstance(nn, int) and isinstance(ndofsPern, int)
        # TODO: this unique storage of X is much more efficient than storing the X pf each element.
        # The code for linear elements must be updated accordingly
        self.X = X.astype(np.double)
        self.nn = nn
        self.ndofs_el = int(nn * ndofsPern)
        self.tot_ndofs = self.ndofs_el
        self.dperN = dperN
        assert dperN.shape == (nn, ndofsPern)
        assert X.shape == (nn, 3)
        try:
            assert [eli.__class__.__name__ == 'SR3DBeams' for eli in el]
            self.GeoExact = True
            self.dofs_trans = np.arange(self.ndofs_el).reshape(self.nn,self.ndofsPern)[:, :3]
            self.dofs_rot = np.arange(self.ndofs_el).reshape(self.nn,self.ndofsPern)[:, 3:]
        except AssertionError:
            self.GeoExact = False
            raise DeprecationWarning

        if not self.GeoExact:
            self.u = zeros(self.ndofs_el,)
            raise NotImplementedError
        else:
            # 3 translationnals dofs per node. The three other rotationnal dofs are handled
            # differently in the GEB theory
            self.u = zeros_like(self.X, dtype=np.double)
            self.rot_dofs_storage =  self.el[0].rot_vars_storage
            if self.rot_dofs_storage == 0:
                # one quaternion per node of the model
                # TO AVOID DATA REPLICAS, ALL THE QUATERNIONS ARE STORED IN AN ARRAY
                # compute quaternions corresponding to an identity tensor
                self.q =  getQuaternions(eye(3)) * np.ones((nn, 4), dtype = np.float)
            elif self.rot_dofs_storage == 1:
                # one rotation vector per npde
                self.v = zeros((self.nn, 3), dtype = np.double)

        self.is_set_package_plot = False
        assert (HasConstraints == 0) or (HasConstraints == 1)
        self.HasConstraints = HasConstraints

        if HasConstraints:
            self.is_set_ContactTable = 0
            self.is_set_master_and_slave_curves = False

        self.HasSphere = False
        # try to identify the connectivity yarn per yarn
        el_per_Y = [[0]]
        Yct = 0
        for ii in range(1, len(self.el)):
            if self.el[ii].nID[0] == self.el[ii-1].nID[1] :
                el_per_Y[Yct].append(ii)
            else:
                Yct += 1
                el_per_Y.append([ii])
        self.el_per_Y = el_per_Y
        self.nY = Yct + 1

        self.debug_mode = False

        # gather information at the model level
        self.nPerEl = array([eli.nID for eli in self.el], dtype = int)
        self.t1 = array([eli.E1 for eli in self.el])
        self.t2 = array([eli.E2 for eli in self.el])
        self.a_crossSec= array([eli.a for eli in self.el])
        self.b_crossSec= array([eli.b for eli in self.el])


        # different formulation of how we enforce contact
        # 0 : simply gN
        # 1 : exp(-gN)
        self.contact_law = 1

    ################################################################################
                            # CONSTRUCTOR
    ################################################################################




    ################################################################################
                            # SETTERS
    ################################################################################

    #------------------------------------------------------------------------------
    #
    #------------------------------------------------------------------------------
    def set_debug_mode_on(self): self.debug_mode = True
    #------------------------------------------------------------------------------
    #
    #------------------------------------------------------------------------------
    def set_u(self, u):
        """
        this method is aimed for the 'classical' finite element where the displacements are
        behaviing as vectorial quantities
        """
        if self.GeoExact: raise NotImplementedError
        assert u.shape == (self.tot_ndofs,)
        self.u = u
    #------------------------------------------------------------------------------
    #
    #------------------------------------------------------------------------------
    def set_planar_directions(self, dir1, dir2):
        assert isinstance(dir1, int)
        assert isinstance(dir2, int)
        assert dir1 != dir2
        self.dir1 = dir1
        self.dir2 = dir2


    #------------------------------------------------------------------------------
    #
    #------------------------------------------------------------------------------
    def setYarnConnectivity(self, ElPerYarn):
        """
        each line of the table gives the id number of the element that constitute the yarn
        """
        assert ElPerYarn.ndim == 2
        # we can provide a np.oject type due to a possible vairable number of elements per yarn
        for ElPerYarni in ElPerYarn:
            assert ElPerYarni.dtype == np.int
        self.ElPerYarn = ElPerYarn


    #------------------------------------------------------------------------------
    #
    #------------------------------------------------------------------------------
    def set_Model_Connectivity(self, Con ):
        """
        gather the connectivity of all the elements (nodes ID per element)
        """
        assert Con.shape == (self.nel, 3)
        self.Con = Con

    #------------------------------------------------------------------------------
    #
    #------------------------------------------------------------------------------
    def create_unset_Contact_Table(self,\
            enforcement,\
            method,\
            contact_type = 1,\
            **kwargs):
        """
        for a constraint problem, the contact table must be set
        a member of the contact_table class will be used to\
        store the information necessary for\
        contact in a synthetic manner
        """
        # 0 for penalty, 1 for LM
        assert enforcement == 0 or enforcement == 1
        self.enforcement = enforcement
        # the contact table must be set before starting the simulation
        assert self.HasConstraints, 'Inconsistency: asked to set the contact table while the problem is without constraints'

        self.ContactTable =ContactTable(self.enforcement, method, **kwargs)
        self.set_master_and_slaves_curves(kwargs["elements_per_curve"],\
                                          kwargs["master_curves_ids"],\
                                          kwargs["slave_curves_ids"])
        self.ContactTable.set_critical_penetration(-np.min([0.5 *\
            eli.b for eli in self.el]))

        if self.enforcement == 0 :
            self.is_set_penalty_options= 0

        if contact_type == 1:
            if not 'LM_per_ctct_el' in kwargs.keys():
                print('1 LM per contact element applied by default')
                # add key to dict
                LM_per_ctct_el = 1
            else:
                LM_per_ctct_el =  kwargs['LM_per_ctct_el']
            self.ContactTable.set_parameter_weak_contact(
                                            kwargs["nintegrationIntervals"],\
                                            kwargs["nxiGQP"],\
                                            kwargs["nthetaGQP"],
                                            LM_per_ctct_el)

        self.ContactTable.construct_storage_arrays()
        self.is_set_ContactTable = True

    #------------------------------------------------------------------------------
    #
    #------------------------------------------------------------------------------
    def create_unset_Contact_Table_with_Sphere(self, alpha):
        ContactTable.ContactWithSphere(\
                enforcement = 0,\
                smooth_cross_section_vectors = 1,\
                method = 'curve_to_curve',\
                alpha = alpha)
    #------------------------------------------------------------------------------
    #
    #------------------------------------------------------------------------------
    def set_penalty_options(self, method_regularization, maxPenAllow,
            kNmin = 10., coeff_mul = 10) :
        assert not self.is_set_penalty_options, 'penalty options already set'
        self.ContactTable.set_penalty_options(method_regularization, maxPenAllow,
            kNmin, coeff_mul)
        self.is_set_penalty_options= 1

    #------------------------------------------------------------------------------
    #
    #------------------------------------------------------------------------------
    def set_plotting_package(self, package_name, ax=[]):
        # convert to lower case to make the comparison case
        # insensitive
        package_name = package_name.lower()
        if package_name not in ['mayavi','matplotlib']: raise NotImplementedError
        self.package_plot= package_name
        self.ax_to_plot_figure = None
        if package_name == 'matplotlib':
            assert ax.__class__.__name__ == 'Axes3DSubplot','axes have not the correct type'
            self.ax_to_plot_figure = ax
        self.is_set_package_plot = True

    #------------------------------------------------------------------------------
    #
    #------------------------------------------------------------------------------
    def set_master_and_slaves_curves(self, elements_per_curves,\
                                      master_curves_ids,\
                                      slave_curves_ids):

        assert not self.is_set_master_and_slave_curves, 'the curves\
        have already been set '
        assert elements_per_curves.shape == ( master_curves_ids.shape[0]+\
                slave_curves_ids.shape[0], 2)
        self.nCurves = master_curves_ids.shape[0]+\
                slave_curves_ids.shape[0]
        self.el_per_curves = elements_per_curves
        self.nmaster_curves = master_curves_ids.shape[0]
        self.nslave_curves = slave_curves_ids.shape[0]
        self.master_curves_ids=master_curves_ids
        self.slave_curves_ids=slave_curves_ids
        assert np.intersect1d(master_curves_ids, slave_curves_ids,
                assume_unique = True).shape[0] == 0

        # the preallocation is made only once
        # coordinates of the limit of the bounding boxes
        self.box_lim = zeros((self.nCurves, 2, 3,), dtype = float)
        # center of the enclosing sphere (enclosing the bounding box)
        self.ctr_box = zeros((self.nCurves, 3), dtype = float)
        # radius
        self.r_sphbox = zeros((self.nCurves, ), dtype = float)

        #TODO : should be  sparse arrays
        self.pairs_curves_to_check = np.zeros( ( self.nmaster_curves ,
            self.nslave_curves), dtype = bool)

        self.is_set_master_and_slave_curves = True
        self.ContactTable.set_curve_to_curve_dynamic_table(\
            self.nCurves,\
-           self.slave_curves_ids, self.master_curves_ids, \
            self.nmaster_curves, self.nslave_curves)


        # which curve belongs to which yarn
        self.curve_in_yarn = np.zeros(self.nCurves, dtype = int)
        Yi = 0
        for ii in range(self.nCurves-1):
            el0, el1 = self.el_per_curves[ii]
            el2, el3 = self.el_per_curves[ii + 1]
            self.curve_in_yarn[ii] = Yi
            if el1 != el2:
                Yi += 1
        # last curve
        el0, el1 = self.el_per_curves[-2]
        el2, el3 = self.el_per_curves[-1]
        if el1 == el2:
            self.curve_in_yarn[-1] = Yi
        else:
            self.curve_in_yarn[-1] = Yi + 1

        # initial check to ensure that we start from a C1 surface
        self.CheckContinuitySurface()

    #------------------------------------------------------------------------------
    #
    #------------------------------------------------------------------------------
    def CheckContinuitySurface(self,):
        # check the continuity of the surface yarn per yarn
        try:
            theta = 2 * np.pi * np.random.rand(1)
            for Y in range(self.nY):
                c_in_Y = np.argwhere(self.curve_in_yarn == Y).ravel()
                for ci in c_in_Y[:-1]:
                    id_c0 = ci
                    id_c1 = ci + 1
                    s0, sxi0, stheta0, n0 = self.SurfPtAndVects_smoothed(id_c0, array([1., theta]))
                    s1, sxi1, stheta1, n1 = self.SurfPtAndVects_smoothed(id_c1, array([0., theta]))
                    assert np.allclose( array(s0), array(s1))
                    assert np.allclose( array(sxi0), array(sxi1))
                    assert np.allclose( array(stheta0), array(stheta1))
                    assert np.allclose( array(n0), array(n1))
        except AssertionError:
            raise ValueError('Some discontinuity in the smoothed surface have been spotted for the\
                    initialconfig')
        else:
            print('\n')
            print('TEST SMOOTHNESS SURFACE PASSED')
            print('\n')



    ################################################################################
                            # SETTERS
    ################################################################################


    ################################################################################
                                # FE ARRAYS
    ################################################################################
    #------------------------------------------------------------------------------
    #
    #------------------------------------------------------------------------------
    def getFandK(self, u, **kwargs):
        """
        get force vector and stiffness of the global finite element assembly
        """
        if self.HasConstraints: Tab = self.ContactTable

        if self.HasConstraints and self.enforcement == 1 :
            nLM = len(Tab.dofs_LM[where(Tab.active_set)].ravel())
            ndo = nLM + self.tot_ndofs
        else:
            ndo = self.tot_ndofs

        K = zeros((ndo, ndo))
        fint = zeros(ndo)
        #------------------------------------------------------------------------------
        # Contribution from internal forces
        #------------------------------------------------------------------------------
        for eli in self.el:
            eli.getFintAndK(self.X[eli.nID], self.u[eli.nID], self.v[eli.nID])
            K[ix_(eli.dofs.ravel(), eli.dofs.ravel()) ] += eli.stiff
            fint[eli.dofs.ravel()] += eli.fint
            # ensure that there is continuity in the interpolation of the rotation field

        #------------------------------------------------------------------------------
        # Contribution from contact between yarns
        #------------------------------------------------------------------------------
        if self.HasConstraints:
            self.weak_enforcement_IntInt2IntInt(fint, K)
            """
            # plot all the contact points
            self.plot_all_contact()
            set_trace()
            """
        #------------------------------------------------------------------------------
        # Contribution from contact with a rigid sphere
        #------------------------------------------------------------------------------
        if self.HasSphere:
            # TODO : brute force approach must be changed
            for id_curve in range(self.nCurves):
                # for the time being only one curve can be in contact
                (Fi, Ki, dofs_ctc_el, CONTRIBUTE) =\
                        self.contact_rigid_sphere(id_curve)
                if CONTRIBUTE:
                    assert norm(Fi) > 1e-10 and norm(Ki) > 1e-10
                    assert np.unique(dofs_ctc_el).shape == dofs_ctc_el.shape
                    fint[dofs_ctc_el] += Fi
                    K[ix_(dofs_ctc_el, dofs_ctc_el)] += Ki

        return fint, K


    #------------------------------------------------------------------------------
    #
    #------------------------------------------------------------------------------
    def broad_phase(self,):
        for i in range(self.nCurves):
            # TODO : beaucoup trop lent et couteux !! je pourrais faire un truc bien plus facile en
            # prenant des points le long de la centroid line
            # idee facile :
            # 5 points le long de la centroid line.
            # calucl distance, r = dmax + 0.5 * a et le centre de la sphere est pour
            #
            self.box_lim[i,] =\
                         self.get_bounding_box_one_curve(\
                         self.el_per_curves[i],\
                         nxi=10,\
                         ntheta = 10)
            # in the middle of the diagonal of the box
            self.ctr_box[i] = 0.5 * (self.box_lim[i,0]+self.box_lim[i,1])
            # from the center to a vertex
            self.r_sphbox[i] = norm(self.box_lim[i,0]-self.ctr_box[i])
            """
            plot_sphere(self.ctr_box[i],\
                    self.r_sphbox[i],
                    color = tuple(np.random.rand(3)),
                    opacity = 0.1)
            """

        Tab = self.ContactTable
        # check intersection of the bounding shperes
        # this will determine which curves will be checked for contact
        # during one step
        for ii in self.slave_curves_ids:
            for jj in self.master_curves_ids:
                if (norm(self.ctr_box[ii] - self.ctr_box[jj]) -\
                        (self.r_sphbox[ii] + self.r_sphbox[jj])) < 1e-10:
                    # the enclosing spheres are close enough
                    # closer look at the bounding boxes to see if they are intersecting
                    bYMii = self.box_lim[ii]
                    bYSjj = self.box_lim[jj]
                    Tab.close_curves[ii, jj] =\
                    collision_AABB(bYMii, bYSjj, eps = 0.05 * self.r_sphbox[i] )



    #------------------------------------------------------------------------------
    #
    #------------------------------------------------------------------------------
    def choose_active_set(self, stp=0):
        """
        return bool to state if there has been some modification of the active set
        """
        if not self.is_set_ContactTable:
            raise ValueError('The contact table must be set! Use set_Contact_Table method ')
        Tab = self.ContactTable
        # ABB intersection
        self.broad_phase()
        # return bool: true if AS was correct
        return self.narrow_phase(stp)

    #------------------------------------------------------------------------------
    #
    #------------------------------------------------------------------------------
    def check_current_active_set(self, stp = 0):
        Tab = self.ContactTable
        # restore attribute
        self.current_active_set_correct= self.choose_active_set(stp)
        AS_correct = self.current_active_set_correct

        if self.enforcement == 0:
            # regularizes the penalty stiffnesses if necessary
            try:
                PenaltyCorrect = Tab.PenaltyRegularization()
            except MaximumPenetrationError:
                self.plot(opacity = 0.1)
                idGQP = np.argwhere(Tab.gN < Tab.critical_penetration)
                pts = self.plot_set_GQP(idGQP)
                pts.glyph.glyph.scale_factor = 0.05
                set_trace()
                self.choose_active_set()
                AS_correct = Tab.PenaltyRegularization()
        if self.enforcement == 0:
            return AS_correct and PenaltyCorrect
        elif self.enforcement == 1:
            return AS_correct
        """
        if self.ContactTable.close_curves[mcurve_id, scurve_id]:
            mlab.clf()
            self.plot_pairs_of_smoothed_surface(self.el_per_curves[pair_curve],\
                    opacity = 0.3)
            for bY in [bYMii, bYSjj]:
                vertices = np.asarray([i for i in product(bY[:,0], bY[:,1], bY[:,2])])
                x,y,z = vertices.T[(0,1,2),]
                glyph= mlab.points3d(x,y,z, color = (0.,1.,0.), scale_factor = 0.1)
                engine = mlab.get_engine()
                from mayavi.modules.outline import Outline
                engine.add_filter(Outline(), glyph)
            x,y,z = self.ctr_box[mcurve_id]
            mlab.text3d(x,y,z,str(mcurve_id),scale = 0.1)
            x,y,z = self.ctr_box[scurve_id]
            mlab.text3d(x,y,z,str(scurve_id),scale=0.1)
            set_trace()
        """
    #------------------------------------------------------------------------------
    #
    #------------------------------------------------------------------------------
    def narrow_phase(self, stp):
        """
        #for pair in np.argwhere(Tab.close_curves):
        #in_parallel = (self.nCurves > 20)
        in_parallel = True
        if not in_parallel:
            [self.narrow_phase_for_a_curve_pair(pair) for pair in np.argwhere(Tab.close_curves)]
        else:
            num_cores = multiprocessing.cpu_count()
            Parallel(n_jobs=num_cores)(delayed(has_shareable_memory)(self.narrow_phase_for_a_curve_pair(pair))\
                    for pair in np.argwhere(Tab.close_curves))
        """

        def plot_problematic_area():
            """
            useful to debug
            """
            id_els = self.el_per_curves[(current_curves,)].ravel()

            xmstr =\
            self.get_surf_point_smoothed_geo(\
                    id_els[(2,3),],
                    asarray(hSol))

            xFG =\
            self.get_surf_point_smoothed_geo(\
                    id_els[(2,3),],
                    hFG)
            for cloud in samp_on_master_candi:
                scatter_3d(cloud.reshape(-1,3), color = tuple(np.random.rand(3)), scale_factor
                        = 1/100)

            self.plot_integration_interval(id_els[(0,1),],\
                    xi_lim_slv,
                    theta_lim_slv,
                    opacity = 1., color = (0.,0.,0.) )

            self.plot_integration_interval(id_els[(2,3),],\
                    xi_lim_mstr,
                    theta_lim_mstr,
                    opacity = 1.,
                    color = (0., 1.,0.))

            mlab.points3d(\
                    xFG[0], xFG[1], xFG[2], color = (1.,0.,0.),\
                    scale_factor = 0.01, name = 'FG')
            mlab.points3d(\
                    samp_on_master_candi[(idx_min)][0],
                    samp_on_master_candi[(idx_min)][1],
                    samp_on_master_candi[(idx_min)][2],
                    color = (1.,0.,0.),\
                    scale_factor = 0.01, name = 'from arr')

            """
            mlab.points3d(\
                    samp_on_master_candi[0,3,:][:,0],
                    samp_on_master_candi[0,3,:][:,1],
                    samp_on_master_candi[0,3,:][:,2],
                    color = (1.,0.,0.),\
                    scale_factor = 0.01, name = 'xi = 0.1')
            """

            mlab.points3d(\
                    xmstr[0], xmstr[1], xmstr[2], color = (0.,1.,0.),\
                    scale_factor = 0.01, name = 'sol on mstr' )

            mlab.points3d(\
                    xGQP[0], xGQP[1], xGQP[2], color = (1.,1.,1.),\
                    scale_factor = 0.01, name = 'pos GQP')


        Tab = self.ContactTable
        # arrays created just to validate cython implementation
        nxiGQP = Tab.nxiGQP
        nthetaGQP= Tab.nthetaGQP
        nuxiGQP, WnuxiGQP =  np.polynomial.legendre.leggauss(nxiGQP)
        nuthetaGQP, WnuthetaGQP =\
        np.polynomial.legendre.leggauss(nthetaGQP)
        nxi_sampled = 20
        ntheta_sampled = 20
        # the master curve has no integration interval
        xi_lim_mstr = array([0,1])
        theta_lim_mstr = array([0, 2 * np.pi])
        # always the same
        xi_mstr = np.linspace(xi_lim_mstr[0],\
                xi_lim_mstr[1],\
                nxi_sampled)
        theta_mstr = np.linspace(theta_lim_mstr[0],\
                theta_lim_mstr[1],\
                ntheta_sampled)

        AS_modified = False

        for slave in self.slave_curves_ids:
            master_close = np.where(Tab.close_curves[slave])
            is_active_slave, idx_in_AS = Tab.query_is_active_slave(slave)
            nmclose = len(master_close[0])

            check_penetration_cells = False

            if is_active_slave and nmclose == 0:
                if np.all(Tab.gN[idx_in_AS]) > 0:
                    Tab.del_contact_element(idx_in_AS)
                    print('el deacti')
                    is_active_slave = False
                    AS_modified = True
                else:
                    self.plot_integration_interval(
                            self.el_per_curves[slave],
                            array([0, 1.]),
                            array([0, 2 * np.pi]),
                            opacity = 1., color = (0.,0.,0.) )
                    plot_AABB(self.box_lim[slave])
                    set_trace()
                    raise ValueError('sudden loss of contact not allowed')

            if  nmclose >0:
                assert np.array_equal(np.unique(master_close), np.sort(
                    np.array(master_close).ravel()))

                id_els_sl = self.el_per_curves[slave]
                # construct slave obb around entire slave curve
                samp_slave = self.sample_surf_point_on_smooth_geo(\
                        id_els_sl,\
                        nxi_sampled,\
                        ntheta_sampled,\
                        array([0,1]),
                        array([0,2*np.pi])).reshape(-1,3)
                obb_slave = build_obb(samp_slave)
                bi = self.control_points(slave)
                obb_slave = build_obb(samp_slave)


                # master obb(s)
                samp_on_master_candi =\
                        zeros( (len(master_close[0]), nxi_sampled , ntheta_sampled,
                    3), dtype = np.float)
                obb_master = []

                for uu, id_m, in enumerate(master_close[0]):
                    samp_on_master_candi[uu]=\
                        self.sample_surf_point_on_smooth_geo(\
                            self.el_per_curves[(id_m),].ravel(),
                            nxi_sampled,\
                            ntheta_sampled,\
                            xi_lim_mstr,\
                            theta_lim_mstr)
                    obb_master.append(build_obb(samp_on_master_candi[uu]))

                # is there intersection between the OBB containing the whole slave curve and the
                # master one ?
                collision_obb = False
                for uu, obb_m in enumerate(obb_master):
                    if collision_btw_obb(obb_m, obb_slave, eps = 1e-10):
                        collision_obb = True

                        """
                        self.plot_integration_interval(
                            self.el_per_curves[slave],
                            array([0, 1.]),
                            array([0, 2 * np.pi]),
                            opacity = 1., color = (0.,0.,0.) )
                        scatter_3d( samp_on_master_candi[uu].reshape(-1,3), color = (1.,1.,1.))
                        scatter_3d( samp_on_master_candi[uu].reshape(-1,3), color = (1.,1.,1.))
                        plot_obb_vertices(obb_m, (1.,0.,0.))
                        plot_obb_vertices(obb_slave, (0.,1.,0.))
                        set_trace()
                        """

                if collision_obb:
                    # a closer look at the penetration is needed
                    check_penetration_cells = True
                    # establish the limit of the integration domain
                    # construct obb for each slave domain
                    obb_slv_II = []
                    # determine which slave_obb has an intersection with a master_obb
                    has_int_xi_lim = []
                    has_int_theta_lim = []
                    for (ii,jj) in product(range(Tab.xi_lim.shape[0]),\
                            range(Tab.theta_lim.shape[0])):
                        samp_slv_II =\
                            self.sample_surf_point_on_smooth_geo(\
                                id_els_sl, nxi_sampled,\
                                ntheta_sampled,\
                                Tab.xi_lim[ii],
                                Tab.theta_lim[jj])
                        obb_slv_II.append(build_obb(samp_slv_II.reshape(-1,3) ))
                        # get teh traid at the beginning of the interval

                        for uu, obb_m in enumerate(obb_master):
                            if collision_btw_obb(obb_slv_II[-1], obb_m):
                                has_int_xi_lim.append(Tab.xi_lim[ii])
                                has_int_theta_lim.append(Tab.theta_lim[jj])

                                # DO NOT DELETE THESE LINES
                                """
                                self.plot_integration_interval(
                                    self.el_per_curves[slave],
                                    Tab.xi_lim[ii],
                                    Tab.theta_lim[jj],
                                    opacity = 1., color = (0.,0.,0.) )

                                plot_obb_vertices(obb_slv_II[-1]  , (0.,0.,0.))
                                # build obb from convex hull to compare
                                obb_hull = build_obb(samp_slv_II.reshape(-1,3), use_convex_hull = True)
                                plot_obb_vertices(obb_hull  , (1.,0.,0.))
                                scatter_3d( samp_on_master_candi[uu].reshape(-1,3), color = (1.,1.,1.))
                                plot_obb_vertices(obb_m, (1.,1.,1.))
                                set_trace()
                                """


                    if len(has_int_xi_lim) == 0 or len(has_int_theta_lim) == 0 :
                        raise ValueError('No slave interval intersecting master curves has not been found')

                    # construct integration interval from intersection between obb
                    has_int_xi_lim = asarray(has_int_xi_lim)
                    has_int_theta_lim = asarray(has_int_theta_lim)

                    # this interval will be the one on which we will construct the cells
                    xi_lim_slv, theta_lim_slv = self.extract_slave_interval(
                            has_int_xi_lim, has_int_theta_lim)

                    # TODO: must be generalized
                    # number of cells along the xi and theta axis
                    nxi_slice = int(Tab.n_cells_xi_per_slice)
                    ntheta_slice = int(Tab.n_cells_theta_per_slice)

                    # cut the interval in unit cell
                    KSI, con = generation_grid_quadri_and_connectivity(
                            xi_lim_slv, theta_lim_slv, nxi_slice, ntheta_slice)
                    # convective coord center of the cell
                    KSIC = 0.5 * (KSI[con[:,0],] + KSI[con[:,2],] )
                    # area in the space of the convective coordinates for each cell (they all have
                    # the same weigth)
                    wxi =  (KSI[con[:,1],] - KSI[con[:,0],])[:,0]
                    wtheta = (KSI[con[:,3],] - KSI[con[:,0],])[:,1]

                    if Tab.enforcement == 1:
                        weak_enforcement_new, weak_constraint = (0,) * 2

                    KSISol = zeros((KSIC.shape[0], 2), dtype = float)
                    gNSol = zeros((KSIC.shape[0]), dtype = float)
                    IDM = zeros((KSIC.shape[0]), dtype = int)

                    # for each cell, measure gN at its center
                    for ii, KSICii in enumerate(KSIC):
                        #TODO: if some info from the previous increment is available, use it
                        xGQP = self.get_surf_point_smoothed_geo(id_els_sl, KSICii)
                        # among the possible master candidates, spot the closest sampled
                        # point from the current GQP
                        d_FG = np.inf
                        # TODO : if I reshape the array of sampled points on the master and I
                        # use unravel_index, I avoid the for loop
                        for uu in range(nmclose):
                            Dist = norm((samp_on_master_candi[uu]- xGQP), axis = 2)
                            idx = np.unravel_index( Dist.argmin(), (samp_on_master_candi[uu].shape[:2]))
                            if Dist[idx] < d_FG:
                                d_FG = Dist[idx]
                                idx_min = (uu, idx[0], idx[1])

                        current_mcurve = master_close[0][idx_min[0]]
                        hFG = array([xi_mstr[idx_min[1]], theta_mstr[idx_min[2]]])
                        current_curves = array([slave, current_mcurve])
                        id_els = self.el_per_curves[(current_curves,)].ravel()
                        # do not remove this assert. Otherwise would be very hard to debug
                        assert np.allclose(samp_on_master_candi[(idx_min)],\
                            self.get_surf_point_smoothed_geo(id_els[(2,3),], hFG))

                        Sol_correct = 0
                        ct_corrections = 0

                        # siolve the minimum distance problem at each center of a cell
                        while not Sol_correct:
                            (gNii, hii,
                                    w_ii,\
                                    ExitCode) = \
                                    self.gN_GQP_IntInt2IntInt(\
                                    current_curves,\
                                    KSICii[0],
                                    wxi[ii],
                                    KSICii[1],
                                    wtheta[ii],
                                    hFG)


                            assert ExitCode == 1
                            if not -1e-10 < hii[0] < 1+1e-10:
                                if hii[0] < 0 :
                                    assert hii[0] > -1.
                                    hFG = np.asarray(hii)
                                    hFG[0] = 1
                                    current_mcurve -= 1
                                    current_curves = array([slave, current_mcurve])

                                elif hii[0] > 1 :
                                    assert hii[0] < 2
                                    hFG = np.asarray(hii)
                                    hFG[0] = 0
                                    current_mcurve += 1
                                    current_curves = array([slave, current_mcurve])
                                ct_corrections += 1
                                if ct_corrections >= 5:
                                    raise ValueError('could not find master curve')
                            else:
                                Sol_correct = 1



                            if gNii < Tab.critical_penetration:
                                # Fatal error
                                plot_problematic_area()

                                set_trace()
                                xGQP = self.get_surf_point_smoothed_geo(id_els_sl,
                                        KSICii)

                                (gNii, hii,
                                    w_ii,\
                                    ExitCode) = \
                                    self.gN_GQP_IntInt2IntInt(\
                                    current_curves,\
                                    KSICii[0],
                                    wxi[ii],
                                    KSICii[1],
                                    wtheta[ii],
                                    hFG)

                            if Tab.enforcement == 1:
                                raise NotImplementedError('deprectd')
                                if is_active_slave:
                                    # interpolate LM
                                    set_trace()
                                    LMnm
                                    # TODO
                                    weak_enforcement_new += LMnm * w_ii
                                else:
                                    weak_constraint += w_ii * gNii

                        KSISol[ii] = hii
                        gNSol[ii] = gNii
                        IDM[ii] = int(current_curves[1])

                    #self.plot_all_cells_centers_and_gap(current_curves[0], KSICii, KSISol, gNSol, IDM)


            # was the slave already present in the contact list ?
            # if not create new entry in the contact table
            if Tab.enforcement == 1:
                raise ValueError('Deprecated')
                if is_active_slave and weak_enforcement > 0:
                     to_deactivate.append(slave)
                else:
                    if weak_constraint < 0:
                        set_trace()
                        to_activate.append(slave)
                        # store the results of the contact detection

            if check_penetration_cells :

                # select all the cells where the gap is small enough as part of
                # the integration domain
                active_cells = np.argwhere(gNSol <  Tab.epsilon).flatten()
                if is_active_slave:
                    if active_cells.shape[0] == 0:
                        set_trace()
                        to_deactivate.append(slave)
                        print('complete deactivation needed')
                    else:
                        id_sl_c = np.argwhere(array(Tab.active_set) == slave).ravel()
                        assert id_sl_c.shape[0] == 1
                        prev_active_cells = Tab.active_cells[int(id_sl_c)]
                        # have the active cells changed ?
                        if not( (len(prev_active_cells) == len(active_cells)) and \
                                (np.allclose(np.sort(prev_active_cells), np.sort(active_cells) ) ) ):
                            #if stp == 39: set_trace()
                            AS_modified = True
                            # update the set of active_cells
                            Tab.modify_set_of_active_cells(
                                    int(id_sl_c), active_cells)
                            print('change in the set of active cells')
                        else:
                            # the active_cells stayed the same
                            print('no change in active cells')
                else:
                    if nmclose >0:
                        if collision_obb:
                            if active_cells.shape[0] > 0:
                                Tab.new_el(slave, active_cells,\
                                        IDM, KSISol, KSI, con )
                                AS_modified = True
            else:
                if is_active_slave:
                    raise ValueError('the element was active but all of its cells for deactivated at once. Dangerous')
        return not AS_modified


    #------------------------------------------------------------------------------
    #
    #------------------------------------------------------------------------------

    def extract_slave_interval(self,  has_int_xi_lim,
            has_int_theta_lim):
        xi_lim_slv = array([np.min(has_int_xi_lim[:,0], axis = 0),\
                np.max(has_int_xi_lim[:,1], axis = 0)]).ravel()
        # Special attention because of the periodicity of the trigonometric
        # circle. The easiest way is to determine the integration interval using
        # degrees
        has_int_theta_lim = np.degrees(has_int_theta_lim)
        theta_lim_slv = array([has_int_theta_lim.min(),
            has_int_theta_lim.max()])
        range_deg = theta_lim_slv[1] - theta_lim_slv[0]
        if range_deg < 1:
            raise ValueError('integration domain too small')
        if range_deg > 180:
            raise ValueError('integration domain too broad')


        return xi_lim_slv, np.radians(theta_lim_slv )


    #------------------------------------------------------------------------------
    #
    #------------------------------------------------------------------------------
    def gN_GQP_IntInt2IntInt( self,\
                     id_curves,\
                     xiGQP,\
                     WxiGQP,\
                     thetaGQP,\
                     WthetaGQP,\
                     hFG):

        id_els = self.el_per_curves[id_curves,].ravel()
        assert id_els.shape == (4,)
        nd_C1 =  self.nPerEl[id_els[(0,1),]].flatten()[(0,1,3),]
        nd_C2 =  self.nPerEl[id_els[(2,3),]].flatten()[(0,1,3),]
        # we will interpolate the cross section vector at the
        # junction between the 2 beams of the same yarns
        # we use the FG stored because there is necesarrily a contact
        assert hFG.shape == (2,)
        # return (gN, hSol, WGQP, ExitCode)
        return ctcVolumeBtoB.LocalDetec_1GQP(\
            self.X[(nd_C1,nd_C2),].reshape(2,9),\
            self.u[(nd_C1,nd_C2),].reshape(2,9),\
            self.v[(nd_C1,nd_C2),].reshape(2,9),\
            self.t1[id_els,],\
            self.t2[id_els,],\
            self.a_crossSec[id_els[(0,2),]],\
            self.b_crossSec[id_els[(0,2),]],\
            xiGQP,thetaGQP, WxiGQP, WthetaGQP,\
            hFG,\
            self.ContactTable.alpha)

    ################################################################################
                                # FE ARRAYS
    ################################################################################




    ################################################################################
                            # GEOMETRICALLY EXACT BEAM ROUTINES
    ################################################################################
    #------------------------------------------------------------------------------
    #
    #------------------------------------------------------------------------------
    def Update_Total_Rotational_Vars(self, DELv):
        assert DELv.shape == (self.nn, 3)
        # update nodal total rotation vectors
        if self.rot_dofs_storage == 0:
            """
            WE RELY HERE ON TABLE 11 [Simo]
            Table 4, page 97 [Simo] IS SINGULAR FOR SOME CASES
            For the update of the rotations, we currently do the following AT EACH NODE:
                1. Retrieve orthogonal tensor from the stored quaternions parameters
                2. from vorticity vector, get unit quaternions. Get corresponding orthogonal tensor
                3. left multiply with 2. LAM
                4. compute new quaternions
            """
            for I in range(self.nn):
                # quaternions corresponding to vorticity vector
                Q = getOrthoMatFromQuaternions(quaternionsFromRotationVect(DELv[I]))
                assert np.allclose(Q, getOrthoTensorFromRotationVector(DELv[I]))
                # current rotation tensor
                LAM = getOrthoMatFromQuaternions(self.q[I])
                # update nodal quaternions from new rotation tensor by left multiplication
                self.q[I] = getQuaternions(dot(Q, LAM ))
                assert isOrthogonal(getOrthoMatFromQuaternions(self.q[I]))
        elif self.rot_dofs_storage == 1:
            for I in range(self.nn):
                # get transfo from stored rotation vector
                T = get_linear_transfo_between_tgt_spaces_of_SO3(self.v[I])
                # we can now safely add the 2 rotation vector because they belong to the same
                # tangent space of SO3
                self.v[I] += T.dot(DELv[I])

        else: raise NotImplementedError
    #------------------------------------------------------------------------------
    #
    #------------------------------------------------------------------------------
    def update_config_geometrically_exact_beams(self, du):
        assert du.shape == (self.nn * 6,)
        # splitting of the increment ot handle rotational dofs apart
        # update the translationnal dofs
        self.u += du.reshape(-1,6)[:, :3]
        # by reshaping this way we direclty get the vorticty vectors
        DELv_I= du[self.dofs_rot]
        #I. Update of the quaternions/Rotation tensors or total rotation vector
        self.Update_Total_Rotational_Vars(DELv_I)
        #II. update strains and stresses at each GQP
        for eli in self.el:
            # note that here the qua
            if self.rot_dofs_storage == 0:
                eli.update_configuration(self.q[eli.nID], self.X[eli.nID],
                                         self.u[eli.nID], DELv_I[eli.nID])
            elif self.rot_dofs_storage == 1:
                eli.update_configuration(self.v[eli.nID], self.X[eli.nID],
                                         self.u[eli.nID], DELv_I[eli.nID])
            else:
                raise NotImplementedError

    ################################################################################
                            # GEOMETRICALLY EXACT BEAM ROUTINES
    ################################################################################


    #------------------------------------------------------------------------------
    #
    #------------------------------------------------------------------------------
    def get_bounding_box_one_curve(self,id_els, nxi, ntheta) :
        """
        note: has to be destinguished from the bounding box of a
        smoothed curve
        """
        return self.get_bounding_box_around_part_smoothed_curve(\
                id_els, nxi, ntheta, array([0,1]), array([0, 2 *
                    np.pi]), endpoint = False)

    #------------------------------------------------------------------------------
    #
    #------------------------------------------------------------------------------
    def get_bounding_box_around_part_smoothed_curve(self, id_els,\
            nxi,ntheta,xi_lim, theta_lim, endpoint = True) :

        pos = self.sample_surf_point_on_smooth_geo(\
                id_els, nxi, ntheta,\
                xi_lim, theta_lim).reshape(-1,3)

        # directly return the vertices of the bounding box
        return getAABBLim(pos)


    #------------------------------------------------------------------------------
    #
    #------------------------------------------------------------------------------
    def are_intersecting_bounding_boxes_integ_segments(self, id_curves,\
         xi_lim, theta_lim, nxi, ntheta) :
        assert id_curves.shape == (2,)
        assert len(xi_lim) == 2
        assert len(theta_lim) == 2
        assert nxi > 1
        assert ntheta > 1
        AABB_master =\
        self.get_bounding_box_around_part_smoothed_curve(\
        self.el_per_curves[id_curves[0]],\
        nxi, ntheta, xi_lim[0], theta_lim[0])

        AABB_slave =\
        self.get_bounding_box_around_part_smoothed_curve(\
        self.el_per_curves[id_curves[1]],\
        nxi, ntheta, xi_lim[1], theta_lim[1])

        return collision_bounding_boxes(AABB_master,\
                AABB_slave)




    ################################################################################
                                    # PLOTTING
    ################################################################################
    def plot(self,\
            diff_color_each_element = True,\
            color=(0.6, 0.6, 0.6),\
            opacity = 1., \
            nS = 10,\
            ntheta = 20,\
            useYarnConnectivity = False,
            manage_rendering = False,
            lock_camera = False):
        assert self.is_set_package_plot, 'package plot must be set as well as the axis'

        f = mlab.gcf()
        f.scene.background = (0.2, 0.2, 0.2)

        if not manage_rendering:
            f.scene.disable_render = True

        if not hasattr(self, "camera_settings"):
            # mayavi constantly update the camera position due to the change in the value of the
            # arrays
            self.camera_settings = {'position' : f.scene.camera.position,
                    'focal_point': f.scene.camera.focal_point,
                    'view_angle' : f.scene.camera.view_angle,
                    'view_up' : f.scene.camera.view_up,
                    'clipping_range':f.scene.camera.clipping_range}

        if not useYarnConnectivity:
            for ii, eli in enumerate(self.el):
                if not hasattr(eli, "color_surface_mayavi"):
                    if diff_color_each_element:
                        eli.color_surface_mayavi = tuple(np.random.rand(3))
                    else:
                        eli.color_surface_mayavi = color

                pos = eli.get_mesh_surface_points(
                                self.X[eli.nID],  self.u[eli.nID],
                                self.v[eli.nID],
                                nS=nS, ntheta=ntheta)

                x,y,z = (pos[:,:,i] for i in range(3))
                if not hasattr(self.el[ii], 'ms'):
                    # source of data of the mesh
                    eli.mesh = mlab.mesh(\
                            pos[:,:,0],
                            pos[:,:,1],
                            pos[:,:,2],
                            color=eli.color_surface_mayavi,
                            opacity=opacity)
                    # handle the source
                    eli.ms = self.el[ii].mesh.mlab_source
                else:
                    eli.ms.set(\
                            x=pos[:,:,0],
                            y=pos[:,:,1],
                            z=pos[:,:,2])
        else:
            UPDATE = True
            if not hasattr(self, "mesh_obj_Y"):
                self.mesh_obj_Y = zeros(len(self.el_per_Y),\
                        dtype = np.object)
                self.mesh_obj_Y_src = zeros(len(self.el_per_Y),\
                        dtype = np.object)
                UPDATE = False


            for ii, eliYi in enumerate(self.el_per_Y):
                POS = np.vstack([ self.el[jj].get_mesh_surface_points(
                        self.X[self.el[jj].nID],  self.u[self.el[jj].nID],
                        self.v[self.el[jj].nID],
                        nS=nS, ntheta=ntheta) for jj in eliYi])

                if not UPDATE:
                    #construct the vtk object
                    self.mesh_obj_Y[ii] = mlab.mesh(\
                                POS[:,:,0],
                                POS[:,:,1],
                                POS[:,:,2],
                                color=color, opacity=opacity)
                else:
                    self.mesh_obj_Y[ii].mlab_source.set(\
                            x=POS[:,:,0],
                            y=POS[:,:,1],
                            z=POS[:,:,2])
        if self.HasSphere:
            self.Sphere.plot()

        if not manage_rendering:
            if lock_camera:
                f.scene.camera.position = self.camera_settings['position']
                f.scene.camera.focal_point = self.camera_settings['focal_point']
                f.scene.camera.view_angle = self.camera_settings['view_angle']
                f.scene.camera.view_up = self.camera_settings['view_up']
                f.scene.camera.clipping_range = self.camera_settings['clipping_range']
            f.scene.disable_render = False


    #------------------------------------------------------------------------------
    #
    #------------------------------------------------------------------------------
    def plot_all_QP_and_ProjStored(self):
        Tab = self.ContactTable
        nxiGQP = Tab.nxiGQP
        nthetaGQP= Tab.nthetaGQP
        nuxiGQP, WnuxiGQP =  np.polynomial.legendre.leggauss(nxiGQP)
        nuthetaGQP, WnuthetaGQP =\
        np.polynomial.legendre.leggauss(nthetaGQP)

        # for the time being the master has always the same limit
        xi_lim_mstr = array([0,1])
        theta_lim_mstr = array([0, 2 * np.pi])
        tol_for_err = 0.1

        tot_active_cells = sum( ([len(Tab.active_cells[i]) for i in range(len(Tab.active_set))]))
        xGQP_arr = zeros((tot_active_cells * nxiGQP * nthetaGQP, 3 ), dtype= float)
        xFG_arr = zeros(xGQP_arr.shape, dtype= float)
        # from the value of the gap
        s =  zeros((tot_active_cells * nxiGQP * nthetaGQP), dtype= float)

        ct = 0
        for ii, id_slv in enumerate(Tab.active_set):
            con = Tab.CON_Cells[ii]
            KSI = Tab.KSI_Cells[ii]

            # WARNING: VERY EASY TO MAKE A MISTAKE WHEN USING jj OR CELL
            # jj : index in the contact table for the active set because we record only what is oing
            # on in the active cells
            # cell: the actual index of the active cell
            for jj, cell in enumerate(Tab.active_cells[ii]):
                # get the limit of the cell
                KSI_Cell = KSI[(con[cell][0],
                    con[cell][1],
                    con[cell][2],
                    con[cell][3]),]

                xi_lim_slv = array([KSI_Cell[:,0].min(), KSI_Cell[:,0].max()])
                theta_lim_slv = array([KSI_Cell[:,1].min(), KSI_Cell[:,1].max()])

                for (kk,ll) in product(range(nxiGQP), range(nthetaGQP)):

                    # first guess for the id of the master curve on which the CPP will be found
                    id_mstr = Tab.ID_master[ii][jj][kk,ll]
                    if id_mstr < 0 :
                        if id_mstr == -999:
                            id_mstr = Tab.ID_master_Ctr[ii][cell]
                        else:
                            raise ValueError

                    current_curves = array([id_slv, id_mstr])
                    xiGQP, WxiGQP= TransformationQuadraturePointsAndWeigths1D(\
                             xi_lim_slv , nuxiGQP[kk], WnuxiGQP[kk])
                    thetaGQP, WthetaGQP = TransformationQuadraturePointsAndWeigths1D(\
                             theta_lim_slv, nuthetaGQP[ll], WnuthetaGQP[ll])

                    hFG = Tab.h[ii][jj, kk, ll][2:]
                    if np.any(np.isnan(hFG)):
                        assert np.all(np.isnan(hFG))
                        hFG = Tab.hCtr[ii][cell]

                    id_els = self.el_per_curves[current_curves,].ravel()
                    xGQP_arr[ct] = self.get_surf_point_smoothed_geo(id_els[(0,1),], array([ xiGQP,\
                            thetaGQP]))
                    xFG_arr[ct] = self.get_surf_point_smoothed_geo(id_els[(2,3),], hFG)

                    gN = Tab.gN[ii][jj][kk,ll]
                    if np.isnan(gN):
                        s[ct] = 0.1
                    else:
                        s[ct] = gN

                    ct += 1


        self.plot(opacity = 0.2, diff_color_each_element = False)


        # Create the points
        x = np.vstack( (xGQP_arr[:,0],  xFG_arr[:,0])).T.ravel()
        y = np.vstack( (xGQP_arr[:,1],  xFG_arr[:,1])).T.ravel()
        z = np.vstack( (xGQP_arr[:,2],  xFG_arr[:,2])).T.ravel()
        # one color for each point. interpolation between them
        s = np.vstack( (s,  s)).T.flatten()
        if not hasattr(self, 'src_QP_FG'):
            self.src_QP_FG= mlab.pipeline.scalar_scatter(x, y, z, s)
            connections = np.arange(2 * xGQP_arr.shape[0]).reshape(-1,2)
            # Connect them
            self.src_QP_FG.mlab_source.dataset.lines = connections
            # The stripper filter cleans up connected lines
            self.lines = mlab.pipeline.stripper(self.src_QP_FG)
        else:
            self.src_QP_FG.mlab_source.set(x=x, y=y, z=z, s=s)

        lines = self.lines
        self.src_QP_FG.update()
        # Finally, display the set of lines
        mlab.pipeline.surface(lines, colormap='Accent', line_width=1, opacity=.4)
        scalar_lut_manager = self.lines.children[0].scalar_lut_manager
        scalar_lut_manager.show_scalar_bar = True
        scalar_lut_manager.data_name = 'gN'
        mlab.points3d(xGQP_arr[:,0],
                xGQP_arr[:,1],
                xGQP_arr[:,2], mode = 'point', color = (1.,1.,1.) )
        """
        mlab.points3d(xFG_arr[:,0],
                xFG_arr[:,1],
                xFG_arr[:,2], mode = 'point', color = (1.,0.,0.) )
        set_trace()
        """

    #------------------------------------------------------------------------------
    #
    #------------------------------------------------------------------------------
    def plot_all_cells_centers_and_gap(self,
            id_slv, KSICii, KSISol, gN_arr, IDM ):


        Tab = self.ContactTable
        nvalue = KSICii.shape[0]
        xGQP_arr = zeros((nvalue, 3 ), dtype= float)
        xFG_arr = zeros(xGQP_arr.shape, dtype= float)
        # get color from the value of the gap
        s =  gN_arr

        for ii in range(nvalue):
            current_curves = array([id_slv, IDM[ii]])
            xi_slv = array(KSICii[ii])
            id_els = self.el_per_curves[current_curves,].ravel()
            xGQP_arr[ii] = self.get_surf_point_smoothed_geo(id_els[(0,1),], KSICii[ii])
            xFG_arr[ii] = self.get_surf_point_smoothed_geo(id_els[(2,3),], KSISol[ii])

        self.plot(opacity = 0.2, diff_color_each_element = False)

        # Create the points to connect
        x = np.vstack( (xGQP_arr[:,0],  xFG_arr[:,0])).T.ravel()
        y = np.vstack( (xGQP_arr[:,1],  xFG_arr[:,1])).T.ravel()
        z = np.vstack( (xGQP_arr[:,2],  xFG_arr[:,2])).T.ravel()
        # one color for each point. interpolation between them
        s = np.vstack( (s,  s)).T.flatten()

        if not hasattr(self, 'src_center_cells'):
            self.src_center_cells= mlab.pipeline.scalar_scatter(x, y, z, s)
            connections = np.arange(2 * xGQP_arr.shape[0]).reshape(-1,2)
            # Connect them
            self.src_center_cells.mlab_source.dataset.lines = connections
            # The stripper filter cleans up connected lines
            self.lines = mlab.pipeline.stripper(self.src_center_cells)
        else:
            self.src_center_cells.mlab_source.set(x=x, y=y, z=z, s=s)

        lines = self.lines
        self.src_center_cells.update()
        # Finally, display the set of lines
        mlab.pipeline.surface(lines, colormap='Accent', line_width=1, opacity=.4)
        scalar_lut_manager = self.lines.children[0].scalar_lut_manager
        scalar_lut_manager.show_scalar_bar = True
        scalar_lut_manager.data_name = 'gN at center'
        mlab.points3d(xGQP_arr[:,0],
                xGQP_arr[:,1],
                xGQP_arr[:,2], mode = 'point', color = (1.,1.,1.) )
        """
        mlab.points3d(xFG_arr[:,0],
                xFG_arr[:,1],
                xFG_arr[:,2], mode = 'point', color = (1.,0.,0.) )
        set_trace()
        """
    #------------------------------------------------------------------------------
    #
    #------------------------------------------------------------------------------
    def plot_set_elements(self,\
            el_to_plot,
            diff_color_each_element = True,\
            color=(1.,1.,0.),\
            opacity = 0.7, \
            nS = 10,\
            ntheta = 15):

        assert self.is_set_package_plot, 'package plot must be set as well as the axis'

        f = mlab.gcf()
        f.scene.disable_render = True
        for ii, eli in enumerate(self.el[el_to_plot,]):
            if diff_color_each_element:
                eli.color_surface_mayavi = tuple(np.random.rand(3))
            else:
                eli.color_surface_mayavi = color
            pos = eli.get_mesh_surface_points(
                            self.X[eli.nID],  self.u[eli.nID],
                            self.v[eli.nID],
                            nS=nS, ntheta=ntheta)

            x,y,z = (pos[:,:,i] for i in range(3))
            eli.mesh =mlab.mesh(x, y, z, color=eli.color_surface_mayavi, opacity=opacity)

    #------------------------------------------------------------------------------
    #
    #------------------------------------------------------------------------------
    def plot_set_nodes(self, set_nodes, color, opacity, scale_factor =
            0.1):
        if isinstance(set_nodes, list) or isinstance(\
                set_nodes, tuple):
            set_nodes = asarray(set_nodes)
        assert set_nodes.ndim ==1 and set_nodes.dtype == int
        x_I = (self.X + self.u)[set_nodes]
        return mlab.points3d(x_I[:,0], x_I[:,1], x_I[:,2],
                scale_factor=scale_factor, color = color, opacity= opacity)

    #------------------------------------------------------------------------------
    #
    #------------------------------------------------------------------------------
    def plot_FEM_nodes(self, color_nodes=(1.,0.,0.), opacity = 1.):
        for eli in (self.el):
            eli.plot_nodes(color, opacity)



    #------------------------------------------------------------------------------
    #
    #------------------------------------------------------------------------------
    def plot_surf_point_smoothed_geo(self, id_els, h , color = (1., 0., 0.) ):
        x = self.get_surf_point_smoothed_geo(id_els, h ,
                                    color = color )
        mlab.points3d(x[0], x[1],x[2], color = color,
                scale_factor=0.1,\
                mode = 'sphere')




    #------------------------------------------------------------------------------
    #
    #------------------------------------------------------------------------------
    def plot_one_smoothed_surface(self, id_els, nS=10, ntheta=10,
            color=(1.,0.,0.), opacity = 0.5):

        pos = self.sample_surf_point_on_smooth_geo(id_els, nS,
                ntheta,\
                xi_lim = array([0,1]),\
                theta_lim = array([0, 2 * np.pi]))
        return mlab.mesh(pos[:, :, 0], pos[:, :, 1], pos[:, :,\
                2], color=color, opacity=opacity)
    #------------------------------------------------------------------------------
    #
    #------------------------------------------------------------------------------
    def plot_pair_integration_interval(self,\
            id_slv, id_mstr, xi_lim_slv, theta_lim_slv,\
            xi_lim_mstr = array([0, 1]),\
            theta_lim_mstr = array([0, 2 * np.pi]),\
            nS=10, ntheta=10,\
            color_sl=(1.,0.,0.), \
            color_mstr = (0.,1.,0.),\
            opacity = 0.5):

        Tab = self.ContactTable
        id_els = self.el_per_curves[(id_slv, id_mstr),]
        # slave integration interval
        ob1 = self.plot_integration_interval(id_els[0], xi_lim_slv,
                theta_lim_slv, color = color_sl, opacity = opacity )
        # master integration interval
        ob2 = self.plot_integration_interval(id_els[1],\
                xi_lim_mstr, theta_lim_mstr, color = color_mstr, opacity =
                opacity )
        return ob1, ob2



    #------------------------------------------------------------------------------
    #
    #------------------------------------------------------------------------------
    def plot_integration_interval(self, id_els, xi_lim, theta_lim,  nS=10, ntheta=20,
            color=(1.,0.,0.), opacity = 0.5):
        pos = self.sample_surf_point_on_smooth_geo(id_els, nS,
                ntheta,\
                xi_lim = xi_lim,\
                theta_lim = theta_lim)
        return mlab.mesh(pos[:, :, 0], pos[:, :, 1], pos[:, :,\
                2], color=color, opacity=opacity)



    #------------------------------------------------------------------------------
    #
    #------------------------------------------------------------------------------
    def plot_perimeter_cross_section(self, ids_els, xi, alpha, ntheta=10,
            color=(1.,0.,0.), opacity = 0.5):

        assert ids_els.shape == (2,)
        assert 0 <= xi <= 1
        theta = np.linspace(0, 2 * np.pi, ntheta)
        b1,b2 = self.el[(ids_els,)]
        pos = zeros(theta.shape + (3,), dtype=np.float)
        t1 = np.ascontiguousarray(np.array([b1.E1, b2.E1]).astype(np.double))
        t2 = np.ascontiguousarray(np.array([b1.E2, b2.E2]).astype(np.double))
        nd_C = asarray([b1.nID[0], b1.nID[1], b2.nID[1]]).ravel()
        a = b1.a
        b = b1.b

        for kk, thetakk in enumerate(theta):
            pos[kk,:] = SmoothingWtSmoothedVect.SurfPoint(\
                self.X[nd_C,].ravel(),\
                self.u[nd_C,].ravel(),\
                self.v[nd_C,].ravel(),\
                t1, t2, a , b , alpha,
                array([xi, thetakk]))

        return mlab.plot3d(pos[:, 0], pos[:, 1], pos[:,\
            2], color=color, opacity=opacity)

    #------------------------------------------------------------------------------
    #
    #------------------------------------------------------------------------------
    def plot_master_rigid_cross_section_elected_for_contact(self):
        assert self.HasConstraints
        Tab = self.ContactTable
        assert Tab.geometry_contact == 'rigid_sec_to_curve'
        for ii in range(Tab.id_els.shape[0]):
            # beam elements to construct curve where the master cross
            # section lies
            id_els = Tab.id_els[ii][0]
            self.plot_perimeter_cross_section(Tab.id_els[ii][0],\
                    Tab.h[ii,0], Tab.alpha, ntheta=10, color=(1.,0.,0.),\
                    opacity = 1)
    #------------------------------------------------------------------------------
    #
    #------------------------------------------------------------------------------
    def plot_smoothed_geometries(self, opacity = 0.5, same_color = True,\
            color = (1.,0.,0.), nxi =10, ntheta = 20 ):
        assert self.package_plot == 'mayavi'
        assert self.is_set_ContactTable
        Tab = self.ContactTable
        for ii in range(self.nCurves):
            if not same_color:
                color = tuple(np.random.rand(3))
            id_els = self.el_per_curves[ii]
            self.plot_one_smoothed_surface(id_els, nxi, ntheta, color , opacity)

    #------------------------------------------------------------------------------
    #
    #------------------------------------------------------------------------------
    def plot_Surface_set_of_element(self, index_elements, color = (0.,0.,0.), nS = 10, ntheta = 20):
        """
        handy method that allow to control which elements are plotted.
        ideal to plot yarns for example
        """
        for ii in index_elements:
            self.el[ii].plotSurface(self.X[self.el[ii].nID], self.u[self.el[ii].nID],
                                       self.v[self.el[ii].nID], nS , ntheta )


    #------------------------------------------------------------------------------
    #
    #------------------------------------------------------------------------------
    def plot_smoothed_centroid_lines(self, same_color = 0, color = (0.,0.,0.)):
        Tab = self.ContactTable
        assert Tab.smoothing
        if not Tab.smoothing_type == 'splines': raise NotImplementedError
        nctc = Tab.gN.shape[0]
        nxi = 10
        xi = np.linspace(0, 1, nxi)
        Phi = zeros((nctc, 2, nxi, 3), dtype = np.float)
        for ii in range(nctc):
            for jj, xjj in enumerate(xi):
                for idc in range(2):
                    Phi[ii, idc, jj] =\
                    self.get_phi_smoothed(ii,idc,xjj)

        if same_color:
            # same color for every pair
            # note here how to replicate a tuple
            color_per_pair = (color,) * nctc
        else:
            color_per_pair = [tuple(np.random.rand(3)) for i in
                    range(nctc)]

        for ii in range(nctc):
            for idc in range(2):
                mlab.plot3d(Phi[ii][idc][:,0],
                            Phi[ii][idc][:,1],
                            Phi[ii][idc][:,2],
                            color = color_per_pair[ii],
                            tube_radius = None)
                            #representation='wireframe')

    #------------------------------------------------------------------------------
    #
    #------------------------------------------------------------------------------
    def plot_final_defo(self, pathToFile):
        with open_file(pathToFile, mode = 'r') as fileh:
            n = -1
            #dofs in displacement
            self.u =\
            fileh.root.TS.Results[n]['u'].reshape(self.nn,3)
            #rotational dofs
            self.v =\
                    fileh.root.TS.Results[n]['rot_vect'].reshape(self.nn,3)
        self.plot()


    #------------------------------------------------------------------------------
    #
    #------------------------------------------------------------------------------
    def Load_Config_at_TS(self, path, n,force_conversion_to_double = False):
        assert n >= 0
        with open_file(path, mode = 'r') as fileh:
            """
            try:
                fileh.root.TS.Results[n]['t']
            except IndexError:
                # get the last commited t before starting over
                n = -1
            """
            fileh.root.TS.Results[n]['t']
            # LOAD RESULTS FROM TIME STEP n
            t = fileh.root.TS.Results[n]['t']
            #dofs in displacement
            self.u = fileh.root.TS.Results[n]['u'].reshape(self.nn,3)
            #rotational dofs
            self.v = fileh.root.TS.Results[n]['rot_vect'].reshape(self.nn,3)
            if force_conversion_to_double:
                self.u = self.u.astype(np.double)
                self.v = self.v.astype(np.double)

            if self.HasSphere:
                # directly set the position of the sphere
                self.Sphere.setCenterPosition(fileh.root.TS.Results[n]['PosCSp'])
        return t



    def Load_Last_TS(self, path, force_conversion_to_double = False):
        with open_file(path, mode = 'r') as fileh:
            nmax= fileh.root.TS.Results.shape[0]
        return self.Load_Config_at_TS(path, nmax -1 ,force_conversion_to_double  )

    #------------------------------------------------------------------------------
    #
    #------------------------------------------------------------------------------
    def FinalAnimation(self, path,
            delay=10,
            plotevery=1,
            Wttitle = 0,
            RecordAnimation = False,
            ncycles = 1,
            useYarnConnectivity = False,
            diff_color_each_element = True,\
            color=(0.6, 0.6, 0.6),\
            opacity = 1., \
            nS = 10,\
            ntheta = 20,\
            size = (1080, 720),
            lock_camera = True):

        with open_file(path, mode = 'r') as fileh:
            tmin = fileh.root.TS.Results[0]['t']
            tmax = fileh.root.TS.Results[-1]['t']
            nstp = fileh.root.TS.Results.shape[0]

        # force reconstruction of the plot (handy if a scene was
        # closed)
        for eli in self.el:
            if hasattr(eli, 'mesh'):
                delattr(eli, 'mesh')
                delattr(eli, 'ms')

        if RecordAnimation :
            import os
            if not os.path.isdir("./Animation"):
                os.mkdir('Animation')
            # Output path for you animation images
            out_path = './Animation'
            out_path = os.path.abspath(out_path)
            #mlab.start_recording(ui=True)
            prefix = 'ani'
            fps = 20
            ext = '.png'
            # allow to order the names of the files in the right order
            padding = len(str(nstp))
            sc = mlab.gcf()
            i = 0
            # force the rendering to be turned off
            sc.scene.disable_render = True
            t = tmin
            n = 0
            while t < tmax:
                # Read data timestep
                if n >= nstp-1:
                    break
                t = self.Load_Config_at_TS(path, n)
                if Wttitle:
                    mlab.title('t = ' + str("%.3f " % (t)))
                print(n)
                print(t)
                self.plot(useYarnConnectivity = useYarnConnectivity,
                            diff_color_each_element = diff_color_each_element,\
                            color=color,\
                            opacity = opacity, \
                            nS = nS,\
                            ntheta = ntheta,
                            manage_rendering = False,
                            lock_camera = lock_camera)
                n += plotevery
                print(t)
                # create zeros for padding index positions for organization
                zeros = '0'*(padding - len(str(i)))
                # concate filename with zero padded index number as suffix
                filename = os.path.join(out_path, '{}_{}{}{}'.format(prefix, zeros, i, ext))
                mlab.savefig(filename = filename , size=size)
                i+=1

        else:
            @mlab.animate(delay=delay, ui=True)
            def animation():
                sc = mlab.gcf()
                cycle_ct = 0
                while cycle_ct <= ncycles:
                    t = tmin
                    n = 0
                    while t < tmax:
                        # Read data timestep
                        if n >=nstp - 1:
                            print('END')
                            # force the generator to quit
                            break
                        t = self.Load_Config_at_TS(path, n)
                        if Wttitle:
                            mlab.title('t = ' + str("%.3f " % (t)))
                        print(n)
                        # the rendering management is taken care of inside
                        self.plot(useYarnConnectivity = useYarnConnectivity,
                                    diff_color_each_element = diff_color_each_element,\
                                    color=color,\
                                    opacity = opacity, \
                                    nS = nS,\
                                    ntheta = ntheta)
                        n += plotevery
                        yield

            animation()
            #mlab.show()

        # make movie from png
        if RecordAnimation:
            import subprocess
            ffmpeg_fname = os.path.join(out_path, '{}_%0{}d{}'.format(prefix, padding, ext))
            cmd = 'ffmpeg -f image2 -r {} -i {} -vcodec mpeg4 -y Animation/{}.mp4'.format(fps,
                                                                                ffmpeg_fname,
                                                                                prefix)
            print(cmd)
            subprocess.check_output(['bash','-c', cmd])

            # Remove temp image files with extension
            #[os.remove(f) for f in os.listdir(out_path) if f.endswith(ext)]
            for f in os.listdir(out_path):
                if f.endswith(ext):
                    os.remove(out_path + '/'  + f)


    #------------------------------------------------------------------------------
    #
    #------------------------------------------------------------------------------

    def MakeVideo(self, path,
            duration, title_movie = 'AGIF',
            useYarnConnectivity = False,
            diff_color_each_element = True,\
            color=(0.6, 0.6, 0.6),\
            opacity = 1., \
            nS = 10,\
            ntheta = 20,\
            lock_camera = True):

        # zulko.github.io/blog/2014/11/29/data-animations-with-python-and-moviepy/
        import  moviepy.editor as mpy

        # force reconstruction of the plot (handy if a scene was
        # closed)
        for eli in self.el:
            if hasattr(eli, 'mesh'):
                delattr(eli, 'mesh')
                delattr(eli, 'ms')
        with open_file(path, mode = 'r') as fileh:
            tmin = fileh.root.TS.Results[0]['t']
            tmax = fileh.root.TS.Results[-1]['t']
            nstp = fileh.root.TS.Results.shape[0]

        # duration of the animation in seconds (it will loop)
        # MAKE A FIGURE WITH MAYAVI
        sc = mlab.gcf()
        sc.scene.movie_maker.record = True
        sc.scene.disable_render = True

        def make_frame(t):
            # relation de proportionnalite de base t/duration = n/nmax
            n = int((t/tmax) * nstp)
            print('n is ')
            print(n)
            self.Load_Config_at_TS(path, n)
            # completely deactivate the rendering of the scene

            self.plot(useYarnConnectivity = useYarnConnectivity,
                            diff_color_each_element = diff_color_each_element,\
                            color=color,\
                            opacity = opacity, \
                            nS = nS,\
                            ntheta = ntheta,
                            manage_rendering = False,
                            lock_camera = lock_camera)
            # reactivate rendering
            #sc.scene.disable_render = False
            return mlab.screenshot(antialiased=True)

        animation = mpy.VideoClip(make_frame, duration=duration)
        # For the export, many options/formats/optimizations are supported
        #animation.write_videofile(title_movie + ".mp4", codec = "h264", fps=10) # export as video
        animation.write_videofile(title_movie + ".mp4", codec = 'mpeg4', fps=10, threads = 4) # export as video
        animation = mpy.VideoClip(make_frame, duration=duration)
        animation.write_gif("sinc.gif", fps=20)


    #------------------------------------------------------------------------------
    #
    #------------------------------------------------------------------------------
    def plot_all_GQP_and_CPP_sphere(self):
        for curve in range(self.nCurves):
            self.plot_all_GQP_and_CPP_sphere_1curve(curve)

    #------------------------------------------------------------------------------
    #
    #------------------------------------------------------------------------------
    def plot_all_GQP_and_CPP_sphere_1curve(self, id_curve):
        self.Sphere.plot(opacity = 0.05)
        self.plot_one_smoothed_surface(self.el_per_curves[id_curve])

        # sample points on the surface of the sphere to get a good first guess
        # for easch GQP
        beta1 = np.linspace(0, 2 * np.pi, 7 , endpoint = False)
        beta2 = np.linspace(0, np.pi, 7 , endpoint = False)
        SPt = self.Sphere.SamplePointsOnSurface(beta1, beta2)

        # Must be changed / generalized
        nGQPxi = self.nGQPxi
        nGQPtheta = self.nGQPtheta
        xi_interval = array([0, 1.])
        theta_interval = array([0, 2 * np.pi])
        nuxiGQP, WnuxiGQP =  np.polynomial.legendre.leggauss(nGQPxi)
        nuthetaGQP, WnuthetaGQP =  np.polynomial.legendre.leggauss(nGQPtheta)

        for ii in range(nGQPxi):
            for jj in range(nGQPtheta):
                BetaSol, foo, Kfoo, gN, ContriGQP, xGQP =\
                self.contri1GQP_SphereToBeam(SPt,\
                                beta1, beta2,\
                                id_curve,\
                                nuxiGQP[ii],\
                                WnuxiGQP[ii],\
                                nuthetaGQP[jj],\
                                WnuthetaGQP[jj],\
                                10.,
                                xi_interval,
                                theta_interval)
                if ContriGQP > 0:
                    color = (0.,0.,1.)
                else:
                    color = (0.,1.,0.)

                Pt = self.Sphere.getPointOnSurface(BetaSol[0],
                        BetaSol[1])
                mlab.plot3d( [xGQP[0],Pt[0]],\
                             [xGQP[1],Pt[1]],
                             [xGQP[2],Pt[2]],
                            color = color )


    #------------------------------------------------------------------------------
    #
    #------------------------------------------------------------------------------
    def plot_active_integration_interval(self, opacity = 0.7):
        Tab = self.ContactTable
        indices = np.argwhere(Tab.active_set)
        if indices.shape[0] > 0:
            for indice in indices:
                id_slv = indice[0]
                xi_lim_slv = Tab.xi_lim[indice[1]]
                theta_lim_slv = Tab.theta_lim[indice[2]]
                color_sl = tuple(np.random.rand(3))
#------------------------------------------------------------------------------
                self.plot_integration_interval(id_els = self.el_per_curves[id_slv],
                        xi_lim = xi_lim_slv,
                    theta_lim = theta_lim_slv,
                    color = color_sl, opacity = opacity )

                """
                id_mstr = Tab.master_ID_forCPP[tuple(indice)]
                color_mstr = tuple(np.random.rand(3))
                # not logical to do so as a slave interval might be projected on different master
                # interval
                self.plot_pair_integration_interval(\
                id_slv, id_mstr, xi_lim_slv, theta_lim_slv,\
                    color_sl=color_sl,\
                    color_mstr =color_mstr ,\
                    opacity = opacity)
                """


    #------------------------------------------------------------------------------
    #
    #------------------------------------------------------------------------------
    def get_all_GQP_and_CPP_coupleII(self):
        Tab = self.ContactTable

        nxiGQP = Tab.nxiGQP
        nthetaGQP= Tab.nthetaGQP
        nuxiGQP, WnuxiGQP =  np.polynomial.legendre.leggauss(nxiGQP)
        nuthetaGQP, WnuthetaGQP =\
        np.polynomial.legendre.leggauss(nthetaGQP)

        AS = np.argwhere(Tab.active_set)
        nGQP_active = AS.shape[0]
        xSLV = zeros((nGQP_active, 3,))
        xMSTR = zeros((nGQP_active, 3,))
        for kk, indices in enumerate(AS):
            id_slv, id_xi_slv, id_theta_slv, id_xi_GQP,\
            id_theta_GQP = indices

            id_mstr = Tab.master_ID_forCPP[tuple(indices)]
            assert id_mstr >= 0
            id_curves = [id_slv, id_mstr]

            # get id of the master curve where the GQP is projected
            id_els = self.el_per_curves[id_curves,].ravel()
            xi_lim_sl = Tab.xi_lim[id_xi_slv]
            theta_lim_sl = Tab.theta_lim[id_theta_slv]

            xisl, WxiGQP = TransformationQuadraturePointsAndWeigths1D(\
                     Tab.xi_lim[id_xi_slv] , nuxiGQP[id_xi_GQP], WnuxiGQP[id_xi_GQP])
            thetasl, WthetaGQP = TransformationQuadraturePointsAndWeigths1D(\
                     Tab.theta_lim[id_theta_slv], nuthetaGQP[id_theta_GQP], WnuthetaGQP[id_theta_GQP])

            xSLV[kk] = \
                    self.get_surf_point_smoothed_geo(\
                    id_els[(0,1),],\
                    array([xisl, thetasl]))
            xMSTR[kk] = \
                    self.get_surf_point_smoothed_geo(\
                    id_els[(2,3),],\
                    Tab.h[tuple(indices)])

            mlab.plot3d( [xMSTR[kk ,0], xSLV[kk,0]],
                    [xMSTR[kk ,1], xSLV[kk,1]],
                    [xMSTR[kk ,2], xSLV[kk,2]],
                    tube_radius = None, opacity = 0.9)



    #------------------------------------------------------------------------------
    #
    #------------------------------------------------------------------------------
    def plot_all_GQP(self,):
        Tab = self.ContactTable
        # TODO !!
        set_trace()

    ################################################################################
                                    # PLOTTING
    ################################################################################






    ################################################################################
                                    # VARIOUS
    ################################################################################
    def control_points(self, id_c):
        id_els = self.el_per_curves[id_c]
        b1,b2 = self.el[id_els,]
        nd_C = asarray([b1.nID[0], b1.nID[1], b2.nID[1]]).ravel()
        x1,x2,x3 = (self.X[nd_C,].ravel() + self.u[nd_C,].ravel()).reshape(3,3)
        alpha = self.ContactTable.alpha
        B0=1/2 * (x2+x1);
        B1=B0+(x2-B0) * alpha;
        B3=1/2 * (x2+x3);
        B2=alpha*x2+(B3)*(1-alpha);
        return array([B0,B1,B2,B3])


    #------------------------------------------------------------------------------
    #
    #------------------------------------------------------------------------------
    def get_surf_point_smoothed_geo(self, id_els, h):
        b1,b2 = self.el[id_els,]
        nd_C = asarray([b1.nID[0], b1.nID[1], b2.nID[1]]).ravel()
        return array( GeoSmooth.SurfPoint(\
                            self.X[nd_C,].ravel(),\
                            self.u[nd_C,].ravel(),\
                            self.v[nd_C,].ravel(),\
                            np.array([b1.E1, b2.E1]),\
                            np.array([b1.E2, b2.E2]),\
                            b1.a , b1.b ,\
                            self.ContactTable.alpha,\
                            h) )

    #------------------------------------------------------------------------------
    #
    #------------------------------------------------------------------------------
    def SurfPtAndVects_smoothed(self, id_c, h):
        id_els = self.el_per_curves[id_c]
        b1,b2 = self.el[id_els,]
        nd_C = asarray([b1.nID[0], b1.nID[1], b2.nID[1]]).ravel()
        #return s, sxi, stheta, n
        return GeoSmooth.SurfAndVects(\
                            self.X[nd_C,].ravel(),\
                            self.u[nd_C,].ravel(),\
                            self.v[nd_C,].ravel(),\
                            np.array([b1.E1, b2.E1]),\
                            np.array([b1.E2, b2.E2]),\
                            b1.a , b1.b ,\
                            self.ContactTable.alpha,\
                            h)


    #------------------------------------------------------------------------------
    #
    #------------------------------------------------------------------------------
    def sample_surf_point_on_smooth_geo(self,\
            id_els, nS, ntheta,\
            xi_lim, theta_lim):
        # version with for loops in full C
        assert id_els.shape == (2,)
        assert xi_lim.shape == (2,)
        assert theta_lim.shape == (2,)
        nd_C =  self.nPerEl[id_els,].flatten()[(0,1,3),]
        pos = zeros((nS, ntheta, 3), dtype = np.double)
        GeoSmooth.samplePointsOnSurface(\
            self.X[nd_C,].ravel(),\
            self.u[nd_C,].ravel(),\
            self.v[nd_C,].ravel(),\
            self.t1[id_els,],\
            self.t2[id_els,],\
            self.a_crossSec[id_els][0],\
            self.b_crossSec[id_els][0],\
            self.ContactTable.alpha,\
            np.linspace(xi_lim[0], xi_lim[1], nS, dtype=float),\
            np.linspace(theta_lim[0], theta_lim[1], ntheta, dtype =float),\
            pos)
        return pos

    #------------------------------------------------------------------------------
    #
    #------------------------------------------------------------------------------
    def sample_point_centroid(self,\
            id_els, nS,\
            xi_lim):
        # version with for loops in full C
        assert id_els.shape == (2,)
        assert xi_lim.shape == (2,)

        nd_C =  self.nPerEl[id_els,].flatten()[(0,1,3),]
        pos = zeros((nS, 3), dtype = np.double)
        GeoSmooth.samplePointsOnCentroid(\
            self.X[nd_C,].ravel(),\
            self.u[nd_C,].ravel(),\
            self.ContactTable.alpha,\
            np.linspace(xi_lim[0], xi_lim[1], nS, dtype=float),\
            pos)
        return pos

    #
    #------------------------------------------------------------------------------
    def get_Eint_From_u(self,uTC):
        # TODO
        # Eint = sum(el.getEint(u[el.dofs ]for el in self.el)

        print('strain energy for beam elements not available yet')
        return 0




    ################################################################################
                                    # VARIOUS
    ################################################################################




    ################################################################################
                            # CONTACT TREATMENT
    ################################################################################



    #------------------------------------------------------------------------------
    #
    #------------------------------------------------------------------------------
    def initialize_Contact_Enforcement(self, choice, **kwargs):
        #choice:
        #    - 0: linear penalty
        #    - 1: Lagrange multipliers
        self.enforcement = choice
        if choice == 0:
            print('####################################')
            print('PENALTY ENFORCEMENT HAS BEEN CHOSEN')
            print('####################################')
        # access the class variable of the contact elemeent
            assert kwargs.has_key('maxPiter'), 'MaxPiter kw missing'
            self.maxPiter = kwargs['maxPiter']
            assert self.maxPiter > 0 and isinstance(self.maxPiter, int)
            assert kwargs.has_key('maxPenAllow')
            assert isinstance(kwargs['maxPenAllow'], float) and kwargs['maxPenAllow'] < 0, 'max pen\
            allowed must be float and negative'
            self.maxPenAllow = kwargs['maxPenAllow']
            self.setndofs(additional_dofs = 0)
        elif choice == 1:
            print('####################################')
            print('LAGRANGE MULTIPLIERS ENFORCEMENT HAS BEEN CHOSEN')
            print('####################################')
            # create attribute
            self.nLagrangeMul = 0
            self.setndofs(additional_dofs = 0)
        else:
            raise NotImplementedError('choice must be 0 for linear penalty methode or 1 for LM method ')

    #------------------------------------------------------------------------------
    #
    #------------------------------------------------------------------------------
    def SurfToSurf_GEB(self, X, u, v_I, index_ctc_element):
        """
        NOTE: contrary to the rest of the methods, this contact scheme has only been developped for
        a parametrization of the finite rotations via rotation vectors
        """

        #------------------------------------------------------------------------------
        #
        #------------------------------------------------------------------------------
        def FGSurfToSurfSmoothed(nxi = 15 , ntheta = 15, plotFG=False):

            method = 0
            h = Tab.h[index_ctc_element]
            if not np.any(np.isnan(h)):
                # meanse the local scheme has already been performed for this pair of contact element
                # TODO:  a refaire !!
                f = AceGenGEB.get_local_residual_GEB3Nodes(Xi, ui, v_Ii,
                                                              t1, t2, ai, bi, h)
                if norm(f) < 1e-3:
                    # the previous converged value of the local scheme is a good first guess
                    return h

            h= zeros(4, dtype=float)
            xi_list = linspace(-1,1,nxi)
            theta_list = linspace(0, 2*np.pi, ntheta)

            #this part is common to the 2 methods. We look for closest distance between the centroid lines
            Phi1 = asarray([beam1.Phi(xii, X[beam1.nID], u[beam1.nID]) for xii in xi_list ] )
            Phi2 = asarray([beam2.Phi(xii, X[beam2.nID], u[beam2.nID]) for xii in xi_list] )
            dist = cdist(Phi1, Phi2, metric='euclidean')
            indices = np.where(dist == np.min(dist))
            h[0] = xi_list[indices[0][0]]
            h[2] = xi_list[indices[1][0]]

            if method == 0:
                # in this case, we sample angles theta, and then we naively look for the closest
                # distance
                # fast but unreliable way of finding a FG for the minimum distance proble. Indeed, when
                # there is no penetration, we will get close to the solution. However, in case of
                # penetration we will get close to the solution corresponding to the intersection of
                # the surface

                # now that xi has been found, one must find a correct fg for theta
                x1 = asarray([beam1.get_x_surf(X[beam1.nID], u[beam1.nID], v_I[beam1.nID], h[0],
                                               thetaii, False)
                                for thetaii in theta_list ] )
                x2 = asarray([beam2.get_x_surf(X[beam2.nID], u[beam2.nID], v_I[beam2.nID], h[2],
                                               thetaii,
                                              False)
                                for thetaii in theta_list ] )
                dist = cdist(x1, x2, metric='euclidean')
                mindist = np.min(dist)
                indices = np.where(dist == mindist)
                h[1] = theta_list[indices[0][0]]
                h[3] = theta_list[indices[1][0]]
            elif method == 1:
                """
                # ONLY LOOKING FOR THE MINIMUM OF gN IS NOT WORKING
                check notes to understand how I manage to do this first guess
                """
                Phi1 = beam1.Phi(h[0], X[beam1.nID], u[beam1.nID])
                Phi2 = beam2.Phi(h[2], X[beam2.nID], u[beam2.nID])
                # point on average centroid line
                A = 0.5 * (Phi1  +\
                            beam2.Phi(h[2], X[beam2.nID], u[beam2.nID]))
                # construct another point very close in terms of xi
                dxi = 0.01
                B = 0.5 * (Phi2 +\
                            beam2.Phi(h[2]+dxi, X[beam2.nID], u[beam2.nID]))
                L_int = Line3D(Point3D(A), Point3D(B))
                # plane containing the cross section of beam1
                P1 = Plane(Point3D(Phi1), normal_vector = beam1.LAMh(h[0],
                                                                   v_I[beam1.nID]).dot(beam1.E3))
                P2 = Plane(Point3D(Phi2), normal_vector = beam2.LAMh(h[2],
                                                                   v_I[beam2.nID]).dot(beam2.E3) )
                # find intersection between the plan containing the cross sections and the above
                # line
                inter1 = P1.intersection(L_int)
                inter2 = P2.intersection(L_int)
                set_trace()
                #TODO: find theta on the cross section attached to Phij (j = 1 or 2 ) giving the
                # point on the permieter the closest to the intersection

            if plotFG:
                beam1.plot_surface_point(X[beam1.nID],
                                   u[beam1.nID],
                                         v_I[beam1.nID], h[:2])
                beam2.plot_surface_point(X[beam2.nID],
                                   u[beam2.nID],
                                         v_I[beam2.nID], h[2:])
                self.plot()

            return h


        Tab= self.ContactTable
        beam1,beam2 = self.el[Tab.id_els[index_ctc_element]]
        for beami in (beam1, beam2):
            assert beami.__class__.__name__== 'SR3DBeams', 'the elements does not have the right type. GEB beam\
        from SR3D class is expected'

        for beami in (beam1, beam2):
            assert beami.nn == 3 , "Only made\ for beams with 3 nodes "

        nID_both = np.concatenate([beam1.nID, beam2.nID])
        assert beam1.nID.shape[0] == beam2.nID.shape[0], 'the beams in contact do not have the same number\
        of nodes'
        # number of nodes per beam element
        nN = beam1.nID.shape[0]
        Xi = np.ascontiguousarray((X[ix_(nID_both)].reshape(2, 3 * nN)).astype(np.double))
        # RESHAPE VERY INEFFICIENT
        ui = np.ascontiguousarray((u.reshape((self.nn,3))[ix_(nID_both)].reshape(2, 3 * nN)).astype(np.double))
        v_Ii = np.ascontiguousarray((v_I[ix_(nID_both)].reshape(2, 3 * nN)).astype(np.double))
        ai = np.ascontiguousarray([beam1.a, beam2.a])
        bi = np.ascontiguousarray([beam1.b, beam2.b])
        v_Ii = np.ascontiguousarray((v_I[ix_(nID_both)].reshape(2, 3 * nN)).astype(np.double))
        t1 = np.ascontiguousarray(np.array([beam1.E1, beam2.E1]).reshape(2,3).astype(np.double))
        t2 = np.ascontiguousarray(np.array([beam1.E2, beam2.E2]).reshape(2,3).astype(np.double))

        # first guess by sampling points on surface OR  start from solution previous iteration
        # TODO : minDist is useless here and must be removed
        h = FGSurfToSurf(plotFG = 0)

        for arr in [Xi,ui, v_Ii, t1,t2,ai,bi,h]:
            assert not np.any(np.isnan(arr))
            assert arr.flags['C_CONTIGUOUS']

        if self.enforcement == 0:
            # Linear Penalty Law
            # penalty stiffness
            if nN ==2:
                func = AceGenGEB.Surf2Surf_GEB2Nodes
            elif nN == 3:
                func = AceGenGEB.Surf2Surf_GEB3Nodes
            h, f, K, gN = func(Xi, ui, v_Ii, t1, t2, ai, bi, h, Tab.kN[index_ctc_element])
        else:
            raise NotImplementedError

        h = np.asarray(h, dtype = np.double)
        f = np.asarray(f, dtype = np.double)
        K = np.asarray(K, dtype = np.double)
        gN = np.float64(gN)
        assert not np.any(np.isnan(h))
        assert not np.any(np.isnan(f))
        assert not np.any(np.isnan(K))
        # check that the convective coordinates of the local problem are not off bound
        for hkk in h[(0,2,),]:
            assert -1. <= hkk <= 1

        # cheap. DO NOT DELETE
        if nN == 2:
            debug = 1
            if debug:
                output = self.test_output_local_scheme(beam1, beam2, h, gN)
                try:
                    assert np.allclose(K, K.T , 1e-13 * Tab.kN[index_ctc_element])
                    assert output == 1
                except AssertionError:
                    raise AssertionError('The contact scheme has failed')
                    set_trace()
                    self.plot_Contact_Pair()
                    self.plot_Contact_Points()

        return (h, f, K, gN, beam1.dofs.ravel(), beam2.dofs.ravel())


    #------------------------------------------------------------------------------
    #
    #------------------------------------------------------------------------------
    def FGSurfToSurf(self, Xi, ui, v_Ii, a , b, t1, t2, index_ctc_element):
        #------------------------------------------------------------------------------
        # first guess to determine the convective coordinated
        # close to the solution of the CPP problem
        #------------------------------------------------------------------------------
        Tab = self.ContactTable
        # sampled points
        nxi = 15; ntheta = 15; plotFG=False
        # Try last iterate as first guess
        h = Tab.h[index_ctc_element]
        if not np.any(np.isnan(h)):
            if not Tab.smooth_cross_section_vectors:
                raise DeprecationWarning
                f =\
                Smooth2Nodes.LocalRes(Xi,\
                        ui, v_Ii, t1, t2, a, b, h, Tab.alpha)
            else:
                f =\
                        SmoothingWtSmoothedVect.Surf2Surf(\
                        Xi,\
                        ui, v_Ii, t1,\
                        t2, a ,b , h, Tab.alpha,\
                        1 ,
                        999.)

            if norm(f) < 1e-3:
                # the previous converged value of the local scheme is a good first guess
                return h, norm(f)

        if np.any(np.isnan(h)):
            h= array([0,0,0,0,-0.1, -0.1], dtype = np.float)
        else:
            # we keep the value of the LM as first guess
            h = array([0,0,0,0,h[4], h[5]])
        # last iterate not present or not good enough.. We go on
        # to find a better first guess
        # ATTENTION: when dealing with the bezier curve the
        # space of the convective parameter goes from 0 to 1
        xi_list = linspace(0,1,nxi)
        theta_list = linspace(0, 2*np.pi, ntheta)
        # this part is independent of the interpolation of the
        # cross section vectors
        indices = self.get_approx_smallest_dist_between_smoothed_centroid(Xi, ui,
            xi_list, alpha)
        indices = np.where(dist == np.min(dist))
        h[0] = xi_list[indices[0][0]]
        h[2] = xi_list[indices[1][0]]

        """
        mlab.points3d(Phi1[:,0], Phi1[:,1], Phi1[:,2])
        mlab.points3d(Phi2[:,0], Phi2[:,1], Phi2[:,2])
        idmin1 = indices[0][0]
        idmin2 = indices[1][0]
        mlab.points3d([Phi1[idmin1,0], Phi2[idmin2,0]],
                              [Phi1[idmin1,1], Phi2[idmin2,1]],
                              [Phi1[idmin1,2], Phi2[idmin2,2]],
                              scale_factor=0.2, color = (1.,0.,0.))
        """
        # TODO: change this part
        # this case, we sample angles theta, and then we naively look for the closest
        # distance
        # fast but unreliable way of finding a FG for the minimum distance proble. Indeed, when
        # there is no penetration, we will get close to the solution. However, in case of
        # penetration we will get close to the solution corresponding to the intersection of
        # the surface
        x1 = zeros((ntheta, 3), dtype = np.float)
        x2 = zeros((ntheta, 3), dtype = np.float)
        if not Tab.smooth_cross_section_vectors:
            for jj, theta in enumerate(theta_list):
                x1[jj] =\
                Smooth2Nodes.PointSmoothedSurface2Nodes(Xi[0], ui[0]
                        , v_Ii[0], t1[0], t2[0], a[0], b[0], Tab.alpha,
                        array([h[0], theta]))
                x2[jj] =\
                Smooth2Nodes.PointSmoothedSurface2Nodes(Xi[1], ui[1]
                        , v_Ii[1], t1[1], t2[1], a[1], b[1], Tab.alpha,
                        array([h[2], theta]))

        else:
            for jj, theta in enumerate(theta_list):
                x1[jj] =\
                SmoothingWtSmoothedVect.SurfPoint(
                        Xi[0], ui[0], v_Ii[0], t1[:2,], t2[:2,], a[0], b[0], Tab.alpha,
                        array([h[0], theta]))
                x2[jj] =\
                SmoothingWtSmoothedVect.SurfPoint( Xi[1], ui[1]
                        , v_Ii[1], t1[2:,], t2[2:,], a[1], b[1], Tab.alpha,
                        array([h[2], theta]))
        dist = cdist(x1, x2, metric='euclidean')
        mindist = np.min(dist)
        indices = np.where(dist == mindist)
        h[1] = theta_list[indices[0][0]]
        h[3] = theta_list[indices[1][0]]
        assert not np.any(np.isnan(h))

        """
        # NICE PLOTTING
        mlab.points3d(x1[:,0], x1[:,1], x1[:,2], scale_factor=0.05)
        mlab.points3d(x2[:,0], x2[:,1], x2[:,2], scale_factor=0.05)

        idmin1 = indices[0][0]
        idmin2 = indices[1][0]
        mlab.points3d([x1[idmin1,0], x2[idmin2,0]],
                              [x1[idmin1,1], x2[idmin2,1]],
                              [x1[idmin1,2], x2[idmin2,2]],
                              scale_factor=0.05,
                              color = (1.,0.,0.))
        self.plot_one_smoothed_geo(index_ctc_element)
        """
        LocalResidualOnly = 1
        f =\
                SmoothingWtSmoothedVect.Surf2Surf(\
                Xi,\
                ui, v_Ii, t1,\
                t2, a ,b , h, Tab.alpha,\
                LocalResidualOnly,
                999.)
        return h, norm(f)

    #------------------------------------------------------------------------------
    #
    #------------------------------------------------------------------------------
    def get_approx_smallest_dist_between_smoothed_centroid(self,
            xi_list, Xi, ui, alpha):
        for arr in [Xi,ui]:assert arr.shape == (2,3)
        assert 0< alpha<1
        nxi = xi_list.shape[0]
        Phi1 = zeros((nxi, 3), dtype = np.float)
        Phi2 = zeros((nxi, 3), dtype = np.float)
        for ii, xii in enumerate(xi_list):
            Phi1[ii] =\
            SmoothingWtSmoothedVect.CentroidPoint(Xi[0], ui[0] , xii,
                    Tab.alpha)
            Phi2[ii] =\
            SmoothingWtSmoothedVect.CentroidPoint(Xi[1], ui[1] , xii,
                    Tab.alpha)
        return cdist(Phi1, Phi2, metric='euclidean').argmin()


    #------------------------------------------------------------------------------
    #
    #------------------------------------------------------------------------------
    def getFG_ctc_at_CPP(self, id_curves):
        X = self.X
        u = self.u
        v_I = self.v
        Tab= self.ContactTable
        h = Tab.h[id_curves[0], id_curves[1]]
        id_els = self.el_per_curves[id_curves,].ravel()
        assert id_els.shape == (4,)
        b1,b2, b3,b4 = self.el[id_els.ravel()]
        # nodes used for the first curve
        nd_C1 = asarray([b1.nID[0], b1.nID[1], b2.nID[1]]).ravel()
        nd_C2 = asarray([b3.nID[0], b3.nID[1], b4.nID[1]]).ravel()
        # number of nodes per beam element
        nN = b1.nID.shape[0]
        Xi = array([X[nd_C1,].ravel(), X[nd_C2,].ravel()])
        ui = array([u[nd_C1,].ravel(), u[nd_C2,].ravel()])
        v_Ii = array([v_I[nd_C1,].ravel(), v_I[nd_C2,].ravel()])
        a = np.ascontiguousarray([b1.a, b3.a])
        b = np.ascontiguousarray([b1.b, b3.b])
        # we will interpolate the cross section vector at the
        # junction between the 2 beams of the same yarns
        t1 = np.ascontiguousarray(np.array([b1.E1, b2.E1, b3.E1,
            b4.E1]).astype(np.double))
        t2 = np.ascontiguousarray(np.array([b1.E2, b2.E2, b3.E2,
            b4.E2]).astype(np.double))
        if not np.any(np.isnan(h)):
            # meanse the local scheme has already been performed for this pair of contact element
            # TODO:  a refaire !!
            f = ctcAtCPP.LocalRes(\
                    Xi, ui, v_Ii,\
                        t1, t2, a, b, h, Tab.alpha)

            if norm(f) < 0.1:
                # the previous converged value of the local scheme is a good first guess
                return h

        h = zeros(6,)
        xi_list = np.linspace(0,1,20)
        theta_list = np.linspace(0, 2 * np.pi,20, endpoint=False)

        nxi = xi_list.shape[0]
        ntheta = theta_list.shape[0]
        Phi1 = zeros((nxi, 3), dtype = np.float)
        Phi2 = zeros((nxi, 3), dtype = np.float)
        for ii, xii in enumerate(xi_list):
            Phi1[ii] =\
            GeoSmooth.CentroidPoint(Xi[0], ui[0] , xii,
                    Tab.alpha)
            Phi2[ii] =\
            GeoSmooth.CentroidPoint(Xi[1], ui[1] , xii,
                    Tab.alpha)

        indices = cdist(Phi1, Phi2, metric='euclidean').argmin()
        # but the indices corresponds to the ravelled array
        indices = np.unravel_index(indices, (nxi,nxi))

        h[0] = xi_list[indices[0]]
        h[2] = xi_list[indices[1]]

        # brute force by just checking distances. Can lead to serious
        # provlems in case of penetration as the distance measured
        # will always be >= 0
        x1= zeros((ntheta , 3), dtype = float)
        x2= zeros((ntheta , 3), dtype = float)
        for ii, thetaii in enumerate(theta_list):
            x1[ii] = GeoSmooth.SurfPoint(
                    Xi[0],\
                    ui[0],\
                    v_Ii[0],\
                    t1[:2],\
                    t2[:2],\
                    a[0] ,b[0],
                    self.ContactTable.alpha,\
                    array([h[0], thetaii]))
            x2[ii] = GeoSmooth.SurfPoint(
                    Xi[1],\
                    ui[1],\
                    v_Ii[1],\
                    t1[2:],\
                    t2[2:],\
                    a[1] ,b[1],
                    self.ContactTable.alpha,\
                    array([h[2],thetaii]))

        indices = cdist(x1, x2, metric='euclidean').argmin()
        # but the indices corresponds to the ravelled array
        indices = np.unravel_index(indices, (ntheta,ntheta))
        h[1] = theta_list[indices[0]]
        h[3] = theta_list[indices[1]]
        return h


    #------------------------------------------------------------------------------
    #
    #------------------------------------------------------------------------------

    def  weak_enforcement_IntInt2IntInt(self,\
                        fint, K):

        def plot_problematic_area():
            #scatter_3d(sampOnMaster.reshape(-1,3), color = (0.,0.,1.))
            xslv =\
            self.get_surf_point_smoothed_geo(\
                    id_els[(0,1),],
                    array([xiGQP, thetaGQP]))

            xmstr =\
            self.get_surf_point_smoothed_geo(\
                    id_els[(2,3),],
                    asarray(hSol))

            xFG =\
            self.get_surf_point_smoothed_geo(\
                    id_els[(2,3),],
                    hFG)


            assert xmstr.shape == (3,)
            assert xslv.shape == (3,)
            """
            line = mlab.plot3d([xslv[0], xmstr[0]],
                        [xslv[1], xmstr[1]],
                        [xslv[2], xmstr[2]],
                        color =
                        (1.,0.,0.),
                        tube_radius = None)
            """
            mlab.points3d(\
                    xFG[0], xFG[1], xFG[2], color = (1.,0.,0.),\
                    scale_factor = 0.01)

            mlab.points3d(\
                    xmstr[0], xmstr[1], xmstr[2], color = (0.,1.,0.),\
                    scale_factor = 0.01)

            mlab.points3d(\
                    xslv[0], xslv[1], xslv[2], color = (1.,1.,1.),\
                    scale_factor = 0.01)

            nxi_sampled = 15
            ntheta_sampled = 20
            xi_mstr = np.linspace(xi_lim_mstr[0],\
                    xi_lim_mstr[1],\
                    nxi_sampled)
            theta_mstr = np.linspace(theta_lim_mstr[0],\
                    theta_lim_mstr[1],\
                    ntheta_sampled)
            # sample on master
            current_sampling_cloud =\
                    self.sample_surf_point_on_smooth_geo(\
                    id_els[(2,3),], nxi_sampled,\
                    ntheta_sampled,\
                    xi_lim_mstr,
                    theta_lim_mstr)

            scatter_3d(current_sampling_cloud.reshape(-1,3), color = (0.,0.,1.),
                    scale_factor= 1/100)

        Tab = self.ContactTable
        nxiGQP = Tab.nxiGQP
        nthetaGQP= Tab.nthetaGQP
        nuxiGQP, WnuxiGQP =  np.polynomial.legendre.leggauss(nxiGQP)
        nuthetaGQP, WnuthetaGQP =\
        np.polynomial.legendre.leggauss(nthetaGQP)

        # for the time being the master has always the same limit
        xi_lim_mstr = array([0,1])
        theta_lim_mstr = array([0, 2 * np.pi])
        tol_for_err = 0.1

        ct = 0
        for ii, id_slv in enumerate(Tab.active_set):
            con = Tab.CON_Cells[ii]
            KSI = Tab.KSI_Cells[ii]
            for jj, cell in enumerate(Tab.active_cells[ii]):
                # get the limit of the cell
                KSI_Cell = KSI[(con[cell][0],
                    con[cell][1],
                    con[cell][2],
                    con[cell][3]),]

                xi_lim_slv = array([KSI_Cell[:,0].min(), KSI_Cell[:,0].max()])
                theta_lim_slv = array([KSI_Cell[:,1].min(), KSI_Cell[:,1].max()])

                for (kk,ll) in product(range(nxiGQP), range(nthetaGQP)):

                    # first guess for the id of the master curve on which the CPP will be found
                    id_mstr = Tab.ID_master[ii][jj][kk,ll]
                    if id_mstr < 0 :
                        if id_mstr == -999:
                            id_mstr = Tab.ID_master_Ctr[ii][cell]
                        else:
                            raise ValueError

                    current_curves = array([id_slv, id_mstr])
                    xiGQP, WxiGQP= TransformationQuadraturePointsAndWeigths1D(\
                             xi_lim_slv , nuxiGQP[kk], WnuxiGQP[kk])
                    thetaGQP, WthetaGQP = TransformationQuadraturePointsAndWeigths1D(\
                             theta_lim_slv, nuthetaGQP[ll], WnuthetaGQP[ll])

                    hFG = Tab.h[ii][jj, kk, ll][2:]
                    if np.any(np.isnan(hFG)):
                        assert np.all(np.isnan(hFG))
                        hFG = Tab.hCtr[ii][cell]

                    if Tab.enforcement == 0 :
                        value_in = Tab.kN[ii]
                    elif Tab.enforcement == 1:
                        value_in = Tab.LM[ii]
                        if Tab.LM_per_ctct_el == 1:
                            assert isinstance(value_in, np.float64)
                        else:
                            assert len(value_in) == len(value_in)

                    sol_correct = 0
                    ct_corrections = 0
                    max_corr = 5
                    # must always be up to date before computing contrubution

                    id_els = self.el_per_curves[current_curves,].ravel()
                    while not sol_correct:
                        (hSol, fkk, Kkk, gN, ExitCode, Contri_sl, fl, iterLoc )=\
                            self.contri1GQP_Int2Int(\
                            current_curves,\
                            hFG,
                            xiGQP,\
                            WxiGQP,\
                            thetaGQP,\
                            WthetaGQP,\
                            value_in)
                        if ExitCode != 1:
                            self.fullLocScheme(current_curves,
                                        array([xiGQP, thetaGQP]) ,
                                        array(hSol))


                        if xi_lim_mstr[0] - 1e-10 <= hSol[0]<= xi_lim_mstr[1] + 1e-10 \
                                and gN > Tab.critical_penetration:
                            sol_correct = 1
                        else:
                            try:
                                assert xi_lim_mstr[0]-tol_for_err <= hSol[0]<= xi_lim_mstr[1] + tol_for_err
                            except AssertionError:
                                set_trace()

                            if gN < Tab.critical_penetration:
                                ExitCode = 3
                                self.plot_all_QP_and_ProjStored()
                                set_trace()
                                break

                            ct_corrections += 1
                            if ct_corrections > max_corr:
                                print('corrections needed')
                                floc, Kloc = self.eval_loc_pder(current_curves,
                                        np.concatenate((array([xiGQP, thetaGQP]) , array(hSol))))
                                self.fullLocScheme(current_curves,
                                        array([xiGQP, thetaGQP]) ,
                                        array(hSol))

                                ExitCode = 3
                                self.plot(opacity = 0.2, diff_color_each_element = False)
                                xGQP = self.get_surf_point_smoothed_geo(id_els[(0,1),],
                                        array([xiGQP, thetaGQP]))
                                xFG= self.get_surf_point_smoothed_geo(id_els[(2,3),], np.asarray(hFG))
                                xSol = self.get_surf_point_smoothed_geo(id_els[(2,3),], array(hSol))
                                mlab.plot3d([xGQP[0], xFG[0]],
                                        [xGQP[1], xFG[1]],
                                         [xGQP[2], xFG[2]], color = (1.,0.,0.),
                                         tube_radius = None )
                                mlab.plot3d([xGQP[0], xSol[0]],
                                        [xGQP[1], xSol[1]],
                                         [xGQP[2], xSol[2]],
                                         color = (1.,1.,0.),
                                         tube_radius = None)
                                self.plot_all_QP_and_FG(plotSol=0)
                                #break
                            if hSol[0] < 0:
                                current_curves[1] -=1
                                hFG = array([1., hSol[1]])
                            elif hSol[0] > 1:
                                current_curves[1] +=1
                                hFG = array([0., hSol[1]])

                            id_els = self.el_per_curves[current_curves,].ravel()

                    try:
                        assert ExitCode == 1
                    except AssertionError:
                        if ExitCode == 3:
                            print('Could not find a projection')
                        xGQP = self.get_surf_point_smoothed_geo(id_els[(0,1),], array([ xiGQP,\
                                thetaGQP]))
                        xFG= self.get_surf_point_smoothed_geo(id_els[(2,3),], np.asarray(hFG))
                        xSol = self.get_surf_point_smoothed_geo(id_els[(2,3),], np.asarray(hSol))

                        self.plot(opacity = 0.2,
                                diff_color_each_element = False)
                        mlab.plot3d([xGQP[0], xFG[0]],
                                [xGQP[1], xFG[1]],
                                 [xGQP[2], xFG[2]], color = (1.,0.,0.),
                                 tube_radius = None )
                        mlab.plot3d([xGQP[0], xSol[0]],
                                [xGQP[1], xSol[1]],
                                 [xGQP[2], xSol[2]],
                                 color = (1.,1.,0.),
                                 tube_radius = None)

                        self.plot_integration_interval(id_els[(0,1),],\
                                xi_lim_slv,
                                theta_lim_slv,
                                opacity = 0.9, color = (0.,0.,0.) )

                        self.plot_integration_interval(id_els[(2,3),],\
                                xi_lim_mstr,
                                theta_lim_mstr,
                                opacity = 0.9,
                                color = (0., 1.,1.))

                        raise MaximumPenetrationError


                    Tab.ID_master[ii][jj][kk,ll] = current_curves[1]
                    Tab.gN[ii][jj][kk,ll] = gN
                    Tab.h[ii][jj][kk,ll][:2] = [xiGQP, thetaGQP]
                    Tab.h[ii][jj][kk,ll][2:] = hSol


                    # nodes used for the first curve
                    nd_C1 =  self.nPerEl[id_els[(0,1),]].flatten()[(0,1,3),]
                    nd_C2 =  self.nPerEl[id_els[(2,3),]].flatten()[(0,1,3),]
                    if Tab.enforcement == 0:
                        dofs_ctc_el = np.concatenate((self.dperN[nd_C1,],self.dperN[nd_C2,])).ravel()
                    elif Tab.enforcement == 1:
                        # add the potential coming from the quadrature point
                        Tab.weak_enforcement[index_GQP[:3]]+=Contri_sl
                        dofs_LM = Tab.dofs_LM[index_GQP[:3]]
                        assert dofs_LM != -999
                        dofs_ctc_el = np.hstack((self.dperN[nd_C1,].ravel(),
                            self.dperN[nd_C2,].ravel(), [dofs_LM]))

                    assert not np.any(np.isnan(fkk))
                    assert not np.any(np.isnan(Kkk))
                    assert np.allclose(array(Kkk), array(Kkk).T)
                    fint[dofs_ctc_el] += array(fkk)
                    K[ix_(dofs_ctc_el, dofs_ctc_el)] += array(Kkk)

        """
        if len(Tab.active_set)>0:
            self.plot_all_QP_and_ProjStored()
            set_trace()
        """




    #------------------------------------------------------------------------------
    #
    #------------------------------------------------------------------------------
    def contri1GQP_Int2Int(\
                        self,\
                        id_curves,
                        hFG,
                        xi_sl,\
                        Wxi_sl,\
                        theta_sl,\
                        Wtheta_sl,\
                        value):

        id_els = self.el_per_curves[id_curves,].ravel()
        assert id_els.shape == (4,)
        nd_C1 =  self.nPerEl[id_els[(0,1),]].flatten()[(0,1,3),]
        nd_C2 =  self.nPerEl[id_els[(2,3),]].flatten()[(0,1,3),]
        # we will interpolate the cross section vector at the
        # junction between the 2 beams of the same yarns
        # we use the FG stored because there is necesarrily a contact
        assert hFG.shape == (2,)

        if self.ContactTable.enforcement ==0:
            SMOOTH_LAW = 0
            return ctcVolumeBtoB.Contri_1GQP_weakPen(\
                self.X[(nd_C1,nd_C2),].reshape(2,9),\
                self.u[(nd_C1,nd_C2),].reshape(2,9),\
                self.v[(nd_C1,nd_C2),].reshape(2,9),\
                self.t1[id_els,],\
                self.t2[id_els,],\
                self.a_crossSec[id_els[(0,2),]],\
                self.b_crossSec[id_els[(0,2),]],\
                xi_sl,theta_sl, Wxi_sl, Wtheta_sl,\
                hFG,\
                self.ContactTable.alpha,\
                value,
                SMOOTH_LAW)
        elif self.ContactTable.enforcement == 1:
            # (hSl, f, K, gN, ExitCode, Contri_sl) =\
            if self.contact_law == 0:
                fun = ctcVolumeBtoB.Contri_1GQP_LM_CST
            else:
                fun = ctcVolumeBtoB.Contri_1GQP_LM_CST_EXP_LAW
            return fun(
                self.X[(nd_C1,nd_C2),].reshape(2,9),\
                self.u[(nd_C1,nd_C2),].reshape(2,9),\
                self.v[(nd_C1,nd_C2),].reshape(2,9),\
                self.t1[id_els,],\
                self.t2[id_els,],\
                self.a_crossSec[id_els[(0,2),]],\
                self.b_crossSec[id_els[(0,2),]],\
                xi_sl,theta_sl, Wxi_sl, Wtheta_sl,\
                hFG,\
                self.ContactTable.alpha,\
                value)

    #------------------------------------------------------------------------------
    #
    #------------------------------------------------------------------------------
    def eval_loc_pder( self,\
                      id_curves,
                      h):

        id_els = self.el_per_curves[id_curves,].ravel()
        assert id_els.shape == (4,)
        nd_C1 =  self.nPerEl[id_els[(0,1),]].flatten()[(0,1,3),]
        nd_C2 =  self.nPerEl[id_els[(2,3),]].flatten()[(0,1,3),]


        return GeoSmooth.getFandKLoc(
                    self.X[(nd_C1,nd_C2),].reshape(2,9),\
                    self.u[(nd_C1,nd_C2),].reshape(2,9),\
                    self.v[(nd_C1,nd_C2),].reshape(2,9),\
                    self.t1[id_els,],\
                    self.t2[id_els,],\
                    self.a_crossSec[id_els[(0,2),]],\
                    self.b_crossSec[id_els[(0,2),]],\
                    h[2:] ,
                    h[0], h[1],
                    self.ContactTable.alpha,\
                    )

    #------------------------------------------------------------------------------
    #
    #------------------------------------------------------------------------------
    def fullLocScheme( self,\
                      id_curves,
                      hQP, hFG):
        tol = 1e-10
        res = 1 + tol
        hM = array(hFG)
        iiter = 0
        set_trace()
        while res > tol:
            h = np.concatenate((hQP, hM))
            f,K = self.eval_loc_pder(id_curves, h)
            hM += -np.linalg.inv(K).dot(f)
            res = np.array(f).dot(np.array(f))
            iiter +=1
            if iiter >= 20:
                raise ValueError
        return hM



    #------------------------------------------------------------------------------
    #
    #------------------------------------------------------------------------------
    def pos1GQP(self, id_curve, nuxiGQP, nuthetaGQP,\
             xi_interval, theta_interval):
        id_els = self.el_per_curves[id_curve]
        xiGQP, WxiGQP = TransformationQuadraturePointsAndWeigths1D(\
                np.array(xi_interval), nuxiGQP, 0)
        thetaGQP, WthetaGQP = TransformationQuadraturePointsAndWeigths1D(\
                np.array(theta_interval), nuthetaGQP, 0)
        return self.get_surf_point_smoothed_geo(id_els, array([ xiGQP,\
                thetaGQP]))

    #------------------------------------------------------------------------------
    #
    #------------------------------------------------------------------------------
    def plot_set_GQP(self, indices):
        Tab = self.ContactTable
        nGQPxi = Tab.nxiGQP
        nGQPtheta = Tab.nthetaGQP
        nuxiGQP, WnuxiGQP =  np.polynomial.legendre.leggauss(nGQPxi)
        nuthetaGQP, WnuthetaGQP =  np.polynomial.legendre.leggauss(nGQPtheta)

        pos_GQP = zeros((indices.shape[0] , 3 ), dtype = float)
        for uu, index in enumerate(indices) :
            id_curve = zeros(2, dtype = int)
            id_curve[0], id_curve[1], ii, jj, kk, ll, mm , nn=\
                    index
            xi_lim = Tab.xi_lim[ii]
            theta_lim = Tab.theta_lim[jj]
            nuxi = nuxiGQP[mm]
            nutheta = nuthetaGQP[mm]
            pos_GQP[uu] = self.pos1GQP(id_curve[0], nuxi, nutheta,\
                    xi_lim, theta_lim)
        return mlab.points3d(pos_GQP[:,0], pos_GQP[:,1], pos_GQP[:,2])




    #------------------------------------------------------------------------------
    #
    #------------------------------------------------------------------------------
    def plot_all_GQP(self):
        raise DeprecationWarning
        nGQPxi = self.nGQPxi
        nGQPtheta = self.nGQPtheta
        nuxiGQP, WnuxiGQP =  np.polynomial.legendre.leggauss(nGQPxi)
        nuthetaGQP, WnuthetaGQP =  np.polynomial.legendre.leggauss(nGQPtheta)
        xi_interval = array([0, 1.])
        theta_interval = array([0, 2 * np.pi])


        pos_GQP = zeros(( self.ContactTable.close_curves.shape[0] *\
                nGQPxi * nGQPtheta , 3 ), dtype = float)

        ct = 0
        for pair_curve in np.argwhere(self.ContactTable.close_curves):
            for jj in range(nGQPxi):
                for kk in range(nGQPtheta):
                    pos_GQP[ct] =\
                            self.pos1GQP(pair_curve[0],\
                                    nuxiGQP[jj],\
                                    nuthetaGQP[kk],\
                                    xi_interval,\
                                    theta_interval)
                    ct += 1
        mlab.points3d(pos_GQP[:,0],\
                       pos_GQP[:,1],\
                        pos_GQP[:,2],\
                        scale_factor = 0.2, color = (1.,0.,0.))

    #------------------------------------------------------------------------------
    #
    #------------------------------------------------------------------------------
    def contact_at_CPP(self, id_curves, contact_detection_only = 0):
        #
        #contact enforcement between smoothed curve associated with
        #ellipses
        #
        X = self.X
        u = self.u
        v_I = self.v
        assert id_curves.shape == (2,)
        Tab = self.ContactTable

        if self.enforcement == 0 :
            if np.isnan(Tab.kN[id_curves[0], id_curves[1]]):
                Tab.kN[id_curves[0], id_curves[1]]=Tab.kNmin
            kN = Tab.kN[id_curves[0], id_curves[1]]
        if self.enforcement == 1 :
            if np.isnan(Tab.LM[id_curves[0], id_curves[1]]):
                raise ValueError('LM not set')

        #------------------------------------------------------------------------------
        def debug1():
            self.plot_pairs_of_smoothed_surface(\
                self.el_per_curves[id_curves,],\
                different_color= True)
            self.plot_surf_point_smoothed_geo(\
                        self.el_per_curves[id_curves[0]],\
                            array(h[:2]),\
                           color = (0., 1., 0.) )
            self.plot_surf_point_smoothed_geo(\
                        self.el_per_curves[id_curves[1]],\
                    array(h[2:4]),\
                           color = (1., 1., 0.) )
        #------------------------------------------------------------------------------
        def Construct_Contri_CtoC():
            h = Tab.h[id_curves[0], id_curves[1]]
            id_els = self.el_per_curves[id_curves,].ravel()
            assert id_els.shape == (4,)
            b1,b2, b3,b4 = self.el[id_els.ravel()]
            # nodes used for the first curve
            nd_C1 = asarray([b1.nID[0], b1.nID[1], b2.nID[1]]).ravel()
            nd_C2 = asarray([b3.nID[0], b3.nID[1], b4.nID[1]]).ravel()
            # number of nodes per beam element
            nN = b1.nID.shape[0]
            Xi = array([X[nd_C1,].ravel(), X[nd_C2,].ravel()])
            ui = array([u[nd_C1,].ravel(), u[nd_C2,].ravel()])
            v_Ii = array([v_I[nd_C1,].ravel(), v_I[nd_C2,].ravel()])
            a = np.ascontiguousarray([b1.a, b3.a])
            b = np.ascontiguousarray([b1.b, b3.b])
            # we will interpolate the cross section vector at the
            # junction between the 2 beams of the same yarns
            t1 = np.ascontiguousarray(np.array([b1.E1, b2.E1, b3.E1,
                b4.E1]).astype(np.double))
            t2 = np.ascontiguousarray(np.array([b1.E2, b2.E2, b3.E2,
                b4.E2]).astype(np.double))
            assert t1.shape == (4,3) and t2.shape == (4,3)
            assert b1.a == b2.a
            assert b3.a == b4.a
            assert b1.b == b2.b
            assert b3.b == b4.b

            h  =\
            self.getFG_ctc_at_CPP(id_curves)

            """
            print('FG is : ')
            print(h[:4])
            print('LM is : ')
            print(Tab.LM[id_curves[0], id_curves[1]] )
            self.FGforh = Tab.h[id_curves[0], id_curves[1]]
            """

            for arr in [Xi,ui, v_Ii, t1,t2,a,b,h]:
                assert not np.any(np.isnan(arr))
                assert arr.flags['C_CONTIGUOUS']

            if self.enforcement == 0:
                fun = ctcAtCPP.Surf2SurfPEN
                h, f, K, gN, ExitCode, fout =\
                        fun(\
                        Xi,\
                        ui, v_Ii, t1,\
                        t2, a ,b , h, Tab.alpha,\
                        0 ,
                        kN)
            elif self.enforcement == 1:
                h, f, K, gN, ExitCode, fout =\
                        ctcAtCPP.Surf2SurfLM(\
                        Xi,\
                        ui, v_Ii, t1,\
                        t2, a ,b , h, Tab.alpha,\
                        0,
                        Tab.LM[id_curves[0], id_curves[1]])


            if ExitCode == 0 : raise ValueError('local scheme did not\
                    converge')

            # we check that at the contact point, the distance between the
            # centroid is not too small. Otherwise, it means a complete
            # penetration occureed
            Phi1 =\
            GeoSmooth.CentroidPoint(Xi[0], ui[0] , h[0],
                    Tab.alpha)
            Phi2 =\
            GeoSmooth.CentroidPoint(Xi[1], ui[1] , h[2],
                    Tab.alpha)
            assert norm(array(Phi1) - array(Phi2)) > 0.1 * b[0]

            return np.asarray(h, dtype = np.double), np.asarray(f,
                    dtype = np.double),\
                    np.asarray(K, dtype = np.double), gN,\
                    ExitCode
        #------------------------------------------------------------------------------

        assert self.ContactTable.smoothing_type , NotImplementedError

        FG =0;
        Tab.h[id_curves[0], id_curves[1]], f, K,\
        Tab.gN[id_curves[0], id_curves[1]],\
        ExitCode =\
            Construct_Contri_CtoC()
        FG = 1;

        # tolerance for xi for out of bounds solution
        eps = 1e-10
        #if the solution of the local scheme is out of bounds
        changes_curve = 0
        if ExitCode == 2:
            if self.enforcement == 0:
                # IMPORTANT: in the case of the penalty, if the scheme
                # has converged out of the bound, we DO NOT try to
                # switch curve because if the next/previous curve is
                # indeed in contact, it will be in the set of
                # the bounding boxes intersecting

                id_els = self.el_per_curves[id_curves,].ravel()
                b1,b2, b3,b4 = self.el[id_els]
                # nodes used for the first curve
                nd_C1 = asarray([b1.nID[0], b1.nID[1], b2.nID[1]]).ravel()
                nd_C2 = asarray([b3.nID[0], b3.nID[1], b4.nID[1]]).ravel()
                dofs_ctc_el = np.hstack((self.dperN[nd_C1,].ravel(),
                    self.dperN[nd_C2,].ravel()))
                return zeros(36, dtype = float),\
                        zeros((36,36), dtype = float),\
                         dofs_ctc_el, False
            elif self.enforcement == 1:
                set_trace()
                while ExitCode == 2:
                    # IMPORTANT : due to the change of curve, the local scheme
                    # might have converged towards an expected solution. So we
                    # force the FG routine to run again !
                    assert ExitCode == 2
                    changes_curve += 1
                    id_prev_curves = np.copy(id_curves)
                    htemp = np.copy(Tab.h[id_curves[0] , id_curves[1]])
                    # adjust first contacting curve
                    if htemp[0] < 0-eps:
                        # go to previous curve
                        htemp[0] = 1
                        id_curves[0] -= 1
                    elif  htemp[0]  > 1 + eps:
                        # go to next curve
                        htemp[0] = 0
                        id_curves[0] += 1

                    # adjust second contacting curve
                    # adjust first contacting curve
                    if htemp[2] < 0-eps:
                        # go to previous curve
                        htemp[2] = 1
                        id_curves[1] -= 1
                    elif  htemp[2]  > 1 + eps:
                        # go to next curve
                        htemp[2] = 0
                        id_curves[1] += 1

                    # we keep the same value of the LM !
                    Tab.LM[id_curves[0], id_curves[1]] = Tab.LM[id_prev_curves[0] , id_prev_curves[1]]
                    # TODO : partie dangereuse avec le transfert des
                    # LM. peut etre faudra t-il changer ca ??
                    # we "activate" the LM for the contact between the
                    # previous curve
                    Tab.activeLM[id_prev_curves[0] ,
                            id_prev_curves[1]] = True
                    #update h
                    Tab.h[id_curves[0], id_curves[1]] = htemp
                    # erase values corresponding to the previously stored
                    # curve
                    Tab.h[id_prev_curves[0] , id_prev_curves[1]] = np.nan
                    Tab.LM[id_prev_curves[0] , id_prev_curves[1]] = np.nan
                    # we deactivate the LM for the contact between the
                    # previous curve
                    Tab.activeLM[id_prev_curves[0] ,
                            id_prev_curves[1]] = False
                    Tab.gN[id_prev_curves[0] , id_prev_curves[1]] = np.nan

                    Tab.h[id_curves[0], id_curves[1]], f, K,\
                    Tab.gN[id_curves[0], id_curves[1]],\
                    ExitCode =\
                        Construct_Contri_CtoC()
                    # debug1( Tab.h[id_curves[0], id_curves[1]])
                    #set_trace()

        assert not np.any(np.isnan(  Tab.h[id_curves[0], id_curves[1]]))
        if not contact_detection_only:
            assert not np.any(np.isnan(f))
            assert not np.any(np.isnan(K))

        h = Tab.h[id_curves[0], id_curves[1]]
        try:
            assert norm(h[0] - 1 / 2) < 1e-12
            assert norm(h[1] - np.pi / 2) < 1e-12
            assert norm(h[2] - 1 / 2) < 1e-12
            assert norm( h[3] - 1.5 * np.pi) < 1e-12
        except AssertionError:
            set_trace()
            Construct_Contri_CtoC()
        if not contact_detection_only:
            id_els = self.el_per_curves[id_curves,].ravel()
            b1,b2, b3,b4 = self.el[id_els]
            # nodes used for the first curve
            nd_C1 = asarray([b1.nID[0], b1.nID[1], b2.nID[1]]).ravel()
            nd_C2 = asarray([b3.nID[0], b3.nID[1], b4.nID[1]]).ravel()
            if self.enforcement == 0:
                dofs_ctc_el = np.hstack((self.dperN[nd_C1,].ravel(),
                    self.dperN[nd_C2,].ravel()))
                if Tab.gN[id_curves[0], id_curves[1]] < 0:
                    CONTRIBUTE = True
                else:
                    CONTRIBUTE = False
            elif self.enforcement == 1:
                dofs_ctc_el = np.hstack((self.dperN[nd_C1,].ravel(),
                    self.dperN[nd_C2,].ravel(),
                    Tab.dofs_LM[id_curves[0], id_curves[1]]))
                CONTRIBUTE = True
                assert dofs_ctc_el.shape == (37,)
                assert np.allclose(K, K.T)
            return (f, K, dofs_ctc_el, CONTRIBUTE)
        if contact_detection_only:
            return (h, gN)

    #------------------------------------------------------------------------------
    #
    #------------------------------------------------------------------------------
    def contri1GQP_SphereToBeam( self,\
                    SPt,\
                    beta1_sample,\
                    beta2_sample,\
                    id_curve,\
                    nuxiGQP,\
                    WnuxiGQP,\
                    nuthetaGQP,\
                    WnuthetaGQP,\
                    kN,\
                    xi_interval, \
                    theta_interval):

        Tab = self.ContactTable
        X = self.X
        u = self.u
        v = self.v
        assert isinstance(id_curve, int)
        # The GQP are the one of the parent interval and they are
        # transformed here
        assert -1. <= nuxiGQP <= 1
        assert -1. <= nuthetaGQP <= 1
        assert 0. <= WnuxiGQP <= 2
        assert 0. <= WnuthetaGQP <= 2

        xiGQP, WxiGQP = TransformationQuadraturePointsAndWeigths1D(\
                np.array(xi_interval), nuxiGQP, WnuxiGQP)
        thetaGQP, WthetaGQP = TransformationQuadraturePointsAndWeigths1D(\
                np.array(theta_interval), nuthetaGQP, WnuthetaGQP)

        Ctr = self.Sphere.C
        radius = self.Sphere.r
        id_els = self.el_per_curves[id_curve,].ravel()
        assert id_els.shape == (2,)
        b1,b2 = self.el[id_els.ravel()]
        # nodes used for the first curve
        nd_C = asarray([b1.nID[0], b1.nID[1], b2.nID[1]]).ravel()
        # number of nodes per beam element
        nN = b1.nID.shape[0]
        Xi = array([X[nd_C,]]).ravel()
        ui = array([u[nd_C,]]).ravel()
        vi = array([v[nd_C,]]).ravel()
        a = b1.a;
        b = b1.b;
        ta = np.ascontiguousarray(np.array([b1.E1, b2.E1]))
        tb = np.ascontiguousarray(np.array([b1.E2, b2.E2]))
        assert ta.shape == (2,3) and tb.shape == (2,3)

        #get closest point on the surface of the surface of the sphere as
        #first guess
        xGQP = self.get_surf_point_smoothed_geo(id_els, array([ xiGQP,\
                thetaGQP]))

        # BROADCASTING
        idx = norm((SPt - xGQP), axis = 2).argmin()
        idx = np.unravel_index(idx, (SPt.shape[:2]))
        # reconstruct the meshgrid
        grid = np.meshgrid(beta1_sample, beta2_sample)
        betaFG= array([grid[0][idx], grid[1][idx]])
        beta1FG = betaFG[0]
        beta2FG = betaFG[1]
        PtFG = self.Sphere.getPointOnSurface(beta1FG, beta2FG)
        BetaFG = array([ beta1FG,  beta2FG])

        (BetaSol, f, K, gN, ExitCode, ContriGQP) =\
                ctcVolumeSphere.curveToSphereVolume1GQP(Xi,ui,vi,\
                ta, tb, a,b,\
                Ctr, radius, xiGQP,\
                thetaGQP, WxiGQP, WthetaGQP,\
                BetaFG, Tab.alpha, kN)

        return np.asarray(BetaSol),\
               np.asarray(f),\
               np.asarray(K),\
               gN,\
               ContriGQP,\
               xGQP


    #------------------------------------------------------------------------------
    #
    #------------------------------------------------------------------------------
    def contact_rigid_sphere(self, id_curve):
        assert isinstance(id_curve, int)
        kN = 10.
        self.nGQPxi = 1
        self.nGQPtheta = 1
        nGQPxi = self.nGQPxi
        nGQPtheta = self.nGQPtheta
        nuxiGQP, WnuxiGQP =  np.polynomial.legendre.leggauss(nGQPxi)
        nuthetaGQP, WnuthetaGQP =  np.polynomial.legendre.leggauss(nGQPtheta)
        xi_interval = array([0, 1.])
        theta_interval = array([0, np.pi])

        # sample points on the surface of the sphere to get a good first guess
        # for easch GQP
        beta1 = np.linspace(0, 2 * np.pi, 8 , endpoint = False)
        beta2 = np.linspace(0, np.pi, 8 , endpoint = False)
        SPt = self.Sphere.SamplePointsOnSurface(beta1, beta2)


        CONTRIBUTE = False
        f = zeros(18, dtype = np.float32)
        K = zeros((18,18), dtype = np.float32)
        for ii in range(nGQPxi):
            for jj in range(nGQPtheta):
                BetaSol, fjj, Kjj, gN, ContriGQP, xGQP =\
                        self.contri1GQP_SphereToBeam(SPt,\
                                        beta1, beta2,\
                                        id_curve,\
                                        nuxiGQP[ii],\
                                        WnuxiGQP[ii],\
                                        nuthetaGQP[jj],\
                                        WnuthetaGQP[jj],\
                                        kN,
                                        xi_interval,
                                        theta_interval)

                # the GQP contributes to the integral
                if ContriGQP > 0 :
                    CONTRIBUTE = True
                    f += np.asarray(fjj)
                    K += np.asarray(Kjj)


        id_els = self.el_per_curves[id_curve,].ravel()
        b1,b2 = self.el[id_els.ravel()]
        # nodes used for the first curve
        nd_C = asarray([b1.nID[0], b1.nID[1], b2.nID[1]]).ravel()
        # nodes used for the first curve
        nd_C = asarray([b1.nID[0], b1.nID[1], b2.nID[1]]).ravel()
        dofs_ctc_el = self.dperN[nd_C,].ravel()

        if CONTRIBUTE:
            print("some GQP active")

        return (f, K, dofs_ctc_el, CONTRIBUTE)


    #------------------------------------------------------------------------------
    #
    #------------------------------------------------------------------------------
    def penetration_checking(self, X, u, v_I, index_ctc_element):
        #contact enforcement between smoothed curve associated with
        #ellipses
        # TODO: avoid the computation of the arrays..
        (h, gN) =\
        self.curve_to_curve_LM(self.X, self.u, self.v,
                index_ctc_element, contact_detection_only = 1)
        self.ContactTable.update_contact_location(index_ctc_element, gN, h )

    #------------------------------------------------------------------------------
    #
    #------------------------------------------------------------------------------
    def test_output_local_scheme(self, beam1, beam2, h, gN):
        #test the consistency of the result of the local scheme run in AceGen
        X = self.X
        u = self.u
        v = self.v
        def get_surf_info_of_1_beam(el, h):
            nID = el.nID
            Xi = np.ascontiguousarray(X[ix_(nID)].astype(np.double))
            ui = np.ascontiguousarray(u[ix_(nID)].astype(np.double))
            v_Ii = np.ascontiguousarray(v[ix_(nID)].astype(np.double))
            return AceGenGEB.get_s_and_tangent_vectors(Xi, ui, v_Ii, el.E1, el.E2, el.a, el.b, h)

        # IMPORTANT DETAIL
        # a test on the comparison of the distance can lead to confusing result ! indeed in case
        # of penetration the Dist found is a local maximum of the distance function but still
        # the first guess can give a larger distance too !
        s1, s1t, n1  = get_surf_info_of_1_beam(beam1, h[:2])
        s2, s2t, n2  = get_surf_info_of_1_beam(beam2, h[2:])
        try:
            for Vector in [s1, s1t, s2, s2t, n1, n2]:
                assert not np.any(np.isnan(asarray([Vector])))
            assert norm(np.cross(n1, n2)) < 1e-10
            assert inner(s1t[0], n2) < 1e-10
            assert inner(s1t[1], n2) < 1e-10
            assert inner(s2t[0], n1) < 1e-10
            assert inner(s2t[1], n1) < 1e-10
            assert np.allclose(abs(gN), norm(array(s1) - array(s2)) )
        except AssertionError:
            print('local scheme has failed or gives inconsistent results')
            set_trace()
            return 0

        return 1

    #------------------------------------------------------------------------------
    #
    #------------------------------------------------------------------------------
    def set_Contact_Table(self, id0, id1, value):
        # the contact table must be set before starting the simulation
        try:
            assert self.HasConstraints
        except AssertionError as e:
            raise e('Inconsistency. You are trying to set the contact table but the problem is without constraints')
        #Ensure that the arguments are iterables
        try:
            iter(id0)
            iter(id1)
        except TypeError as e:
            # not iterable
            raise e('the arguments of the function must be iterables')
        else:
            # iterable
            id0 = asarray([id0]).flatten()
            id1 = asarray([id1]).flatten()

        assert id0.shape == id1.shape == value.shape
        if id0.shape[0] == 0:
            print('EMPTY CONTACT TABLE --> THE CONTACT BETWEEN YARNS IS NOT ENFORCED')
            self.is_set_ContactTable = True
        else:
            assert id0.dtype == np.int
            assert id1.dtype == np.int
            assert id0.shape == id1.shape
            # a same element can be in contact several time. Hence the False
            assert np.intersect1d(id0, id1, assume_unique = False).shape[0] == 0
            # value is the default value that is attributed to the penalty stiffness or the lagrange
            # multiplier
            assert (isinstance(value, float) or isinstance(value,int) )


        if self.ConstantContactTable:
            # an element cannot be in contact with itself.
            try:
                assert not hasattr(self, 'ContactTable')
            except AssertionError as e:
                raise e('the FEM already had a contact table !')
            # initialize empty contact table object
            self.ContactTable = ContactTable(self.enforcement)
            # fill it
            self.ContactTable.add_row(id0, id1, value = value * ones_like(id0))
        else:
            raise NotImplementedError

        if self.enforcement == 1:
            self.ContactTable.regenerate_dofs(self.ndofs_el - 1)
            self.set_nLagrangeMul()
            self.setndofs(self.nLagrangeMul)
        self.is_set_ContactTable = True



    #------------------------------------------------------------------------------
    #
    #------------------------------------------------------------------------------
    def set_nLagrangeMul(self):
        # useful if the number of dofs changes along the simulation for example change in active set
        # when Largrange Multiplier method is used
        self.nLagrangeMul= int(where(self.ContactTable.activeLM == True)[0].shape[0])
        # update the number of total dofs
        self.setndofs(self.nLagrangeMul)


    #------------------------------------------------------------------------------
    #
    #------------------------------------------------------------------------------
    def add_Sphere(self, Sphere):
        self.HasSphere = True
        self.Sphere = Sphere

    ################################################################################
                            # CONTACT TREATMENT
    ################################################################################

#------------------------------------------------------------------------------
#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
                            # MODEL CLASS
#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
#------------------------------------------------------------------------------


#------------------------------------------------------------------------------
#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
                            # Generation of initial geometry
#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
#------------------------------------------------------------------------------
def generate_regular_lattice(nYH, nYV, nnYH, nnYV, xinterval,
        yinterval, get_connectivity_curves):

    assert xinterval.ndim == 1 and xinterval.shape == (2,)
    assert yinterval.ndim == 1 and yinterval.shape == (2,)
    # spacing between NODES along nodes in horizontal direction
    spnH = (xinterval[1] - xinterval[0])/ (nnYH )
    # spacing between NODES along nodes in vertical direction
    spnV = (yinterval[1] - yinterval[0]) / (nnYV )
    c = 4
    xH = np.linspace(xinterval[0] - c * spnH, xinterval[1] + c * spnH,
            nnYH +
            2)
    yV = np.linspace(yinterval[0] - c*spnV, yinterval[1] +  c *spnV,
            nnYV + 2)

    # spacing between YARNS in horizontal direction
    spYH = (xinterval[1] - xinterval[0])/ (nYH )
    # spacing between YARNS in horizontal direction
    spYV = (yinterval[1] - yinterval[0])/ (nYV )
    yH =  np.linspace(yinterval[0] , yinterval[1] , nYH)
    xV =  np.linspace(xinterval[0] , xinterval[1] , nYV)

    # we add an additionnal node at each end to impose BCs easily
    # number of nodes for each type of yarn
    nnH = xH.shape[0]
    nnV = yV.shape[0]

    # TODO: there is a pb I had to add a yarn
    XYH = zeros((nYH , nnH, 3), dtype = float)
    XYH[:,:,0] = xH
    for i in range(nYH ): XYH[i,:,1] = yH[i]

    XYV = zeros((nYV , nnV, 3), dtype = float)
    XYV[:,:,1] = yV
    for i in range(nYV): XYV[i,:,0] = xV[i]


    nnYH_tot = np.product(XYH.shape[:2])
    nnYV_tot = np.product(XYH.shape[:2])
    # node on each yarn
    ndYH = np.arange(nnYH_tot).reshape(XYH.shape[:2])
    ndYV = np.arange(nnYH_tot, nnYH_tot + nnYV_tot).reshape(XYV.shape[:2])
    # connectivity (node per element for each yarn)
    conYH = zeros((ndYH.shape[0], ndYH.shape[1] - 1, 2), dtype = int)
    conYH[:,:,0] = ndYH[:,:-1]
    conYH[:,:,1] = ndYH[:,1:]
    conYV = zeros((ndYV.shape[0], ndYV.shape[1] - 1, 2), dtype = int)
    conYV[:,:,0] = ndYV[:,:-1]
    conYV[:,:,1] = ndYV[:,1:]
    # gather all the coordinates of the nodes
    X = np.concatenate((XYH.reshape(-1,3),XYV.reshape(-1,3)))
    # connectivity of all the elements
    con = np.concatenate((conYH.reshape(-1,2),conYV.reshape(-1,2)))
    # to which yarn each element is associated
    nel_per_YH = conYH.shape[1]
    nel_per_YV = conYV.shape[1]
    el_per_YH = np.arange(nel_per_YH * nYH).reshape(nYH, nel_per_YH)
    el_per_YV = el_per_YH[-1][-1] + 1 + np.arange(nel_per_YV * nYV).reshape(nYV, nel_per_YV)

    which_yarn_is_el = np.hstack((el_per_YH.ravel(),
        el_per_YV.ravel()))
    which_fam_of_yarn_is_el = np.hstack((zeros_like(el_per_YH.ravel()),
        ones_like(el_per_YV.ravel())))

    if get_connectivity_curves:
        ncurve = lambda nel : nel - 1
        # element per curves
        con_curves_YH = zeros((nYH, ncurve(nel_per_YH), 2  ), dtype=int)
        con_curves_YV = zeros((nYV, ncurve(nel_per_YV), 2  ), dtype=int)
        for i in range(nYH):
            con_curves_YH[i,:,0] = el_per_YH[i][:-1]
            con_curves_YH[i,:,1] = el_per_YH[i][1:]
        for i in range(nYV):
            con_curves_YV[i,:,0] = el_per_YV[i][:-1]
            con_curves_YV[i,:,1] = el_per_YV[i][1:]
        assert con_curves_YV[0][0][0] == con_curves_YH[-1][-1][-1] + 1

        # last output gives the number of elements
        return XYH, XYV, conYH, conYV, ndYH, ndYV, nnYH, nnYV, X, con,\
        con.shape[0], which_yarn_is_el, which_fam_of_yarn_is_el,\
        con_curves_YH, con_curves_YV
    else:
        # last output gives the number of elements
        return XYH, XYV, conYH, conYV, ndYH, ndYV, nnYH, nnYV, X, con,\
        con.shape[0], which_yarn_is_el, which_fam_of_yarn_is_el

