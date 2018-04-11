
from Common.BoundaryConditions import BC, Static_Loading
from Common.Utilities import *
from Common.UserDefErrors import FailedConvergence, Failure_Adaptive_TS
import numpy as np
from pdb import set_trace
from numpy import dot, ix_, zeros, array, ones, asarray, inner, copy,\
zeros_like, allclose, where
from numpy.linalg import norm
import numpy.linalg as LA
# from scipy.sparse import coo_matrix
# we will able to catch warnings as errors afterwards
from tables import Float32Col, Int32Col, open_file, IsDescription, StringCol, Int32Atom, Float32Atom
import os
import matplotlib.pylab as plt
from pyquaternion import Quaternion
import time
from mayavi import mlab


#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
                            # SOLVER CLASS
#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
class Solver():
    """
    Contains solvers for static and dynamic analysis
    """
    ################################################################################
                            # CONSTRUCTOR
    ################################################################################
    def __init__(
            self,
            Loading,
            FEM,
            maxiter,
            tol,
            Static=1,
            SolverType='NR',
            wtConstraints=None,
            max_failure_authorized = 5,
            max_active_set_changes_authorized = 5,
            **kwargs):

        assert FEM.__class__.__name__ == 'Model'
        assert isinstance(tol, float)
        assert isinstance(maxiter, int)
        assert isinstance(Static, bool) or Static == 0 or Static == 1

        try:
            assert Static
        except AssertionError:
            raise NotImplementedError('Dynamic analysis not implemented yet ')

        # true if the analysis is a static one
        self.Static = Static
        # store reference to Loading object
        self.Loading = Loading
        # store reference to finite element assembly object
        self.FEM = FEM
        # tolerance for convergence of the NL solver
        self.tol = tol
        self.set_tot_ndofs(FEM.tot_ndofs)
        self.SolverType = SolverType
        self.maxiter = maxiter
        self.is_set_initial_u = False
        # do not ask for results recovery if set to true
        self.ignore_previous_results= False
        self.max_failure_authorized = max_failure_authorized
        self.max_active_set_changes_authorized =\
                max_active_set_changes_authorized

        if FEM.HasConstraints:
            assert FEM.is_set_ContactTable, 'set contact table first'
            if "store_contact_history" in kwargs.keys():
                self.StoreContactHistory = kwargs["store_contact_history"]
            else:
                self.StoreContactHistory = True
        else:
            self.StoreContactHistory= False

        # default method to compute the residual. Can be changed via setter method
        self.variant_to_compute_residual = 0

        self.successful_step = 0


        tmax= Loading.tfinal
        ndofs = FEM.ndofs_el

        if not self.Static:
            # generate "unique" output filename
            filename =  str("%.fsec" %
                        (tmax)) + str("%d" %
                        (ndofs)) + str("%.f%.f" %
                        (a0, a1)) + str("%.f%.f%.f" %
                        (Alpha, Beta, Gamma)) + '.h5'
        else:
            filename =  str("%.fsec" %
                        (tmax)) + str("%ddofs" %
                        (ndofs)) + '.h5'

        self.pathfile = './Results/' + filename
        self.control_inf_norm = True
        self.restore_to_prev_converge_if_AS_wrong = False

    ################################################################################
                            # CONSTRUCTOR
    ################################################################################




    ################################################################################
                                # SETTER
    ################################################################################
    def set_tot_ndofs(self, ndofs):
        # useful if the number of dofs changes along the simulation for example change in active set
        # when Largrange Multiplier method is used
        assert isinstance(ndofs, int)
        self.tot_ndofs = ndofs

    #------------------------------------------------------------------------------
    #
    #------------------------------------------------------------------------------
    def set_initial_u(self, initial_u):
        # handy in the case we want to set an initial displacement
        assert initial_u.shape == (self.tot_ndofs,)
        self.is_set_initial_u = True
        self.initial_u = initial_u

    def set_variant_to_compute_residual(self, method_chosen):
        try:
            assert method_chosen in [0,1]
        except AssertionError:
            raise NotImplementedError
        self.variant_to_compute_residual = method_chosen

    def set_control_inf_norm(self, arg):
        assert isinstance(arg, bool)
        self.control_inf_norm = arg

    def set_restore_to_prev_converge_if_AS_wrong(self, arg):
        assert isinstance(arg, bool)
        self.restore_to_prev_converge_if_AS_wrong = arg
    ################################################################################
                                # SETTER
    ################################################################################






    ################################################################################
                              # NONLINEAR SOLVERS
    ################################################################################
    #------------------------------------------------------------------------------

    #                   STATIC ANALYSIS

    #------------------------------------------------------------------------------
    #------------------------------------------------------------------------------

    #------------------------------------------------------------------------------
    def Solve(self):
        Loading = self.Loading
        FEM = self.FEM
        # if geometrically exact beam elements are used, a different strategy must be adopted to
        # update the displacements.
        GeoExact= FEM.GeoExact

        self.charact_dim = 0.5 * np.min([FEM.a_crossSec, FEM.b_crossSec])

        #------------------------------------------------------------------------------
            # INITIALIZE SOLVER AND CREATE OUTPUT FILE TO STORE SIM. RESULTS
        #------------------------------------------------------------------------------
        LoadResults, Success, t = self.getOutputFile(\
            Loading.tfinal, 'NR', FEM.ndofs_el)

        if LoadResults:
            if Success:
                return
            else:
                ContinueFailed = input(
                        "0 : just load failed analysis and return \n" +\
                        "1 : continue analysis from failed TS \n")
                if ContinueFailed == '0':
                    return
                elif ContinueFailed == '1':
                    # the last non failed step
                    self.restore_previously_converged()
                else:
                    raise ValueError('Incorrect input')
        else:

            self.set_tot_ndofs(FEM.tot_ndofs)
            fext = zeros(FEM.ndofs_el, dtype=np.float)
            step_start = 0
        if FEM.HasConstraints:
            Tab = FEM.ContactTable
            if FEM.enforcement==0:
                assert FEM.is_set_penalty_options, 'penalty parameters not\
            initialized'

            # CHOOSE INITIAL ACTIVE SET
            # for the other steps, the active set will be taken at the
            # begin of the step as the one from the previoulsy
            # converged step
            FEM.choose_active_set()
        #------------------------------------------------------------------------------
            # INITIALIZE SOLVER AND CREATE OUTPUT FILE TO STORE SIM. RESULTS
        #------------------------------------------------------------------------------



        tn = 0
        self.stp = 0
        while True:
            converged = 0
            failure_stp = 0

            while not converged:
                t = tn + Loading.DELt
                if t > Loading.tfinal:
                    self.SuccessSignal()
                    return
                (duG, dfext, di, fr) = Loading.get_duG_dfext(t, tn, FEM.ndofs_el)
                try:
                    (u, fr) = self.solve_TS_NR(duG, fext, di ,fr)
                    converged = 1
                    self.CommitTS(t, di, fr, u, fext)
                    self.stp += 1
                    self.adaptDELt(1)
                    fextn = fext
                except FailedConvergence as error:
                    failure_stp += 1
                    self.adaptDELt(0)
                    if failure_stp >= self.max_failure_authorized:
                        FEM.plot()
                        set_trace()
                        raise Failure_Adaptive_TS(t)
                    else:
                        self.restore_previously_converged()
            tn = t


    #------------------------------------------------------------------------------
    #
    #------------------------------------------------------------------------------
    def restore_previously_converged(self, recover_AS = True):
        # stored cumulative displacement during TS
        FEM = self.FEM
        with open_file(self.pathfile, mode = 'r') as fileh:
            # last converged
            nlc = fileh.root.TS.Results.shape[0] - 1
            if nlc == -1:
                # means we restore the system in its initial state
                print('restore initial config ')
                ulc = zeros(FEM.u.shape)
                #rotational dofs
                vlc = zeros(FEM.v.shape)
            elif nlc < -1:
                raise ValueError('cannot restore system')
            else:
                ulc = fileh.root.TS.Results[nlc]['u'].reshape(self.FEM.nn,3)
                #rotational dofs
                vlc = fileh.root.TS.Results[nlc]['rot_vect'].reshape(self.FEM.nn,3)
                if FEM.HasSphere:
                    # directly set the position of the sphere
                    FEM.Sphere.setCenterPosition(fileh.root.TS.Results[nlc]['PosCSp'])

            if FEM.HasConstraints:
                # can be turned off if we want to do a step with a different active set
                if recover_AS:
                    Tab = FEM.ContactTable
                    Tab.active_set[:] = 0
                    # the AS has been saved with argwhere and flattened becaue we cannot save tuples in PyTables
                    AS = tuple( fileh.root.TS.active_set[nlc].reshape(-1,3).T)
                    Tab.active_set[AS] = True
                    Tab.dofs_LM[:] = -999
                    Tab.regenerate_dofs(FEM.ndofs_el)
                    #Tab.gN[AS] = fileh.root.TS.Results[nlc]['gN']
                    if self.FEM.enforcement == 1:
                        Tab.LM[AS] = fileh.root.TS.LM[nlc]
                    elif self.FEM.enforcement == 0:
                        Tab.kN[AS] = fileh.root.TS.kN[nlc]
                    else:
                        raise ValueError('case not recognized')

        DEL_u = FEM.u - ulc
        DEL_v = FEM.v - vlc
        DEL_U = zeros((FEM.nn , 6), dtype = float)
        DEL_U[:,:3] = DEL_u
        DEL_U[:,3:] = DEL_v
        FEM.update_config_geometrically_exact_beams(- DEL_U.ravel())
        print('System restored to previous converged TS')


    #------------------------------------------------------------------------------
    #
    #------------------------------------------------------------------------------
    def adaptDELt(self, Code):
        if Code == 0:
            # the step has failed and the BCs must be applied less
            # "brutally"
            factor = 0.5
            self.Loading.DELt *= factor
            print('Decrease DELt by a factor of %d'%(1./factor))
        elif Code == 1:
            if self.iter < 10:
                if self.iter <= 1:
                    print('Increase DELt')
                    self.successful_step = 0
                    self.Loading.DELt *= 1.05
                self.successful_step += 1
                if self.successful_step == 5:
                    print('Increase DELt')
                    self.Loading.DELt *= 1.05
                    self.successful_step = 0

        else:
            raise ValueError('unknown case')


    #------------------------------------------------------------------------------
    #
    #------------------------------------------------------------------------------
    def solve_TS_NR(self, duG, fext, di ,fr):
        Loading = self.Loading
        FEM = self.FEM
        if FEM.HasConstraints:
            Tab = FEM.ContactTable
        GeoExact= FEM.GeoExact

        #------------------------------------------------------------------------------
        def get_r_A(u=None):
            """
            get residual and stiffness
            """
            fint, A = FEM.getFandK(u,\
                    iiter = self.iter,\
                    outer_loop_ctc = Outer_Loop_counter,\
                    stp = self.stp)
            #assert np.allclose(A, A.T)
            return fint-fext , A
        #------------------------------------------------------------------------------


        CONSTRAINTS_OK = 0
        Outer_Loop_counter = 0
        self.printBeginStepMsg(self.stp)
        self.DEL_u =  zeros((self.tot_ndofs,), dtype=np.float)
        while not CONSTRAINTS_OK:
            # the update of the config must be made iteration per iteration
            du = zeros((self.tot_ndofs,), dtype=np.float)
            if Outer_Loop_counter == 0:
                if FEM.HasConstraints and\
                    FEM.enforcement == 1:
                    fr_init = np.copy(fr)
                else:
                    fr_init = fr
                if FEM.HasSphere:
                    FEM.Sphere.Move(Loading.CurrentSphereDisplacement)
                    print('Position of the sphere:')
                    print(FEM.Sphere.C)
                # different from 0 only during the first loop
                du[di] += duG[di]
                FEM.update_config_geometrically_exact_beams(du[:FEM.ndofs_el])

            if FEM.HasConstraints and FEM.enforcement == 1:
                # remove the rows corresponding to the LM of
                # the previous loop
                fext = fext[:FEM.ndofs_el]
                du = du[:FEM.ndofs_el]

                # the free indices are constant for one outer loop with the LM technique
                # np.where PERFECT here
                frLM = Tab.dofs_LM[where(Tab.active_set)].ravel()
                fr = np.hstack((fr_init, frLM ))
                assert np.all(frLM >= 0)
                # change the sizes of the arrays bcse of the
                # presence of LM
                du = np.hstack((du,\
                    zeros_like(frLM)))
                fext= np.hstack((fext,\
                    zeros_like(frLM)))

            self.iter = 0
            r, A = get_r_A()
            res = 1 + self.tol
            while res > self.tol:
                if self.iter >= self.maxiter:
                    raise FailedConvergence('no conv after maxiter')
                du[fr] = np.linalg.solve(A[ix_(fr, fr)], -r[fr])

                if self.control_inf_norm:
                    infnorm = norm(du[fr_init], np.inf)
                    if infnorm > 100. * self.charact_dim:
                        FEM.plot()
                        set_trace()
                        print ('inf norm too high')
                        raise FailedConvergence('failed stp')
                    if infnorm > self.charact_dim:
                        du[fr] /= (1+1e-5) *  (infnorm / self.charact_dim)
                        print('reduction')

                FEM.update_config_geometrically_exact_beams(du[:FEM.ndofs_el])
                self.DEL_u += du[:FEM.ndofs_el]


                if FEM.HasConstraints and FEM.enforcement == 1:
                    # the free indices are constant for one outer loop with the LM technique
                    Tab.update_LM_value(du)
                if self.iter == 0 and Outer_Loop_counter == 0:
                    du[di] = 0
                r, A = get_r_A()

                res = self.scalar_residual(r, di, fr, fext)
                print('res = %s' % str(res))
                self.iter += 1

            # verifies correctness of the active set
            if FEM.HasConstraints:
                CONSTRAINTS_OK =\
                FEM.check_current_active_set(self.stp)

                if not CONSTRAINTS_OK:
                    # valid only for the model with 3 yarns
                    self.printWrongActiveSet()
                    Outer_Loop_counter += 1
                    if Outer_Loop_counter >=\
                    self.max_active_set_changes_authorized:
                        raise FailedConvergence(' Could not find an appropriate\
                                active set')

                    if self.restore_to_prev_converge_if_AS_wrong:
                        print('AS wrong -> return previous state')
                        self.restore_previously_converged(recover_AS = False)


            else:
                CONSTRAINTS_OK = 1


        return (FEM.u, fr)


    #------------------------------------------------------------------------------

    #                   DYNAMIC ANALYSIS

    #------------------------------------------------------------------------------
    def ExplicitCentralDiff(self, t, a0=0, a1=0, updateC=0):
        """
        CENTRAL DIFFERENCE METHOD IN TIME
        From Belytschko p.313
        CAREFUL : Explicit Time Integration from Bathe p.771 only valid for linear system.
        Otherwise, must replace KU by fint !

        VALID FOR NON CONSTANT TS TOO
        """
        Wov = self.Wov
        LoadResults, Success, Lastnconv = self.getOutputFile(
            t, 'Implicit', self._Ndofs, a0, a1)

        if LoadResults:
            if Success:
                return
            else:
                ContinueFailed = input(
                    "0 : just load failed analysis and return \n 1 : continue analysis from failed TS \n ")
                if not ContinueFailed:
                    return
                else:
                    n0 = Lastnconv

        ndofs = Wov._Ndofs
        un = Wov._u
        unP1 = np.zeros(ndofs)
        vn = np.zeros(ndofs)
        vnP1 = np.zeros(ndofs)
        an = np.zeros(ndofs)
        anP1 = np.zeros(ndofs)
        n = 0

        # INITIALISATION OF THE SOLVER FOR t = 0 AND n = 0

        (fextn, duG, di, fr, v, vimp) = self.Loading.getCurrentBCs(0, ndofs)
        vfr = np.delete(np.arange(ndofs), (vimp))
        fint, Kint, M, C = Wov.getforce(un, a0, a1, updateC)
        vn[vimp] = v[vimp]
        r = fextn - fint - inner(C, vn)
        an[fr] = LA.solve(M[ix_(fr, fr)], r[fr])
        # commit t = 0
        self.CommitTS(n=0,
                      di=di,
                      fr=fr,
                      uTC=un,
                      vTC=vn,
                      aTC=an,
                      fintTC=fint,
                      fextTC=fextn,
                      Wov=Wov)
        # to initialise the algortihm, we must infer from the data unM1 and from it we can
        # construct vnM05
        DtM05 = (t[1] - t[0])
        unM1 = un - DtM05 * vn + -1.5 * DtM05**2 * an
        # from the formula 5.8 Book DeBorst, Criesfield
        vM05 = zeros(ndofs)
        vM05[vfr] = (un - unM1)[vfr] / (-DtM05)
        tM05 = 0.5 * (- DtM05)

        for n in range(t.shape[0] - 1):
            tP1 = t[n + 1]
            tn = t[n]
            tP05 = 0.5 * (tP1 + tn)
            DtP05 = tP1 - tn
            # get the BCs for the current simulation time + the indices : di for dirichlet indices,
            # fr for free ones
            (fext, duG, di, fr, v, vimp) = self.Loading.getCurrentBCs(tP1, ndofs, tn)
            vfr = np.delete(np.arange(ndofs), (vimp))
            unP1[di] = un[di] + duG[di]
            # first update of the partial velocities
            vP05 = zeros_like(unP1)
            vP05[vfr] = (vn + (tP05 - tn) * an)[vfr]
            vP05[vimp] = v[vimp]
            unP1[fr] = un[fr] + DtP05 * vP05[fr]
            fint, Kint, M, C = Wov.getforce(unP1, a0, a1, updateC)
            r = fext - fint - inner(C, vP05)
            # update acceleration
            anP1[vimp] = 0
            anP1[vfr] = LA.solve(M[ix_(vfr, vfr)], r[vfr])
            vnP1[vfr] = vP05[vfr] + (tP1 - tP05) * anP1[vfr]
            vnP1[vimp] = v[vimp]

            selfCheckEnergyConsTS(raiseifFailed=1)

            self.CommitTS(n=n + 1,
                          di=di,
                          fr=fr,
                          uTC=unP1,
                          vTC=vnP1,
                          aTC=anP1,
                          fintTC=fint,
                          fextTC=fext,
                          Wov=Wov)

            # check the energy conservation of the TS

            # update for next TS
            un = np.copy(unP1)
            vn = np.copy(vnP1)
            an = np.copy(anP1)
            vM05 = np.copy(vP05)

            print('TS ' + repr(n + 1) + '\n')

        print('------------------------------------------------------------------------------ ')
        print('END OF ANAYSIS')
        print('------------------------------------------------------------------------------ ')

    #------------------------------------------------------------------------------

    #------------------------------------------------------------------------------
    def AlphaHHT(
            self,
            tmax,
            NumericalParam='',
            Alpha=0,
            Gamma=0,
            Beta=0,
            a0=0,
            a1=0,
            updateC=0,
            ConstantDt=1,
            NTS=0):
        # from DeBorst And Criesfield book
        # from the book of Laursen: alpha = 1 means no numerical dissipation added by the HHT TI
        # while Alpha = 0 means very high dissipation
        # note that the Newmark's time integrator can be recovered by using

        if not NumericalParam == '':
            print('Set of numerical parameters set according to family of algo provided')
            # provide numerical parameteres that control stability and
            # dissipation by hand
            if NumericalParam == 'OptimDissip':
                Alpha = -0.1
                Beta = 0.3025
                Gamma = 0.6
            elif NumericalParam == 'TR':
                Alpha = 0
                Beta = 0.25
                Gamma = 0.5
            else:
                raise ValueError('NumericalParam not recognized')

        if Alpha == 0:
            if Gamma == 0:
                Gamma = 0.5
                # if Gamma = 0.5, we have the trapezoidal rule
                Beta = 0.5 * Gamma

            if Gamma == 0.5 and Beta == 0.25:
                print('TRAPEZOIDAL RULE')
            else:
                print('DISSIPATIVE SCHEME CHOSEN')
        else:
            # if the parameters follow the next guidelines, unconditonnal stability is guaranteed
            # and second order accuracy in time is assured for linear systems
            if Gamma == 0 and Beta == 0:
                if Alpha < 0 and Alpha > -1. / 3.:
                    Beta = 0.25 * (1 - Alpha)**2
                    Gamma = 0.5 - Alpha
                else:
                    raise Warning('The algorithmic parameters provided for the Alpha method do not\
                                  follow the guidelines given by Criesfield ')
            else:
                if Alpha == 0.1 and Gamma == 0.6 and Beta == 0.3025:
                    print('Optimal high frequency damping (Laursen 1997, Hilber 1977)')

        if ConstantDt:
            assert NTS > 0
            assert not isinstance(tmax, np.ndarray)
            Dt = tmax * (NTS)**(-1)
        else:
            raise NotImplementedError

        # CREATE OUTPUT FILE
        LoadResults, Success, t = self.getOutputFile(
            tmax, 'AlphaHHT', self._Ndofs, a0, a1, Alpha, Beta, Gamma)

        if LoadResults:
            if Success:
                return
            else:
                ContinueFailed = input(
                    "0 : just load failed analysis and return \n 1 : continue analysis from failed TS \n ")
                if not ContinueFailed:
                    return
                else:
                    un, vn, an, tn, Dt = self.LoadTS()
                    t = tn + Dt
                    unP1 = zeros_like(un)
                    vnP1 = zeros_like(un)
                    anP1 = zeros_like(un)
                    set_trace()

        def getuNP1Tilde():
            return un + Dt * vn + 0.5 * Dt**2 * (1 - 2 * Beta) * an

        def getvNP1Tilde():
            # 6.3.5 Belytschko
            return vn + (1 - Gamma) * Dt * an

        Wov = self.Wov
        if a0 != 0 or a1 != 0:
            Damping = 1
        else:
            Damping = 0

        def get_r_A():
            unPAlpha = (1 + Alpha) * unP1 - Alpha * un
            fint, Kint, M, C = Wov.getforce(unPAlpha, a0, a1, updateC=updateC)
            # velocity and acceleration update have the same formulas as in the
            # Newmark scheme
            anP1[fr] = (Beta * Dt**2)**(-1) * (unP1 - uTilde)[fr]
            vnP1[fr] = (vTilde + Gamma * Dt * anP1)[fr]
            vnP1[vimp] = v[vimp]
            # vnPAlpha = (1 + Alpha) * vnP1 - Alpha * vn
            # 'algorithmic' stiffness matrix
            # BE CAREFUL WITH THE EXPRESSION GIVEN BY BELYTSCHLO THAT WITH THE SPLITTING OF KINT
            # VALID ONLY FOR LINEAR SYSTEM
            # Here, we have evaluated the stiffness in unPAlpha, so we can use
            # it directly
            A = (1 + Alpha) * Kint + (Beta * Dt**2)**(-1) * M
            # 'algorithmic' force residual
            r = fint + inner(M, anP1) - ((1 + Alpha) * fextnP1 - Alpha * fextn)
            if Damping:
                # The damping force acts only on the dirichlet indices
                # we add the damping force to the force residual
                # we must use vnP1
                r[fr] += inner(C, vnP1)[fr]
                # modify corresponding term in dynamic stiffness matrix
                # derived from sympy !!
                # A += C*Gamma*(Alpha + 1)/(Beta*Dt)
                A[ix_(fr, fr)] += C[ix_(fr, fr)] * Gamma / (Beta * Dt)
            return r, A, fint, vnP1, anP1

        if not LoadResults:
            # handy in the case where u is not 0 at t = 0
            un = np.copy(Wov._u)
            """
            # THE FOLLOWING CAUSES ERRORS!
            unP1 = np.zeros(Wov._Ndofs, dtype = np.float32)
            vn = np.zeros(Wov._Ndofs,  dtype = np.float32)
            vnP1 = np.zeros(Wov._Ndofs,  dtype = np.float32)
            an = np.zeros(Wov._Ndofs,  dtype = np.float32)
            anP1 = np.zeros(Wov._Ndofs,  dtype = np.float32)
            """
            unP1 = np.zeros(Wov._Ndofs)
            vn = np.zeros(Wov._Ndofs)
            vnP1 = np.zeros(Wov._Ndofs)
            an = np.zeros(Wov._Ndofs)
            anP1 = np.zeros(Wov._Ndofs)

            # compute initial acceleration
            # surtout si on commence avec une position penetree !
            (fextn, duG, di, fr, v, vimp) = self.Loading.getCurrentBCs(0, Wov._Ndofs)
            # no duG at t = 0 !
            # un[di] += duG[di]
            vn[vimp] = v[vimp]
            fint, Kint, M, C = Wov.getforce(un, a0, a1, updateC=updateC)
            # VALID FOR v0 != 0
            r = fextn - fint - inner(C, vn)
            # estimation of the acceleration at t=0 (not mentionned in the book of Crieslielf but
            # present in the one of Belytschko)
            an[fr] = LA.solve(M[ix_(fr, fr)], r[fr])
            self.CommitTS(0, 0, di, fr, un, vn, an, fint, fextn, Wov)

            tn = 0
            t = Dt

        # we solve for displacements and pder at t(n + 1)
        while t < tmax:
            assert Dt > 0 and tn < t
            # the 2 following expressions are constant during TS because based
            # on previous converged TS only
            uTilde = getuNP1Tilde()
            vTilde = getvNP1Tilde()
            # get the BCs for the current simulation time + the indices : di for dirichlet indices,
            # fr for free ones
            (fextnP1, duG, di, fr, v, vimp) = self.Loading.getCurrentBCs(
                t, Wov._Ndofs, tn)
            # estimate of the displacement at current time step n + 1
            unP1[fr] = np.copy(uTilde)[fr]
            vnP1[fr] = vTilde[fr]
            DEL_u = zeros_like(un)
            DEL_u[di] += duG[di]
            # Approximation of the velocity at the dirichlet BCs
            vnP1[di] = duG[di] * (Dt)**(-1)
            vnP1[vimp] = v[vimp]
            # force the velocity where needded
            MaxPenet = 1 + self.MaxPenAllow
            PenCt = 0
            while MaxPenet > self.MaxPenAllow:
                # iteration counter
                ii = 0
                res = 1 + self._tol
                r, A, fint, vnP1, anP1 = get_r_A()
                while res > self._tol:
                    # solve for increment in displacements
                    dufr = LA.solve(A[ix_(fr, fr)], -r[fr])
                    # inf norm
                    iN = LA.norm(dufr, np.inf)
                    if iN > Wov._Yarns[0]._r:
                        print ('dufr shrunk')
                        # CAREFUL ONLY THE FREE INDICES MUST BE SHRUNK
                        red = (iN) * (Wov._Yarns[0]._r)**(-1)
                        if red <= 1:
                            raise ValueError
                        elif red > 10:
                            self.FailureSignal()
                            raise ValueError('du too important')
                        dufr /= red
                    DEL_u[fr] += dufr
                    unP1 = un + DEL_u
                    r, A, fint, vnP1, anP1 = get_r_A()
                    # new convergence criterion from belytschko's book
                    maxF = max(norm(fint), norm(fextnP1), norm(inner(M, anP1)))
                    if maxF > 1e-8:
                        res = norm(r[fr]) * maxF**(-1)
                    else:
                        res = norm(r[fr])
                    ii += 1
                    if ii > self.maxiter:
                        self.FailureSignal()
                        raise ValueError(
                            'MaxNIter')

                if self.PenReguAllowed:
                    MaxPenet = Wov.CtcEL.PenaltyRegu(
                        -self.MaxPenAllow, u=unP1, v=vnP1, a=anP1, fext=fextnP1, FEAss=Wov)
                    PenCt += 1
                    if PenCt > self.maxPiter:
                        raise ValueError(
                            'Maximum penetration remains too high')
                else:
                    MaxPenet = self.MaxPenAllow - 1
            self.CommitTS(t, Dt, di, fr, unP1, vnP1, anP1, fint, fextnP1, Wov)
            # update for next TS
            un = np.copy(unP1)
            vn = np.copy(vnP1)
            an = np.copy(anP1)
            fextn = np.copy(fextnP1)
            tn = np.copy(t)

            self.printConvTSMsg(t, ii)
            # left in the end ON PURPOSE otherwise ew go over tmax and the BCs
            # suddenly disappear !
            t += Dt
        self.SuccessSignal()
    ################################################################################
                              # NONLINEAR SOLVERS
    ################################################################################






    ################################################################################
                            # OUTPUT FILE AND RESULT STORAGE
    ################################################################################
    def getOutputFile(
            self,
            tmax,
            SolverName,
            ndofs,
            a0=0,
            a1=0,
            Alpha=0,
            Beta=0,
            Gamma=0):
        # use the hd5 file to avoid enromous ammount of memory needed if a preallocation with a
        # numpy array is done

        path = str(self.pathfile)
        if os.path.isfile(self.pathfile):
            # the file has already been made. If the outcome was successful needless to make a
            # new run
            # open the file and read stored values fom disk
            fileh = open_file(self.pathfile, mode="r")
            if self.ignore_previous_results:
                print('A result file corresponding to the same analysis'+\
                      'has already been done but will be deleted')
                fileh.close()
                os.remove(self.pathfile)
            else:
                Info = fileh.root.TS.Info
                try:
                    Success = Info[0]['Success']
                    if Success == 1:
                        end = ' and ended with SUCCESS'
                    elif Success == 0:
                        end = ' and FAILED prematurely'
                    else:
                        raise ValueError('Success must be stored as a bool')
                    fileh.close()
                    base = 'Analysis with same parameters has already been done'
                    print(base + end)
                    LoadResults = input(
                        "1 : load results and continue \n0 : start analysis from scratch(0) ? \n")
                except IndexError:
                    fileh.close()
                    os.remove(self.pathfile)
                    print('Output file found but corrupted')
                    LoadResults = 0
                except IOError:
                    os.remove(self.pathfile)
                    print('Output file found but corrupted')
                    LoadResults = 0

                if LoadResults =='1':
                    return LoadResults, Success, 0
                else:
                    try:
                        os.remove(self.pathfile)
                    except OSError:
                        pass

        GeoExact = self.FEM.GeoExact
        nn = self.FEM.nn
        rot_dofs_storage= self.FEM.rot_dofs_storage
        # create data structure
        class TSRes(IsDescription):
            if not GeoExact:
                u = Float32Col(shape=(ndofs,))
            else:
                u = Float32Col(shape=(nn * 3, ))
                if rot_dofs_storage == 0:
                    # quaternions storage
                    q = Float32Col(shape=(nn,4 ))
                elif rot_dofs_storage == 1:
                    rot_vect = Float32Col(shape=(nn,3 ))
                else:
                    raise NotImplementedError

            v = Float32Col(shape=(ndofs,))
            a = Float32Col(shape=(ndofs,))
            fd = Float32Col(shape=(ndofs,))
            t = Float32Col()
            Eint = Float32Col()
            Ekin = Float32Col()
            Wext = Float32Col()
            D = Float32Col()
            Dt = Float32Col()
            NAct = Int32Col(shape=(1,))
            if self.FEM.HasSphere:
                # store the successive positions of the center of the sphere
                PosCSp = Float32Col(shape=(3,))

        # create data structure to store all the numerical parameter used
        # during analysis
        class RunInfo(IsDescription):
            name = StringCol(32)
            Success = Int32Col()
            a0 = Int32Col()
            a1 = Int32Col()
            Alpha = Int32Col()
            Beta = Int32Col()
            Gamma = Int32Col()
            RunTime = Float32Col()
            InitialEnergy = Float32Col()

        # create directory for result storage if it doesnt exist
        if not os.path.isdir("./Results"):
            os.mkdir('Results')
        else:
            # delete file with results if the same simulation has already been done
            if os.path.isfile(self.pathfile):
                os.remove(self.pathfile)

        fileh = open_file(path, mode="w")
        # get the root of the file structure
        root = fileh.root
        # will contain the result of each converged time step
        fileh.create_group(root, 'TS')
        # table with different numpy arrays that have a fixed length (number of
        # dofs)

        # the first stable is easier to construct because we know the number of dofs
        # and each numpy arrays that will be contained will be of fixed size
        fileh.create_table("/TS", 'Results', TSRes, 'Results Sim')
        # second table with more generic info on the Simulation
        TabInfo = fileh.create_table(
            "/TS", 'Info', RunInfo, 'Info on the Simu')

        if self.FEM.HasConstraints:
            # --> we cannot know the number of active contact points in advance.
            # this is why we create this array with variable dimensions
            # in the case we wan to store the positions of the contact points
            """
            fileh.create_vlarray(
                fileh.root.TS, 'xCtc', Float32Col(
                    (2, 3)), "xC")
            """
            # Array with VARIABLE length
            # --> we cannot know the number of active contact points in advance.
            # store normal gap
            fileh.create_vlarray(fileh.root.TS, 'active_set', Int32Atom(  shape= () ), "active_set")
            fileh.create_vlarray(fileh.root.TS, 'gN', Float32Atom(), "gN")
            #fileh.create_vlarray(fileh.root, 'index_elements', tables.Int32Atom(shape=((2,))), "index_elements")
            if self.FEM.enforcement == 0:
                fileh.create_vlarray(fileh.root.TS, 'kN', Float32Atom(), "kN")
            elif self.FEM.enforcement == 1:
                fileh.create_vlarray(fileh.root.TS, 'LM',
                        Float32Atom(), "LM")
            else:
                raise ValueError('unknown case')

        if not self.Static:
            a0 = np.nan
            a1 = np.nan
            Alpha = np.nan
            Beta = np.nan
            Gamma = np.nan
        # complete the info tab with the numerical parameters
        row = TabInfo.row
        """
        row['name'] = filename
        row['a0'] = a0
        row['a1'] = a1
        row['Alpha'] = Alpha
        row['Beta'] = Beta
        row['Gamma'] = Gamma
        """
        # do not forget to append the array and to flush the Table to free
        # memory !
        row.append()
        TabInfo.flush()
        fileh.close()
        assert not fileh.isopen
        print('Output file successfully created !')

        return 0, 0, 0

    #------------------------------------------------------------------------------
    #
    #------------------------------------------------------------------------------
    def CommitTS(self, t, di, fr, uTC, fextTC, **kwargs):
        # TC : shorthand for 'to commit'
        # In this function we commit the result of one time step

        # construct copies of the array
        FEM = self.FEM
        if uTC.ndim != 1:
            uTC = uTC.ravel()
        if fextTC.ndim != 1:
            fextTC = fextTC.ravel()
        #####################################
        # commit current TS and save data on disk
        #####################################
        with open_file(self.pathfile, mode="a") as fileh:
            """
            if not hasattr(self, 'Eatt0') and t == 0:
                # store initial energy
                # note : useful for energy conservation checking if it is
                # different from 0
                self.Eatt0 = Ekin + Eint
                # modify value without touching the row
                fileh.root.TS.Info.cols.InitialEnergy[0] = float(self.Eatt0)
            elif not hasattr(self, 'Eatt0') and t > 0:
                raise ValueError('t = 0 has been skept !')
            """

            if not hasattr(self, 'fextstrd'):
                self.fextstrd = fextTC.ravel()
                self.ustrd = uTC.ravel()
                #self.Wextstrd = np.copy(Wext)
            else:
                self.fextstrd[:] = fextTC
                self.ustrd[:] = uTC
                #self.Wextersotrted [:] = uTC

            if FEM.HasConstraints:
                # append the information on contact elements
                Tab = FEM.ContactTable
                # count the number of contact points that are penetrated
                # (active)
                # the structures of TS.gN and TS.kN is a vlarray that allows to store arrays of differetns
                # lengths

                # IMPORTANT NOTEL: tuple(np.argwhere(Tab.active_set).T) == np.where(Tab.active_set)
                AS = Tab.active_set
                # has to flatten to be saved
                fileh.root.TS.active_set.append(AS)
                #fileh.root.TS.gN.append(Tab.gN[AS])
                if self.FEM.enforcement == 1:
                    fileh.root.TS.LM.append(Tab.LM)
                elif self.FEM.enforcement == 0:
                    fileh.root.TS.kN.append(Tab.kN)
                else:
                    raise ValueError('case not recognized')

            tab = fileh.root.TS.Results
            # get reference to pointer to current row in table
            row = tab.row
            # commit the result of the current time step
            row['u'] = uTC.ravel()
            row['t'] = t
            if FEM.HasSphere:
                row['PosCSp'] = FEM.Sphere.C
            if self.FEM.GeoExact:
                if self.FEM.rot_dofs_storage == 0:
                    row['q'] = self.FEM.q
                elif self.FEM.rot_dofs_storage == 1:
                    row['rot_vect'] = self.FEM.v
                else: raise NotImplementedError
            row.append()
            tab.flush()
        assert not fileh.isopen

    #------------------------------------------------------------------------------
    #

    #------------------------------------------------------------------------------
    #
    #------------------------------------------------------------------------------
    def LoadTS(self, n):
        with open_file(self.pathfile, mode='r') as fileh:
            #dofs in displacement
            self.FEM.u =\
            fileh.root.TS.Results[n]['u'].reshape(self.nn,3)
            #rotational dofs
            self.FEM.v =\
                    fileh.root.TS.Results[n]['rot_vect'].reshape(self.nn,3)


    #------------------------------------------------------------------------------
    #
    #------------------------------------------------------------------------------
    def Load_full_contact_history(self):
        with open_file(self.pathfile, mode = 'r') as fileh:
            nstp = fileh.root.TS.Results.shape[0]
        nLM = zeros(nstp , dtype = np.int)

        try:
            with open_file(self.pathfile, mode='r') as fileh:
                for i in range(nstp):
                    if self.FEM.enforcement == 1:
                        nLM[i] = fileh.root.TS.LM[i].shape[0]
        finally:
            fileh.close()
        return nLM

    ################################################################################
                            # OUTPUT FILE AND RESULT STORAGE
    ################################################################################




    ################################################################################
                            # PRINTING MESSAGES
    ################################################################################
    #------------------------------------------------------------------------------
    #
    #------------------------------------------------------------------------------
    def printConvTSMsg(self, t, iternb):
        print(' t = %.9f converged in %d iters' % (t, iternb))

    #------------------------------------------------------------------------------
    #
    #------------------------------------------------------------------------------
    def printBeginStepMsg(self,stp):
        assert isinstance(stp, int)
        print('\n')
        print('###################################################')
        print('################### STEP %d #######################'%(stp))
        print('###################################################')
        print('\n')
    #------------------------------------------------------------------------------
    #
    #------------------------------------------------------------------------------
    def printWrongActiveSet(self):
        print('\n')
        print('     ################### refining AS  #######################')
        print('\n')
    #------------------------------------------------------------------------------
    #
    #------------------------------------------------------------------------------
    def FailureSignal(self):
        # called when the NR scheme fails
        fileh = open_file(self.pathfile, mode="a")
        # store initial energy
        # note : useful for energy conservation checking if it is different
        # from 0
        tab = fileh.root.TS.Info
        tab[0]['Success'] = 0
        tab.flush()
        fileh.close()
        assert not fileh.isopen
        raise ValueError('FAILURE SIGNAL : NR did not converge')

    #------------------------------------------------------------------------------
    #
    #------------------------------------------------------------------------------
    def SuccessSignal(self):
        fileh = open_file(self.pathfile, mode="a")
        # NOTE : THIS IS THE WAY TO MODIFY A CERTAIN ENTRY OF THE FILE STORED ON THE DISK OTHERWISE
        # THE MODIFICATION IS NOT EFFECTIVE
        # store initial energy
        # note : useful for energy conservation checking if it is different
        # from 0
        fileh.root.TS.Info.cols.Success[0] = 1
        fileh.close()
        assert not fileh.isopen
    ################################################################################
                            # PRINTING MESSAGES
    ################################################################################





    ################################################################################
                            #  VARIOUS
    ################################################################################
    def scalar_residual(self, r, di, fr, fext ):
        method = self.variant_to_compute_residual
        if method == 0 :
            # not adapted when the material parameters are very high and an external loading is
            # applied!
            if fr.shape[0] > 0:
                return norm(r[fr]) / fr.shape[0]
            else:
                return norm(r)
        elif method == 1 :
            # suitable when external loading is applied
            # in agreement with the method of Lars. Except that in this case, the external forces
            # must be updates at the dirichlet dofs
            return norm(r) / norm(fext)
        else:
            # TODO
            raise NotImplementedError

    #------------------------------------------------------------------------------
    #
    #------------------------------------------------------------------------------
    def CheckEnergyConsTS(self, Ekin, Eint, D, Wext, raiseifFailed=0):
        #print('E0 : ' + str(self.Eatt0))
        try:
            err = norm(Eint - Wext + Ekin + D - self.Eatt0)
            assert err <= max(0.1 * self.tol, 0.1 *
                              max(np.abs([Eint, Wext, Ekin])))
            # print('ERROR ON ENERGY : '+ str(err) )
            return 1
        except AssertionError:
            if raiseifFailed:
                raise ValueError('Divergence')
            else:
                print('ERROR ON ENERGY : ' + str(err))
                # we might shrink DELt
                return 0

    #------------------------------------------------------------------------------
    #
    #------------------------------------------------------------------------------
    def set_ignore_previous_results(self, val):
        """
        handy if we launch several simulation without requiring the input from the
        user
        """
        assert isinstance(val, bool)
        self.ignore_previous_results= val


    ################################################################################
                            #  VARIOUS
    ################################################################################


#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
                            # SOLVER CLASS
#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
