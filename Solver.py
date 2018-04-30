
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
import pickle


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

        filename =  str("%.fsec" %
                        (tmax)) + str("%ddofs" %
                        (ndofs)) + '.h5'

        self.pathfile = './Results/' + filename
        if self.FEM.HasConstraints:
            self.pathfile_contact = ' '.join(self.pathfile.split('.h5')[:-1]) + '_contact' + '.p'
            self.pathfile_latest_cells_info = ' '.join(self.pathfile.split('.h5')[:-1]) + '_cells' + '.p'


        self.control_inf_norm = True
        self.restore_to_prev_converge_if_AS_wrong = False
        self.stp = 0
        self.tn = 0
        self.record_results = True


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

    #------------------------------------------------------------------------------
    #
    #------------------------------------------------------------------------------
    def set_variant_to_compute_residual(self, method_chosen):
        try:
            assert method_chosen in [0,1]
        except AssertionError:
            raise NotImplementedError
        self.variant_to_compute_residual = method_chosen

    #------------------------------------------------------------------------------
    #
    #------------------------------------------------------------------------------
    def set_control_inf_norm(self, arg):
        assert isinstance(arg, bool)
        self.control_inf_norm = arg

    #------------------------------------------------------------------------------
    #
    #------------------------------------------------------------------------------
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
        if self.record_results:
            LoadResults, Success, t = self.getOutputFile(\
                Loading.tfinal, 'NR', FEM.ndofs_el)
        else:
            LoadResults, Success, t = (0,0,0)

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
                    fext = FEM.fext
                else:
                    raise ValueError('Incorrect input')
        else:
            self.set_tot_ndofs(FEM.tot_ndofs)
            fext = zeros(FEM.ndofs_el, dtype=np.float)
            step_start = 0
        if FEM.HasConstraints:
            Tab = FEM.ContactTable
            # CHOOSE INITIAL ACTIVE SET
            # for the other steps, the active set will be taken at the
            # begin of the step as the one from the previoulsy
            # converged step
            FEM.choose_active_set()
        #------------------------------------------------------------------------------
            # INITIALIZE SOLVER AND CREATE OUTPUT FILE TO STORE SIM. RESULTS
        #------------------------------------------------------------------------------

        while True:
            converged = 0
            failure_stp = 0
            while not converged:
                t = float(self.tn + Loading.DELt)
                if t > Loading.tfinal:
                    self.SuccessSignal()
                    return
                (duG, dfext, di, fr) = Loading.get_duG_dfext(t, self.tn, FEM.ndofs_el)
                try:
                    (u, fr) = self.solve_TS_NR(duG, fext, di ,fr)
                    converged = 1
                    if self.record_results:
                        self.CommitTS(t, di, fr, u, fext)
                    self.stp += 1
                    self.adaptDELt(1)
                    self.fext = fext
                except FailedConvergence as error:
                    failure_stp += 1
                    self.adaptDELt(0)
                    if failure_stp >= self.max_failure_authorized:
                        FEM.plot()
                        set_trace()
                        raise Failure_Adaptive_TS(t)
                    else:

                        if self.record_results:
                            self.restore_previously_converged()
                        else:
                            raise ValueError('cannot restore because recording results has been\
                                    turned off')
            self.tn = t


        print('Assess continuity surface final config')
        FEM.CheckContinuitySurface()



    #------------------------------------------------------------------------------
    #
    #------------------------------------------------------------------------------
    def restore_previously_converged(self, recover_AS = True):
        # stored cumulative displacement during TS
        FEM = self.FEM
        with open_file(self.pathfile, mode = 'r') as fileh:
            # last converged increment
            nlc = fileh.root.TS.Results.shape[0] - 1
            if nlc == -1:
                # means we restore the system in its initial state
                raise NotImplementedError
        self.LoadTS(nlc)

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
                        "1 : load stored results\n0 : start analysis from scratch \n")
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

            fext = Float32Col(shape=(ndofs,))
            t = Float32Col()
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


        if not self.Static:
            a0 = np.nan
            a1 = np.nan
            Alpha = np.nan
            Beta = np.nan
            Gamma = np.nan
        # complete the info tab with the numerical parameters
        row = TabInfo.row
        # do not forget to append the array and to flush the Table to free
        # memory !
        row.append()
        TabInfo.flush()
        fileh.close()
        assert not fileh.isopen


        if self.FEM.HasConstraints:
            # will be used for storage
            # the last dict is to store kN or the values of the LM
            h_H = dict()
            gN_H = dict()
            active_cells_H = dict()
            active_set_H= dict()
            ID_master_H= dict()
            value = dict()

            # we will need to store the results for all increments in
            with open(self.pathfile_contact, "wb") as f:
                pickle.dump((h_H, gN_H, active_cells_H, active_set_H , ID_master_H, value), f)

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

            tab = fileh.root.TS.Results
            # get reference to pointer to current row in table
            row = tab.row
            # commit the result of the current time step
            row['u'] = uTC.ravel()
            row['t'] = t
            row['fext'] = fextTC
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

        if FEM.HasConstraints:
            # store the contact history in another file due to the specificity of the contact
            # sceheme (change in the number of active set and so on)

            with open(self.pathfile_contact, "rb") as f :
                h_H, gN_H, active_cells_H, active_set_H, ID_master_H, value = pickle.load(f)


            # append the information on contact elements
            Tab = FEM.ContactTable
            # add new key for the time step to the dictionnary
            key = str(self.stp)
            active_set_H[key] = Tab.active_set
            h_H[key] = Tab.h
            gN_H[key] = Tab.gN
            active_cells_H[key] = Tab.active_cells
            ID_master_H[key] = Tab.ID_master

            if active_set_H is h_H:
                raise ValueError('references to the same object !')
            #if len(Tab.active_cells) > 0: set_trace()
            if FEM.enforcement == 0:
                value[key] = Tab.kN
            elif FEM.enforcement == 1 :
                value[key] = Tab.LM

            with open(self.pathfile_contact,  "wb") as f:
                pickle.dump((h_H, gN_H, active_cells_H, active_set_H, ID_master_H,  value), f)

            # for the info on the cells, only the last converged increment is of interest.
            # Everything can be stored in a dict
            cell_info = dict()
            for attr in 'ID_master_Ctr hCtr KSI_Cells CON_Cells '.split(' '):
                cell_info[attr] = getattr(Tab, attr)

            with open(self.pathfile_latest_cells_info, "wb") as f:
                pickle.dump(cell_info, f)

    #------------------------------------------------------------------------------
    #

    #------------------------------------------------------------------------------
    #
    #------------------------------------------------------------------------------
    def LoadTS(self, n):
        self.step = n
        with open_file(self.pathfile, mode='r') as fileh:
            #dofs in displacement
            self.FEM.u =\
                fileh.root.TS.Results[n]['u'].reshape(self.FEM.nn,3).astype(np.double)
            #rotational dofs
            self.FEM.v =\
                    fileh.root.TS.Results[n]['rot_vect'].reshape(self.FEM.nn,3).astype(np.double)

            self.FEM.fext =\
                    fileh.root.TS.Results[n]['fext']

        if self.FEM.HasConstraints:
            Tab = self.FEM.ContactTable
            key = str(n)
            if not os.path.isfile(self.pathfile_contact):
                raise IOError('Cannot open file ')

            with open(self.pathfile_contact, mode="rb") as f:
                h_H, gN_H, active_cells_H, active_set_H, ID_master_H, value = pickle.load(f)
                if self.FEM.enforcement ==0 :
                    Tab.kN = value[key]
                elif self.FEM.enforcement == 1 :
                    Tab.LM = value[key]
                set_trace()
                Tab.active_set = active_set_H[key]
                Tab.active_cells = active_cells_H[key]
                Tab.gN = gN_H[key]
                Tab.h = h_H[key]
                Tab.ID_master = ID_master_H[key]


            with open(self.pathfile_latest_cells_info, "rb") as f:
                cell_info = pickle.load(f)

            for attr in 'ID_master_Ctr hCtr KSI_Cells CON_Cells '.split(' '):
                setattr(Tab, attr, cell_info[attr])




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
