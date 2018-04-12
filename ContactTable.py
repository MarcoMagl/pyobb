from __future__ import division
import numpy as np
from numpy import zeros, ones, ones_like, where, ix_, array
from pdb import set_trace
from Common.UserDefErrors import MaximumPenetrationError

#------------------------------------------------------------------------------
#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
#------------------------------------------------------------------------------
class ContactTable():
    """
    store info to handle contact for example which pair of elements might be in contact
    """

    #------------------------------------------------------------------------------
    #
    #------------------------------------------------------------------------------
    def __init__(self, enforcement,\
            method, contact_type = 1,\
            gN_treshold_not_checked = 1.,\
            **kwargs):

        # penalty or LM method
        assert enforcement == 0 or enforcement == 1, NotImplementedError('Only Penalty and LM implemented')
        self.enforcement = enforcement
        assert "alpha" in kwargs.keys(), "a scalar\
        parameter is needed for the contruction of the\
        splines "
        self.smoothing = True
        self.smoothing_type = 'splines'
        assert "alpha" in kwargs.keys(), "a scalar\
        parameter is needed for the contruction of the\
        splines "
        assert "smooth_cross_section_vectors" in\
        kwargs.keys(), "choose if the smooth section\
        vector of successive beams are smoothed or not"
        self.alpha=kwargs["alpha"]
        assert 0 < self.alpha < 1
        self.smooth_cross_section_vectors =\
        kwargs["smooth_cross_section_vectors"]
        assert method.lower() == 'curve_to_curve', 'Not Implemented'
        self.geometry_contact = method.lower()

        if self.enforcement == 0 :
            self.gN_treshold_not_checked = gN_treshold_not_checked
        elif self.enforcement == 1:
            self.default_value_LM = 1.
        self.contact_type = contact_type
        if contact_type == 0:
            # pointwise contact between curves
            setattr(self, 'active_set_type', 'pointwise')
        elif contact_type == 1:
            # weak enforcement of the contct inequality constraint
            self.is_set_parameter_weak_enforcement = 0
            setattr(self, 'active_set_type', 'segmental')
        else:
            raise NotImplementedError

        self.value_epsilon_set = False
        self._epsilon = None


    #------------------------------------------------------------------------------
    #
    #------------------------------------------------------------------------------
    def set_default_value_LM(self, value):
        assert self.enforcement == 1, 'not set to work with LM'
        self.default_value_LM = value


    #------------------------------------------------------------------------------
    #
    #------------------------------------------------------------------------------
    def query_is_active_slave(self, id_s):
        return id_s in self.active_set

    #------------------------------------------------------------------------------
    #
    #------------------------------------------------------------------------------
    def set_parameter_weak_contact(self, nintegrationIntervals = np.array([1,1]) ,\
            nxiGQP = 1 , nthetaGQP = 1, LM_per_ctct_el = 0):
        assert not self.is_set_parameter_weak_enforcement, 'already\
        set'
        assert nxiGQP >= 1
        assert nthetaGQP >= 1
        self.nxiGQP = nxiGQP
        self.nthetaGQP = nthetaGQP
        assert nintegrationIntervals.shape == (2,) and \
        np.all(nintegrationIntervals >= 1)
        self.nII = nintegrationIntervals
        self.is_set_parameter_weak_enforcement = True

        if self.enforcement == 1:
            self.LM_per_ctct_el = LM_per_ctct_el

    #------------------------------------------------------------------------------
    #
    #------------------------------------------------------------------------------
    def set_curve_to_curve_dynamic_table(self, ncurves,\
            slave_curves_ids, master_curves_ids, nmaster_curves,
            nslave_curves):
        """
        creation of the arrays necessary for the storing of the
        information for contact between beams
        """
        self.ncurves = ncurves
        self.pairs_to_check= np.zeros( ( ncurves,
            ncurves), dtype = bool)
        self.close_curves = np.zeros( ( ncurves,
            ncurves), dtype = bool)
        self.nmaster_curves = len(master_curves_ids)
        self.nslave_curves = len(slave_curves_ids)

    #------------------------------------------------------------------------------
    #
    #------------------------------------------------------------------------------
    def construct_storage_arrays(self):
        ncurves = self.ncurves
        if self.contact_type == 0 :
            self.h = np.nan * np.ones(self.close_curves.shape +\
                    (6,), dtype = float)
            self.gN = 1 * np.ones(self.close_curves.shape , dtype = float)
            if self.enforcement == 0:
                self.kN = self.kNmin * np.ones(self.close_curves.shape , dtype = float)
            elif self.enforcement == 1:
                self.dofs_LM = -999 * np.ones(self.close_curves.shape , dtype = int)
                self.LM = np.nan * np.ones(self.close_curves.shape , dtype = float)
                self.activeLM = np.zeros(self.close_curves.shape , dtype = bool)
            else:
                raise NotImplementedError
        else:
            assert self.is_set_parameter_weak_enforcement

            xi_lim = np.linspace(0, 1, self.nII[0] + 1)
            theta_lim = np.linspace(0, 2 * np.pi, self.nII[1] + 1)
            if self.nII[0] == 1:
                self.xi_lim= array([xi_lim])
            else:
                self.xi_lim =  np.vstack((xi_lim[:-1], xi_lim[1:])).T

            if self.nII[1] == 1:
                self.theta_lim= array([theta_lim])
            else:
                self.theta_lim =  np.vstack((theta_lim[:-1],
                    theta_lim[1:])).T


            # https://stackoverflow.com/questions/29839350/numpy-append-vs-python-append
            # way faster to deal with list and convert to np.array if needed

            """
            # list of int
            self.ID_slave = np.array([], dtype = int)
            # list of list of int
            self.ID_master= np.array([], dtype = int)
            # list of np.array of dim GP_xi * GP_theta
            self.gN =  np.array([], dtype = float)
            # list of array of dim dim GP_xi * GP_thetal * 4
            self.h = np.array([],  dtype = float)
            # list of array of dim dim GP_xi * GP_theta
            self.wGP = np.array([],  dtype = float)
            # list of bool
            self.active_set = np.array([], dtype = np.bool)
            """
            for attr in 'ID_slave ID_master ID_master_Ctr gN \
                    active_cells KSI_Cells CON_Cells hCtr h active_set'.split(' '):
                setattr(self, attr, [])
            """
            # list of int
            self.ID_slave = []
            # list of list of int
            self.ID_master= []
            # list of np.array of dim GP_xi * GP_theta
            self.gN = []
            # list of array of dim dim GP_xi * GP_thetal * 4
            self.h = []
            self.active_cells = []
            """

            if self.enforcement == 1:
                # list of array of dim Nctc * nLM_per_ctc
                self.LM =  [] #np.array([], dtype = float)
                # list of array of dim Nctc * nLM_per_ctc
                self.dofs_LM = np.array([], dtype = int)
            elif self.enforcement == 0:
                self.kN =  [] # np.array([], dtype = float)
            else:
                raise NotImplementedError

            # list of float
            self.weak_enforcement= np.array([], dtype = float)
            self.weak_constraint= np.array([], dtype = float)
    #------------------------------------------------------------------------------
    #
    #------------------------------------------------------------------------------
    def set_penalty_options(self,
            method_regularization,
            maxPenAllow,
            kNmin, coeff_mul) :
        assert maxPenAllow < 0
        self.maxPenAllow = maxPenAllow
        assert self.enforcement == 0, 'Only for penalty like method'
        assert method_regularization == 0 or method_regularization == 1
        self.method_regularization = method_regularization
        assert isinstance(maxPenAllow, float) and maxPenAllow<0
        self.maxPenAllow = maxPenAllow
        # coeff_mul can have 2 different uses: whether it is the constant value that mutliply wach
        # time the stiffness of the penalty springs or, it is used to ensure that there is some
        # minimum increase of the penalty stiffness when a more elaborate regularization is used
        assert coeff_mul > 1, 'One must increase the stiffness of the penalty springs. Hence\
            must be >1'
        self.Multiplier_Pen = coeff_mul
        self.kNmin = kNmin


    #------------------------------------------------------------------------------
    #
    #------------------------------------------------------------------------------
    def set_critical_penetration(self, critical_penetration):
        assert critical_penetration < 0
        self.critical_penetration = critical_penetration



    #------------------------------------------------------------------------------
    #
    #------------------------------------------------------------------------------
    def new_el(self, slave, active_cells,\
            IDM, hCtr, KSI_Cells, CON_Cells ):
        self.active_set.append(slave)
        self.active_cells.append(active_cells)
        self.ID_master_Ctr.append(IDM)
        self.hCtr.append(hCtr)
        self.KSI_Cells.append(KSI_Cells)
        self.CON_Cells.append(CON_Cells)

        new_act_cell = active_cells.shape[0]
        self.ID_master.append(-999* ones((new_act_cell ,  self.nxiGQP, self.nthetaGQP), dtype = np.int))
        self.h.append(np.nan * ones((new_act_cell, self.nxiGQP, self.nthetaGQP,4), dtype = np.float))
        self.gN.append(np.nan *ones((new_act_cell, self.nxiGQP, self.nthetaGQP), dtype = np.float))
        self.kN.append(self.kNmin)


        nAct = len(self.active_set)
        # list of arrays
        for attr in 'active_set active_cells ID_master_Ctr hCtr KSI_Cells CON_Cells ID_master h gN'.split(' '):
            assert len(getattr(self, attr)) == nAct


    #------------------------------------------------------------------------------
    #
    #------------------------------------------------------------------------------
    def update_LM_value(self, du):
        """
        called inside the NR procedure
        """
        active_LM = np.where(self.active_set)
        self.LM[active_LM] += du[self.dofs_LM[active_LM]]
    #------------------------------------------------------------------------------
    #
    #------------------------------------------------------------------------------
    def deactivateLM(self,ids):
        # ids is just an index to make the implementation adaptable
        # for any type of implementation
        assert self.enforcement == 1, 'This method is made for LM enforcement of the constraints'
        print('DEACTIVATION')
        assert np.all(self.active_set[ids]==True), 'LM activated was already inactive'
        self.active_set[ids]=False
        self.dofs_LM[ids] = -999
        self.LM[ids] = np.nan

    #------------------------------------------------------------------------------
    #
    #------------------------------------------------------------------------------
    def activateLM(self,ids):
        assert self.enforcement == 1, 'This method is made for LM enforcement of the constraints'
        print('ACTIVATION')
        assert not np.any(self.active_set[ids] == True), 'One of the LM activated was already active !'
        self.active_set[ids] = True
        self.LM[ids] =self.default_value_LM

    #------------------------------------------------------------------------------
    #
    #------------------------------------------------------------------------------
    def PenaltyRegularization(self ):
        """
        perform the penalty regularization if a linear penalty law is used
        If a regularization has been performed return true
        """
        Method = self.method_regularization
        maxPenAllow = self.maxPenAllow
        mul = self.Multiplier_Pen
        IsCorrectPenStiff = True
        """
        if np.argwhere(self.active_set).shape[0] > 0:
            set_trace()
        """

        """
        if self.contact_type == 0 :
            ToRegularize = np.where(self.gN < maxPenAllow)
        else:
            # if one of the GQP is too much penetrated, we increase
            # the penalty for all the GQP
        """
        for ii in range( len(self.gN)):
            ToIncr = np.argwhere(self.gN[ii] < maxPenAllow)
            if ToIncr.shape[0] > 0:
                print('Some increase penalty stiff')
                if Method == 0:
                    # multiply the stiffness in a naive manner
                    assert mul > 0
                    if self.contact_type == 0:
                        self.kN[ToIncr] *= mul
                        IsCorrectPenStiff = False
                    elif self.contact_type == 1:
                        # there is only one kN per pair of curves but
                        # several GQP and possible several integration
                        # intervals
                        self.kN[ii] *= mul
                        """
                        pen_adjusted = 0
                        # a slave curve can be in contact only once. This allows an efficient
                        # sorting of the intergation intervals that might need a regulariaztion if the
                        # constraint is too much violated in the average sense !
                        # plus, with unique we are sure to only regularize once each interval
                        for index in np.unique(ToIncr[:,:3], axis =0):
                            if np.average(self.gN[tuple(index)]) < self.maxPenAllow:
                                self.kN[tuple(index)] *= ( 1 + mul/(self.nxiGQP * self.nthetaGQP))
                                pen_adjusted += 1
                                IsCorrectPenStiff = False
                        print('stiff of %d elements adjusted'%(pen_adjusted))
                        """

                    else:
                        raise NotImplementedError
                elif Method == 1:
                    mul_min =\
                        np.min(\
                        (self.gN[ToIncr] * (maxPenAllow)**(-1),
                        mul * ones(ToIncr.shape[0])), axis = 0)
                    # penalty regularization like Durville
                    if self.contact_type == 0:
                        self.kN[ToDecr] *= mul_min
                    elif self.contact_type == 1:
                        # there is only one kN per pair of curves but
                        # several GQP and possiblu several integration
                        # intervals
                        self.kN[ToDecr[:-2],] *= mul_min
                    else:
                        raise NotImplementedError
                else:
                    raise NotImplementedError

            """
            if ToDecr[0].shape[0] > 0:
                print('Some Decrease penalty stiff')
                #corrected = np.max(self.kNmin,\
                #        0.1 * ones(ToDecr.shape[0]), axis = 0)
                corrected = self.kNmin
                # penalty regularization like Durville
                if self.contact_type == 0:
                    self.kN[ToDecr] = corrected
                elif self.contact_type == 1:
                    # there is only one kN per pair of curves but
                    # several GQP and possiblu several integration
                    # intervals
                    self.kN[ToDecr[:-2],] = corrected
                else:
                    raise NotImplementedError
            """

            display_message = False
            if display_message:
                max_penet = np.min(self.gN)
                if max_penet < self.critical_penetration:
                    raise MaximumPenetrationError
                print('MAXIMUM PENETRATION IS ' + repr(-np.min(self.gN)))
                print('MAXIMUM PEN STIFF IS ' + repr(np.max(self.kN)))
            else:
                if len(self.gN[ii]) > 0:
                    if np.min(self.gN[ii]) <  self.critical_penetration:
                        raise MaximumPenetrationError

        assert not np.any(np.asarray(self.kN) / self.kNmin > 1e7),\
        'relative increase of the penalty stiffness too important'

        return IsCorrectPenStiff

    #------------------------------------------------------------------------------
    #
    #------------------------------------------------------------------------------
    @property
    def epsilon(self):
        assert self.value_epsilon_set, 'value of epsilon not set'
        return self._epsilon
    #------------------------------------------------------------------------------
    #
    #------------------------------------------------------------------------------
    @epsilon.setter
    def epsilon(self, value):
        assert value > 0
        self.value_epsilon_set = True
        # without underscore infiinte number of recursion
        self._epsilon = value

    #------------------------------------------------------------------------------
    #
    #------------------------------------------------------------------------------
    def regenerate_dofs(self, dofs_start):
        # this must be run when the Active set has been updated in case the LM method has been
        # chosen. The dofs associated with any LM may change but its value is kept in memory which
        # is very practical when we go from one time step to the next one
        assert not self.enforcement == 0
        assert type(dofs_start) == np.int
        # first of all the inactive constraints will have no attributed dof
        #Inact = np.where(self.active_set == False)
        # cannot use np.nan with integers. This value means the LM is
        # not activated
        #assert np.all(self.dofs_LM[Inact] == -999)
        act = np.where(self.active_set)
        nLM = act[0].shape[0]
        self.dofs_LM[act] =\
            np.arange(dofs_start, dofs_start + nLM, dtype =
                    int)

    #------------------------------------------------------------------------------
    #
    #------------------------------------------------------------------------------



class ContactWithSphere(ContactTable):

    def __init__(self, enforcement,\
            method,
            **kwargs):

        contact_type = 1
        ContactTable(enforcement, method, **kwargs)
