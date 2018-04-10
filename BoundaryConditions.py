from __future__ import division
import numpy as np
from numpy import array, zeros, where, ones, any, diff, asarray
from pdb import set_trace
from itertools import count
from scipy.spatial import Delaunay
from scipy.sparse import coo_matrix, lil_matrix


#-------------------------------------------------------------------------------
# For Static Problems only
#-------------------------------------------------------------------------------
class BC():
    """
    Contain information on a boundary condition applied in dynamic or static analysis
    """
    def __init__( self, Type, Values, **kwargs):
        """
        Create BC instance. Bcs must be applied in any cases USING THE pT3d OBJECTS
        CAPITAL TO UNDERSTAND THE IMPLEMENTATION
        Type 1 : Dirichlet BCs
        Type 2 : Neuman BCs
        The Type 0 is left alone on purpose because it will be used to spot the dofs without BCs
        """
        self._ID = self._ids.next()
        if Type == 'Dirichlet' or Type == 'Neuman' or Type == 'Velocity':
            assert kwargs.has_key('locDofs')
            assert kwargs.has_key('PtsID')
            ############################################ ##########################
            self.Values = Values
            PtsID = kwargs.get('PtsID')
            locDofs= kwargs.get('locDofs')
            dofs = self.FEAss.getDOFSFromNodesID(PtsID)
            if not isinstance(Values, np.ndarray):
                Values = asarray(Values)
            if not isinstance(locDofs, np.ndarray):
                locDofs = asarray(locDofs)
            assert where(locDofs)[0].shape[0] == Values.shape[0]
            # extract only the local dofs of interest for each selected point
            if dofs.ndim == 2 :
                dofs = dofs[:, where(locDofs)[0]]
            elif dofs.ndim == 1:
                # only one pt is subject to BC
                dofs = dofs[where(locDofs)[0]]
            else:
                raise ValueError('dofs shape is not correct')
            # broadcasting the values !
            self.Values = (Values * np.ones_like(dofs)).ravel()
            self.dofs = (dofs).ravel()
            assert self.Values.shape == self.dofs.shape

            if Type == 'Dirichlet':
                self.Type = 1
            elif Type == 'Neuman':
                self.Type = 2
            elif Type == 'Velocity':
                self.Type = 4

        elif Type == 'DispRigidSphere':
            # can only be a displacement
            self.Type = 3
            Values = asarray([Values]).ravel()
            assert Values.shape[0] == 3
            assert not np.array_equal(Values, zeros(3))
            self.Values = Values

        else:
            raise ValueError('Incorrect instantiation of BC object')



#-------------------------------------------------------------------------------
# fOR sTATIC pROBLEMS ONLY
#-------------------------------------------------------------------------------

class Static_Loading():
    """
    A simple way of handling BCs is to introduce some fictitious time (from t = 0s to t=1s for
    example) and also a time stepping that will allow to introduce more or less time step per phase
    of loading
    TODO : A riks solver solver could handle that in a cleaner way
    """
    def __init__(self, t0s, tmaxs, types ,dofs, values, Interval, timestepping,  **kwargs):
        nBCs = t0s.shape[0]
        try:
            for arr in [t0s, tmaxs, types, dofs, values]:
                assert arr.shape == (nBCs,)
        except AssertionError:
            raise AssertionError('incorrect size of one of the input for construction of BCs')
        """
        Type 1 : Dirichlet BCs
        Type 2 : Neuman BCs
        Type 3 : moving a rigid obstacle
        """
        assert types.dtype == np.int
        # different BCs might not affect the same number of dofs. Thus the array of dofs affected
        # has a second dimension that varies
        assert dofs.dtype == np.object
        self.t0s = t0s
        self.tmaxs = tmaxs
        self.types = types
        self.dofs = dofs
        self.values = values
        # the user decides indirectly what will be the DELt that will
        # be used in the first place
        assert (Interval.shape == (2,))
        self.DELt = (Interval[1] - Interval[0]) / timestepping
        assert Interval[0] == 0.
        # the last time interval must give the maximum time of the analysis
        self.tfinal = Interval[-1]
        assert np.all(self.t0s >= 0)
        assert np.all(self.tmaxs <= self.tfinal )

    #-------------------------------------------------------------------------------
    #
    #-------------------------------------------------------------------------------
    def get_duG_dfext(self, t, tn, ndofs):
        assert t >=0
        assert tn >=0
        # used to check that a single dofs is not subjected to 2 bcs
        dofs_wt_bcs = zeros((ndofs,), dtype = np.bool_)
        # increment in displacement
        duG = zeros((ndofs,), dtype = np.float_)
        # increment in external force
        dfext = zeros((ndofs,), dtype = np.float_)
        # dofs subjected to Dirichlet BCs
        di = []

        for BC_idx in range(self.t0s.shape[0]):
            t0 = self.t0s[BC_idx]
            tmax = self.tmaxs[BC_idx]
            if t0 <= t <= tmax:
                if tn <= t0:
                    # the BC was not yet active at the previous TS
                    del_lbda = (t - t0) / (tmax - t0)
                else:
                    del_lbda = (t - tn) / (tmax - t0)
                if self.types[BC_idx] == 3:
                    # move rigid obstacle
                    assert self.values[BC_idx].shape == (3,)
                    self.CurrentSphereDisplacement = del_lbda * self.values[BC_idx]
                elif self.types[BC_idx]==1 or self.types[BC_idx]==2:
                    dofs = self.dofs[BC_idx]
                    # check that the dofs were not subjected to BCs already
                    assert np.all(dofs_wt_bcs[dofs] == 0)
                    dofs_wt_bcs[dofs] = True
                    if self.types[BC_idx]==1:
                        # Dirichlet
                        duG[dofs] = del_lbda * self.values[BC_idx]
                        di.append(dofs)
                    elif self.types[BC_idx]==2:
                        # Neuman
                        dfext[dofs] = del_lbda * self.values[BC_idx]
                    else:
                        raise ValueError('This type of BC is not applicable for static anaylsis')
        if len(di) > 0:
            di = np.concatenate([dik.ravel() for dik in di])
        else:
            di = array([])
        # generate the free indices for the current step
        fr = np.delete(np.arange(ndofs), (di))
        assert dfext.shape == duG.shape
        assert di.shape[0] + fr.shape[0] == dfext.shape[0]
        return duG, dfext, di, fr



#-------------------------------------------------------------------------------
# For Dynamic Problems only
#-------------------------------------------------------------------------------
class BC_DYN(BC):
    """
    We just add t0 and tEnd to make a BC suitable for implementation in dynamic problems
    """
    _ids = count(0)
    _Ndofs = 0
    def __init__( self, Type, Values, t0, tEnd, FEAss, **kwargs):
        """
        Create BC instance. Bcs must be applied in any cases USING THE pT3d OBJECTS
        CAPITAL TO UNDERSTAND THE IMPLEMENTATION
        Type 1 : Dirichlet BCs
        Type 2 : Neuman BCs
        The Type 0 is left alone on purpose because it will be used to spot the dofs without BCs
        """
        self._ID = self._ids.next()
        self.FEAss = FEAss
        # init parent class
        BC.__init__(self, Type, Values, **kwargs)
        assert t0 <= tEnd
        self.t0 = t0
        self.tEnd = tEnd

#-------------------------------------------------------------------------------
#
#-------------------------------------------------------------------------------
class ComplexLoading_DYN():
    def __init__(self, BCs):
        """
        BCs for dynamic problems
        For the BC instances, I just have to add t0 and tEnd to make them dynamic !!
        """
        # check the input
        for BC in BCs:
           assert BC.__class__.__name__ == 'BC_DYN'
        self.BCs = BCs

#-------------------------------------------------------------------------------
#
#-------------------------------------------------------------------------------


    def getCurrentBCs( self, t , ndofs, tM1 = np.nan):
        """
        use the current time to compute the load factor
        for each BC, it t0 < t < tend, then lmabda = (1 / (tend - t0) * t - (t0/ (tend - t0)))
        else --> 0
        """

        duG = zeros(ndofs, dtype=np.float)
        fext = zeros(ndofs, dtype=np.float)
        v = zeros(ndofs, dtype = np.float)
        # list of dofs subjected to Dirichlet BCs
        di = []
        # dofs with velocity imposed
        vimp = np.empty(0, dtype=np.int)
        di = np.empty(0, dtype=np.int)

        if t != 0:
            try:
                assert not np.isnan(tM1)
            except AssertionError:
                raise ('tnM1 should ne provided if t > 0 !')

        for BC in self.BCs:
            if BC.t0 <= t <= BC.tEnd:
                # the load is applied linearly as a function of time between t0 and tEnd
                # the corresponding loading factor is lbda
                if t == 0 :
                    del_lbda = 0
                else:
                    # pas besoin d'ordonnees a l'origine vu que l'on soustrait la valeur d'une
                    # droite entre deux points
                    # del_lbda = (BC.tEnd - BC.t0)**(-1) * (t - t_discr[n-1]) - (BC.t0 * (BC.tEnd - BC.t0) )
                    del_lbda = (BC.tEnd - BC.t0)**(-1) * (t - tM1)
                if BC.Type == 1:
                    # Dirichlet BCs
                    duG[BC.dofs] = del_lbda * BC.Values
                    di = np.concatenate((di, BC.dofs))
                elif BC.Type == 2:
                    # Neuman BCs
                    fext[BC.dofs] = del_lbda * BC.Values
                elif BC.Type == 3:
                    # moving the rigid sphere
                    try:
                        assert BC.FEAss.HasSphere
                    except AssertionError as e:
                        raise e('Model must have a rigid sphere attached')
                    self.MoveSphereDYN( BC, t, BC.FEAss.Sphere, tM1)

                elif BC.Type == 4:
                    # the velocity of certain node is imposed
                    vimp = np.concatenate((vimp, BC.dofs))
                    v[BC.dofs] = BC.Values
                else:
                    raise NotImplementedError()

        di = np.asarray(di).flatten()
        #assert that the dirichlet dofs and the dofs with velocity imposed do not intersect
        if vimp.shape[0] != 0:
            assert np.intersect1d(di, vimp).shape[0] == 0

        # generate free indices
        fr = np.delete(np.arange(ndofs), (di))
        return (fext, duG, di, fr, v, vimp)


#-------------------------------------------------------------------------------
#
#-------------------------------------------------------------------------------


    def MoveSphereDYN(self,BC, t, Sphere, tM1 ):
        assert BC.Type == 3
        if BC.t0 <= t[n] <= BC.tEnd:
            if n == 0:
                del_lbda = (BC.tEnd - BC.t0)**(-1) * t
            else:
                del_lbda = (BC.tEnd - BC.t0)**(-1) * (t - tM1)
        else:
            del_lbda = 0
        Sphere.Move(del_lbda * BC.Values)


#-------------------------------------------------------------------------------
#
#-------------------------------------------------------------------------------
