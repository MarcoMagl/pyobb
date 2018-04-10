import cython
cimport numpy as np
import numpy as np
from libc.stdlib cimport malloc, free
from libc.math cimport sin,cos, sqrt, acos, atan2, pow
cimport cython
from numpy.linalg import norm

DTYPE = np.int
# "ctypedef" assigns a corresponding compile-time type to DTYPE_t. For
# every type in the numpy module there's a corresponding compile-time
# type with a _t-suffix.
ctypedef np.int_t DTYPE_t
ctypedef np.double_t double_t

# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%#
cdef extern void SurfPointFUN(double v[5001],double X[9],double u[9]
     ,double Vec[9],double ta[2][3],double tb[2][3],double (*a),double (*b),double
     (*xi),double (*theta),double (*alph),double s[3])

cdef extern void LocalDetectAtGQPFUN(double v[5005],double X[2][9],double u[2][9]
     ,double Vec[2][9],double ta[4][3],double tb[4][3],double a[2],double b[2]
     ,double (*alph),double (*gN),double (*xiGQP),double (*thetaGQP),double (*WxiGQP
     ),double (*WthetaGQP),double hFG[2],double h[2],double (*WEIGTHGQP),int
     (*ExitCode))



# ---------------------------------------------------------#
# sample points on a surface
# ---------------------------------------------------------#

def samplePointsOnSurface(
    np.ndarray[double, ndim=1, mode="c"] X not None,
    np.ndarray[double, ndim=1, mode="c"] u not None,
    np.ndarray[double, ndim=1, mode="c"] Vec not None,
    np.ndarray[double, ndim=2, mode="c"] t1 not None,
    np.ndarray[double, ndim=2, mode="c"] t2 not None,
    double a, double b, double alpha,
    np.ndarray[double, ndim=1, mode="c"] xis not None,
    np.ndarray[double, ndim=1, mode="c"] thetas not None,
    np.ndarray[double, ndim=3, mode="c"] POS not None,
    ):

    cdef:
        double v[5001];
        int i;
        int j;
        int nxis = xis.shape[0]
        int nthetas = thetas.shape[0]

    for i in range(nxis):
        for j in range(nthetas):
            SurfPointFUN(v, &X[0],&u[0], &Vec[0],
                    <double (*)[3]>&t1[0,0],
                    <double (*)[3]>&t2[0,0],
                    &a, &b,
                    &xis[i],
                    &thetas[j],
                    &alpha,
                    <double (*)>&POS[i,j,0])
    # NO NEED TO RETURN THE ARRAY !!!

#------------------------------------------------------------------------------
#
#------------------------------------------------------------------------------
def LocalDetec_1GQP_C(
    np.ndarray[double, ndim=2, mode="c"] X not None,
    np.ndarray[double, ndim=2, mode="c"] u not None,
    np.ndarray[double, ndim=2, mode="c"] v not None,
    np.ndarray[double, ndim=2, mode="c"] t1 not None,
    np.ndarray[double, ndim=2, mode="c"] t2 not None,
    np.ndarray[double, ndim=1, mode="c"] a not None,
    np.ndarray[double, ndim=1, mode="c"] b not None,
    double xiGQP,
    double thetaGQP,
    double WxiGQP,
    double WthetaGQP,
    np.ndarray[double, ndim=1, mode="c"] hFG not None,
    double alpha,
    ):

    cdef:
        double vecAG[5005];
        double h[2]; 
        double gN;
        int i;
        int j;
        int ExitCode;
        double WEIGTHGQP;

    for i in range(2):
        h[i] = 0.;


    LocalDetectAtGQPFUN(
            vecAG,
            <double (*)[9]>&X[0,0], 
            <double (*)[9]>&u[0,0], 
            <double (*)[9]>&v[0,0], 
            <double (*)[3]>&t1[0,0], 
            <double (*)[3]>&t2[0,0], 
            &a[0],
            &b[0],
            &alpha,
            &gN,
            &xiGQP,
            &thetaGQP,
            &WxiGQP,
            &WthetaGQP,
            &hFG[0],
            h,
            &WEIGTHGQP,
            &ExitCode) 

    return (gN, h, WEIGTHGQP , ExitCode)



#------------------------------------------------------------------------------
#
#------------------------------------------------------------------------------
def LocalDetec_1GQP(
    np.ndarray[DTYPE_t, ndim=1] id_curves,
    np.ndarray[DTYPE_t, ndim=2] nPerEl,
    np.ndarray[DTYPE_t, ndim =2] el_per_curves,
    np.ndarray[double, ndim=2, mode="c"] X not None,
    np.ndarray[double, ndim=2, mode="c"] u not None,
    np.ndarray[double, ndim=2, mode="c"] v not None,
    np.ndarray[double, ndim=2, mode="c"] t1 not None,
    np.ndarray[double, ndim=2, mode="c"] t2 not None,
    np.ndarray[double, ndim=1, mode="c"] a not None,
    np.ndarray[double, ndim=1, mode="c"] b not None,
    double xiGQP,
    double thetaGQP,
    double WxiGQP,
    double WthetaGQP,
    np.ndarray[double, ndim=1, mode="c"] hFG not None,
    double alpha
    ):

    id_els = el_per_curves[id_curves,].ravel()
    assert id_els.shape == (4,), 'id_els has not the right size'
    nd_C1 =  nPerEl[id_els[(0,1),]].flatten()[(0,1,3),]
    nd_C2 =  nPerEl[id_els[(2,3),]].flatten()[(0,1,3),]
    nd_C = np.concatenate((nd_C1, nd_C2))

    # (gN, h, WEIGTHGQP , ExitCode)
    return LocalDetec_1GQP_C(\
        X[nd_C,].reshape(2,9),\
        u[nd_C,].reshape(2,9),\
        v[nd_C,].reshape(2,9),\
        t1[id_els,],\
        t2[id_els,],\
        a[id_els[(0,2),]],
        b[id_els[(0,2),]],
        xiGQP,
        thetaGQP,
        WxiGQP,
        WthetaGQP,
        hFG,
        alpha)

# ---------------------------------------------------------#
# get point on smoothed surface 
# ---------------------------------------------------------#
def SurfPoint_C(
    np.ndarray[double, ndim=1, mode="c"] X not None,
    np.ndarray[double, ndim=1, mode="c"] u not None,
    np.ndarray[double, ndim=1, mode="c"] Vec not None,
    np.ndarray[double, ndim=2, mode="c"] t1 not None,
    np.ndarray[double, ndim=2, mode="c"] t2 not None,
     double a, double b, double alpha,
    np.ndarray[double, ndim=1, mode="c"] h not None):

    cdef:
        double v[5001];
        double s[3];
        double xi = h[0];
        double theta = h[1];

    SurfPointFUN(v, &X[0],&u[0], &Vec[0],
            <double (*)[3]>&t1[0,0],
            <double (*)[3]>&t2[0,0],
            &a, &b, &xi, &theta, &alpha, s)

    return s


def get_surf_point(\
        np.ndarray[DTYPE_t, ndim=1] id_els,
        np.ndarray[double, ndim=1, mode="c"] h not None,
        np.ndarray[DTYPE_t, ndim=2] nPerEl,
        np.ndarray[double, ndim=2, mode="c"] X not None,
        np.ndarray[double, ndim=2, mode="c"] u not None,
        np.ndarray[double, ndim=2, mode="c"] v not None,
        np.ndarray[double, ndim=2, mode="c"] t1 not None,
        np.ndarray[double, ndim=2, mode="c"] t2 not None,
        np.ndarray[double, ndim=1, mode="c"] a not None,
        np.ndarray[double, ndim=1, mode="c"] b not None,
        double alpha):

    cdef np.ndarray nd_C =  nPerEl[id_els,].flatten()[(0,1,3),]

    return SurfPoint_C(\
        X[nd_C,].ravel(),\
        u[nd_C,].ravel(),\
        v[nd_C,].ravel(),\
        t1[id_els,],\
        t2[id_els,],\
        a[id_els][0],\
        b[id_els][0],\
        alpha,\
        h)


#------------------------------------------------------------------------------
#
#------------------------------------------------------------------------------


def TransfoQP(\
        double[:] Interval,
        double xiGQP,
        double wiGQP):
    assert Interval.shape[0] == 2, 'Interval should have a length of 2'
    assert Interval[0] < Interval[1],'Incorrect interval'
    assert -1 < xiGQP < 1, 'xi is out of its bound'
    #return  the coordinate of the GQP in the current 1D interval
    # and associated weigth
    return 0.5 * (Interval[0] + Interval[1] + xiGQP * (Interval[1] -
        Interval[0])), 0.5 * (Interval[1] - Interval[0])


#------------------------------------------------------------------------------
#
#------------------------------------------------------------------------------
def sample_surf_point_on_smooth_geo(\
        np.ndarray[DTYPE_t, ndim=1] id_els,
        np.ndarray[double, ndim=1] xi_lim,
        np.ndarray[double, ndim=1] theta_lim,
        np.ndarray[DTYPE_t, ndim=2] nPerEl,
        np.ndarray[double, ndim=2, mode="c"] X not None,
        np.ndarray[double, ndim=2, mode="c"] u not None,
        np.ndarray[double, ndim=2, mode="c"] v not None,
        np.ndarray[double, ndim=2, mode="c"] t1 not None,
        np.ndarray[double, ndim=2, mode="c"] t2 not None,
        np.ndarray[double, ndim=1, mode="c"] a not None,
        np.ndarray[double, ndim=1, mode="c"] b not None,
        double alpha,
        int nS,
        int ntheta):

    """
    # with memoryviews indexing does not work but can be used with other function
    def sample_surf_point_on_smooth_geo(\
            int[:] id_els,
            double[:] xi_lim,
            double[:] theta_lim,
            int[:] nPerEl,
            double[: ,:] X,
            double[:, :] u,
            double[:, :] v,
            double[:, :] t1,
            double[:, :] t2,
            double[:, :] a,
            double[:, :] b,
            double alpha,
            int nS, int ntheta): 
    # version with for loops in full C
    """
    """
    assert id_els.shape == (2,)
    assert xi_lim.shape == (2,)
    assert theta_lim.shape == (2,)
    """
    cdef nd_C = np.zeros(3,  dtype = np.int)
    nd_C = nPerEl[id_els,].flatten()[(0,1,3),]
    cdef pos = np.zeros((nS, ntheta, 3), dtype = np.double)

    samplePointsOnSurface(\
        X[nd_C,].ravel(),\
        u[nd_C,].ravel(),\
        v[nd_C,].ravel(),\
        t1[id_els,],\
        t2[id_els,],\
        a[id_els][0],\
        b[id_els][0],\
        alpha,\
        np.linspace(xi_lim[0], xi_lim[1], nS, dtype=float),\
        np.linspace(theta_lim[0], theta_lim[1], ntheta, dtype =float),\
        pos)
    return pos



#------------------------------------------------------------------------------
#
#------------------------------------------------------------------------------
def narrow_phase_fun(
        int enforcement, 
        int nxiGQP,
        int nthetaGQP,
        np.ndarray[double, ndim=2, mode="c"] xi_lim  not None,
        np.ndarray[double, ndim=2, mode="c"] theta_lim  not None,
        np.ndarray[DTYPE_t, ndim=1] slave_curves_ids,
        np.ndarray[DTYPE_t, ndim =2] close_curves,
        np.ndarray[DTYPE_t, ndim =2] el_per_curves,
        np.ndarray[DTYPE_t, ndim =2] nPerEl,
        np.ndarray[double, ndim=2, mode="c"] X not None,
        np.ndarray[double, ndim=2, mode="c"] u not None,
        np.ndarray[double, ndim=2, mode="c"] v not None,
        np.ndarray[double, ndim=2, mode="c"] t1 not None,
        np.ndarray[double, ndim=2, mode="c"] t2 not None,
        np.ndarray[double, ndim=1, mode="c"] a not None,
        np.ndarray[double, ndim=1, mode="c"] b not None,
        double alpha,
        np.ndarray[double, ndim=5, mode="c"] gN not None,
        np.ndarray[double, ndim=6, mode="c"] h not None,
        np.ndarray[double, ndim=3, mode="c"] LM not None,
        np.ndarray[DTYPE_t, ndim=5, mode="c"] master_ID_forCPP not None
        ):


        cdef np.ndarray sum_w_gn= np.zeros(
                (LM.shape[0], LM.shape[1],LM.shape[2]),
                dtype = np.double)
        cdef np.ndarray sum_w_LM =\
        np.zeros( (LM.shape[0], LM.shape[1],
            LM.shape[2]), dtype = np.double)

        cdef:
            int AS_MODIFIED = False;

        # preallocation
        nuxiGQP, WnuxiGQP =  np.polynomial.legendre.leggauss(nxiGQP)
        nuthetaGQP, WnuthetaGQP =\
        np.polynomial.legendre.leggauss(nthetaGQP)
        cdef:
            int nxi_sampled = 10;
            int ntheta_sampled = 20;
        # the master curve has no integration interval
        cdef np.ndarray xi_lim_mstr =\
                np.array([0,1], dtype = np.double)
        cdef np.ndarray theta_lim_mstr =\
                np.array([0, 2 * np.pi], dtype = np.double)
        # always the same
        cdef np.ndarray xi_mstr = np.linspace(xi_lim_mstr[0],\
                xi_lim_mstr[1],\
                nxi_sampled)
        cdef np.ndarray theta_mstr = np.linspace(theta_lim_mstr[0],\
                theta_lim_mstr[1],\
                ntheta_sampled)

        cdef int slave
        cdef int iii, uu, id_m, ii, jj, mm, nn ;
        cdef float sum_w_gn_ijkl, sum_w_LM_ijkl

        cdef int nslave_ids = len(slave_curves_ids);

        for iii in range(nslave_ids):
            slave = slave_curves_ids[iii];
            master_close = np.argwhere(close_curves[slave])
            if master_close.shape[0] >0:
                samp_on_master_candi =\
                    np.zeros( (len(master_close),\
                    nxi_sampled , ntheta_sampled,\
                        3), dtype = np.float)

                id_els_sl = el_per_curves[slave]

                for uu, id_m, in enumerate(master_close):
                    id_els_m = el_per_curves[(id_m),].ravel()
                    samp_on_master_candi[uu]=\
                        sample_surf_point_on_smooth_geo(\
                            id_els_m,\
                            xi_lim_mstr,\
                            theta_lim_mstr,
                            nPerEl, X, u, v,
                            t1, t2,
                            a, b, alpha,
                            nxi_sampled,
                            ntheta_sampled )

                # iterate over the slave integration domains
                for ii in range(xi_lim.shape[0]):
                    for jj in range(theta_lim.shape[0]):
                        xi_lim_slv = xi_lim[ii]
                        theta_lim_slv = theta_lim[jj]
                        index_slave_II = (slave, ii,jj)

                        if enforcement == 1 :
                            sum_w_gn_ijkl = 0
                            sum_w_LM_ijkl = 0

                        for mm in range(nxiGQP):
                            xiGQP, WxiGQP = TransfoQP(\
                                     xi_lim_slv,
                                     nuxiGQP[mm], WnuxiGQP[mm])
                            for nn in range(nthetaGQP):
                                #index_GQP = (id_curves[0], ii, jj,  mm,nn)
                                # fixed slave coordinates
                                thetaGQP, WthetaGQP = TransfoQP(\
                                         theta_lim_slv,
                                         nuthetaGQP[nn],
                                         WnuthetaGQP[nn])
                                hSl = np.array([xiGQP, thetaGQP])

                                xGQP = get_surf_point(\
                                                id_els_sl,
                                                hSl,
                                                nPerEl, X, u, v,
                                                t1, t2,
                                                a, b, alpha)

                                d_FG = np.inf

                                for uu in range(master_close.shape[0]):
                                    Dist = norm((samp_on_master_candi[uu]- xGQP), axis = 2)
                                    idx = np.unravel_index( Dist.argmin(), (samp_on_master_candi[uu].shape[:2]))
                                    if Dist[idx] < d_FG:
                                        d_FG = Dist[idx]
                                        idx_dmin = uu
                                        idx_hmin = idx

                                hFG = np.array(\
                                        [xi_mstr[idx_hmin[0]],\
                                        theta_mstr[idx_hmin[1]]])
                                current_mcurve= int(master_close[idx_dmin])
                                current_curves = np.array([slave, current_mcurve])
                                """
                                print(xiGQP)
                                print(WxiGQP)
                                print(thetaGQP)
                                print(WthetaGQP)
                                """

                                (gNgqp, hSol,wi_gqp,ExitCode) = \
                                    LocalDetec_1GQP(\
                                    current_curves, nPerEl, el_per_curves,\
                                    X, u, v,
                                    t1, t2,
                                    a, b,
                                    xiGQP, thetaGQP,WxiGQP, WthetaGQP, hFG, alpha )


                                index_GQP = (current_curves[0], ii, jj,  mm,nn)
                                try:
                                    assert ExitCode == 1
                                except AssertionError:
                                    print('local scheme failed')
                                    print('index of GQP is:')
                                    print(index_GQP)

                                h[index_GQP] = hSol
                                gN[index_GQP] = gNgqp
                                master_ID_forCPP[index_GQP] = current_mcurve

                                if enforcement == 1:
                                    sum_w_gn[index_GQP[:3]] += wi_gqp * gNgqp
                                    sum_w_LM[index_GQP[:3]] += wi_gqp * LM[index_GQP[:3]] 
        return sum_w_gn, sum_w_LM
