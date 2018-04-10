import numpy as np
from netCDF4 import Dataset

cimport numpy as np
cimport cython

class Test():
    def __init__(self):
        return

    def calc_sum(self, np.ndarray[float, ndim=3] a):

        cdef int i, j, n
        cdef float a_cum

        a_cum = 0
        for i in range(a.shape[0]):
            for j in range(a.shape[1]):
                for n in range(a.shape[2]):
                    a_cum = a_cum+(a[i,j,n])
        return a_cum

    def useless_cython2(self, year):
        from netCDF4 import Dataset
        f = Dataset('air.sig995.'+year+'.nc')
        a = f.variables['air'][:]
        a_cum = calc_sum(a)

        d = np.array(a_cum)
        d.tofile(year+'.bin')
        print(year)
        return d
