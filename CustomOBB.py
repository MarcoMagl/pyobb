from pyobb.obb import OBB
import numpy as np


def CustomOBB(OBB):
    def __init__(self, triad, points ):
        assert triad.shape == (3,3)
        OBB.__init__(self)
        self.rotation = array(triad )

        # apply the rotation to all the position vectors of the array
        # TODO : this operation could be vectorized with tensordot
        p_primes = asarray([self.rotation.dot(p) for p in points])
        self.min = npmin(p_primes, axis=0)
        self.max = npmax(p_primes, axis=0)



