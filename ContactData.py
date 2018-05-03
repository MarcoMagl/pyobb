import numpy as np

class ContactData(object):
    def __init__(self, slave, cells, nGP, master, xi, ct, gN, value):
        self.slave = slave
        #LM or kN
        self.value = value
        self.ct = ct
        self.ncells = cells.shape
        self.nGP = nGP.shape
        self.cells = cells
        self.master = master
        self.gN= gN
        self.xi = xi
        self.check_consistency_data()

    def check_consistency_data(self):
        assert isinstance(self.slave, int)
        assert isinstance(self.ct, int)
        assert self.cells.dtype == int
        assert self.cells.ndim == 1
        base_shape = (self.ncells, self.nGP[0], self.nGP[1])
        assert self.master.shape == base_shape
        assert self.xi.shape == base_shape + (2,)
        assert self.gN.shape == base_shape


    def query_cell_info(qcell):
        indx = np.argwhere(self.cells == qcell)
        if indx.shape[0] == 0:
            return 0 , None
        else:
            return self.master[indx], self.xi[indx], self.gN[indx]


    def add_cell_to_data(self, cell, gN, xi):
        raise NotImplementedError












