from distutils.core import setup
from distutils.extension import Extension
from Cython.Distutils import build_ext
import numpy

setup(
    cmdclass={
        'build_ext': build_ext},
    ext_modules=[
        Extension(
            "narrow_phase",
            sources=[
                "narrow_phase.pyx",
                "SurfPoint.c",
                "/Users/marco.magliulo/Softwares/AceFEM_Trial_MacOSX/AceGen/Include/C/SMSUtility.c"],
            include_dirs=[
                numpy.get_include(),
                '/Users/marco.magliulo/Softwares/AceFEM_Trial_MacOSX/AceGen/Include/C','.'])])

