
# class Error is derived from super class Exception
class Error(Exception):

    # Error is derived class for Exception, but
    # Base class for exceptions in this module
    pass

class FailedConvergence(Error):

    # Raised when an operation attempts a state
    # transition that's not allowed.
    def __init__(self, msg):
        # Error message thrown is saved in msg
        self.msg = msg


class Failure_Adaptive_TS(Error):

    # Raised when an operation attempts a state
    # transition that's not allowed.
    def __init__(self, t):
        # Error message thrown is saved in msg
        self.msg = "No convergence of the solver at t = " + str(t)

class MaximumPenetrationError(Error):
    def __init__(self):
        print('CRITICAL PENETRATION ')



