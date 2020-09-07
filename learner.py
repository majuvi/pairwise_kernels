import numpy as np
from scipy.sparse.linalg import LinearOperator, minres
from rlscore.learner.cg_kron_rls import CGKronRLS
from rlscore.measure import cindex
from matvec import pko_kronecker, pko_cartesian, pko_linear, pko_poly2d, pko_symmetric, pko_antisymmetric, pko_mlpk
from validation import setting_kernels_heterogeneous, setting_kernels_homogeneous

# Save AUC in validation set(s) over iterations by calling corresponding matvec methods
class SaveAUC(object):

    def __init__(self, Ys, mvs):
        self.Ys = Ys
        self.mvs = mvs
        self.aucs = []
        self.iterations = 0

    def __call__(self, A):
        aucs = [cindex(Y, mv.matvec(A)) for Y, mv in zip(self.Ys, self.mvs)]
        self.aucs.append(aucs)
        self.iterations += 1
        print("+", end="")

# RLScore object rls assumes the callback object cb is called by cb.callback(rls)
class RLScoreSaveAUC(SaveAUC):

    def callback(self, rls):
        self(rls.A)

    def finished(self, rls):
        print()

# Raise in the callback function to stop penalty minimization iterations
class StopIteration(Exception):
    pass

# Minimum residual iteration with coefficient vector A assumes the callback object cb is called by cb(A)
class CheckStop(object):

    def __init__(self, Y, mv, limit=10):
        self.Y = Y
        self.mv = mv
        self.best_auc = 0
        self.iterations = 0
        self.no_updates = 0
        self.limit = limit
        self.aucs = []

    def __call__(self, A):
        P = self.mv.matvec(A)
        current_auc = cindex(self.Y, P)
        self.aucs.append(current_auc)
        if current_auc > self.best_auc:
            self.iterations += (self.no_updates + 1)
            self.best_auc = current_auc
            self.no_updates = 0
        else:
            self.no_updates += 1
        if self.no_updates > self.limit:
            raise StopIteration()

# RLScore object rls assumes the callback object cb is called by cb.callback(rls)
class RLScoreStop(CheckStop):

    def callback(self, rls):
        self(rls.A)

    def finished(self, rls):
        pass

