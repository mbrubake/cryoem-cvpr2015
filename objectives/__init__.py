from sum import SumObjectives

from null import NullPrior

from car import CARPrior
from sparsity import SparsityPrior
from gaussian import GaussianPrior
from negative import NegativePenalty
from mixexpgauss import MixExpGaussPrior
from centered import CenteredPenalty
from likelihood import UnknownRSLikelihood

def eval_objective(estr):
    return eval(estr)
