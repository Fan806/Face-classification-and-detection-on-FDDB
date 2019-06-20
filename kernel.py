import numpy as np
import numpy.linalg as la


class Kernel(object):
    """Implements list of kernels from
    http://en.wikipedia.org/wiki/Support_vector_machine
    """
    def linear(self):
        return lambda x, y: np.inner(x, y)

    def gaussian(self, sigma):
        return lambda x, y: \
            np.exp(-np.sqrt(la.norm(x-y) ** 2 / (2 * sigma ** 2)))
