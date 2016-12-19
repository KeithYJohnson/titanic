from ipdb import set_trace as st
from create_simple_nn_params import *
import numpy as np

def grad_check(costfn, gradfn, *params):
    cost = costfn(*params)
    analytical_grad  = gradfn(*params)

    numerical_grad = np.zeros(analytical_grad.shape)
    epsilon = 1e-4
    for index, x in np.ndenumerate(numerical_grad):
        perturb = np.zeros(analytical_grad.shape)
        perturb[index] = epsilon

        #Why?  Cuz *** TypeError: 'tuple' object does not support item assignment
        plus_params = list(params)
        plus_perturbed = plus_params[0] + perturb
        plus_params[0] = plus_perturbed

        minus_params = list(params)
        minus_perturbed = minus_params[0] - perturb
        minus_params[0] = minus_perturbed

        Jplus = costfn(*plus_params)
        Jminus = costfn(*minus_params)

        weight_grad = (Jplus - Jminus) / (2 * epsilon)
        numerical_grad[index] = weight_grad

        elementwise_error_diff(weight_grad, analytical_grad, index)


def elementwise_error_diff(numerical_grad, analytical_grad, index, threshold=1e-5):
    reldiff = abs(numerical_grad - analytical_grad[index]) / max(1, abs(numerical_grad), abs(analytical_grad[index]))
    if reldiff > threshold:
        print("Gradient check failed.")
        print("gradient error found at index %s" % str(index))
        print("Your gradient: %f \t Numerical gradient: %f" % (analytical_grad[index], numerical_grad))
        return

## SEE http://cs231n.github.io/neural-networks-3/
def relative_error_diff(numerical_grad, analytical_grad):
    diff = abs(numerical_grad - analytical_grad) / np.maximum(1, abs(numerical_grad), abs(analytical_grad))
    print("relative_error_diff: ", diff)


def norm_diff(numerical_grad, analytical_grad, max_diff):
    print('\nnp.linalg.norm(x - analytical_grad)/np.linalg.norm(x + analytical_grad);')
    print('should be less than the max_diff of: ', max_diff)
    diff = np.linalg.norm(numerical_grad - analytical_grad) / np.linalg.norm(numerical_grad + analytical_grad);
    print('diff: ', diff, '\n\n')
if __name__ == '__main__':
    # Simple example to test that the gradient checking code runs correctly
    print('running simple grad check')
    grad_check(
        lambda x: np.sum(x ** 2),
        lambda x: x * 2,
        np.random.randn(4,5)
    )
