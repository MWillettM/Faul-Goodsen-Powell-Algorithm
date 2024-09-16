This is a Python implementation of the Faul-Goodsen-Powell algorithm which produces an interpolant for d-dimensional data using the multiquadric radial basis functions. It works well for even very high dimensional data.

**INSTALLATION/REQUIREMENTS:**

Python  and Numpy

**ALGORITHM DESCRIPTION** - inputs - FGP(data, values, q, c, error)

- Data centers ($x_i$) and values at those points ($f_i$)

- Error

- Two parameters for the algorithm - $q$ and $c$:

- $c>0$, using a smaller value ($O(10^{-1})$ or smaller) is advised. This is the 'shape parameter' for the multiquadric.

- $q>0$, using a value of q=30 is standard - feel free to go between 5 and 50. A rule of thumb is that smaller q means each iteration is quicker, but we may need more iterations for convergence overally. 

**ALGORITHM DESCRIPTION** - outputs

- Iteration count - k

The interpolant, s(x) is of the form 

$s(x) = \sum_i^n \lambda_i \phi(\|x-x_i\|) + \alpha$

where the $x_i$ are the data centers and $\phi(x) = (x^2+c^2)^{\frac{1}{2}}$ is the multiquadric radial basis function.

The algorithm returns the coefficients lambda_i and the value of alpha.

You can try the DEMO version or use the FULL IMPLEMENTATION version as well.

The DEMO version lets you vary the distribution that the test data is drawn from:
The unit d-ball
The unit d-cube
The unit d-Normal
The integer grid in d-dimensions.
