import numpy as np
import matplotlib.pyplot as plt
import pygeoinf as inf


# Set up Euclidean space and a simple Gaussian distribution
X = inf.EuclideanSpace(6)
mu = inf.GaussianMeasure.from_standard_deviation(X, 0.1)

# Print out some of its properties
print(mu.expectation)
print(mu.sample())


# Draw random samples, and estimate the covariance of two elements numerically.
n = 10000
xs = mu.samples(n)

i = 2
j = 3

cov_ij = 0
for x in xs:
    xi = x[i]
    xj = x[j]
    cov_ij += (xi - mu.expectation[i]) * (xj - mu.expectation[j])
cov_ij /= n

print(cov_ij)


# Set up another space and an affine mapping between them
Y = inf.EuclideanSpace(3)

A = inf.LinearOperator(X, Y, lambda x: 2 * x[: Y.dim])
a = Y.basis_vector(0)

# Transform the samples under the mapping
ys = []
for x in xs:
    ys.append(A(x) + a)

# Estimate the variance of the ith component
i = 0

mean = 0
for y in ys:
    yi = y[i]
    mean += yi
mean /= n


var = 0
for y in ys:
    yi = y[i]
    var += (yi - mean) ** 2
var /= n - 1
std = np.sqrt(var)

print(std)

# Transform the distributions directly
nu = mu.affine_mapping(operator=A, translation=a)

print(nu.expectation)
print(nu.covariance)
print(nu.sample())
