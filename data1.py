import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
import pandas as pd

# Define the angle in radians for rotation (pi/5 radians, which is 36 degrees)
angle = np.pi / 5

# Set the stretch factor for the x-axis (stretching by a factor of 5)
stretch = 5

# Define the number of data points to generate (200 data points)
m = 200

# Set a random seed for reproducibility
np.random.seed(3)

# Generate random data points with dimensions 'm' by 2, scaled down by 10
X = np.random.randn(m, 2) / 10

# Stretch the data points along the x-axis by a factor of 'stretch'
X = X.dot(np.array([[stretch, 0], [0, 1]]))  # stretch

# Rotate the stretched data points counterclockwise by an 'angle' (in radians)
X = X.dot([[np.cos(angle), np.sin(angle)], [-np.sin(angle), np.cos(angle)]])  # rotate

X_std = StandardScaler().fit_transform(X)
np.mean(X_std, axis=0)


def log_likelihood(evals):

    Lmax = len(evals)
    ll = np.arange(0.0, Lmax)

    for L in range(Lmax):

        group1 = evals[0 : L + 1]  # Divide Eigenvalues in two groups
        group2 = evals[L + 1 : Lmax]

        mu1 = np.mean(group1)
        mu2 = np.mean(group2)

        # eqn (20.30)
        sigma = (np.sum((group1 - mu1) ** 2) + np.sum((group2 - mu2) ** 2)) / Lmax

        ll_group1 = np.sum(multivariate_normal.logpdf(group1, mu1, sigma))
        ll_group2 = np.sum(multivariate_normal.logpdf(group2, mu2, sigma))

        ll[L] = ll_group1 + ll_group2  
    return ll