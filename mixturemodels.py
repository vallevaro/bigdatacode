import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
import matplotlib as mpl

import itertools
from scipy import linalg


# color_iter = itertools.cycle(['navy', 'c', 'cornflowerblue', 'darkorange'])
color_iter = itertools.cycle(["r", "g", "b"])
prop_cycle = plt.rcParams["axes.prop_cycle"]
color_iter = prop_cycle.by_key()["color"]


from sklearn.cluster import KMeans
from sklearn.datasets import make_blobs
from sklearn.mixture import GaussianMixture

from matplotlib.colors import LogNorm

if 0:
    K = 5
    blob_centers = np.array([[0.2, 2.3], [-1.5, 2.3], [-2.8, 1.8], [-2.8, 2.8], [-2.8, 1.3]])
    blob_std = np.array([0.4, 0.3, 0.1, 0.1, 0.1])
    X, y = make_blobs(n_samples=2000, centers=blob_centers, cluster_std=blob_std, random_state=7)

if 0:
    X1, y1 = make_blobs(n_samples=1000, centers=((4, -4), (0, 0)), random_state=42)
    X1 = X1.dot(np.array([[0.374, 0.95], [0.732, 0.598]]))
    X2, y2 = make_blobs(n_samples=250, centers=1, random_state=42)
    X2 = X2 + [6, -8]
    X = np.r_[X1, X2]
    y = np.r_[y1, y2]
    K = 3

if 1:
    # two off-diagonal blobs
    X1, _ = make_blobs(n_samples=1000, centers=((4, -4), (0, 0)), random_state=42)
    X1 = X1.dot(np.array([[0.374, 0.95], [0.732, 0.598]]))
    # three spherical blobs
    blob_centers = np.array([[-4, 1], [-4, 3], [-4, -2]])
    s = 0.5
    blob_std = np.array([s, s, s])
    X2, _ = make_blobs(n_samples=1000, centers=blob_centers, cluster_std=blob_std, random_state=7)

    X = np.r_[X1, X2]
    K = 5


def plot_gaussian_ellipse(gm, X):
    Y = gm.predict(X)
    means = gm.means_
    covariances = gm.covariances_
    K, D = means.shape
    if gm.covariance_type == "tied":
        covariances = np.tile(covariances, (K, 1, 1))
    splot = plt.subplot(1, 1, 1)
    for i, (mean, covar, color) in enumerate(zip(means, covariances, color_iter)):
        if gm.covariance_type == "spherical":
            covar = covar * np.eye(D)
        if gm.covariance_type == "diag":
            covar = np.diag(covar)
        v, w = linalg.eigh(covar)
        v = 2.0 * np.sqrt(2.0) * np.sqrt(v)
        u = w[0] / linalg.norm(w[0])
        plt.scatter(X[Y == i, 0], X[Y == i, 1], 0.8, color=color)
        angle = np.arctan(u[1] / u[0])
        angle = 180.0 * angle / np.pi  # convert to degrees
        ell = mpl.patches.Ellipse(mean, v[0], v[1], 180.0 + angle)
        ell.set_clip_box(splot.bbox)
        ell.set_alpha(0.3)
        splot.add_artist(ell)


def plot_data(X):
    plt.plot(X[:, 0], X[:, 1], "k.", markersize=2)


def plot_centroids(centroids, weights=None, circle_color="w", cross_color="k"):
    if weights is not None:
        centroids = centroids[weights > weights.max() / 10]
    plt.scatter(
        centroids[:, 0], centroids[:, 1], marker="o", s=30, linewidths=8, color=circle_color, zorder=10, alpha=0.9
    )
    plt.scatter(
        centroids[:, 0], centroids[:, 1], marker="x", s=50, linewidths=50, color=cross_color, zorder=11, alpha=1
    )


def plot_gaussian_mixture(clusterer, X, resolution=1000, show_ylabels=True):
    mins = X.min(axis=0) - 0.1
    maxs = X.max(axis=0) + 0.1
    xx, yy = np.meshgrid(np.linspace(mins[0], maxs[0], resolution), np.linspace(mins[1], maxs[1], resolution))

    Z = -clusterer.score_samples(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)

    plt.contourf(xx, yy, Z, norm=LogNorm(vmin=1.0, vmax=30.0), levels=np.logspace(0, 2, 12))
    plt.contour(xx, yy, Z, norm=LogNorm(vmin=1.0, vmax=30.0), levels=np.logspace(0, 2, 12), linewidths=1, colors="k")

    # plot decision boundaries
    if 0:
        Z = clusterer.predict(np.c_[xx.ravel(), yy.ravel()])
        Z = Z.reshape(xx.shape)
        plt.contour(xx, yy, Z, linewidths=2, colors="r", linestyles="dashed")

    plt.plot(X[:, 0], X[:, 1], "k.", markersize=2)
    # plot_centroids(clusterer.means_, clusterer.weights_)

    plt.xlabel("$x_1$", fontsize=14)
    if show_ylabels:
        plt.ylabel("$x_2$", fontsize=14, rotation=0)
    else:
        plt.tick_params(labelleft=False)


def plot_assignment(gm, X):
    # plt.figure(figsize=(8, 4))
    plt.figure()
    plt.scatter(X[:, 0], X[:, 1])
    y_pred = gm.predict(X)
    K, D = gm.means_.shape
    for k in range(K):
        color = next(color_iter)
        plt.plot(X[y_pred == k, 0], X[y_pred == k, 1], "o", color=color)


def make_plot(gm, X, name):
    ttl = name
    # plt.figure(figsize=(8, 4))
    plt.figure()
    plot_gaussian_mixture(gm, X)
    fname = f"figures/gmm_2d_{name}_contours.pdf"
    plt.title(ttl)
    plt.tight_layout()
    plt.axis("equal")
    #plt.savefig(fname, dpi=300)
    plt.show()
