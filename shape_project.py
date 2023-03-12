import geomstats
import math
import geomstats.backend as gs
from geomstats.geometry.spd_matrices import *
from geomstats.geometry.lie_group import *
from geomstats.datasets.utils import load_connectomes
from geomstats.visualization.spd_matrices import Ellipses
import networkx as nx
import matplotlib.pyplot as plt
import numpy as np
import numpy.linalg as npla
from geomstats.learning.geodesic_regression import GeodesicRegression
from geomstats.geometry.pre_shape import PreShapeSpace, KendallShapeMetric
from geomstats.learning.frechet_mean import FrechetMean
from geomstats.geometry.special_euclidean import SpecialEuclidean
from geomstats.geometry.lie_group import LieGroup
from geomstats.geometry.general_linear import GeneralLinear
from shape_project import *

def unsqueeze(mat):
    return gs.reshape(mat,(1,mat.shape[0],mat.shape[1]))

def plot_mats(data,noisy_mats,ncol):
    T = noisy_mats.shape[0]
    figs,axs=plt.subplots(int(math.ceil((T+1)/(1*ncol))), ncol, figsize=(20,9))
    axs[0][0].imshow(data)
    axs[0][0].title.set_text("Input")
    for im_id,noisy_mat in enumerate(noisy_mats):
        axs[int((im_id+1)/ncol)][(im_id+1)%ncol].imshow(noisy_mat)
        axs[int((im_id+1)/ncol)][(im_id+1)%ncol].title.set_text(f'T={im_id+1}')
    return figs

def denoise_mats(x_0,noisy_samples):
    for ind,noisy_mat in enumerate(noisy_samples):
        if ind == 0:
            denoising_mats=unsqueeze(npla.solve(x_0,noisy_mat))
        else:
            denoising_mats=gs.concatenate((denoising_mats,unsqueeze(npla.solve(noisy_samples[ind-1],noisy_mat))),axis=0)
    return denoising_mats

def plot_results(test,noisy,pred):
    figs,axs=plt.subplots(1, 3, figsize=(9,9))
    axs[0].imshow(test)
    axs[0].title.set_text("Test matrix")
    axs[1].imshow(noisy)
    axs[1].title.set_text("Noisy matrix")
    axs[2].imshow(pred)
    axs[2].title.set_text("Denoised matrix")
    return figs

def compute_sqr_dist(a, b, metric):
    """Compute the Bures-Wasserstein squared distance.
        
    Compute the Riemannian squared distance between all 
    combinations of healthy SPD matrices and 
    schizophrenic SPD matrices.

    Parameters
    ----------
    healthy_spd : array-like, shape=[..., n, n]
        Point.
    schiz_spd : array-like, shape=[..., n, n]
        Point.
    n : int
        Size of matrix.

    Returns
    -------
    sqrd_dist : array-like, shape=[...]
        Riemannian squared distance of all SPD combinations.
    """
    sqrd_dist = []
    for i in range(len(a[:])):
        for j in range(len(b[:])):
            sqrd_dist.append(metric.squared_dist(a[i], b[j]))
    return sqrd_dist