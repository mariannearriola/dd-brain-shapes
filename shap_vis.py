import geomstats
import geomstats.backend as gs
from geomstats.geometry.spd_matrices import *
from geomstats.geometry.stiefel import Stiefel
from geomstats.datasets.utils import load_connectomes
from geomstats.visualization.spd_matrices import Ellipses
import networkx as nx
import matplotlib.pyplot as plt
import numpy as np
import numpy.linalg as npla

def compute_sqr_dist(healthy_spd, schiz_spd, n):
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
    spd_bures = SPDBuresWassersteinMetric(n)
    sqrd_dist = []
    for i in range(len(healthy_spd[:])):
        for j in range(len(schiz_spd[:])):
            sqrd_dist.append(spd_bures.squared_dist(healthy_spd[i], schiz_spd[j]))
    return sqrd_dist

def plot_graphs_spatial(nx_graph, subgraph_id, title):
    """Plots spatial graphs.
    
    Plots graph network with title and appropriate subgraph.
    
    Parameters
        ----------
        nx_graph : graph object
            Graph object used for plotting.
        subgraph_id : int
            ID for subgraph plot
        title : string
            Title of plot.
    """
    plt.subplot(subgraph_id)
    degrees = [n for n in nx.degree_centrality(nx_graph).values()]
    nx.draw(nx_graph,pos=None,with_labels=False,node_color=degrees)
    plt.title(title)
    
def plot_graphs_spectral(nx_graph,subgraph_id,title):
    """Plots spectral graphs.
    
    Plots spectral graph network with title and appropriate subgraph.
    
    Parameters
        ----------
        nx_graph : graph object
            Graph object used for plotting.
        subgraph_id : int
            ID for subgraph plot
        title : string
            Title of plot.
    """
    evals,_ = npla.eig(nx.adjacency_matrix(nx_graph).todense())
    plt.subplot(subgraph_id)
    plt.hist(evals,range=(0,np.max(evals)))
    plt.xlabel('Eigenvalue')
    plt.ylabel('Frequency')
    plt.title(title)