from sklearn.base import ClusterMixin

from sklearn.utils import check_random_state
import numpy


from sklearn.utils import check_array
from sklearn.utils.validation import check_is_fitted
from tslearn.preprocessing import TimeSeriesScalerMeanVariance
from tslearn.utils import to_time_series_dataset, check_dims
#from tslearn.metrics import cdist_normalized_cc, y_shifted_sbd_vec
from tslearn.bases import BaseModelPackage, TimeSeriesBaseEstimator

from .utils import (TimeSeriesCentroidBasedClusteringMixin,
                    _check_no_empty_cluster, _compute_inertia,
                    _check_initial_guess, EmptyClusterError)

__author__ = 'Romain Tavenard romain.tavenard[at]univ-rennes2.fr'


class KShape(ClusterMixin, TimeSeriesCentroidBasedClusteringMixin,
             BaseModelPackage, TimeSeriesBaseEstimator):
    """KShape clustering for time series.

    KShape was originally presented in [1]_.

    Parameters
    ----------
    n_clusters : int (default: 3)
        Number of clusters to form.

    max_iter : int (default: 100)
        Maximum number of iterations of the k-Shape algorithm.

    tol : float (default: 1e-6)
        Inertia variation threshold. If at some point, inertia varies less than
        this threshold between two consecutive
        iterations, the model is considered to have converged and the algorithm
        stops.

    n_init : int (default: 1)
        Number of time the k-Shape algorithm will be run with different
        centroid seeds. The final results will be the
        best output of n_init consecutive runs in terms of inertia.

    verbose : bool (default: False)
        Whether or not to print information about the inertia while learning
        the model.

    random_state : integer or numpy.RandomState, optional
        Generator used to initialize the centers. If an integer is given, it
        fixes the seed. Defaults to the global
        numpy random number generator.

    init : {'random' or ndarray} (default: 'random')
        Method for initialization.
        'random': choose k observations (rows) at random from data for the
        initial centroids.
        If an ndarray is passed, it should be of shape (n_clusters, ts_size, d)
        and gives the initial centers.

    Attributes
    ----------
    cluster_centers_ : numpy.ndarray of shape (sz, d).
        Centroids

    labels_ : numpy.ndarray of integers with shape (n_ts, ).
        Labels of each point

    inertia_ : float
        Sum of distances of samples to their closest cluster center.

    n_iter_ : int
        The number of iterations performed during fit.

    Notes
    -----
        This method requires a dataset of equal-sized time series.

    Examples
    --------
    >>> from tslearn.generators import random_walks
    >>> X = random_walks(n_ts=50, sz=32, d=1)
    >>> X = TimeSeriesScalerMeanVariance(mu=0., std=1.).fit_transform(X)
    >>> ks = KShape(n_clusters=3, n_init=1, random_state=0).fit(X)
    >>> ks.cluster_centers_.shape
    (3, 32, 1)

    References
    ----------
    .. [1] J. Paparrizos & L. Gravano. k-Shape: Efficient and Accurate
       Clustering of Time Series. SIGMOD 2015. pp. 1855-1870.
    """

    def __init__(self, n_clusters=3, max_iter=100, tol=1e-6, n_init=1,
                 verbose=False, random_state=None, init='random'):
        self.n_clusters = n_clusters
        self.max_iter = max_iter
        self.tol = tol
        self.random_state = random_state
        self.n_init = n_init
        self.verbose = verbose
        self.init = init

    def _is_fitted(self):
        """
        Check if the model has been fit.

        Returns
        -------
        bool
        """

        check_is_fitted(self,
                        ['cluster_centers_', 'norms_', 'norms_centroids_'])
        return True

    def _shape_extraction(self, X, k):
        sz = X.shape[1]
        Xp = y_shifted_sbd_vec(self.cluster_centers_[k], X[self.labels_ == k],
                               norm_ref=-1,
                               norms_dataset=self.norms_[self.labels_ == k])
        S = numpy.dot(Xp[:, :, 0].T, Xp[:, :, 0])
        Q = numpy.eye(sz) - numpy.ones((sz, sz)) / sz
        M = numpy.dot(Q.T, numpy.dot(S, Q))
        _, vec = numpy.linalg.eigh(M)
        mu_k = vec[:, -1].reshape((sz, 1))

        # The way the optimization problem is (ill-)formulated, both mu_k and
        # -mu_k are candidates for barycenters
        # In the following, we check which one is best candidate
        dist_plus_mu = numpy.sum(numpy.linalg.norm(Xp - mu_k, axis=(1, 2)))
        dist_minus_mu = numpy.sum(numpy.linalg.norm(Xp + mu_k, axis=(1, 2)))
        if dist_minus_mu < dist_plus_mu:
            mu_k *= -1

        return mu_k

    def _update_centroids(self, X):
        for k in range(self.n_clusters):
            self.cluster_centers_[k] = self._shape_extraction(X, k)
        self.cluster_centers_ = TimeSeriesScalerMeanVariance(
            mu=0., std=1.).fit_transform(self.cluster_centers_)
        self.norms_centroids_ = numpy.linalg.norm(self.cluster_centers_,
                                                  axis=(1, 2))

    def _cross_dists(self, X):
        return 1. - cdist_normalized_cc(X, self.cluster_centers_,
                                        norms1=self.norms_,
                                        norms2=self.norms_centroids_,
                                        self_similarity=False)

    def _assign(self, X):
        dists = self._cross_dists(X)
        self.labels_ = dists.argmin(axis=1)
        _check_no_empty_cluster(self.labels_, self.n_clusters)
        self.inertia_ = _compute_inertia(dists, self.labels_)

    def _fit_one_init(self, X, rs):
        if hasattr(self.init, '__array__'):
            self.cluster_centers_ = self.init.copy()
        elif self.init == "random":
            indices = rs.choice(X.shape[0], self.n_clusters)
            self.cluster_centers_ = X[indices].copy()
        else:
            raise ValueError("Value %r for parameter 'init' is "
                             "invalid" % self.init)
        self.norms_centroids_ = numpy.linalg.norm(self.cluster_centers_,
                                                  axis=(1, 2))
        self._assign(X)
        old_inertia = numpy.inf

        for it in range(self.max_iter):
            old_cluster_centers = self.cluster_centers_.copy()
            self._update_centroids(X)
            self._assign(X)
            if self.verbose:
                print("%.3f" % self.inertia_, end=" --> ")

            if numpy.abs(old_inertia - self.inertia_) < self.tol or \
                    (old_inertia - self.inertia_ < 0):
                self.cluster_centers_ = old_cluster_centers
                self._assign(X)
                break

            old_inertia = self.inertia_
        if self.verbose:
            print("")

        self._iter = it + 1

        return self

    def fit(self, X, y=None):
        """Compute k-Shape clustering.

        Parameters
        ----------
        X : array-like of shape=(n_ts, sz, d)
            Time series dataset.

        y
            Ignored
        """
        X = check_array(X, allow_nd=True)

        max_attempts = max(self.n_init, 10)

        self.labels_ = None
        self.inertia_ = numpy.inf
        self.cluster_centers_ = None

        self.norms_ = 0.
        self.norms_centroids_ = 0.

        self.n_iter_ = 0

        X_ = to_time_series_dataset(X)
        self._X_fit = X_
        self.norms_ = numpy.linalg.norm(X_, axis=(1, 2))

        _check_initial_guess(self.init, self.n_clusters)

        rs = check_random_state(self.random_state)

        best_correct_centroids = None
        min_inertia = numpy.inf
        n_successful = 0
        n_attempts = 0
        while n_successful < self.n_init and n_attempts < max_attempts:
            try:
                if self.verbose and self.n_init > 1:
                    print("Init %d" % (n_successful + 1))
                n_attempts += 1
                self._fit_one_init(X_, rs)
                if self.inertia_ < min_inertia:
                    best_correct_centroids = self.cluster_centers_.copy()
                    min_inertia = self.inertia_
                    self.n_iter_ = self._iter
                n_successful += 1
            except EmptyClusterError:
                if self.verbose:
                    print("Resumed because of empty cluster")
        self.norms_centroids_ = numpy.linalg.norm(self.cluster_centers_,
                                                  axis=(1, 2))
        self._post_fit(X_, best_correct_centroids, min_inertia)
        return self

    def fit_predict(self, X, y=None):
        """Fit k-Shape clustering using X and then predict the closest cluster
        each time series in X belongs to.

        It is more efficient to use this method than to sequentially call fit
        and predict.

        Parameters
        ----------
        X : array-like of shape=(n_ts, sz, d)
            Time series dataset to predict.

        y
            Ignored

        Returns
        -------
        labels : array of shape=(n_ts, )
            Index of the cluster each sample belongs to.
        """
        return self.fit(X, y).labels_

    def predict(self, X):
        """Predict the closest cluster each time series in X belongs to.

        Parameters
        ----------
        X : array-like of shape=(n_ts, sz, d)
            Time series dataset to predict.

        Returns
        -------
        labels : array of shape=(n_ts, )
            Index of the cluster each sample belongs to.
        """
        X = check_array(X, allow_nd=True)
        check_is_fitted(self,
                        ['cluster_centers_', 'norms_', 'norms_centroids_'])

        X_ = check_dims(X, X_fit_dims=self.cluster_centers_.shape)
        X_ = TimeSeriesScalerMeanVariance(mu=0., std=1.).fit_transform(X_)
        dists = self._cross_dists(X_)
        return dists.argmin(axis=1)

# # Numba testing

# from numba import jit, objmode, njit, prange
# #numba.set_num_threads(2)

# @jit()
# def normalized_cc(s1, s2, norm1=-1., norm2=-1.):
#     #assert s1.shape[1] == s2.shape[1]
#     sz = s1.shape[0] #~100
#     # Compute fft size based on tip from https://stackoverflow.com/questions/14267555/
#     #fft_sz = 1 << (2 * sz - 1).bit_length()
#     fft_sz = 1 << numpy.int(2**(numpy.ceil(numpy.log2(2*sz-1))))# bit_length doesn't work with numba
                                                                
#     denom = 0.
    

#     if norm1 < 0.:
#         norm1 = numpy.linalg.norm(s1)
#     if norm2 < 0.:
#         norm2 = numpy.linalg.norm(s2)

#     denom = norm1 * norm2
#     if denom < 1e-9:  # To avoid NaNs
#         denom = numpy.inf

#     # cc = numpy.real(numpy.fft.ifft(numpy.fft.fft(s1, fft_sz, axis=0) *
#     #                                numpy.conj(numpy.fft.fft(s2, fft_sz, axis=0)), axis=0))
#     # cc = numpy.vstack((cc[-(sz-1):], cc[:sz])) 
#     # Numba doesn't like axis argument, so use transposing instead. Oh, and also apparently needs object mode...
#     with objmode(cc='float64[:,:]'):
#         cc = numpy.real((numpy.fft.ifft(((numpy.fft.fft(s1.T, fft_sz)).T *  numpy.conj((numpy.fft.fft(s2.T, fft_sz))).T).T)).T)
#     cc = numpy.vstack((cc[-(sz-1):], cc[:sz]))

#     return numpy.real(cc).sum(axis=-1) / denom

# @jit()
# def cdist_normalized_cc(dataset1, dataset2, norms1, norms2, self_similarity):
#     #assert dataset1.shape[2] == dataset2.shape[2]
#     dists = numpy.empty((dataset1.shape[0], dataset2.shape[0]))

#     # if (norms1 < 0.).any(): #For some strange reason numba complains about this block. Since it is unnecessary, can comment out.
#     # #    norms1 = numpy.linalg.norm(dataset1, axis=(1, 2)) #axis keyword not liked by numba
#     #     norms1 = numpy.sqrt(numpy.sum((dataset1*dataset1), axis=(1, 2)))
#     # if (norms2 < 0.).any():
#     # #    norms2 = numpy.linalg.norm(dataset2, axis=(1, 2)) #axis keyword not liked by numba
#     #     norms2 = numpy.sqrt(numpy.sum((dataset2*dataset2), axis=(1, 2)))

#     for i in range(dataset1.shape[0]): #~250k - 1 million
#         for j in range(dataset2.shape[0]): #~100 - 1000
#             if self_similarity and j < i: #self_similarity always false, could switch it off.
#                 dists[i, j] = dists[j, i]
#             elif self_similarity and i == j:
#                 dists[i, j] = 0.
#             else:
#                 dists[i, j] = normalized_cc(dataset1[i], dataset2[j], norm1=norms1[i], norm2=norms2[j]).max()
#     return dists


# @jit()
# def y_shifted_sbd_vec(ref_ts, dataset, norm_ref, norms_dataset):
#     #assert dataset.shape[1] == ref_ts.shape[0] and dataset.shape[2] == ref_ts.shape[1]
#     sz = dataset.shape[1]
#     dataset_shifted = numpy.zeros((dataset.shape[0], dataset.shape[1], dataset.shape[2]))

#     if norm_ref < 0:
#         norm_ref = numpy.linalg.norm(ref_ts)
#     if (norms_dataset < 0.).any():
#     #    norms_dataset = numpy.linalg.norm(dataset, axis=(1, 2)) #axis keyword not liked by numba
#         norms_dataset = numpy.sqrt(numpy.sum((dataset*dataset), axis=(1, 2)))

#     for i in range(dataset.shape[0]): #~250k - 1 million
#         cc = normalized_cc(ref_ts, dataset[i], norm1=norm_ref, norm2=norms_dataset[i])
#         idx = numpy.argmax(cc)
#         shift = idx + 1 - sz
#         if shift > 0:
#             dataset_shifted[i, shift:] = dataset[i, :-shift, :]
#         elif shift < 0:
#             dataset_shifted[i, :shift] = dataset[i, -shift:, :]
#         else:
#             dataset_shifted[i] = dataset[i]

#     return dataset_shifted

# # Other parallelization possibilities

from joblib import Parallel, delayed

def normalized_cc(s1, s2, norm1=-1., norm2=-1.):
    assert s1.shape[1] == s2.shape[1]
    sz = s1.shape[0] #~100
    # Compute fft size based on tip from https://stackoverflow.com/questions/14267555/
    fft_sz = 1 << (2 * sz - 1).bit_length()
    denom = 0.
    

    if norm1 < 0.:
        norm1 = numpy.linalg.norm(s1)
    if norm2 < 0.:
        norm2 = numpy.linalg.norm(s2)

    denom = norm1 * norm2
    if denom < 1e-9:  # To avoid NaNs
        denom = numpy.inf

    cc = numpy.real(numpy.fft.ifft(numpy.fft.fft(s1, fft_sz, axis=0) *
                                   numpy.conj(numpy.fft.fft(s2, fft_sz, axis=0)), axis=0))
    cc = numpy.vstack((cc[-(sz-1):], cc[:sz]))
    return (numpy.real(cc).sum(axis=-1) / denom)

def do_parallel(dataset1_row, dataset2, norms1_row, norms2):
    temp = numpy.zeros(dataset2.shape[0])
    for j in range(dataset2.shape[0]):
        temp[j] = normalized_cc(dataset1_row, dataset2[j], norms1_row, norms2[j]).max()
    return(temp)


def cdist_normalized_cc(dataset1, dataset2, norms1, norms2, self_similarity): #self_similarity always false, so unused
    assert dataset1.shape[2] == dataset2.shape[2]
    dists = numpy.empty((dataset1.shape[0], dataset2.shape[0]))

    if (norms1 < 0.).any():
        norms1 = numpy.linalg.norm(dataset1, axis=(1, 2))
    if (norms2 < 0.).any():
        norms2 = numpy.linalg.norm(dataset2, axis=(1, 2))

    results = Parallel(n_jobs=-1)(delayed(do_parallel)(dataset1[i],dataset2,norms1[i],norms2) for i in range(dataset1.shape[0]))
    for i in range(dataset1.shape[0]):
        dists[i] = results[i]

    # results = Parallel()(delayed(normalized_cc)(
    #     dataset1[i],dataset2[j],norms1[i],norms2[j]) for i in range(dataset1.shape[0]) for j in range(dataset2.shape[0]))
    
    # dists = numpy.asarray(results).reshape(dataset1.shape[0],dataset2.shape[0])
    return dists


def y_shifted_sbd_vec(ref_ts, dataset, norm_ref, norms_dataset):
    assert dataset.shape[1] == ref_ts.shape[0] and dataset.shape[2] == ref_ts.shape[1]
    sz = dataset.shape[1]
    dataset_shifted = numpy.zeros((dataset.shape[0], dataset.shape[1], dataset.shape[2]))

    if norm_ref < 0:
        norm_ref = numpy.linalg.norm(ref_ts)
    if (norms_dataset < 0.).any():
        norms_dataset = numpy.linalg.norm(dataset, axis=(1, 2))

    for i in range(dataset.shape[0]): #~250k - 1 million
        cc = normalized_cc(ref_ts, dataset[i], norm1=norm_ref, norm2=norms_dataset[i])
        idx = numpy.argmax(cc)
        shift = idx + 1 - sz
        if shift > 0:
            dataset_shifted[i, shift:] = dataset[i, :-shift, :]
        elif shift < 0:
            dataset_shifted[i, :shift] = dataset[i, -shift:, :]
        else:
            dataset_shifted[i] = dataset[i]

    return dataset_shifted