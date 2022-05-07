import numpy as np
import torch
from scipy.stats import entropy
from sklearn.metrics import pairwise_distances


class TSNE:
    """ Pytorch implementation of T-distributed Stochastic Neighbor Embedding. """

    def __init__(self, n_components=2, perplexity=30.0, n_iter=1000, verbose=0):
        self.n_components = n_components
        self.perplexity = perplexity
        self.n_iter = n_iter

    def fit_transform(self, X) -> np.ndarray:
        """
        Fit X into an embedded space and return that transformed output.

        :param X: np.ndarray (n_samples, n_features) with training data
        :return: Y np.ndarray embedding of the training data in low-dimensional space.
        """
        cond_probs = self._calculate_cond_probs(X)
        n = cond_probs.shape[0]
        cond_probs = (cond_probs + cond_probs.T) / (2. * n)

        target_probs = torch.from_numpy(cond_probs[np.triu_indices(n)]).float()
        mask_p_probs = torch.nonzero(target_probs)

        Y = torch.randn((n, self.n_components), dtype=torch.float32, requires_grad=True)
        optimizer = torch.optim.SGD([Y], lr=0.9, momentum=0.9)
        kl_loss = torch.nn.KLDivLoss(reduction='sum')

        for it in range(self.n_iter):
            q_probs = self._calculate_q_probs(Y)
            kl_div = kl_loss(torch.log(q_probs[mask_p_probs]), target_probs[mask_p_probs])
            kl_div.backward()
            optimizer.step()
            optimizer.zero_grad()

        return Y.cpu().detach()

    def fit(self, X):
        self.fit_transform(X)
        return self

    @staticmethod
    def _calculate_q_probs(Y) -> torch.FloatTensor:
        n = Y.shape[0]
        q_pairwise_distance = torch.cdist(Y, Y, 2) ** 2
        q_probs = 1 / (1 + q_pairwise_distance)
        q_probs.fill_diagonal_(0.)
        q_probs = q_probs / q_probs.sum()
        return q_probs[np.triu_indices(n)]

    def _calculate_cond_probs_by_idx(self, idx, sigma) -> np.ndarray:
        probs = np.exp(-self.pairwise_distance[idx] / (2 * (sigma ** 2)))
        probs[idx] = 0.
        probs /= probs.sum()
        return probs

    def _calculate_cond_probs(self, X) -> np.ndarray:
        self.pairwise_distance = pairwise_distances(X, X) ** 2
        sigmas = self._calculate_sigmas(X)
        cond_probs = [self._calculate_cond_probs_by_idx(i, sigmas[i]) for i in range(len(X))]
        return np.asarray(cond_probs)

    @staticmethod
    def _calculate_perplexity(probs) -> float:
        mask = ~np.isclose(probs, 0.0)
        entropy_ = entropy(probs[mask])
        return 2 ** entropy_

    def _calculate_sigmas(self, X, init_val=1.0, tol=1e-5, n_iter=1000) -> np.ndarray:
        """
        Perform a binary search over each sigma_i
        until Perp(P_i) = our desired perplexity.

        :param X: np.ndarray (n_samples, n_features) with training data
        :param init_val: Initial value (middle point) in BinSearch
        :param tol: Float, once our guess is this close to target, stop
        :param n_iter: Maximum num. iterations to search for
        :return:
        """
        sigmas = []

        for i in range(len(X)):
            lower = 0
            upper = 2000.
            mid = init_val
            diff = np.inf
            it = 0

            while (np.abs(diff) > tol) and (it < n_iter):
                diff = TSNE._calculate_perplexity(self._calculate_cond_probs_by_idx(i, mid)) - self.perplexity
                if diff < 0:
                    lower = mid
                else:
                    upper = mid
                it += 1
                mid = (lower + upper) / 2

            sigmas.append(mid)
        return np.asarray(sigmas)
