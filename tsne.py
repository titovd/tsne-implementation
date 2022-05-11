import numpy as np
import torch
from scipy.stats import entropy
from sklearn.metrics import pairwise_distances


class TSNE:
    """ Pytorch and native implementation of T-distributed Stochastic Neighbor Embedding. """

    def __init__(self, n_components=2, perplexity=30.0, n_iter=1000, method='pytorch'):
        self.n_components = n_components
        self.perplexity = perplexity
        self.n_iter = n_iter
        self.method = method

    def fit_transform(self, X):
        """
        Fit X into an embedded space and return that transformed output.

        :param X: np.ndarray (n_samples, n_features) with training data
        :return: Y np.ndarray embedding of the training data in low-dimensional space.
        """
        cond_probs = self._calculate_cond_probs(X)
        n = cond_probs.shape[0]
        cond_probs = (cond_probs + cond_probs.T) / (2. * n)
        
        if self.method == "pytorch":
            return self._fit_pytorch(cond_probs)
        elif self.method == "native":
            return self._fit_native(cond_probs)
        else:
            raise ValueError("Unknown method")

    def fit(self, X):
        self.fit_transform(X)
        return self
    
    def _fit_pytorch(self, cond_probs):
        """ t-SNE training using Pytorch. """
        n = cond_probs.shape[0]
        
        target_probs = torch.from_numpy(cond_probs[np.triu_indices(n)]).float()
        mask_p_probs = torch.nonzero(target_probs)
        
        Y = torch.randn((n, self.n_components), dtype=torch.float32, requires_grad=True)
        optimizer = torch.optim.SGD([Y], lr=0.9, momentum=0.9)

        for it in range(self.n_iter):
            q_probs = self._calculate_q_probs(Y)
            q_probs = q_probs[np.triu_indices(n)]
            
            kl_div = self._calculate_kldiv_loss(
                target_probs[mask_p_probs], 
                torch.log(q_probs[mask_p_probs])
            )
            kl_div.backward()
            
            optimizer.step()
            optimizer.zero_grad()
        return Y.cpu().detach()
    
    def _fit_native(self, cond_probs):
        """ t-SNE training using native implementation. """
       
        n = cond_probs.shape[0]
        Y = np.random.normal(0., 1e-4, [n, self.n_components])
        Y_momentum_1 = np.copy(Y)
        Y_momentum_2 = np.copy(Y)
        learning_rate = 0.9
        momentum = 0.9
        
        for it in range(self.n_iter):
            q_probs, inv_dist = self._calculate_q_probs(torch.from_numpy(Y).float(), return_inv_q=True)
            q_probs, inv_dist = q_probs.numpy(), inv_dist.numpy()
            
            grad = TSNE._calculate_grad(cond_probs, q_probs, Y, inv_dist)
            Y -= learning_rate * grad
            Y += momentum * (Y_momentum_1 - Y_momentum_2)
            
            Y_momentum_2 = np.copy(Y_momentum_1)
            Y_momentum_1 = np.copy(Y)
        return Y
            
    @staticmethod            
    def _calculate_grad(p_distr, q_distr, Y, inv_q_dist):
        """ Gradients calculation from original paper. """
        n = p_distr.shape[0]
        n_components = Y.shape[1]
        
        pq_diff = (p_distr - q_distr).reshape(n, n, 1)
        y_diff = Y.reshape(n, 1, n_components) - Y.reshape(1, n, n_components)
        dists_diff = inv_q_dist.reshape(n, n, 1)
        grad = 4. * (pq_diff * y_diff * dists_diff).sum(axis=1)
        return grad
        
    @staticmethod
    def _calculate_kldiv_loss(p_distr, q_distr):
        """ KL-div loss calculation. """
        return torch.sum(p_distr * (p_distr.log() - q_distr))
    
    @staticmethod
    def _calculate_q_probs(Y, return_inv_q=False):
        """ Compute matrix of joint probabilities with entries q_ij."""
        q_pairwise_distance = torch.cdist(Y, Y, 2) ** 2
        q_probs = 1. / (1. + q_pairwise_distance)
        q_probs.fill_diagonal_(0.)
        q_probs = q_probs / q_probs.sum()
        if return_inv_q:
            return q_probs, 1. / (1. + q_pairwise_distance)
        return q_probs

    def _calculate_cond_probs_by_idx(self, idx, sigma):
        deg = -self.pairwise_distance[idx] / (2 * (sigma ** 2))
        probs = np.exp(deg - np.max(deg))
        probs[idx] = 0.
        probs /= probs.sum()
        return probs

    def _calculate_cond_probs(self, X):
        """ Compute joint probabilities matrix with entries p_ij."""
        self.pairwise_distance = pairwise_distances(X, X) ** 2
        sigmas = self._calculate_sigmas(X)
        cond_probs = [self._calculate_cond_probs_by_idx(i, sigmas[i]) for i in range(len(X))]
        return np.asarray(cond_probs)

    @staticmethod
    def _calculate_perplexity(probs):
        """ Calculation of perplexity. """
        mask = ~np.isclose(probs, 0.0)
        entropy_ = entropy(probs[mask])
        return 2 ** entropy_

    def _calculate_sigmas(self, X, init_val=1.0, tol=1e-5, n_iter=1000):
        """
        Perform a binary search over each sigma_i
        until Perp(P_i) = our desired perplexity.

        :param X: np.ndarray (n_samples, n_features) with training data
        :param init_val: Initial value (middle point) in BinSearch
        :param tol: Float, once our guess is this close to target, stop
        :param n_iter: Maximum num. iterations to search for
        :return: List of sigmas_i
        """
        sigmas = []

        for i in range(len(X)):
            lower = 0
            upper = 2000.
            mid = init_val
            diff = np.inf
            it = 0

            while (np.abs(diff) > tol) and (it < n_iter):
                diff = TSNE._calculate_perplexity(self._calculate_cond_probs_by_idx(i, mid)) \
                    - self.perplexity
                if diff < 0:
                    lower = mid
                else:
                    upper = mid
                mid = (lower + upper) / 2
                it += 1

            sigmas.append(mid)
        return np.asarray(sigmas)
