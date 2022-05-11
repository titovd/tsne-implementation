from tsne import TSNE

import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs
import os


def visualize(X_hat, y, title='Данные в латентном пространстве', fname='fig.png'):
    plt.figure(figsize=(10, 6))
    plt.scatter(X_hat[:, 0], X_hat[:, 1], c=y, cmap='Set1')
    plt.xlabel('Координата 1')
    plt.ylabel('Координата 2')
    plt.title(title)
    plt.savefig(os.path.join('figures', fname))


def main():
    # small synthetic dataset: 10 features, 200 objects
    n_samples = 200
    n_features = 10
    n_components = 2
    
    X, y = make_blobs(
        n_samples=n_samples, 
        n_features=n_features, 
        centers=5,
    )
    
    # pytorch implementation
    tsne = TSNE(n_components=n_components, method='pytorchsdsd')
    X_hat = tsne.fit_transform(X)
    visualize(X_hat, y, 'Данные в латентном пространстве, Pytorch-реализация.', 'pytorch-vis.png')
    
    # native implementation
    tsne = TSNE(n_components=n_components, method='native')
    X_hat = tsne.fit_transform(X)
    visualize(X_hat, y, 'Данные в латентном пространстве, собственная реализация.', 'native-vis.png')


if __name__ == '__main__':
    main()
