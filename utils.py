import numpy as np
from typing import Tuple
import os
from PIL import Image
import matplotlib.pyplot as plt
import torch
from torchvision import transforms
import cv2
import random

plt.ion()


# RITORNA LA SINGOLA IMMAGINE SIMILE, DEVI POI USARE L'INDICE ACTOR DENTRO AL FILE actors_list.txt PER TROVARE IL NOME DELL'ATTORE
def get_image(actor, row):
    TRAIN_PATH = "../archive/Dataset_nuovo/"

    actors = os.listdir(TRAIN_PATH)
    actors.sort()
    convert_tensor = transforms.ToTensor()

    imgs = os.listdir(TRAIN_PATH + actors[actor])
    print(actor, row, len(actors), len(imgs))
    src = TRAIN_PATH + "{}/{}".format(actors[actor], imgs[row])
    image = Image.open(src)
    image = image.convert("RGB")
    img_flat = convert_tensor(image)

    return img_flat


def get_w_h(n: int, max_ratio: int = 2):
    values, rests = [], []
    r = int(np.sqrt(n))
    for i in reversed(range(1, r + 1)):
        if (n // i) / i > max_ratio and i < r:
            break
        rests.append(n % i)
        values.append(n // i)
    return r - np.argmin(rests), values[np.argmin(rests)]


def show_eigenfaces(eigenfaces: np.ndarray, size: Tuple, max_components: int = 25):
    """
    Plots ghostly eigenfaces.

    Parameters
    ----------
    eigenfaces: ndarray
        eigenfaces (eigenvectors of face covariance matrix).
    size: tuple
        the size of each face image like (h, w).

    Returns
    -------
    None
    """

    num_eigenfaces = eigenfaces.shape[1]
    (w, h) = get_w_h(min(max_components, num_eigenfaces))

    fig, ax = plt.subplots(nrows=h, ncols=w, sharex="col", sharey="row", figsize=(5, 5))
    for i in range(h):
        for j in range(w):
            f = np.array(eigenfaces[:, j + i * w])
            f = np.reshape(f, newshape=size)
            ax[i, j].imshow(f, cmap="gray")
            ax[i, j].grid(False)
            ax[i, j].axis("off")
            ax[i, j].set_title(f"eig {j+i*w}")
            ax[i, j].set_aspect("equal")

    plt.subplots_adjust(wspace=0.05, hspace=0.4)


def show_3d_faces_with_class(points: np.ndarray, labels: np.ndarray):
    """
    Plots 3d data in colorful point (color is class).

    Parameters
    ----------
    points: ndarray
        3d points to plot (shape: (n_samples, 3)).
    labels: ndarray
        classes (shape: (n_samples,)).

    Returns
    -------
    None
    """
    fig = plt.figure(figsize=(8, 8))
    ax = fig.add_subplot(111, projection="3d")
    ax.scatter(points[:, 0], points[:, 1], points[:, 2], alpha=0.5, c=labels, s=60)
    plt.show(block=True)


def show_nearest_neighbor(
    X_train: np.array,
    Y_train: np.ndarray,
    X_test: np.array,
    Y_test: np.ndarray,
    nearest_neighbors: np.array,
):
    # visualize nearest neighbors
    _, (ax0, ax1) = plt.subplots(1, 2)

    while True:
        # extract random index
        test_idx = np.random.randint(0, X_test.shape[0])

        X_cur, Y_cur = X_test[test_idx], Y_test[test_idx]

        actor = int(
            nearest_neighbors[test_idx] / 20
        )  # PER PRENDERE INDICE NOME ATTORE E IMMAGINE SIMILE!!!!
        row = nearest_neighbors[test_idx] - actor * 20

        X = get_image(actor, row)
        X_cur_pred, Y_cur_pred = X, Y_train[nearest_neighbors[test_idx]]

        ax0.imshow(X_cur.permute(1, 2, 0))  # , cmap='gray')
        ax0.set_title(f"Test face - ID {int(Y_cur)}")

        color = "r" if Y_cur != Y_cur_pred else "g"

        ax1.imshow(X_cur_pred.permute(1, 2, 0))  # , cmap='gray')
        ax1.set_title(f"Nearest neighbor - ID {int(Y_cur_pred)}", color=color)

        # plot faces
        plt.waitforbuttonpress()


class NearestNeighbor:
    def __init__(self, n_neighbors=8):
        self._X_db, self._Y_db = None, None
        self._num_neighbors = n_neighbors

    def fit(
        self,
        X: np.ndarray,
        y: np.ndarray,
    ):
        """
        Fit the model using X as training data and y as target values
        """
        self._X_db = X
        self._Y_db = y

    def predict(self, X: np.ndarray):
        """
        Finds the 1-neighbor of a point. Returns predictions as well as indices of
        the neighbors of each point.
        """

        n_neighbors = self._num_neighbors

        # predict test faces
        predictions = np.zeros((1,))

        distances = np.sum(np.square(self._X_db - X), axis=1)
        # print(distances)

        sorted_indexes = distances.argsort()
        # print(sorted_indexes)
        # print(distances[sorted_indexes])

        # nearest neighbor classification
        # nearest_neighbor = np.argmin(distances)
        nearest_neighbors = sorted_indexes[:n_neighbors]  # PER LA MEDIANA!!!!
        nearest_neighbors_indexes = nearest_neighbors.copy()
        print(nearest_neighbors_indexes)
        pred = {}
        for i in range(n_neighbors):
            nearest_neighbors[i] = self._Y_db[nearest_neighbors[i]]

            if nearest_neighbors[i] not in pred.keys():
                pred[nearest_neighbors[i]] = 1
            else:
                pred[nearest_neighbors[i]] += 1

        max_votes = max(pred.values())
        max_voted = 0
        for i in range(n_neighbors):
            if pred[nearest_neighbors[i]] == max_votes:
                max_voted = nearest_neighbors[i]
                break

        # median = int(np.median(nearest_neighbors))
        # print(median)

        # predictions = self._Y_db[max_voted]
        predictions = max_voted

        return predictions, nearest_neighbors, nearest_neighbors_indexes
