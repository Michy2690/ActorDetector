"""
Eigenfaces main script.
"""

import numpy as np
import torch
from utils import show_eigenfaces
from utils import show_nearest_neighbor
from utils import get_dataset
from utils import PCA
from utils import NearestNeighbor
from lib import VGGFaceCNN
#from data_io import get_faces_dataset

import matplotlib.pyplot as plt
plt.ion()

def main():
    X_train, Y_train, X_test, Y_test = get_dataset()
    ncomp = [i for i in range(10, 200)]
    acc = []
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    #cpu = torch.device('cpu')
    model = VGGFaceCNN().double().to(device)
    model_dict = torch.load('models/vggface_CNN.pth', map_location=lambda storage, loc: storage)
    model.load_state_dict(model_dict)
    print("CARICATO")
    x_train = model(X_train.to(device))
    x_train = x_train.cpu().detach().numpy()
    print(x_train.shape)
    #x_train = x_train.reshape((16*5, 4096))#512*7*7))
    x_test = model(X_test.to(device))
    x_test = x_test.cpu().detach().numpy()
    #x_test = x_test.reshape((16*3, 4096))#512*7*7))
    #x_train = np.asarray(x_train.to(cpu)).reshape((2,512*7*7))
    #x_test = np.asarray(x_test.to(cpu)).reshape((2,512*7*7))
    y_train = np.asarray(Y_train)
    y_test = np.asarray(Y_test)
    
        

    # number of principal components
    n_components = 30

    # fit the PCA transform
    eigpca = PCA(n_components)
    eigpca.fit(x_train, verbose=False)

    # project the training data
    proj_train = eigpca.transform(x_train)

    # project the test data
    proj_test = eigpca.transform(x_test)

    # fit a 1-NN classifier on PCA features
    nn = NearestNeighbor()
    nn.fit(proj_train, y_train)

    # Compute predictions and indices of 1-NN samples for the test set
    predictions, nearest_neighbors = nn.predict(proj_test)

    # Compute the accuracy on the test set
    test_set_accuracy = float(np.sum(predictions == y_test))/len(predictions)
    acc.append(test_set_accuracy)
    print(f'Test set accuracy: {test_set_accuracy}')
    """
    nn = NearestNeighbor()
    nn.fit(x_train, y_train)
    predictions, nearest_neighbors = nn.predict(x_test)
    test_set_accuracy = float(np.sum(predictions == y_test))/len(predictions)
    print(f'Accuracy: {test_set_accuracy}') 
    """
    #plt.plot(ncomp, acc)
    #plt.waitforbuttonpress()
    #plt.close()
    # Show results.
    show_nearest_neighbor(X_train, Y_train,
                          X_test, Y_test, nearest_neighbors)


if __name__ == '__main__':
    main()

