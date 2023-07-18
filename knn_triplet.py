"""
Eigenfaces main script.
"""

import numpy as np
import torch
from utils import show_eigenfaces
from utils import show_nearest_neighbor
from utils import get_data
from utils import PCA
from utils import NearestNeighbor
from lib import VGGFaceCNN
#from data_io import get_faces_dataset

import matplotlib.pyplot as plt
plt.ion()

def main():
    X_test, Y_test = get_data()
    
    Y_train = torch.zeros((544*10, ))
    actor_index = 0
    for i in range(544):
        for c in range(10):
            Y_train[actor_index] = i
            actor_index += 1

    ncomp = [i for i in range(10, 100)]
    acc = []
    x_test = torch.zeros((X_test.shape[0], 4096)).double()
    #device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    #cpu = torch.device('cpu')
    model = VGGFaceCNN().double()#.to(device)
    model_dict = torch.load('Weights_trained/vggface_triplet_trained_5.pth', map_location=lambda storage, loc: storage)
    model.load_state_dict(model_dict)
    print("CARICATO")
    x_train = torch.load("data_triplet.txt")
    x_train = x_train.detach().numpy()#x_train.cpu().detach().numpy()
    print(x_train.shape)
    #x_train = x_train.reshape((16*5, 4096))#512*7*7))
    _, x_test[:28] = model(X_test[:28])#.to(device))
    _, x_test[28:] = model(X_test[28:])#.to(device))
    x_test = x_test.detach().numpy()#x_test.cpu().detach().numpy()
    #x_test = x_test.reshape((16*3, 4096))#512*7*7))
    #x_train = np.asarray(x_train.to(cpu)).reshape((2,512*7*7))
    #x_test = np.asarray(x_test.to(cpu)).reshape((2,512*7*7))
    y_train = np.asarray(Y_train)
    y_test = np.asarray(Y_test)
    
    print("Senza PCA")
    nn = NearestNeighbor()
    nn.fit(x_train, y_train)

    predictions, nearest_neighbors = nn.predict(x_test)
    test_set_accuracy = float(np.sum(predictions == y_test))/len(predictions)
    print(f'Test set accuracy: {test_set_accuracy}')

    show_nearest_neighbor(None, Y_train,
                          X_test, Y_test, nearest_neighbors)

    print("Con PCA")
    max = 0
    max_acc = 0    

    # number of principal components
    for i in range(15, 26):
        print(i)
        n_components = i

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
        #print(f'Test set accuracy: {test_set_accuracy}, {i}')

        if test_set_accuracy > max_acc:
            max_acc = test_set_accuracy
            max = i
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

    n_components = max
    print(f"Max acc: {max_acc}, n_comp: {max}")

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
    # Show results.
    show_nearest_neighbor(None, Y_train,
                          X_test, Y_test, nearest_neighbors)


if __name__ == '__main__':
    main()

