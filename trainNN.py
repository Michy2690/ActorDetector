"""
VGG FACE network pre-trained with LFW dataset
The last layer has been modified to 16 output parameters in order to classify with the actors in our dataset
In this script we performed a fine-tuning of the last two layers FC and saved final weights in a new folder.
"""
import argparse
import torch
import torch.nn.functional as F
import numpy as np
import cv2
from lib import VGGFace
import torchvision
import torchvision.transforms as transforms
import torch.optim as optim
from tqdm.notebook import tqdm
import os


if __name__ == '__main__':
    # Set up a parser for command line arguments
    parser = argparse.ArgumentParser("VGGFace demo script")
    parser.add_argument('--img', type=str, default='data/depp.jpg', help='input image file')
    # TODO: add CUDA acceleration
    parser.add_argument('--cuda', dest='cuda', action='store_true', help='use CUDA acceleration')
    parser.add_argument('--no-cuda', dest='cuda', action='store_false', help='do NOT use CUDA acceleration')
    parser.set_defaults(cuda=True)
    args = parser.parse_args()

    #names_celeba = 2622

    # Get names list
    names = [line.rstrip('\n') for line in open('data/names.txt')]
    print(len(names))

    # Build VGGFace model and load pre-trained weights
    model = VGGFace().double()
    model_dict = torch.load('models/vggface.pth', map_location=lambda storage, loc: storage)
    model.load_state_dict(model_dict)

    #TRAIN ADDING NEW DATA
    mean = (0.4913997551666284, 0.48215855929893703, 0.4465309133731618)
    std  = (0.24703225141799082, 0.24348516474564, 0.26158783926049628)
    transform = transforms.Compose(
        [transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std) # your transforms here
    ])

    DATASET_PATH = "./Dataset"
    actors = os.listdir(DATASET_PATH+'/TRAIN')

    classes = {i: actor for i, actor in enumerate(actors)}
    #print(classes)

    train_set = torchvision.datasets.ImageFolder(DATASET_PATH+'/TRAIN', transform=transform)
    train_loader = torch.utils.data.DataLoader(train_set, batch_size=16)#, shuffle=True)

    test_set = torchvision.datasets.ImageFolder(DATASET_PATH+'/TEST', transform=transform)
    test_loader = torch.utils.data.DataLoader(test_set, batch_size=16)#, shuffle=True)

    epochs = 8
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    layers_before = []
    layers_new = []

    i=1
    for layer in model.children():
        for param in layer.parameters():
            if i > 28:
                layers_new.append(param)
            else:
                layers_before.append(param)
            i+=1

    #opt_old = optim.SGD(layers_before, lr=0.001)
    opt = optim.SGD(layers_new, lr=0.005)#, momentum=0.9)
    crit = torch.nn.CrossEntropyLoss().to(device) # loss criterion

    for e in range(epochs):
        # magic progress bar printer
        pbar = tqdm(total=len(train_loader), desc=f'Epoch {e} - 0%')
        
        # training loop
        for i, (x, y) in enumerate(train_loader):
            
            # forward pass goes here
            opt.zero_grad()
            x, y = x.double().to(device), y.to(device)
            #y = torch.Tensor([i+names_celeba for i in y]).long().to(device)
            output = model(x)
            loss = crit(output, y)
            loss.backward()
            #opt_old.step()
            opt.step()

            # logging functions
            pbar.update(1)
            pbar.set_description(f'Epoch {e} - {round(i/len(train_loader) * 100)}% -- loss {loss.item():.2f}')
        
        # evaluation loop
        corr = 0
        with torch.no_grad():
            for x, y in test_loader:
                x, y = x.double().to(device), y.to(device)
                #y = torch.Tensor([i+names_celeba for i in y]).long().to(device)
                output = model(x)
                output = torch.max(output, 1)
                _, output = output
                corr += (output == y).sum().item()
        print(f"Accuracy for epoch {e}:{corr / len(test_set)}")


    # Save trained weights
    model_file = os.path.join('./Weights_trained', 'vggface_trained.pth')
    print("#. Save VGGFace weights at {}".format(model_file))
    r = input("Salvare i pesi correnti? (s/n): ")
    if r == 's':
        torch.save(model.state_dict(), model_file)
        print("Salvato")

    # Set model to evaluation mode
    model.eval()

#-------------------------------------------------------
	# Load test image and resize to 224x224
    img = cv2.imread(args.img)
    img = cv2.resize(img, (224, 224))

    # Forward test image through VGGFace
    img = torch.Tensor(img).permute(2, 0, 1).view(1, 3, 224, 224).double().to(device)
    img -= torch.Tensor(np.array([129.1863, 104.7624, 93.5940])).double().view(1, 3, 1, 1).to(device)
    predictions = F.softmax(model(img), dim=1)
    score, index = predictions.max(-1)
    print("Predicted id: {} (probability: {})".format(names[index], score.item()))

#-------------------------------------------------------
    # Load test image and resize to 224x224
    img = cv2.imread("data/ah.jpg")
    img = cv2.resize(img, (224, 224))

    # Forward test image through VGGFace
    img = torch.Tensor(img).permute(2, 0, 1).view(1, 3, 224, 224).double().to(device)
    img -= torch.Tensor(np.array([129.1863, 104.7624, 93.5940])).double().view(1, 3, 1, 1).to(device)
    predictions = F.softmax(model(img), dim=1)
    score, index = predictions.max(-1)
    print("Predicted id: {} (probability: {})".format(names[index], score.item()))

#-------------------------------------------------------
    # Load test image and resize to 224x224
    img = cv2.imread("data/scarlett.jpg")
    img = cv2.resize(img, (224, 224))

    # Forward test image through VGGFace
    img = torch.Tensor(img).permute(2, 0, 1).view(1, 3, 224, 224).double().to(device)
    img -= torch.Tensor(np.array([129.1863, 104.7624, 93.5940])).double().view(1, 3, 1, 1).to(device)
    predictions = F.softmax(model(img), dim=1)
    score, index = predictions.max(-1)
    print("Predicted id: {} (probability: {})".format(names[index], score.item()))
