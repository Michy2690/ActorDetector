import torch
import numpy as np
from lib import VGGFaceCNN
import torchvision
import torchvision.transforms as transforms
import torch.optim as optim
from tqdm.notebook import tqdm
import os
import pandas as pd
from Dataset_Triplet import Dataset_Scrub_Triplet


if __name__ == "__main__":
    # Build VGGFace model and load pre-trained weights
    model = VGGFaceCNN().double()
    # model_dict = torch.load('models/vggface_CNN.pth', map_location=lambda storage, loc: storage)
    model_dict = torch.load(
        "Weights_trained/vggface_triplet_trained_5.pth",
        map_location=lambda storage, loc: storage,
    )
    model.load_state_dict(model_dict)

    # TRAIN ADDING NEW DATA
    mean = (0.4913997551666284, 0.48215855929893703, 0.4465309133731618)
    std = (0.24703225141799082, 0.24348516474564, 0.26158783926049628)
    # transform = transforms.Compose(
    #    [transforms.ToTensor(),
    #    transforms.Normalize(mean=mean, std=std) # your transforms here
    # ])
    transform = transforms.Compose(
        [
            transforms.RandomHorizontalFlip(),
            # transforms.RandomApply([
            # transforms.ColorJitter(brightness=0.5,
            #                        contrast=0.5,
            #                        saturation=0.5,
            #                        hue=0.1)
            # ], p=0.8),
            transforms.RandomGrayscale(p=0.2),
            transforms.GaussianBlur(kernel_size=9),
            transforms.ToTensor(),
            transforms.Normalize((0.5,), (0.5,)),
        ]
    )

    DATASET_PATH = "./Dataset_nuovo"

    train_data = pd.read_csv("dataset.csv")
    dataset = Dataset_Scrub_Triplet(train_data, DATASET_PATH, True, transform)

    # train_set = torchvision.datasets.ImageFolder(dataset, transform=transform)
    train_loader = torch.utils.data.DataLoader(
        dataset, batch_size=16, shuffle=False, num_workers=4
    )

    # test_set = torchvision.datasets.ImageFolder(dataset, transform=transform)
    # test_loader = torch.utils.data.DataLoader(test_set, batch_size=32, shuffle=True)

    epochs = 8
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("device:", device)
    model = model.to(device)
    layers_before = []
    layers_new = []

    i = 1
    for layer in model.children():
        for param in layer.parameters():
            if i > 26:
                layers_new.append(param)
            else:
                layers_before.append(param)
            i += 1

    # opt_old = optim.SGD(layers_before, lr=0.001)
    opt = optim.SGD(layers_new, lr=0.012)  # , momentum=0.8)
    # opt = optim.SGD(model.parameters(), lr=0.012)
    crit = torch.nn.TripletMarginLoss().to(device)  # loss criterion
    max = 0
    max_dict = None
    for e in range(epochs):
        # magic progress bar printer
        pbar = tqdm(total=len(train_loader), desc=f"Epoch {e} - 0%")

        running_loss = []
        # training loop
        for i, (anchor_img, positive_img, negative_img, anchor_label) in enumerate(
            tqdm(train_loader, desc="Training", leave=False)
        ):
            # forward pass goes here
            opt.zero_grad()
            anchor_img = anchor_img.double().to(device)
            positive_img = positive_img.double().to(device)
            negative_img = negative_img.double().to(device)
            # y = torch.Tensor([i+names_celeba for i in y]).long().to(device)
            _, anchor_out = model(anchor_img)
            _, positive_out = model(positive_img)
            _, negative_out = model(negative_img)
            loss = crit(anchor_out, positive_out, negative_out)
            loss.backward()
            # opt_old.step()
            opt.step()
            running_loss.append(loss.cpu().detach().numpy())
            if i % 20 == 0:
                print(i)
        print(
            "Epoch: {}/{} — Loss: {:.4f}".format(e + 1, epochs, np.mean(running_loss))
        )

        # Save trained weights
        model_file = os.path.join("./Weights_trained", "vggface_triplet_trained_7.pth")
        print("#. Save VGGFace weights at {}".format(model_file))
        # r = input("Salvare i pesi correnti? (s/n): ")
        # if r == 's':
        torch.save(model.state_dict(), model_file)
        print("Salvato")

        print(np.sum(running_loss))
        with open("Weights_trained/risMarco.txt", "a") as f:
            f.write(
                "\nmean: {}\nsum: {}".format(
                    np.mean(running_loss), np.sum(running_loss)
                )
            )
            f.close()
