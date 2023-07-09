import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
import torch.multiprocessing as mp
from torch.utils.data.distributed import DistributedSampler
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.distributed import init_process_group, destroy_process_group
import os
import pandas as pd
import torchvision.transforms as transforms
from DatasetTriplet_Par import Dataset_Scrub_Triplet
from lib import VGGFaceCNN

def ddp_setup(rank, world_size):
    """
    Args:
        rank: Unique identifier of each process
        world_size: Total number of processes
    """
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "12355"
    init_process_group(backend="nccl", rank=rank, world_size=world_size)
    torch.cuda.set_device(rank)

class Trainer:
    def __init__(
        self,
        model: torch.nn.Module,
        train_data: DataLoader,
        optimizer: torch.optim.Optimizer,
        gpu_id: int,
        save_every: int,
        loss_crit: torch.nn,
    ) -> None:
        self.gpu_id = gpu_id
        self.model = model.to(gpu_id)
        self.train_data = train_data
        self.optimizer = optimizer
        self.save_every = save_every
        self.loss_crit = loss_crit
        self.model = DDP(model, device_ids=[gpu_id])
        self.running_loss = []

    def _run_batch(self, anchor_img, positive_img, negative_img):
        self.optimizer.zero_grad()
        anchor_img = anchor_img.double()
        positive_img = positive_img.double()
        negative_img = negative_img.double()
        _, anchor_out = self.model(anchor_img)
        _, positive_out = self.model(positive_img)
        _, negative_out = self.model(negative_img)
        
        loss = self.loss_crit(anchor_out, positive_out, negative_out)
        loss.backward()
        self.optimizer.step()
        self.running_loss.append(loss.cpu().detach().numpy())
        #print(np.sum(self.running_loss))

    def _run_epoch(self, epoch):
        b_sz = len(next(iter(self.train_data))[0])
        print(f"[GPU{self.gpu_id}] Epoch {epoch} | Batchsize: {b_sz} | Steps: {len(self.train_data)}")
        self.train_data.sampler.set_epoch(epoch)
        i = 0
        for anchor_img, positive_img, negative_img, anchor_label in self.train_data:
            #print("carico")
            anchor_img = anchor_img.to(self.gpu_id)
            positive_img = positive_img.to(self.gpu_id)
            negative_img = negative_img.to(self.gpu_id)
            self._run_batch(anchor_img, positive_img, negative_img)
            i+=1
            if i % 15 == 0:
                print(i)

    def _save_checkpoint(self, epoch):
        ckp = self.model.module.state_dict()
        PATH = "./Weights_triplet_all_layers/checkpoint_{}.pth".format(epoch+15)
        torch.save(ckp, PATH)
        print(f"Epoch {epoch} | Training checkpoint saved at {PATH}")

        with open("Weights_triplet_all_layers/ris.txt", "a") as f:
            f.write("{}\nsum: {}\nmean: {}\n".format(epoch+15, np.sum(self.running_loss), np.mean(self.running_loss)))
            f.close()
        self.running_loss = []

    def train(self, max_epochs: int):
        for epoch in range(max_epochs):
            self._run_epoch(epoch)
            if self.gpu_id == 0 and epoch % self.save_every == 0:
                self._save_checkpoint(epoch)


def load_train_objs():
    transform = transforms.Compose([transforms.RandomHorizontalFlip(),
                    transforms.RandomGrayscale(p=0.2),
                    transforms.GaussianBlur(kernel_size=9),
                    transforms.ToTensor(),
                    transforms.Normalize((0.5,), (0.5,))
    ])

    DATASET_PATH = "../Dataset_nuovo"

    train_data = pd.read_csv('dataset.csv')
    train_set = Dataset_Scrub_Triplet(train_data, DATASET_PATH, True, transform)    # load your dataset
    model = VGGFaceCNN().double()
    #model_dict = torch.load('models/vggface_CNN.pth', map_location=lambda storage, loc: storage)
    model_dict = torch.load('Weights_triplet_all_layers/checkpoint_14.pth', map_location=lambda storage, loc: storage)
    model.load_state_dict(model_dict)

    layers_new = []
    i=1
    for layer in model.children():
        for param in layer.parameters():
            if i > 26:
                layers_new.append(param)
            i+=1

    optimizer = torch.optim.SGD(model.parameters(), lr=1e-3)
    loss_crit = torch.nn.TripletMarginLoss()

    return train_set, model, optimizer, loss_crit


def prepare_dataloader(dataset: Dataset, batch_size: int):
    return DataLoader(
        dataset,
        batch_size=batch_size,
        pin_memory=True,
        shuffle=False,
        sampler=DistributedSampler(dataset)
    )


def main(rank: int, world_size: int, save_every: int, total_epochs: int, batch_size: int):
    ddp_setup(rank, world_size)
    dataset, model, optimizer, loss_crit = load_train_objs()
    train_data = prepare_dataloader(dataset, batch_size)
    trainer = Trainer(model, train_data, optimizer, rank, save_every, loss_crit)
    trainer.train(total_epochs)
    destroy_process_group()


if __name__ == "__main__":
    save_every = 1
    total_epochs = 15
    batch_size = 16
    
    world_size = torch.cuda.device_count()
    mp.spawn(main, args=(world_size, save_every, total_epochs, batch_size), nprocs=world_size)
