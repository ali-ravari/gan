import torch
import pytorch_lightning as pl
from argparse import ArgumentParser
from torchvision.datasets import DatasetFolder
from PIL import Image
from torch.utils.data import DataLoader
from torchvision.transforms import ToTensor
from painting_gan import PaintingGan

def main():
    parser = ArgumentParser()
    parser.add_argument("--data", default="resizedImages", type=str, help="image directory")# -- means optional
    args = parser.parse_args()
    
    dataset = DatasetFolder(args.data, Image.open, transform= ToTensor(), extensions=(".jpg",))
    num_datapoints = len(dataset)
    test_datapoints = int(0.1*num_datapoints) # 10 percent of the data for training
    train_datapoints = num_datapoints - test_datapoints
    train, test = torch.utils.data.random_split(dataset, (train_datapoints, test_datapoints))
    train_dl = DataLoader(train, 32, shuffle=True, num_workers=11)
    test_dl = DataLoader(test, 32, num_workers=11)
    Model = PaintingGan(num_classes=len(dataset.classes))

    trainer = pl.Trainer(gpus= 1, max_epochs=100)
    trainer.fit(Model, train_dataloader= train_dl, val_dataloaders= test_dl)



if __name__ == "__main__":
    main()
