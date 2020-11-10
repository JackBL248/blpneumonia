import torch
from torch import nn
import torch.optim as optim

from args import parser
from data_prep import get_transforms, get_dataloader
from model import Model


def main():
    args = parser.parse_args()
    # create transform objects for data augmentation
    train_transforms, test_transforms = get_transforms(
        args.rotation, args.scale, args.shear)
    # create dataloaders
    train_dataloader = get_dataloader(
        args.root_path, "train", train_transforms, args.batch, args.workers)
    val_dataloader = get_dataloader(
        args.root_path, "val", test_transforms, args.batch, args.workers)
    test_dataloader = get_dataloader(
        args.root_path, "test", test_transforms, args.batch, args.workers)

    dataloaders = {"train": train_dataloader, "val": val_dataloader}

    # set CUDA as device if available
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    # define model architecture, dropout, cost function and optimizer
    model = Model(args.model, args.dropout, device, args.log)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.model.parameters(), lr=args.lr)

    # write the model arch, lr, dropout
    # and data augmentation parameters to file
    with open(args.log, "a+") as f:
        f.write("model architecture:%s learning rate:%.3f dropout:%.2f\n" % (
            args.model,
            args.lr,
            args.dropout
            )
        )
        f.write("rotation:%d shear:%d scale: " % (
            args.rotation,
            args.shear,
            )
        )
        for i in args.scale:
            f.write("%d " % i)
        f.write("\n")

    # train model
    model.train(
        criterion,
        optimizer,
        dataloaders,
        args.epochs,
        args.patience
    )
    # test model
    model.test(test_dataloader)


if __name__ == '__main__':
    main()
