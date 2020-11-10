import copy
import time

import torch
import torch.nn as nn
from torchvision import models


class Model(object):

    def __init__(self, model_type, dropout, device, log):
        self.device = device
        self.log = log
        if model_type == "alexnet":
            self.model = models.alexnet(pretrained=False)
            self.model.classifier[6] = nn.Sequential(
                nn.Dropout(dropout),
                nn.Linear(
                    in_features=4096,
                    out_features=2,
                ),
                nn.LogSoftmax(dim=1)
            )
            self.model = self.model.to(self.device)

        elif model_type == "resnet18":
            self.model = models.resnet18(pretrained=False)
            num_ftrs = self.model.fc.in_features
            classifier = nn.Sequential(
                nn.Dropout(dropout),
                nn.Linear(
                    in_features=num_ftrs,
                    out_features=2
                ),
                nn.LogSoftmax(dim=1)
            )
            self.model.fc = classifier
            self.model = self.model.to(self.device)

        elif model_type == "resnet34":
            self.model = models.resnet34(pretrained=False)
            num_ftrs = self.model.fc.in_features
            classifier = nn.Sequential(
                nn.Dropout(dropout),
                nn.Linear(
                    in_features=num_ftrs,
                    out_features=2
                ),
                nn.LogSoftmax(dim=1)
            )
            self.model.fc = classifier
            self.model = self.model.to(self.device)

        elif model_type == "resnet50":
            self.model = models.resnet50(pretrained=False)
            num_ftrs = self.model.fc.in_features
            classifier = nn.Sequential(
                nn.Dropout(dropout),
                nn.Linear(
                    in_features=num_ftrs,
                    out_features=2
                ),
                nn.LogSoftmax(dim=1)
            )
            self.model.fc = classifier
            self.model = self.model.to(self.device)

        elif model_type == "densenet121":
            self.model = models.densenet121(pretrained=False)
            classifier = nn.Sequential(
                nn.Dropout(dropout),
                nn.Linear(
                    in_features=1024,
                    out_features=2
                ),
                nn.LogSoftmax(dim=1)
            )
            self.model.classifier = classifier
            self.model = self.model.to(self.device)

        else:
            raise ValueError("Model type not recognised. \
                Please use one of the following models: alexnet, resnet18, \
                resnet34, resnet50 or densenet121")

    def train(
        self,
        criterion,
        optimizer,
        dataloaders,
        num_epochs,
        patience,
    ):
        """
        Trains a neural network with early stopping.
        -----------------------
        Args: PyTorch model, training criterion, optimizer, num epochs
        and early stopping patience

        Returns: trained model with the best weights saved
        -----------------------
        """
        # time the operation
        since = time.time()

        best_model_wts = copy.deepcopy(self.model.state_dict())
        best_acc = 0.0
        best_loss = 1000.00
        # Set counter for early stopping
        early_stop_counter = 0

        for epoch in range(num_epochs):
            print('Epoch {}/{}'.format(epoch, num_epochs - 1))
            print('-' * 10)

            # Each epoch has a training and validation phase
            for phase in ['train', 'val']:
                if phase == 'train':
                    self.model.train()  # Set model to training mode
                else:
                    self.model.eval()   # Set model to evaluate mode

                running_loss = 0.0
                running_corrects = 0

                # Iterate over data.
                for inputs, labels in dataloaders[phase]:
                    # convert labels to type Tensor.long
                    labels = torch.Tensor.long(labels)

                    inputs = inputs.to(self.device)
                    labels = labels.to(self.device)

                    # zero the parameter gradients
                    optimizer.zero_grad()

                    # forward
                    # track history if only in train
                    with torch.set_grad_enabled(phase == 'train'):
                        outputs = self.model(inputs)
                        _, preds = torch.max(outputs, 1)
                        loss = criterion(outputs, labels)

                        # backward + optimize only if in training phase
                        if phase == 'train':
                            loss.backward()
                            optimizer.step()

                    # statistics
                    running_loss += loss.item() * inputs.size(0)
                    running_corrects += torch.sum(preds == labels.data)
                # calculate loss and accuracy for epoch
                epoch_loss = running_loss / len(dataloaders[phase])
                epoch_acc = running_corrects.double() / len(dataloaders[phase])

                # print and write epoch info to file
                text = '{} Loss: {:.4f} Acc: {:.4f}'.format(
                    phase, epoch_loss, epoch_acc)
                print(text)
                with open(self.log, "a+") as f:
                    f.write(text+"\n")

                # deep copy the model
                if phase == 'val':
                    if epoch_acc > best_acc:
                        best_acc = epoch_acc
                        best_model_wts = copy.deepcopy(self.model.state_dict())

                    if epoch_loss < best_loss:
                        best_loss = epoch_loss
                        # Reset early stopping counter to zero if there is
                        # improvement in loss
                        early_stop_counter = 0

                    elif epoch_loss >= best_loss:
                        # Increment early stopping counter if there is no
                        # improvement in loss
                        early_stop_counter += 1

                # If loss has not improved for 15 epochs, stop the training
            if early_stop_counter == patience:
                print("No improvement in loss for 15 epochs. \n")
                break

        time_elapsed = time.time() - since
        # print and write to file info on time and best val accuracy
        time_text = 'Training complete in {:.0f}m {:.0f}s'.format(
            time_elapsed // 60, time_elapsed % 60)
        best_val_text = 'Best val Acc: {:4f}'.format(best_acc)
        print(time_text)
        print(best_val_text)
        with open(self.log, "a+") as f:
            f.write(time_text+"\n")
            f.write(best_val_text+"\n")

        # load best model weights
        self.model.load_state_dict(best_model_wts)


def test(
    self,
    dataloader
):
    """
    Tests a trained neural network on a test dataset.
    -----------------------
    Args: dataloader

    Returns: results of the model run on the test dataset in the form of:
    - accuracy
    - sensitivity (True Positive rate)
    - specificity (True Negative rate)
    -----------------------
    """
    total = len(dataloader)
    test_correct = 0
    positives = 0
    positives_pred = 0
    negatives = 0
    negatives_pred = 0

    for data, labels in dataloader:
        data = data.to(self.device)
        labels = labels.to(self.device)
        outputs = self.model(data)
        _, preds = torch.max(outputs, 1)
        num_correct = int(torch.sum(preds == labels))
        test_correct += num_correct
        for label, pred in zip(labels, preds):
            if label == 1:
                # Calculate Sensitivity
                positives += 1
            if pred == 1:
                positives_pred += 1
            else:
                # Calculate Specificity
                negatives += 1
                if pred == 0:
                    negatives_pred += 1

    test_accuracy = test_correct / total
    test_sensitivity = positives_pred / positives
    test_specificity = negatives_pred / negatives
    # print and write test results to file
    test_acc_text = 'Test Acc: {:4f}'.format(test_accuracy)
    test_sens_text = 'Test Acc: {:4f}'.format(test_sensitivity)
    test_spec_text = 'Test Acc: {:4f}'.format(test_specificity)
    print(test_acc_text)
    with open(self.log, "a+") as f:
        f.write(test_acc_text+"\n")
        f.write(test_sens_text+"\n")
        f.write(test_spec_text+"\n")
        f.write("\n")
    return test_accuracy, test_sensitivity, test_sensitivity
