"""
Script to perform hyperparameter optimization on a model
"""
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.models as models
import torchvision.transforms as transforms

import argparse
import logging
import os
import sys
from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)
logger.addHandler(logging.StreamHandler(sys.stdout))


def test(model, test_loader, criterion, device):
    """Evaluate model"""
    # set model to evaluation mode
    model.eval()

    # initialize variables to keep track of loss and accuracy during testing
    running_loss = 0
    running_corrects = 0

    # iterate over test data in batches
    for inputs, labels in test_loader:
        # move inputs and labels to GPU if available
        inputs = inputs.to(device)
        labels = labels.to(device)

        # forward pass through the network
        outputs = model(inputs)

        # calculate loss
        loss = criterion(outputs, labels)

        # calculate number of correct predictions
        _, preds = torch.max(outputs, 1)
        running_loss += loss.item() * inputs.size(0)
        running_corrects += torch.sum(preds == labels.data).item()

    # calculate average loss and accuracy over all test data
    total_loss = running_loss / len(test_loader.dataset)
    total_acc = running_corrects / len(test_loader.dataset)

    # log testing results
    logger.info(f"Testing Loss: {total_loss}")
    logger.info(f"Testing Accuracy: {total_acc}")


def train(model,
          train_loader,
          validation_loader,
          criterion,
          optimizer,
          device,
          early_stopping):
    """Train model"""

    # set number of epochs and initialize some variables
    epochs = 2
    best_loss = 1e6
    image_dataset = {'train': train_loader, 'valid': validation_loader}
    loss_counter = 0

    # loop over epochs
    for epoch in range(epochs):
        # loop over training and validation phases
        for phase in ['train', 'valid']:
            print(f"Epoch {epoch}, Phase {phase}")

            # set model to appropriate mode
            if phase == 'train':
                model.train()
            else:
                model.eval()

            # initialize some variables for this phase
            running_loss = 0.0
            running_corrects = 0
            running_samples = 0

            # iterate over batches of data
            for step, (inputs, labels) in enumerate(image_dataset[phase]):
                # move inputs and labels to GPU if available
                inputs = inputs.to(device)
                labels = labels.to(device)

                # forward pass through the network
                outputs = model(inputs)

                # calculate loss
                loss = criterion(outputs, labels)

                # take a backward step and update parameters during training phase
                if phase == 'train':
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()

                # calculate number of correct predictions
                _, preds = torch.max(outputs, 1)
                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data).item()
                running_samples += len(inputs)

                # optionally log progress during training
                if running_samples % 2000 == 0:
                    accuracy = running_corrects / running_samples
                    print("Images [{}/{} ({:.0f}%)] Loss: {:.2f} Accuracy: {}/{} ({:.2f}%)".format(
                            running_samples,
                            len(image_dataset[phase].dataset),
                            100.0 * (running_samples / len(image_dataset[phase].dataset)),
                            loss.item(),
                            running_corrects,
                            running_samples,
                            100.0 * accuracy,
                        )
                    )

                # break loop to train and test on subset of dataset
                if running_samples > (0.2 * len(image_dataset[phase].dataset)):
                    break

            # calculate average loss and accuracy over this phase
            epoch_loss = running_loss / running_samples
            epoch_acc = running_corrects / running_samples

            # update best loss if appropriate
            if phase == 'valid':
                if epoch_loss < best_loss:
                    best_loss = epoch_loss
                else:
                    loss_counter += 1

            # log progress for this phase
            logger.info('{} loss: {:.4f}, acc: {:.4f}, best loss: {:.4f}'.format(phase,
                                                                                        epoch_loss,
                                                                                        epoch_acc,
                                                                                        best_loss))
        # check for early stopping
        if loss_counter == early_stopping:
            logger.info('Early stopping')
            break

    return model


def net():
    """Network initialization"""
    # initialize pre-trained ResNet-50 model
    model = models.resnet50(pretrained=True)

    # freeze parameters so we don't backpropagate through them
    for param in model.parameters():
        param.requires_grad = False

    # replace fully-connected layer to match number of classes (133 in this case)
    num_features = model.fc.in_features
    model.fc = nn.Sequential(
                   nn.Linear(num_features, 128),
                   nn.ReLU(inplace=True),
                   nn.Linear(128, 133))
    return model


def model_fn(model_dir):
    """
    Loads a PyTorch model from the specified directory and returns it in eval mode.
    """
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model_path = os.path.join(model_dir, "model.pth")

    # Load the saved model parameters onto the CPU/GPU
    checkpoint = torch.load(model_path, map_location=device)
    model = net()
    model.load_state_dict(checkpoint)
    model.to(device)

    logger.info('Model loaded successfully')
    return model.eval()


def create_data_loaders(data, batch_size):
    """Create data loader"""
    train_data_path = os.path.join(data, 'train')
    test_data_path = os.path.join(data, 'test')
    validation_data_path = os.path.join(data, 'valid')

    train_transform = transforms.Compose([
        transforms.RandomResizedCrop((224, 224)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])

    test_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])

    train_data = torchvision.datasets.ImageFolder(
        root=train_data_path,
        transform=train_transform
    )
    train_data_loader = torch.utils.data.DataLoader(
        train_data,
        batch_size=batch_size,
        shuffle=True,
    )

    test_data = torchvision.datasets.ImageFolder(
        root=test_data_path,
        transform=test_transform
    )
    test_data_loader = torch.utils.data.DataLoader(
        test_data,
        batch_size=batch_size,
        shuffle=False,
    )

    validation_data = torchvision.datasets.ImageFolder(
        root=validation_data_path,
        transform=test_transform,
    )
    validation_data_loader = torch.utils.data.DataLoader(
        validation_data,
        batch_size=batch_size,
        shuffle=False,
    )

    return train_data_loader, test_data_loader, validation_data_loader


def main(args):
    logger.info(f'Hyperparameters are LR: {args.learning_rate}, Batch Size: {args.batch_size}, Early Stopping: {args.early_stopping_rounds}')
    logger.info(f'Data Paths: {args.data}')

    model = net()
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    criterion = nn.CrossEntropyLoss(ignore_index=133)
    optimizer = optim.Adam(model.fc.parameters(), lr=args.learning_rate)
    train_loader, test_loader, validation_loader = create_data_loaders(args.data,
                                                                       args.batch_size)
    model = model.to(device)

    logger.info("Training the model")
    model = train(model,
                  train_loader,
                  validation_loader,
                  criterion,
                  optimizer,
                  device,
                  early_stopping=args.early_stopping_rounds)

    logger.info("Testing the model")
    test(model, test_loader, criterion, device)

    logger.info("Saving the model")
    torch.save(model.state_dict(), os.path.join(args.model_dir, "model.pth"))


if __name__=='__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--learning-rate',
                        type=float,
                        default=0.001)
    parser.add_argument('--batch-size',
                        type=int,
                        default=32)
    parser.add_argument('--early-stopping-rounds',
                        type=int,
                        default=10)
    parser.add_argument('--data', type=str,
                        default=os.environ['SM_CHANNEL_TRAINING'])
    parser.add_argument('--model_dir',
                        type=str,
                        default=os.environ['SM_MODEL_DIR'])
    parser.add_argument('--output_dir',
                        type=str,
                        default=os.environ['SM_OUTPUT_DATA_DIR'])

    args = parser.parse_args()
    print(args)

    main(args)