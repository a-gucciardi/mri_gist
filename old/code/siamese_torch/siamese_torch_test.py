from __future__ import print_function
import argparse, time, pickle
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from collections import defaultdict
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

import random
import gc
import wandb
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
from torchvision import transforms as T
from torch.optim.lr_scheduler import StepLR

from datasets import load_dataset, DatasetDict, Dataset
from torch.utils.data import DataLoader
from PIL import Image
import utils as ut

import siamese_models as sm
# Currently using simple dataset : BCELoss is sufficient (no triplet loss yet)
# Tracking enablable with wandb

BASE_EPOCH = 5
BASE_LR = 0.01
BASE_GAMMA = 0.7
BASE_LOG = 10 # how many batches to wait before logging training status

def train(args, model, device, train_loader, optimizer, epoch):
    model.train()
    # we are using BCELoss 
    criterion = nn.BCELoss()
    # running_loss, last_loss = 0., 0.

    for batch_idx, i in enumerate(train_loader):
        images_1 = i['slice1'].to(device, dtype=torch.float32)#.unsqueeze(1)
        images_2 = i['slice2'].to(device, dtype=torch.float32)#.unsqueeze(1)
        targets = i['label'].to(device, dtype=torch.float32)

        optimizer.zero_grad()
        #print("shapes", images_1.shape, images_2.shape)
        outputs = model(images_1, images_2).squeeze()
        train_loss = criterion(outputs, targets)
        train_loss.backward()
        optimizer.step()

        # running_loss += train_loss.item()

        correct = 0
        pred = torch.where(outputs > 0.5, 1, 0)  # get the index of the max log-probability
        correct += pred.eq(targets.view_as(pred)).sum().item()
        train_acc = 100. * correct / len(images_1) # 64 batch size

        if batch_idx % BASE_LOG == BASE_LOG-1:
            # last_loss = train_loss / 32 # loss per batch
            print(f"Epoch {epoch} Batch {batch_idx+1}\t {100. * batch_idx / len(train_loader):.1f}%\t Loss {train_loss.item():.6f}\t Train Accuracy {train_acc}")
            # print(outputs, targets)

            # print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}\t Accuracy: ({:.0f}%)'.format(
            #     epoch, batch_idx * len(images_1), len(train_loader.dataset),
            #     100. * batch_idx / len(train_loader), train_loss.item(), train_acc))
            # # if args.dry_run:
            # running_loss = 0.

    return train_loss, train_acc

def evaluate(model, device, data_loader):
    model.eval()
    all_targets = []
    all_predictions = []

    with torch.no_grad():
        for i in data_loader:
            images_1 = i['slice1'].to(device, dtype=torch.float32)#.unsqueeze(1)
            images_2 = i['slice2'].to(device, dtype=torch.float32)#.unsqueeze(1)
            targets = i['label'].to(device, dtype=torch.float32)

            outputs = model(images_1, images_2).squeeze()
            predictions = (outputs > 0.5).float()

            all_targets.extend(targets.cpu().numpy())
            all_predictions.extend(predictions.cpu().numpy())

    all_targets = np.array(all_targets)
    all_predictions = np.array(all_predictions)

    accuracy = accuracy_score(all_targets, all_predictions)
    precision = precision_score(all_targets, all_predictions)
    recall = recall_score(all_targets, all_predictions)
    f1 = f1_score(all_targets, all_predictions)

    return {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1
    }

def test(model, device, test_loader):
    model.eval()
    test_loss, correct = 0, 0
    criterion = nn.BCELoss()

    all_targets = []
    all_predictions = []

    with torch.no_grad():
        for i in test_loader:
            images_1 = i['slice1'].to(device, dtype=torch.float32)#.unsqueeze(1)
            images_2 = i['slice2'].to(device, dtype=torch.float32)#.unsqueeze(1)
            targets = i['label'].to(device, dtype=torch.float32)

            outputs = model(images_1, images_2).squeeze()
            test_loss += criterion(outputs, targets).sum().item()  # sum up batch loss

            predictions = (outputs > 0.5).float()
            all_targets.extend(targets.cpu().numpy())
            all_predictions.extend(predictions.cpu().numpy())

    all_targets = np.array(all_targets)
    all_predictions = np.array(all_predictions)

    accuracy = accuracy_score(all_targets, all_predictions)
    precision = precision_score(all_targets, all_predictions)
    recall = recall_score(all_targets, all_predictions)
    f1 = f1_score(all_targets, all_predictions)
    cm = confusion_matrix(all_targets, all_predictions)

    # print(f'\nVal set: Average loss: {test_loss:.4f}')
    # print(f'Accuracy: {accuracy:.4f}')
    # print(f'Precision: {precision:.4f}')
    # print(f'Recall: {recall:.4f}')
    # print(f'F1 Score: {f1:.4f}')
    # print('Confusion Matrix:')
    # print(cm)

    return test_loss, {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'confusion_matrix': cm,
        'all_targets': all_targets,
        'all_predictions': all_predictions
    }

def compute_confusion_matrix(model, device, data_loader):
    model.eval()
    all_preds = []
    all_targets = []
    with torch.no_grad():
        for i in data_loader:
            images_1 = i['slice1'].to(device, dtype=torch.float32).unsqueeze(1)
            images_2 = i['slice2'].to(device, dtype=torch.float32).unsqueeze(1)
            targets = i['label'].to(device, dtype=torch.float32)

            outputs = model(images_1, images_2).squeeze()
            predictions = (outputs > 0.5).float()

            all_targets.extend(targets.cpu().numpy())
            all_preds.extend(predictions.cpu().numpy())

    return confusion_matrix(all_targets, all_preds)

def visualize_tsne(model, device, test_loader, epoch, wandb=True):
    """
        Attempt at embeddings(after siamese pass) visualization with dim reduction. 
    """
    model.eval()
    features = []
    labels = []

    with torch.no_grad():
        for i in test_loader:
            images_1 = i['slice1'].to(device, dtype=torch.float32).unsqueeze(1)
            images_2 = i['slice2'].to(device, dtype=torch.float32).unsqueeze(1)
            targets = i['label'].to(device, dtype=torch.float32)

            # Extract features
            feat1 = model.forward_once(images_1)
            feat2 = model.forward_once(images_2)
            combined_features = torch.cat((feat1, feat2), dim=1)

            features.extend(combined_features.cpu().numpy())
            labels.extend(targets.cpu().numpy())

    features = np.array(features)
    labels = np.array(labels)

    # t-SNE Visualization
    tsne = TSNE(n_components=2, random_state=0)
    reduced_features = tsne.fit_transform(features)
    print(reduced_features.shape)

    # plt.figure(figsize=(10, 8))
    fig, axes = plt.subplots(3, 1, figsize=(20, 8))
    scatter = axes[0].scatter(reduced_features[:, 0], reduced_features[:, 1], c=labels, cmap='viridis')
    axes[0].set_title(f't-SNE Visualization - Epoch {epoch}')
    axes[1].imshow(images_1[0].cpu().squeeze().numpy())
    axes[1].set_title(f'image 1 - Epoch {epoch}')
    axes[2].imshow(images_2[0].cpu().squeeze().numpy())
    axes[2].set_title(f'image 2 - Epoch {epoch}')
    plt.colorbar(scatter)
    plt.savefig(f'tsneepoch{epoch}.png')
    plt.close()

    # Feature Importance (assuming the last layer is named 'fc')
    importance = model.fc[-1].weight.abs().cpu().detach().numpy()
    plt.figure(figsize=(12, 6))
    plt.bar(range(len(importance[0])), importance[0])
    plt.title(f'Feature Importance - Epoch {epoch}')
    plt.savefig(f'feature_importanceepoch{epoch}.png')
    plt.close()

    # You can add more visualizations here

    if wandb:
        wandb.log({
            "t-SNE Visualization": wandb.Image(f'tsneepoch{epoch}.png'),
            "Feature Importance": wandb.Image(f'feature_importanceepoch{epoch}.png')
        })

def compute_saliency_maps(model, image1, image2):
    model.eval()
    image1.requiresgrad()
    image2.requiresgrad()

    output = model(image1, image2)
    output_sum = output.sum()

    output_sum.backward()

    saliency1 = image1.grad.abs()
    saliency2 = image2.grad.abs()

    return saliency1, saliency2

def visualize_saliency_maps(model, device, test_loader, epoch, num_examples=5, wandb=True):
    """
        Attempt at saliency map creation and visualization.
    """
    all_images_1, all_images_2, all_labels = [], [], []

    for i in test_loader:
        images_1 = i['slice1'].to(device, dtype=torch.float32).unsqueeze(1)
        images_2 = i['slice2'].to(device, dtype=torch.float32).unsqueeze(1)
        targets = i['label'].to(device, dtype=torch.float32)
        all_images_1.extend(images_1)
        all_images_2.extend(images_2)
        all_labels.extend(targets)
        if len(all_images_1) >= 100:  # Collect 100 examples
            break

    # Randomly select num_examples
    indices = random.sample(range(len(all_images_1)), num_examples)

    fig, axes = plt.subplots(num_examples, 4, figsize=(20, 5*num_examples))

    for i, idx in enumerate(indices):
        image_1 = all_images_1[idx].unsqueeze(0).to(device, dtype=torch.float32)#.unsqueeze(1)
        image_2 = all_images_2[idx].unsqueeze(0).to(device, dtype=torch.float32)#.unsqueeze(1)
        label = all_labels[idx].item()

        saliency1, saliency2 = compute_saliency_maps(model, image_1, image_2)

        axes[i, 0].imshow(image_1.squeeze().cpu().detach().numpy(), cmap='gray')
        axes[i, 0].set_title(f'Image 1 (Label: {label})')
        axes[i, 1].imshow(saliency1.squeeze().cpu().detach().numpy(), cmap='hot')
        axes[i, 1].set_title('Saliency Map 1')
        axes[i, 2].imshow(image_2.squeeze().cpu().detach().numpy(), cmap='gray')
        axes[i, 2].set_title('Image 2')
        axes[i, 3].imshow(saliency2.squeeze().cpu().detach().numpy(), cmap='hot')
        axes[i, 3].set_title('Saliency Map 2')

    plt.tight_layout()
    plt.savefig(f'saliency_mapsepoch{epoch}.png')
    plt.close()

    if wandb:
        wandb.log({f"Saliency Maps Epoch {epoch}": wandb.Image(f'saliency_mapsepoch{epoch}.png')})

def main():
    # Training settings
    parser = argparse.ArgumentParser(description='PyTorch Siamese network Example')
    parser.add_argument('--batch-size', type=int, default=32, metavar='N',
                        help='input batch size for training (default: 32)')
    parser.add_argument('--epochs', type=int, default=5, metavar='N',
                        help='number of epochs to train (default: 10)')
    parser.add_argument('--no-cuda', action='store_true', default=False,
                        help='disables CUDA training')
    parser.add_argument('--seed', type=int, default=1, metavar='S',
                        help='random torch seed (default: 1)')
    parser.add_argument('--save', action='store_true', default=True,
                        help='For Saving the current Model')
    parser.add_argument('--wandb', action='store_true', default=False,
                        help='Weight N Biases tracking')
    parser.add_argument('--name', type=str, default="siamese run",
                        help='wandb run name')
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    # device
    use_cuda = not args.no_cuda and torch.cuda.is_available()
    if use_cuda:
        device = torch.device("cuda:2")
    else:
        device = torch.device("cpu")

    #### DATA load
    print("Loading data, might take a few minutes the first time")
    ds_folds = ut.siamese_noise_dataset_fold_range(test_size=0.1, noise_size=5, n_splits=2)
    print("Data load completed")
    ### 

    # list models architectures
    # models = [SiameseNetworkResnet(), SiameseNetworkMobnet(), SiameseNetworkNext(), SiameseNetworkVGGnet(), SiameseNetworkEffnet()]
    # models = [sm.SiameseNetworkResnet()]
    # models = {
    # 'ResNet': SiameseNetworkResnet,
    # 'MobileNet': SiameseNetworkMobnet,
    # 
    # models = {'ResNet': sm.SiameseNetworkResnet(), 'MobileNet': sm.SiameseNetworkMobnet(), 'ResNeXt': sm.SiameseNetworkNext(),
    #           'VGGNet': sm.SiameseNetworkVGGnet(), 'EffNet': sm.SiameseNetworkEffnet() }
    # models = {'ResNet': sm.SiameseNetworkResnetTest(), 'ResNeXt': sm.SiameseNetworkNextTest(), 'EffNet': sm.SiameseNetworkEffnetTest(),
    #           'MobNet': sm.SiameseNetworkMobnetTest(), 'VGGNet': sm.SiameseNetworkVGGnetTest(), 'VIT': sm.SiameseNetworkVitTest()}
    models = {'VGGNet': sm.SiameseNetworkVGGnetTest()}
    # train for each fold
    # for m in models:
    for model_name in models:
         # store results per fold
        fold_results = defaultdict(list)
        best_models = []
        # print(f"\nTraining with {m}")
        print(f"\nTraining with {model_name}")
        for fold, ds in enumerate(ds_folds):
            print(f"Training on fold {fold+1}")

            train_loader = torch.utils.data.DataLoader(ds['train'].with_format("torch"), batch_size=args.batch_size)
            val_loader = torch.utils.data.DataLoader(ds['valid'].with_format("torch"), batch_size=args.batch_size)

            # Initialize model, optimizer, and scheduler for each fold
            # model = models[m]
            if model_name == "ResNet": model = sm.SiameseNetworkResnet()
            if model_name == "MobileNet": model = sm.SiameseNetworkMobnet()
            if model_name == "ResNeXt": model = sm.SiameseNetworkNext()
            if model_name == "VGGNet": model = sm.SiameseNetworkVGGnet()
            if model_name == "EffNet": model = sm.SiameseNetworkEffnet()
            if model_name == "VIT": model = sm.SiameseNetworkVitTest()

            model.to(device)
            optimizer = optim.Adadelta(model.parameters(), lr=BASE_LR)
            scheduler = StepLR(optimizer, step_size=1, gamma=BASE_GAMMA)

            config = {
                # "architecture": f"{m}",
                "architecture": f"{model_name}",
                "dataset": "MRI_T1",
                "learning_rate": BASE_LR,
                "epochs": BASE_EPOCH,
                "optimizer": optimizer,
                "loss": nn.BCELoss()
            }

            if args.wandb:
                # run = wandb.init(project="mri_siamese", config=config, name=f"{args.name}_{m}_fold{fold+1}", reinit=True)
                run = wandb.init(project="mri_siamese", config=config, name=f"{args.name}_{model_name}_fold{fold+1}", reinit=True)

            best_val_loss = float('inf')
            best_model = None

            for epoch in range(1, BASE_EPOCH + 1):
                train_loss, train_accuracy = train(args, model, device, train_loader, optimizer, epoch)
                val_loss, val_metrics = test(model, device, val_loader)

                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    best_model = model.state_dict()

                if args.wandb:
                    wandb.log({
                        "Fold": fold+1,
                        "Epoch": epoch,
                        "Training loss": train_loss,
                        "Training accuracy": train_accuracy,
                        "Validation loss": val_loss,
                        "Validation accuracy": val_metrics['accuracy'],
                        "Validation precision": val_metrics['precision'],
                        "Validation recall": val_metrics['recall'],
                        "Validation F1 score": val_metrics['f1']
                    })
                scheduler.step()

            # Store the best model for this fold
            print(f"Best model from fold {fold+1}")
            best_models.append(best_model)

            # Store results for this fold
            fold_results['val_loss'].append(best_val_loss)
            fold_results['val_accuracy'].append(val_metrics['accuracy'])
            fold_results['val_precision'].append(val_metrics['precision'])
            fold_results['val_recall'].append(val_metrics['recall'])
            fold_results['val_f1'].append(val_metrics['f1'])

            if args.wandb:
                wandb.log({
                    "Fold": fold+1,
                    "Best Validation loss": best_val_loss,
                    "Best Validation accuracy": val_metrics['accuracy'],
                    "Best Validation precision": val_metrics['precision'],
                    "Best Validation recall": val_metrics['recall'],
                    "Best Validation F1 score": val_metrics['f1']
                })
                run.finish()

            torch.cuda.empty_cache()

        # Average results across all folds
        print("\nAverage validation results across all folds:")
        print(f"Validation Loss: {np.mean(fold_results['val_loss']):.4f} (+/- {np.std(fold_results['val_loss']):.4f})")
        print(f"Validation Accuracy: {np.mean(fold_results['val_accuracy']):.4f}")  # (+/- {np.std(fold_results['val_accuracy']):.4f})")
        print(f"Validation Precision: {np.mean(fold_results['val_precision']):.4f}")# (+/- {np.std(fold_results['val_precision']):.4f})")
        print(f"Validation Recall: {np.mean(fold_results['val_recall']):.4f}")      # (+/- {np.std(fold_results['val_recall']):.4f})")
        print(f"Validation F1 Score: {np.mean(fold_results['val_f1']):.4f}")        # (+/- {np.std(fold_results['val_f1']):.4f})")

        # best model based on validation performance
        best_fold = np.argmin(fold_results['val_loss'])
        best_overall_model = best_models[best_fold]
        if args.save:
            torch.save(best_overall_model, model_path:=f"./models/kfoldtest_lil2/{model_name}_s5.pt")
            print(f"Best model saved to {model_path}")

        print("\nFinal Evaluation:")
        # usage of the test set only once with the best val model
        test_loader = torch.utils.data.DataLoader(ds_folds[0]['test'].with_format("torch"), batch_size=args.batch_size)
        final_model = model.to(device)
        final_model.load_state_dict(best_overall_model)
        final_model.eval()
        test_loss, test_metrics = test(final_model, device, test_loader)

        print(f"\nFinal Test Results for: {model_name}")
        print(f"Test Loss: {test_loss:.4f}")
        print(f"Test Accuracy: {test_metrics['accuracy']:.4f}")
        print(f"Test Precision: {test_metrics['precision']:.4f}")
        print(f"Test Recall: {test_metrics['recall']:.4f}")
        print(f"Test F1 Score: {test_metrics['f1']:.4f}")
        print("Confusion Matrix:")
        print(test_metrics['confusion_matrix'])

        if args.wandb:
            wandb.init(project="mri_siamese", name=f"{args.name}_{model_name}_eval_final", reinit=True)
            wandb.log({
                "Final Test Loss": test_loss,
                "Final Test Accuracy": test_metrics['accuracy'],
                "Final Test Precision": test_metrics['precision'],
                "Final Test Recall": test_metrics['recall'],
                "Final Test F1 Score": test_metrics['f1'],
                "Final Test Confusion Matrix": wandb.plot.confusion_matrix(
                    preds=test_metrics['all_predictions'],
                    y_true=test_metrics['all_targets'],
                    class_names=['0', '1']
                )
            })
            wandb.finish()



# if __name__ == '__main__':
#     start = time.time()
#     main()
#     end = time.time()
#     print(f"\n Time : {end-start:.3f}")