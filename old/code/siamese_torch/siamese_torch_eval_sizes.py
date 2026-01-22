from __future__ import print_function
import argparse, time, pickle
import numpy as np
from collections import defaultdict
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR

import utils as ut
import siamese_models as sm

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


def train_and_evaluate_sizes(noise_sizes_train=[1,2,3], noise_sizes_eval=range(10), args=None):
    if args is None:
        args = argparse.Namespace(
            batch_size=32,
            epochs=BASE_EPOCH,
            no_cuda=False,
            seed=1,
            save=True,
            wandb=False,
            name="siamese run"
        )

    torch.manual_seed(args.seed)
    use_cuda = not args.no_cuda and torch.cuda.is_available()
    device = torch.device("cuda:2" if use_cuda else "cpu")
    
    models = {'VGGNet': sm.SiameseNetworkVGGnetTest()}
    all_results = {}

    for train_noise_size in noise_sizes_train:
        print(f"\nTraining with noise size {train_noise_size}")
        ds_folds = ut.siamese_noise_dataset_fold_range(test_size=0.1, noise_size=train_noise_size, n_splits=2)
        model_results = defaultdict(lambda: defaultdict(list))
        
        for model_name in models:
            print(f"\nTraining {model_name}")
            for fold, ds in enumerate(ds_folds):
                print(f"Training on fold {fold+1}")

                train_loader = torch.utils.data.DataLoader(ds['train'].with_format("torch"), batch_size=args.batch_size)
                val_loader = torch.utils.data.DataLoader(ds['valid'].with_format("torch"), batch_size=args.batch_size)

                if model_name == "VGGNet": 
                    model = sm.SiameseNetworkVGGnetTest()
                model.to(device)
                
                optimizer = optim.Adadelta(model.parameters(), lr=BASE_LR)
                scheduler = StepLR(optimizer, step_size=1, gamma=BASE_GAMMA)
                best_val_loss = float('inf')
                best_model = None

                for epoch in range(1, BASE_EPOCH + 1):
                    train_loss, train_accuracy = train(args, model, device, train_loader, optimizer, epoch)
                    val_loss, val_metrics = test(model, device, val_loader)

                    if val_loss < best_val_loss:
                        best_val_loss = val_loss
                        best_model = model.state_dict()

                    scheduler.step()

                if args.save:
                    model_path = f"./models/kfoldtest_lil2/{model_name}_train{train_noise_size}_fold{fold+1}.pt"
                    torch.save(best_model, model_path)
                    print(f"Best model saved to {model_path}")

                model.load_state_dict(best_model)
                model.eval()
                
                for eval_noise_size in noise_sizes_eval:
                    eval_ds = ut.siamese_noise_dataset_fold_range(test_size=0.1, noise_size=eval_noise_size, n_splits=2)[0]
                    eval_loader = torch.utils.data.DataLoader(eval_ds['test'].with_format("torch"), batch_size=args.batch_size)
                    test_loss, test_metrics = test(model, device, eval_loader)
                    
                    model_results[fold][eval_noise_size].append({
                        'loss': test_loss,
                        'accuracy': test_metrics['accuracy'],
                        'precision': test_metrics['precision'],
                        'recall': test_metrics['recall'],
                        'f1': test_metrics['f1']
                    })

                torch.cuda.empty_cache()

            avg_results = {}
            for eval_noise_size in noise_sizes_eval:
                avg_results[eval_noise_size] = {
                    metric: np.mean([fold[eval_noise_size][0][metric] for fold in model_results.values()])
                    for metric in ['loss', 'accuracy', 'precision', 'recall', 'f1']
                }

            all_results[train_noise_size] = avg_results

        print(f"\nResults for model trained with noise size {train_noise_size}:")
        for eval_noise_size, metrics in avg_results.items():
            print(f"\nEvaluation noise size {eval_noise_size}:")
            print(f"Test Loss: {metrics['loss']:.4f}")
            print(f"Test Accuracy: {metrics['accuracy']:.4f}")
            print(f"Test Precision: {metrics['precision']:.4f}")
            print(f"Test Recall: {metrics['recall']:.4f}")
            print(f"Test F1 Score: {metrics['f1']:.4f}")

    return all_results

def main():
    start = time.time()
    evals = [i for i in range(10)]
    results = train_and_evaluate_sizes(noise_sizes_train=evals, noise_sizes_eval=range(10))
    
    with open('noise_size_results_all.pkl', 'wb') as f:
        pickle.dump(results, f)
    
    end = time.time()
    print(f"\nTotal time: {end-start:.3f}")

if __name__ == '__main__':
    main()
