import time

import torch.nn
from torch.utils.data import DataLoader

from mlp.helpers.helper_evaluation import compute_accuracy


def train_model(model, num_epochs, train_loader, valid_loader,
                test_loader, optimizer, device: str):
    start_time = time.time()
    minibatch_loss_list, train_acc_list, valid_acc_list = [], [], []

    for epoch in range(num_epochs):
        model.train()
        for idx, (features, targets) in enumerate(train_loader):

            features = features.to(device)
            targets = targets.to(device)

            # FORWARD AND BACKWARD PASS
            logits = model(features)
            loss = torch.nn.functional.cross_entropy(logits, targets)
            optimizer.zero_grad()

            loss.backward()

            # Update model parameters
            optimizer.step()

            # LOGGING
            minibatch_loss_list.append(loss.item())
            if not idx % 50:
                print(f'Epoch: {epoch + 1:03d}/{num_epochs:03d} '
                      f'| Batch {idx:04d}/{len(train_loader):04d} '
                      f'| Loss: {loss:.4f}')

        model.eval()
        with torch.no_grad():
            train_acc = compute_accuracy(model, train_loader, device)
            valid_acc = compute_accuracy(model, valid_loader, device)
            train_acc_list.append(train_acc)
            valid_acc_list.append(valid_acc)

            print(f'Epoch: {epoch + 1:03d}/{num_epochs:03d} '
                  f'| Train: {train_acc :.2f}% '
                  f'| Validation: {valid_acc :.2f}%')

        elapsed = (time.time() - start_time)/60
        print(f'Time elapsed: {elapsed:.2f} min')

    elapsed = (time.time() - start_time) / 60
    print(f'Total Training Time: {elapsed:.2f} min')

    test_acc = compute_accuracy(model, test_loader, device=device)
    print(f'Test accuracy {test_acc :.2f}%')

    return minibatch_loss_list, train_acc_list, valid_acc_list

