import torch

from torch import nn
from torch.optim import lr_scheduler, Adam
from tqdm import tqdm


def run_epochs(epochs, train_data, val_data, model, threshold=0.5):
    optimizer = Adam(model.parameters(), lr = 0.1)

    scheduler = lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="min",
        factor=0.1,
        patience=10,
        threshold=0.0001
    )
    device = "cuda" if torch.cuda.is_available() else "cpu"

    loss_fucntion = nn.BCEWithLogitsLoss()

    train_loss = []
    train_acc = []
    val_loss = []
    val_acc = []

    for epoch in range(epochs):

        model.train()
        running_train_loss = []
        true_answer = 0

        train_loop = tqdm(train_data, leave=False)

        for img, target in train_loop:

            img = img.to(device)
            target = target.to(device)

            optimizer.zero_grad()

            pred = model(img)
            loss = loss_fucntion(pred, target)

            loss.backward()
            optimizer.step()

            running_train_loss.append(loss.item())
            mean_train_loss = sum(running_train_loss)/len(running_train_loss)

            pred = (torch.sigmoid(pred) > threshold).float()
            true_answer += (pred == target).sum().item()

            train_loop.set_description(f"Epoch: [{epoch+1}/{epochs}], train_loos: {mean_train_loss:.4f}")

        total_predictions = len(train_data.dataset) * target.size(1)
        running_train_acc = true_answer/total_predictions

        train_loss.append(mean_train_loss)
        train_acc.append(running_train_acc)
        

        model.eval()
        with torch.no_grad():
            running_val_loss = []
            true_answer = 0
            val_loop = tqdm(val_data, leave=False)
            for img, target in val_loop:

                img = img.to(device)
                target = target.to(device)

                pred = model(img)
                loss = loss_fucntion(pred, target)

                running_val_loss.append(loss.item())
                mean_val_loss = sum(running_val_loss)/len(running_val_loss)

                pred = (torch.sigmoid(pred) > threshold).float()
                true_answer += (pred == target).sum().item()

            total_predictions = len(val_data.dataset) * target.size(1)
            running_val_acc = true_answer/total_predictions

            val_loss.append(mean_val_loss)
            val_acc.append(running_val_acc)

        scheduler.step(mean_val_loss)

        print(f"Epoch: [{epoch+1}/{epochs}], train_loos: {mean_train_loss:.4f}, train_acc: {running_train_acc:.4f}, val_loos: {mean_val_loss:.4f},  val_acc: {running_val_acc:.4f}")

    return [
            train_loss,
            train_acc,
            val_loss,
            val_acc
    ]
            
