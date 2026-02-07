import torch

from torch import nn
from torch.optim import lr_scheduler, Adam
from tqdm import tqdm


def run_epochs(epochs, train_data, val_data, model, threshold=0.5, early_stop=False):
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

    best_loss = None
    best_version = None
    count_steps = 0
    count_steps_without_better = 12

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

        if epoch % 1 == 0:
            state = {
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "num_cls": model.num_cls,
                "optimizer_state_dict": optimizer.state_dict(),
                "scheduler_state_dict": scheduler.state_dict(),
                "metrics": {"train_loos": mean_train_loss, "train_acc": running_train_acc, "val_loos": mean_val_loss,  "val_acc": running_val_acc}
            }
            model.save_checkpoint(state, epoch)
        
        if best_loss is None:
            best_loss = mean_val_loss
            best_version = epoch
        
        if mean_val_loss < best_loss - best_loss*threshold:
            best_loss = mean_val_loss
            best_version = epoch
            count_steps = 0
    
        if count_steps >= count_steps_without_better and early_stop:
            print(f"Stopped train on {epoch} epoch")
            print(f"Best loss: {best_loss}\nBest version model: {best_version}")
            break

        count_steps += 1

    return [
            train_loss,
            train_acc,
            val_loss,
            val_acc
    ]

def test_model(test_data, model, threshold=0.5):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    loss_fucntion = nn.BCEWithLogitsLoss()
    running_test_loss = []

    model.eval()
    with torch.no_grad():
        true_answer = 0
        test_loop = tqdm(test_data, leave=False)
        for img, target in test_loop:

            img = img.to(device)
            target = target.to(device)

            pred = model(img)
            loss = loss_fucntion(pred, target)

            running_test_loss.append(loss.item())
            mean_test_loss = sum(running_test_loss)/len(running_test_loss)

            pred = (torch.sigmoid(pred) > threshold).float()
            true_answer += (pred == target).sum().item()

        total_predictions = len(test_data.dataset) * target.size(1)
        running_test_acc = true_answer/total_predictions
    
    return f"Loss: {mean_test_loss}\nAccuracy: {running_test_acc}"

        
