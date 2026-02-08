import numpy as np
import torch

from torch import nn
from torch.optim import lr_scheduler, Adam
from tqdm import tqdm
from sklearn.metrics import f1_score


def compute_pos_weight(train_data, num_classes, device):
    pos = torch.zeros(num_classes)
    total = 0

    for _, targets in train_data:
        pos += targets.sum(dim=0)
        total += targets.size(0)

    neg = total - pos
    pos_weight = neg / (pos + 1e-6)

    return pos_weight.to(device)



def run_epochs(
               epochs, 
               train_data, 
               val_data, model, 
               threshold=0.5, 
               early_stop=False, 
               optimizer_state=None, 
               scheduler_state=None,
               ft=False
               ):
    optimizer = Adam(model.parameters(), lr = 0.001)
    optimizer = optimizer if optimizer_state is None else optimizer.load_state_dict(optimizer_state)

    scheduler = lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="min",
        factor=0.5,
        patience=4,
        threshold=0.01,
        min_lr=1e-6
    )

    scheduler = scheduler if scheduler_state is None else scheduler.load_state_dict(scheduler_state)

    device = "cuda" if torch.cuda.is_available() else "cpu"

    loss_fucntion = nn.BCEWithLogitsLoss(
        pos_weight=compute_pos_weight(train_data, 33, device)
        )

    best_loss = None
    best_version = None
    count_steps = 0
    patience = 12

    train_loss = []
    train_acc = []
    f1_train = []
    val_loss = []
    val_acc = []
    f1_val = []

    for epoch in range(epochs):

        model.train()
        running_train_loss = []
        running_train_acc = 0
        all_targets = []
        all_preds = []
        f1_score_train = 0

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

            batch_acc = (pred == target).float().mean().item()
            running_train_acc += batch_acc

            all_targets.append(target.cpu().numpy())
            all_preds.append(pred.cpu().numpy())

            train_loop.set_description(f"Epoch: [{epoch+1}/{epochs}], train_loos: {mean_train_loss:.4f}")

            

        all_targets = np.vstack(all_targets)
        all_preds = np.vstack(all_preds)
        f1_score_train = f1_score(all_targets, all_preds, average="samples")

        running_train_acc /= len(train_data)

        train_loss.append(mean_train_loss)
        train_acc.append(running_train_acc)
        f1_train.append(f1_score_train)
        

        model.eval()
        with torch.no_grad():
            running_val_loss = []
            all_targets = []
            all_preds = []
            running_val_acc = 0
            f1_score_val = 0
            val_loop = tqdm(val_data, leave=False)
            for img, target in val_loop:

                img = img.to(device)
                target = target.to(device)

                pred = model(img)
                loss = loss_fucntion(pred, target)

                running_val_loss.append(loss.item())
                mean_val_loss = sum(running_val_loss)/len(running_val_loss)

                pred = (torch.sigmoid(pred) > threshold).float()

                batch_acc = (pred == target).float().mean().item()
                running_val_acc += batch_acc

                all_targets.append(target.cpu().numpy())
                all_preds.append(pred.cpu().numpy())

            running_val_acc /= len(val_data)

            all_targets = np.vstack(all_targets)
            all_preds = np.vstack(all_preds)
            f1_score_val = f1_score(all_targets, all_preds, average="samples")

            val_loss.append(mean_val_loss)
            val_acc.append(running_val_acc)
            f1_val.append(f1_score_val)

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
            if ft:
                model.save_checkpoint(state, epoch, ft=True)
            else:
                model.save_checkpoint(state, epoch)
        
        if best_loss is None:
            best_loss = mean_val_loss
            best_version = epoch
        
        if mean_val_loss < best_loss - best_loss*threshold:
            best_loss = mean_val_loss
            best_version = epoch
            count_steps = 0
    
        if count_steps >= patience and early_stop:
            print(f"Stopped train on {epoch} epoch")
            print(f"Best loss: {best_loss}\nBest version model: {best_version}")
            break

        count_steps += 1

    return [
            train_loss,
            train_acc,
            f1_train,
            val_loss,
            val_acc,
            f1_val
    ]

def test_model(test_data, model, threshold=0.5):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    loss_fucntion = nn.BCEWithLogitsLoss()
    running_test_loss = []
    all_targets = []
    all_preds = []

    model.eval()
    with torch.no_grad():
        true_answer = 0
        total = 0
        running_test_acc = 0
        test_loop = tqdm(test_data, leave=False)
        for img, target in test_loop:

            img = img.to(device)
            target = target.to(device)

            pred = model(img)
            loss = loss_fucntion(pred, target)

            running_test_loss.append(loss.item())
            mean_test_loss = sum(running_test_loss)/len(running_test_loss)

            pred = (torch.sigmoid(pred) > threshold).float()

            batch_acc = (pred == target).float().mean().item()
            running_test_acc += batch_acc

            all_targets.append(target.cpu().numpy())
            all_preds.append(pred.cpu().numpy())


        all_targets = np.vstack(all_targets)
        all_preds = np.vstack(all_preds)
        f1 = f1_score(all_targets, all_preds, average="samples")

        running_test_acc /= len(test_data)
    
    return f"Loss: {mean_test_loss}\nAccuracy: {running_test_acc}\nF1 score: {f1}"

        
