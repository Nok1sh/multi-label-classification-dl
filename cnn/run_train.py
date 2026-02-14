import numpy as np
import torch

from torch import nn
from torch.optim import lr_scheduler, AdamW
from tqdm import tqdm
from sklearn.metrics import f1_score, average_precision_score


device = "cuda" if torch.cuda.is_available() else "cpu"


def run_epochs(
               epochs, 
               train_data, 
               val_data, model, 
               ft=False,
               lr=1e-3,
               use_scheduler=True
               ):
    
    if ft:
        lr = 3e-4
    optimizer = AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    
    if use_scheduler:
        scheduler = lr_scheduler.OneCycleLR(
            optimizer,
            max_lr=lr,
            steps_per_epoch=len(train_data),
            epochs=epochs
        )
    else:
        scheduler = None

    loss_fucntion = nn.BCEWithLogitsLoss()

    train_loss = []
    train_acc = []
    val_loss = []
    val_acc = []
    f1_val_samples = []
    f1_val_macro = []
    f1_val_weights = []
    mAP_val = []

    last_epoch_probs = None
    last_epoch_targets = None


    for epoch in range(epochs):

        # train
        model.train()
        all_targets, all_preds, label_accuracy_train, mean_train_loss, probs = one_epoch_validation(
            model, 
            loss_fucntion, 
            train_data,
            evaluate=False,
            optimizer=optimizer,
            scheduler=scheduler,
            epoch=epoch,
            epochs=epochs
            )

        train_loss.append(mean_train_loss)
        train_acc.append(label_accuracy_train)

        # Validation
        with torch.no_grad():
            model.eval()
            all_targets, all_preds, label_accuracy_val, mean_val_loss, probs = one_epoch_validation(model, loss_fucntion, val_data)

        val_loss.append(mean_val_loss)
        val_acc.append(label_accuracy_val)

        last_epoch_probs = probs
        last_epoch_targets = all_targets

        f1_samples = f1_score(all_targets, all_preds, average="samples")
        f1_macro = f1_score(all_targets, all_preds, average="macro")
        f1_weighted = f1_score(all_targets, all_preds, average="weighted")

        mAP = average_precision_score(
            last_epoch_targets,
            last_epoch_probs,
            average="macro"
        )

        print(f"F1 samples:  {f1_samples:.4f}")
        print(f"F1 macro:    {f1_macro:.4f}")
        print(f"F1 weighted: {f1_weighted:.4f}")
        print(f"Mean Average Precision: {mAP:.4f}")

        f1_val_samples.append(f1_samples)
        f1_val_macro.append(f1_macro)
        f1_val_weights.append(f1_weighted)
        mAP_val.append(mAP)

        print(f"Epoch: [{epoch+1}/{epochs}], train_loos: {mean_train_loss:.4f}, train_acc: {label_accuracy_train:.4f}, val_loos: {mean_val_loss:.4f},  val_acc: {label_accuracy_val:.4f}")

        if epoch % 3 == 0:
            state = {
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "num_cls": model.num_cls,
                "threshold": model.threshold_probability,
                "optimizer_state_dict": optimizer.state_dict(),
                "metrics": {"train_loos": mean_train_loss, "train_acc": label_accuracy_train, "val_loos": mean_val_loss,  "val_acc": label_accuracy_val}
            }
            if ft:
                model.save_checkpoint(state, epoch, ft=True)
            else:
                model.save_checkpoint(state, epoch)
    
    thresholds = np.arange(0.1, 0.9, 0.05)

    best_f1 = 0
    best_threshold = 0.5

    for t in thresholds:
        preds = (last_epoch_probs > t).astype(int)
        f1 = f1_score(last_epoch_targets, preds, average="macro")
                
        if f1 > best_f1:
            best_f1 = f1
            best_threshold = t
    
    model.threshold_probability = best_threshold

    state = {
        "epoch": epochs-1,
        "model_state_dict": model.state_dict(),
        "num_cls": model.num_cls,
        "threshold": model.threshold_probability,
        "optimizer_state_dict": optimizer.state_dict(),
        "metrics": {"train_loos": mean_train_loss, "train_acc": label_accuracy_train, "val_loos": mean_val_loss,  "val_acc": label_accuracy_val}
    }
    if ft:
        model.save_checkpoint(state, epochs-1, ft=True)
    else:
        model.save_checkpoint(state, epochs-1)

    return [
            train_loss,
            train_acc,
            val_loss,
            val_acc,
            f1_val_samples,
            f1_val_macro,
            f1_val_weights,
            mAP_val
    ]

def test_model(test_data, model):
    loss_fucntion = nn.BCEWithLogitsLoss()

    with torch.no_grad():
        model.eval()
        all_targets, all_preds, label_accuracy_test, mean_test_loss, probs = one_epoch_validation(model, loss_fucntion, test_data)

    f1 = f1_score(all_targets, all_preds, average="samples")

    
    return f"Loss: {mean_test_loss}\nAccuracy: {label_accuracy_test}\nF1 score samples: {f1}"

        

def one_epoch_validation(model, loss_fucntion, data, evaluate=True, optimizer=None, scheduler=None, epoch=None, epochs=None):

    threshold = model.threshold_probability

    running_loss = []
    all_targets = []
    all_preds = []
    data_loop = tqdm(data, leave=False)
    for img, target in data_loop:

        img = img.to(device)
        target = target.to(device)

        if evaluate:
            pred = model(img)
            loss = loss_fucntion(pred, target)

        else:
            optimizer.zero_grad()

            pred = model(img)
            loss = loss_fucntion(pred, target)

            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            if scheduler is not None:
                scheduler.step()

        running_loss.append(loss.item())
        mean_loss = sum(running_loss)/len(running_loss)

        all_targets.append(target.cpu().numpy())
        all_preds.append(pred.cpu().numpy())

        if not evaluate:
            data_loop.set_description(f"Epoch: [{epoch+1}/{epochs}], train_loos: {mean_loss:.4f}")

    all_targets = np.vstack(all_targets)
    all_preds = np.vstack(all_preds)

    probs = 1 / (1 + np.exp(-all_preds))

    all_preds = (probs > threshold).astype(int)

    intersection = (all_preds * all_targets).sum(axis=1)
    union = ((all_preds + all_targets) > 0).sum(axis=1)

    label_accuracy = (intersection / (union + 1e-6)).mean()
    
    return all_targets, all_preds, label_accuracy, mean_loss, probs