import matplotlib.pyplot as plt


def learning_curve(history):
    fig, axes = plt.subplots(1, 2, figsize=(14, 8))

    train_loss, train_acc, val_loss, val_acc = history
    
    axes[0].set_title("Loss")
    axes[0].plot(train_loss, color="blue", label="train")
    axes[0].plot(val_loss, color="red", label="validation")
    axes[0].legend()
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Loss")

    axes[1].set_title("Accuracy")
    axes[1].plot(train_acc, color="blue", label="train")
    axes[1].plot(val_acc, color="red", label="validation")
    axes[1].legend()
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Accuracy")

    fig.suptitle("Train History")
    plt.tight_layout()
    plt.show()
