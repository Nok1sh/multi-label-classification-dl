import matplotlib.pyplot as plt
import streamlit as st
import numpy as np
import cv2
import torch

from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
from pytorch_grad_cam.utils.image import show_cam_on_image

from cnn.base_model import ResNetModel
from data.dataset_class import MultiLabelDataset


device = "cuda" if torch.cuda.is_available() else "cpu"

classes = MultiLabelDataset.CLASSES

def learning_curve(history):
    fig, axes = plt.subplots(1, 3, figsize=(14, 8))

    train_loss, train_acc, val_loss, val_acc, f1_train, f1_val = history
    
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

    axes[2].set_title("F1 score")
    axes[2].plot(f1_train, color="blue", label="train")
    axes[2].plot(f1_val, color="red", label="validation")
    axes[2].legend()
    axes[2].set_xlabel("Epoch")
    axes[2].set_ylabel("F1 score")

    fig.suptitle("Train History")
    plt.tight_layout()
    plt.show()


def predict_model(model, img):

    tags = []

    img = ResNetModel.TRANSFORM(img).unsqueeze(0).to(device)

    pred = model.predict_proba(img)

    cls_ind = 0
    for cls, p in zip(classes.values(), pred.numpy()[0]):
        if p >= 0.5:
            tags.append((cls, round(float(p), 2), cls_ind))
        cls_ind += 1
        
    return sorted(tags, key=lambda x: x[1], reverse=True)

def visualize_activity_map(model, img, cls_id):

    target_layer = [model.model.layer4[-1].conv2]

    cam = GradCAM(
        model=model.model,
        target_layers=target_layer
    )

    image = ResNetModel.TRANSFORM(img).unsqueeze(0).to(device)

    targets = [ClassifierOutputTarget(cls_id)]

    cam = GradCAM(model=model.model, target_layers=[model.model.layer4[-1].conv2])

    grayscale_cam = cam(input_tensor=image, targets=targets)[0]

    h, w, _ = img.shape
    grayscale_cam = cv2.resize(grayscale_cam, (w, h))

    rgb_img = np.float32(img) / 255.0

    visualization = show_cam_on_image(rgb_img, grayscale_cam, use_rgb=True)

    return visualization
