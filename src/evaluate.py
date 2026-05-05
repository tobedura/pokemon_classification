import os
import torch
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score


def evaluate(model, loader, class_names, device, exp_name):
    model.eval()
    all_preds, all_labels = [], []

    with torch.no_grad():
        for images, labels in loader:
            images = images.to(device)
            preds = model(images).argmax(dim=1).cpu()
            all_preds.extend(preds.tolist())
            all_labels.extend(labels.tolist())

    accuracy = accuracy_score(all_labels, all_preds)
    report = classification_report(all_labels, all_preds, target_names=class_names, output_dict=True)

    print(f"\n[{exp_name}] Test Accuracy: {accuracy:.4f}")
    print(classification_report(all_labels, all_preds, target_names=class_names))

    _save_report(report, accuracy, exp_name)
    _save_confusion_matrix(all_labels, all_preds, class_names, exp_name)

    return accuracy, report


def _save_report(report, accuracy, exp_name):
    os.makedirs("results/metrics", exist_ok=True)
    df = pd.DataFrame(report).transpose()
    df.to_csv(f"results/metrics/{exp_name}_report.csv")


def _save_confusion_matrix(labels, preds, class_names, exp_name):
    cm = confusion_matrix(labels, preds)
    fig, ax = plt.subplots(figsize=(20, 20))
    sns.heatmap(cm, xticklabels=class_names, yticklabels=class_names,
                cmap="Blues", ax=ax, fmt="d", annot=len(class_names) <= 30)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title(f"Confusion Matrix - {exp_name}")
    os.makedirs("results/plots", exist_ok=True)
    fig.savefig(f"results/plots/{exp_name}_confusion_matrix.png", bbox_inches="tight")
    plt.close(fig)
