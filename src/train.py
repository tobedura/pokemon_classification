import os
import torch
import torch.nn as nn
from torch.optim import Adam
from torch.optim.lr_scheduler import CosineAnnealingLR
import matplotlib.pyplot as plt
import json


def train(model, train_loader, val_loader, config, device, exp_name):
    model = model.to(device)
    optimizer = Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=config["lr"])
    scheduler = CosineAnnealingLR(optimizer, T_max=config["epochs"])
    criterion = nn.CrossEntropyLoss()

    history = {"train_loss": [], "val_loss": [], "val_acc": []}
    best_acc = 0.0

    for epoch in range(config["epochs"]):
        # 학습
        model.train()
        train_loss = 0.0
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            loss = criterion(model(images), labels)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()

        # 검증
        val_loss, val_acc = _evaluate(model, val_loader, criterion, device)
        scheduler.step()

        history["train_loss"].append(train_loss / len(train_loader))
        history["val_loss"].append(val_loss)
        history["val_acc"].append(val_acc)

        print(f"[{exp_name}] Epoch {epoch+1}/{config['epochs']} | "
              f"Train Loss: {train_loss/len(train_loader):.4f} | "
              f"Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.4f}")

        # 가장 좋은 모델 저장
        if val_acc > best_acc:
            best_acc = val_acc
            os.makedirs("models", exist_ok=True)
            torch.save(model.state_dict(), f"models/{exp_name}_best.pth")

    _save_history(history, exp_name)
    _plot_learning_curve(history, exp_name)
    print(f"\n[{exp_name}] 학습 완료 | Best Val Acc: {best_acc:.4f}")
    return history


def _evaluate(model, loader, criterion, device):
    model.eval()
    total_loss, correct, total = 0.0, 0, 0
    with torch.no_grad():
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            total_loss += criterion(outputs, labels).item()
            correct += (outputs.argmax(1) == labels).sum().item()
            total += labels.size(0)
    return total_loss / len(loader), correct / total


def _save_history(history, exp_name):
    os.makedirs("results/metrics", exist_ok=True)
    with open(f"results/metrics/{exp_name}_history.json", "w") as f:
        json.dump(history, f, indent=2)


def _plot_learning_curve(history, exp_name):
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    epochs = range(1, len(history["train_loss"]) + 1)

    axes[0].plot(epochs, history["train_loss"], label="Train Loss")
    axes[0].plot(epochs, history["val_loss"], label="Val Loss")
    axes[0].set_title("Loss")
    axes[0].set_xlabel("Epoch")
    axes[0].set_xticks(epochs)
    axes[0].legend()

    axes[1].plot(epochs, history["val_acc"], label="Val Accuracy")
    axes[1].set_title("Accuracy")
    axes[1].set_xlabel("Epoch")
    axes[1].set_xticks(epochs)
    axes[1].legend()

    fig.suptitle(exp_name)
    os.makedirs("results/plots", exist_ok=True)
    fig.savefig(f"results/plots/{exp_name}_learning_curve.png")
    plt.close(fig)
