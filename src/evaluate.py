import torch
import json
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, classification_report
from dataset import get_dataloaders
from model import build_model

def evaluate():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    _, val_loader, class_names = get_dataloaders()

    model = build_model(len(class_names))
    model.load_state_dict(torch.load("results/best_model.pt", map_location=device))
    model.to(device).eval()

    all_preds, all_labels = [], []
    with torch.no_grad():
        for images, labels in val_loader:
            images = images.to(device)
            outputs = model(images)
            preds = outputs.argmax(1).cpu().numpy()
            all_preds.extend(preds)
            all_labels.extend(labels.numpy())

    report = classification_report(all_labels, all_preds, target_names=class_names, output_dict=True)
    with open("results/classification_report.json", "w") as f:
        json.dump(report, f, indent=2)

    print(classification_report(all_labels, all_preds, target_names=class_names))

    cm = confusion_matrix(all_labels, all_preds)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt="d", xticklabels=class_names, yticklabels=class_names, cmap="Blues")
    plt.xlabel("Predicted"); plt.ylabel("True")
    plt.title("Confusion Matrix")
    plt.tight_layout()
    plt.savefig("results/confusion_matrix.png")

if __name__ == "__main__":
    evaluate()
