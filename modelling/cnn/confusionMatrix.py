import os
import torch
import torchvision.transforms as transforms
from torchvision import datasets, models
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, classification_report, f1_score
import matplotlib.pyplot as plt
import numpy as np
from collections import defaultdict

# ===== Configuration =====
TEST_DIR = r"C:\Users\evanb\Testing"
MODEL_PATH = r"C:\Users\evanb\city_model.pth"
IMG_SIZE = (224, 224)
BATCH_SIZE = 32
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ===== Load Test Data =====
transform = transforms.Compose([
    transforms.Resize(IMG_SIZE),
    transforms.ToTensor(),
])

test_dataset = datasets.ImageFolder(TEST_DIR, transform=transform)
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)
class_names = test_dataset.classes
num_classes = len(class_names)

# ===== Load Model (ResNet18 assumed) =====
model = models.resnet18(pretrained=False)
model.fc = torch.nn.Linear(model.fc.in_features, num_classes)
model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
model.to(DEVICE)
model.eval()

# ===== Inference =====
all_preds = []
all_labels = []

with torch.no_grad():
    for images, labels in test_loader:
        images, labels = images.to(DEVICE), labels.to(DEVICE)
        outputs = model(images)
        _, preds = torch.max(outputs, 1)
        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())

# ===== Per-Class Accuracy and Overall Accuracy =====
per_class_correct = defaultdict(int)
per_class_total = defaultdict(int)

for label, pred in zip(all_labels, all_preds):
    if label == pred:
        per_class_correct[label] += 1
    per_class_total[label] += 1

print("📊 Accuracy by City:")
total_correct = 0
total_images = 0
accuracies = []

for i, class_name in enumerate(class_names):
    correct = per_class_correct[i]
    total = per_class_total[i]
    acc = (correct / total) if total > 0 else 0
    total_correct += correct
    total_images += total
    accuracies.append((acc, i))
    print(f" - {class_name}: {acc * 100:.2f}% ({correct}/{total})")

overall_accuracy = total_correct / total_images
print(f"\n✅ Overall Accuracy: {overall_accuracy * 100:.2f}% ({total_correct}/{total_images})\n")

# ===== F1 Score =====
macro_f1 = f1_score(all_labels, all_preds, average="macro")
print(f"🎯 Macro F1 Score: {macro_f1:.4f}")

report = classification_report(all_labels, all_preds, target_names=class_names, zero_division=0)
print("\n🧾 Classification Report (Per-Class F1 Scores):")
print(report)

# ===== Confusion Matrix =====
cm = confusion_matrix(all_labels, all_preds)

# ===== Top 10 Most Accurate Classes =====
top_accurate_classes = sorted(accuracies, reverse=True)[:10]
top_accurate_indices = [idx for _, idx in top_accurate_classes]
top_accurate_labels = [class_names[i] for i in top_accurate_indices]
cm_accurate = cm[np.ix_(top_accurate_indices, top_accurate_indices)]

fig1, ax1 = plt.subplots(figsize=(10, 10))
disp1 = ConfusionMatrixDisplay(confusion_matrix=cm_accurate, display_labels=top_accurate_labels)
disp1.plot(ax=ax1, cmap='Blues', xticks_rotation=45)
plt.title("Top 10 Most Accurate Classes")
plt.tight_layout()

save_path1 = r"C:\Users\evanb\TravelHunters\confusion_top_accurate.png"
plt.savefig(save_path1)
plt.show()
print(f"📈 Saved top accurate confusion matrix to: {save_path1}")

# ===== Top 10 Classes Most Frequently Mistaken for Another =====
mistaken_as_other = np.zeros(num_classes)

for true_label, pred_label in zip(all_labels, all_preds):
    if true_label != pred_label:
        mistaken_as_other[true_label] += 1

mistaken_indices = np.argsort(mistaken_as_other)[::-1][:10]
mistaken_labels = [class_names[i] for i in mistaken_indices]
cm_mistaken = cm[np.ix_(mistaken_indices, mistaken_indices)]

fig2, ax2 = plt.subplots(figsize=(10, 10))
disp2 = ConfusionMatrixDisplay(confusion_matrix=cm_mistaken, display_labels=mistaken_labels)
disp2.plot(ax=ax2, cmap='Reds', xticks_rotation=45)
plt.title("Top 10 Most Mistaken-For-Another Classes")
plt.tight_layout()

save_path2 = r"C:\Users\evanb\TravelHunters\confusion_most_mistaken.png"
plt.savefig(save_path2)
plt.show()
print(f"⚠️ Saved most mistaken confusion matrix to: {save_path2}")
