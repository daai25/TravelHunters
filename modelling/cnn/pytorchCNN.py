import os
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
from torchvision import models
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import random
from tqdm import tqdm
import multiprocessing

# ==== CONFIGURATION ====
TRAIN_DIR = r"C:\Users\evanb\Training"
VAL_DIR = r"C:\Users\evanb\Validation"
IMG_SIZE = (224, 224)
BATCH_SIZE = 32
EPOCHS = 8
LEARNING_RATE = 0.001
NUM_WORKERS = 2

# ==== DATASET ====
class CityDataset(Dataset):
    def __init__(self, image_label_pairs, transform=None):
        self.image_label_pairs = image_label_pairs
        self.transform = transform

    def __len__(self):
        return len(self.image_label_pairs)

    def __getitem__(self, idx):
        img_path, label = self.image_label_pairs[idx]
        image = Image.open(img_path).convert("RGB")
        if self.transform:
            image = self.transform(image)
        return image, label

# ==== LOAD DATA ====
def load_data(folder_path, city_to_idx):
    all_data = []
    for city in sorted(os.listdir(folder_path)):
        city_path = os.path.join(folder_path, city)
        if not os.path.isdir(city_path):
            continue
        image_files = [f for f in os.listdir(city_path) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
        image_paths = [os.path.join(city_path, f) for f in image_files]

        label = city_to_idx.get(city)
        if label is None:
            print(f"⚠️ Warning: City '{city}' found in data folder but not in training folder, skipping.")
            continue

        labels = [label] * len(image_paths)
        combined = list(zip(image_paths, labels))
        all_data.extend(combined)
    return all_data

# ==== MODEL ====
def get_model(num_classes):
    model = models.resnet18(pretrained=True)
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    return model

# ==== TRAIN ====
def train_model(model, loader, criterion, optimizer):
    model.train()
    for epoch in range(EPOCHS):
        total_loss = 0.0
        for images, labels in tqdm(loader, desc=f"Epoch {epoch + 1}/{EPOCHS}"):
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        print(f"Epoch {epoch + 1} Loss: {total_loss / len(loader):.4f}")

# ==== EVALUATE ====
def evaluate_model(model, loader, city_names):
    model.eval()
    total = 0
    correct = 0
    city_correct = {city: 0 for city in city_names}
    city_total = {city: 0 for city in city_names}

    with torch.no_grad():
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, preds = torch.max(outputs, 1)

            correct += (preds == labels).sum().item()
            total += labels.size(0)

            for label, pred in zip(labels, preds):
                city = city_names[label]
                city_total[city] += 1
                if label == pred:
                    city_correct[city] += 1

    print(f"\n✅ Overall Accuracy: {100 * correct / total:.2f}% on {total} test images\n")
    for city in city_names:
        if city_total[city] > 0:
            acc = 100 * city_correct[city] / city_total[city]
            print(f"🏙️ {city:25s} — {acc:.1f}% accuracy ({city_correct[city]}/{city_total[city]})")
        else:
            print(f"🏙️ {city:25s} — No test data")

# ==== MAIN ====
def main():
    transform = transforms.Compose([
        transforms.Resize(IMG_SIZE),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(10),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],
                             [0.229, 0.224, 0.225])
    ])

    print("🔄 Loading city names from training data...")
    city_names = sorted([f for f in os.listdir(TRAIN_DIR) if os.path.isdir(os.path.join(TRAIN_DIR, f))])
    city_to_idx = {city: idx for idx, city in enumerate(city_names)}

    print("🔄 Loading and preparing training data...")
    train_data = load_data(TRAIN_DIR, city_to_idx)

    print("🔄 Loading and preparing validation data...")
    val_data = load_data(VAL_DIR, city_to_idx)

    train_set = CityDataset(train_data, transform)
    val_set = CityDataset(val_data, transform)

    train_loader = DataLoader(train_set, batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS)
    val_loader = DataLoader(val_set, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS)

    print(f"🔧 Starting training on {len(city_names)} cities, {len(train_data)} training images...")

    model = get_model(len(city_names)).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=LEARNING_RATE, momentum=0.9)

    train_model(model, train_loader, criterion, optimizer)

    torch.save(model.state_dict(), r"C:\Users\evanb\city_model.pth")
    print("✅ Model saved to city_model.pth")

    evaluate_model(model, val_loader, city_names)

if __name__ == "__main__":
    multiprocessing.freeze_support()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    main()
