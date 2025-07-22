import os
import torch
import torchvision.transforms as transforms
from torchvision import models
from PIL import Image
import sqlite3

# ===== Konfiguration =====
MODEL_PATH = "../../cnn/city_model.pth"
IMG_SIZE = (224, 224)
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
num_classes = 127

# ===== Modell laden =====
model = models.resnet18(weights=None)
model.fc = torch.nn.Linear(model.fc.in_features, num_classes)
model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
model.to(DEVICE)
model.eval()

# ===== Bild einlesen und vorverarbeiten =====
transform = transforms.Compose([
    transforms.Resize(IMG_SIZE),
    transforms.ToTensor(),
])
img_path = input("Pfad zum Bild eingeben: ")
img = Image.open(img_path).convert("RGB")
img_tensor = transform(img).unsqueeze(0).to(DEVICE)

# ===== Vorhersage =====
with torch.no_grad():
    output = model(img_tensor)
    probs = torch.softmax(output, dim=1).cpu().numpy()[0]
    top5_idx = probs.argsort()[-5:][::-1]
    top5_probs = probs[top5_idx]

# ===== Klassennamen laden =====
db_path = "../../../data_acquisition/database/travelhunters.db"
conn = sqlite3.connect(db_path)
cursor = conn.cursor()
cursor.execute("SELECT name FROM city")
class_names = [row[0] for row in cursor.fetchall()]
conn.close()

# Entferne die Städte: Baa Atoll und Dhigurah
class_names = [name for name in class_names if name not in ["Baa Atoll", "Dhigurah"]]

# ===== Top-5-Ausgabe =====
print("Top 5 Vorhersagen:")
for rank, (idx, prob) in enumerate(zip(top5_idx, top5_probs), 1):
    city = class_names[idx] if idx < len(class_names) else "unbekannt"
    print(f"{rank}. {city}: {prob*100:.2f}%")