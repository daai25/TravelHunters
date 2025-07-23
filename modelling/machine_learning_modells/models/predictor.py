import torch
import torch.nn.functional as F
from torchvision import transforms, models
from PIL import Image
import os

# --- Configuration ---
MODEL_PATH = r"C:\Users\evanb\TravelHunters\modelling\cnn\city_model.pth"
IMAGE_SIZE = (224, 224)
NUM_CLASSES = 127  # Adjust to match your number of classes
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# List of cities
city_names = [
    "Agios Ioannis Mykonos", "Agios Sostis Mykonos", "Agios Stefanos", "Agrari", "Akrotiri",
    "Amed", "Amsterdam", "Ano Mera", "Auckland", "Baa Atoll",
    "Bangkok", "Barcelona", "Beijing", "Berlin", "Bogota",
    "Brisbane", "Budapest", "Buenos Aires", "Cairo", "Cala Llenya",
    "Cancún", "Canggu", "Cape Town", "Casablanca", "Chicago",
    "Copenhagen", "Dal", "Dhangethi", "Dhidhdhoo", "Dhiffushi",
    "Dhigurah", "Dubai", "Eidsvoll", "Elia", "Es Cana",
    "Fenfushi", "Fira", "Fulidhoo", "Gaafu Alifu Atoll", "Gardermoen",
    "Geneva", "Gjerdrum", "Gystad", "Hangnaameedhoo", "Helsinki",
    "Hong Kong", "Ibiza Town", "Imerovigli", "Jessheim", "Johannesburg",
    "Kintamani", "Klofta", "Klouvas", "Kuala Lumpur", "Kuta",
    "Las Vegas", "Lima", "Lisbon", "London", "Los Angeles",
    "Madrid", "Makunudhoo", "Male City", "Mandhoo", "Marrakech",
    "Meedhoo", "Meemu Atoll", "Megalokhori", "Melbourne", "Miami Beach",
    "Montréal", "Mumbai", "Mushimasgali", "Mýkonos City", "Nannestad",
    "New Delhi", "New York", "Nika Island", "Noonu", "North Male Atoll",
    "Nusa Dua", "Oia", "Osaka", "Oslo", "Paris",
    "Payangan", "Perivolos", "Perth", "Phuket Town", "Platis Yialos Mykonos",
    "Playa d'en Bossa", "Playa del Carmen", "Plintri", "Portinatx", "Prague",
    "Puerto de San Miguel", "Raa Atoll", "Rio de Janeiro", "Rome", "San Antonio",
    "San Antonio Bay", "San Francisco", "Sant Joan de Labritja", "Santa Agnès de Corona", "Santa Eularia des Riu",
    "Santiago de Compostela", "Sao Paulo", "Selemadeg", "Seminyak", "Seoul",
    "Shanghai", "Singapore", "South Male Atoll", "Stockholm", "Super Paradise Beach",
    "Sydney", "Tabanan", "Talamanca", "Thundufushi", "Tokyo","Toronto", "Tourlos", 
    "Tulum", "Ubud", "Uluwatu", "Vancouver", "Vienna", "Zürich"]

# --- Rebuild the model architecture ---
def build_model():
    model = models.resnet18(weights=None)
    model.fc = torch.nn.Linear(model.fc.in_features, NUM_CLASSES)
    return model.to(device)

model = build_model()
model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
model.eval()

# --- Image Preprocessing ---
transform = transforms.Compose([
    transforms.Resize(IMAGE_SIZE),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225])
])

# --- Prediction ---
def predict_image(image_path):
    if not os.path.exists(image_path):
        print("File not found:", image_path)
        return

    image = Image.open(image_path).convert("RGB")
    input_tensor = transform(image).unsqueeze(0).to(device)

    with torch.no_grad():
        output = model(input_tensor).squeeze()
        probs = F.softmax(output, dim=0)
        top_prob, top_idx = torch.topk(probs, 1)

    idx = top_idx.item()
    city = city_names[idx] if idx < len(city_names) else f"Class {idx}"
    #print(f"{city} ({top_prob.item():.2f})")
    return city, round(top_prob.item(), 2)


# --- Terminal input ---
if __name__ == "__main__":
    #image_path = input("Enter the full path to the image: ").strip()
    #image_path = r"C:\Users\evanb\Downloads\20230423_151618.jpg".strip()
    image_path = input().strip()
    city, confidence = predict_image(image_path)
    print(city)
    print(confidence)
