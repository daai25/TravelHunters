import torch
import clip
from PIL import Image
import os
import numpy as np
from tqdm import tqdm

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Verwende Gerät: {device}")
model, preprocess = clip.load("ViT-B/32", device=device)
print("CLIP-Modell geladen.")

def compute_city_embeddings(city_image_folder):
    city_embeddings = {}
    print(f"Berechne Embeddings für Städte in: {city_image_folder}")
    for city in tqdm(os.listdir(city_image_folder), desc="Städte"):
        city_path = os.path.join(city_image_folder, city)
        embeddings = []
        print(f"  Verarbeite Stadt: {city}")
        for img_file in tqdm(os.listdir(city_path), desc=f"Bilder in {city}", leave=False):
            img_path = os.path.join(city_path, img_file)
            image = preprocess(Image.open(img_path)).unsqueeze(0).to(device)
            with torch.no_grad():
                image_embedding = model.encode_image(image)
            embeddings.append(image_embedding.cpu().numpy())
        city_embeddings[city] = np.mean(embeddings, axis=0)
        print(f"  -> {len(embeddings)} Bilder verarbeitet für {city}")
    print("Alle Städte-Embeddings berechnet.")
    return city_embeddings

def match_image_to_city(uploaded_image_path, city_embeddings):
    print(f"Vergleiche hochgeladenes Bild: {uploaded_image_path}")
    image = preprocess(Image.open(uploaded_image_path)).unsqueeze(0).to(device)
    with torch.no_grad():
        image_embedding = model.encode_image(image).cpu().numpy().squeeze()

    similarities = {}
    for city, emb in city_embeddings.items():
        sim = np.dot(image_embedding, emb.T) / (np.linalg.norm(image_embedding) * np.linalg.norm(emb))
        similarities[city] = sim

    print("Ähnlichkeiten berechnet.")
    return sorted(similarities.items(), key=lambda x: x[1], reverse=True)

city_embeddings = compute_city_embeddings("../../data_acquisition/database/extracted_images")
top_matches = match_image_to_city("user_upload.jpeg", city_embeddings)
print("Top 3 Städte-Vorschläge:", top_matches[:3])