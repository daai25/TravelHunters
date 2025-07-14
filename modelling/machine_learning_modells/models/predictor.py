'''
This script uses the classifier to predict a designation based on an input image.
'''


import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import load_img, img_to_array

# --- Configuration ---
MODEL_PATH = r"city_classifier_model.h5"
IMAGE_SIZE = (224, 224)  # Based on the error, this is what your model expects

# List of cities for the predictor
city_names = [
    "Agios Ioannis Mykonos",
    "Agios Sostis Mykonos",
    "Agios Stefanos",
    "Agrari",
    "Akrotiri",
    "Amed",
    "Amsterdam",
    "Ano Mera",
    "Auckland",
    "Baa Atoll",
    "Bangkok",
    "Barcelona",
    "Beijing",
    "Berlin",
    "Bogota",
    "Brisbane",
    "Budapest",
    "Buenos Aires",
    "Cairo",
    "Cala Llenya",
    "Cancún",
    "Canggu",
    "Cape Town",
    "Casablanca",
    "Chicago",
    "Copenhagen",
    "Dal",
    "Dhangethi",
    "Dhidhdhoo",
    "Dhiffushi",
    "Dhigurah",
    "Dubai",
    "Eidsvoll",
    "Elia",
    "Es Cana",
    "Fenfushi",
    "Fira",
    "Fulidhoo",
    "Gaafu Alifu Atoll",
    "Gardermoen",
    "Geneva",
    "Gjerdrum",
    "Gystad",
    "Hangnaameedhoo",
    "Helsinki",
    "Hong Kong",
    "Ibiza Town",
    "Imerovigli",
    "Jessheim",
    "Johannesburg",
    "Kintamani",
    "Klofta",
    "Klouvas",
    "Kuala Lumpur",
    "Kuta",
    "Las Vegas",
    "Lima",
    "Lisbon",
    "London",
    "Los Angeles",
    "Madrid",
    "Makunudhoo",
    "Male City",
    "Mandhoo",
    "Marrakech",
    "Meedhoo",
    "Meemu Atoll",
    "Megalokhori",
    "Melbourne",
    "Miami Beach",
    "Montréal",
    "Mumbai",
    "Mushimasgali",
    "Mýkonos City",
    "Nannestad",
    "New Delhi",
    "New York",
    "Nika Island",
    "Noonu",
    "North Male Atoll",
    "Nusa Dua",
    "Oia",
    "Osaka",
    "Oslo",
    "Paris",
    "Payangan",
    "Perivolos",
    "Perth",
    "Phuket Town",
    "Platis Yialos Mykonos",
    "Playa d'en Bossa",
    "Playa del Carmen",
    "Plintri",
    "Portinatx",
    "Prague",
    "Puerto de San Miguel",
    "Raa Atoll",
    "Rio de Janeiro",
    "Rome",
    "San Antonio",
    "San Antonio Bay",
    "San Francisco",
    "Sant Joan de Labritja",
    "Santa Agnès de Corona",
    "Santa Eularia des Riu",
    "Santiago de Compostela",
    "Sao Paulo",
    "Selemadeg",
    "Seminyak",
    "Seoul",
    "Shanghai",
    "Singapore",
    "South Male Atoll",
    "Stockholm",
    "Super Paradise Beach",
    "Sydney",
    "Tabanan",
    "Talamanca",
    "Thundufushi",
    "Tokyo",
    "Toronto",
    "Tourlos",
    "Tulum",
    "Ubud",
    "Uluwatu",
    "Vancouver",
    "Vienna",
    "Zürich"]

# --- Load Model ---
model = load_model(MODEL_PATH)

# --- Predict Function ---
def predict_image(image_path):
    # Load and preprocess image (no flattening)
    img = load_img(image_path, target_size=IMAGE_SIZE)
    img_array = img_to_array(img)
    img_array = img_array / 255.0  # Normalize
    img_array = np.expand_dims(img_array, axis=0)  # Shape: (1, 150, 150, 3)

    print("Processed input shape:", img_array.shape)

    # Predict
    prediction = model.predict(img_array)
    print("Raw Model Output:", prediction)
    #print(max(prediction[0]))

    # Interpret prediction
    # Interpret prediction
    if prediction.shape[-1] > 1:
        top_indices = np.argsort(prediction[0])[::-1][:3]  # Sort descending and take top 3
        first, second, third = top_indices
        print("Top 3 Cities to visit:")
        print("1st:", city_names[first])
        print("2nd:", city_names[second])
        print("3rd:", city_names[third])
    else:
        # Binary classification fallback
        predicted_class = (prediction[0][0] > 0.5).astype("int")
        first = predicted_class
        second = third = None  # Not applicable
        print("Predicted Class (binary):", predicted_class)


# --- Run Manually ---
if __name__ == "__main__":
    image_path = input("Enter the full path to the image: ").strip()
    predict_image(image_path)
    

