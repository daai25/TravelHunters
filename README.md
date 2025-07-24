# TravelHunters 🌍✈️

An intelligent travel recommendation system that suggests hotels and destinations based on user preferences – using both image and text input. The project combines web scraping, a modern database, and advanced machine learning (NLP & CNN) to deliver personalized, inspiring travel recommendations.

---

## Project Overview

TravelHunters brings together multiple data sources and AI models:

- **Hotels:** Over 2,000 hotels from Booking.com with images, ratings, and amenities
- **Destinations:** 129 destinations with Wikipedia integration and curated images
- **Activities:** GetYourGuide activities for selected destinations
- **Images:** 10,000+ destination images from Google, Wikipedia, GetYourGuide, Duckduckgo

The system provides recommendations through:

- **Semantic Text Search:** Finds hotels matching user wishes (e.g. "family friendly, pool, beach") using a multilingual SentenceTransformer (Alibaba-NLP/gte-multilingual-base)
- **Image Recognition:** Users upload a photo, and a CNN predicts the city to inspire relevant suggestions
- **Hybrid Search:** Combines both inputs for even more personalized results

---

## Features

### Data Acquisition

- Automated scraping of hotel and activity data from Booking.com and GetYourGuide
- Bulk download and processing of destination images (Google, Wikipedia, Duckduckgo)
- Wikipedia integration for destination info
- Data validation and cleaning pipelines

### Machine Learning Models

- **Semantic Search:** Alibaba-NLP/gte-multilingual-base for multilingual, context-aware hotel matching
- **CNN City Classifier:** PyTorch/Keras model for city recognition from images
- **Hybrid Recommendation:** Combines image and text for best results
- **Evaluation Metrics:** Top-3-Accuracy, overall accuracy, user satisfaction

### Backend & API

- **Flask APIs:** For both hotel recommender and image classifier (models stay in memory for fast response)
- **SQLite Database:** Structured storage for hotels, destinations, and activities

### Frontend

- **Modern React App:** Upload images, enter wishes, get instant recommendations
- **Dark/Light Mode, Language Switch (EN/DE)**
- **Direct booking links, ratings, amenities, and more**

---

## Project Structure

- `data_acquisition/` – Web scraping scripts and data collection tools
- `database/` – SQLite database and schema definitions
- `modelling/` – Machine Learning models (NLP & CNN) and evaluation scripts
- `docs/` – Project documentation (Quarto, Markdown, images)
- `travelhunters-frontend/` – React frontend

### Key Files

- `modelling/machine_learning_modells/models/hotel_recommender.py` – Flask API for hotel recommendations
- `modelling/cnn/predictor.py` – Flask API for city prediction from images
- `docs/pics/` – Plots and KPI graphics for documentation
- `conda.yml` – Python environment configuration

---

## Getting Started

### Prerequisites

- Python 3.8+
- Conda package manager
- Node.js & npm (for frontend)
- Git

### Installation

1. **Clone the repository:**
   ```bash
   git clone <repository-url>
   cd TravelHunters
   ```

2. **Create conda environment:**
   ```bash
   conda env create -f conda.yml
   conda activate travelhunters
   ```

3. **Install frontend dependencies:**
   ```bash
   cd travelhunters-frontend
   npm install
   ```

4. **Set up environment variables:**
   ```bash
   cp .env.template .env
   # Edit .env with your API keys and configurations
   ```

---

## Usage

### Data Collection

1. **Scrape hotel data:**
   ```bash
   cd data_acquisition
   scrapy crawl booking_spider
   ```

2. **Download images:**
   ```bash
   python download_json_images.py
   ```

3. **Merge and process data:**
   ```bash
   cd mergingjson
   python merging_json_booking.py
   python merging_json_activity.py
   ```

### Model Training and Evaluation

1. **Train models:**
   ```bash
   cd modelling
   # Run model training scripts for NLP and CNN
   ```

2. **Evaluate performance:**
   ```bash
   cd modelling/machine_learning_modells/models
   python hotel_recommender_tester.py
   ```

---

## Running Backend & Frontend (in parallel)

To use the full application, you need to run both the backend (API) and frontend **in parallel in two terminals**:

1. **Start backend (ML API):**
   ```bash
   cd modelling/machine_learning_modells/models
   python unified_travel_api.py
   ```

2. **Start frontend:**
   ```bash
   cd travelhunters-frontend
   npm start
   ```

---

### Frontend: Required Packages

The following Node.js packages are required for the frontend (installed automatically via `npm install`):

- **react**
- **react-dom**
- **react-scripts**
- **axios**
- **@mui/material**
- **@emotion/react**
- **@emotion/styled**
- **react-dropzone**
- **react-router-dom**
- **dotenv**
- **(see `package.json` for the full list)**

**Install all dependencies:**
```bash
cd travelhunters-frontend
npm install
```

---

## Documentation

The project includes comprehensive documentation built with Quarto:

- **Project Charter:** Project scope and objectives
- **Data Report:** Data collection and quality analysis  
- **Modeling Report:** ML model development and selection (NLP & CNN)
- **Evaluation Report:** Performance metrics and results

**Build documentation:**
```bash
cd docs
quarto render
```

---

## Key Performance Indicators

- **Top-3-Accuracy (Hotel Recommender):** 60 %
- **Overall Accuracy (CNN City Classifier):** 78 %
- **Combined (Hybrid) Top-3-Accuracy:** 70 %
- **Data Pipeline Success Rate:** 95 %
- **User Satisfaction:** 4.5/5 (90 %)

---

## Team

**Data Science Summer School 2025 – ZHAW School of Engineering**

- Leona Kryeziu
- Evan Blazo
- Jolan Felber
- Jakub Baranec

---

## License

This project is licensed under the MIT License – see the [LICENSE](LICENSE) file for details.

---

## References

- [Alibaba-NLP/gte-multilingual-base](https://huggingface.co/Alibaba-NLP/gte-multilingual-base)
- Booking.com API Documentation
- Scrapy Documentation
- Scikit-learn Documentation
- PyTorch & Keras Documentation