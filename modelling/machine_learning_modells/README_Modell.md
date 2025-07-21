# TravelHunters Modelling Module

## 🎯 Goal
Build recommendation models that suggest the perfect hotel based on user preferences, with a strong focus on embedding-based approaches for best results.

## 🚀 Quick Start

```bash
# Install dependencies
pip install -r requirements.txt

# Interactive demo (main application)
python demo.py

# Embedding-based recommendation (CLI)
python models/hotel_recommender.py

# Automated evaluation
python models/hotel_recommender_tester.py
```

## 🏗️ System Overview

- **Database:** SQLite with >8,000 hotels (`booking_worldwide`)
- **Models:** Embedding-based (SentenceTransformer), parameter-based, text-based, hybrid
- **Recommendation:** Semantic search using vector similarity (Cosine Similarity)
- **CLI Tools:** Direct hotel recommendation and automated evaluation
- **Evaluation:** Top-3-Accuracy with test cases

## 🔑 Main Features

- **hotel_recommender.py:** CLI tool that suggests the Top 3 hotels for any user query (semantic search, price filter, formatted output)
- **hotel_recommender_tester.py:** Automated test cases, shows input/output and calculates Top-3-Accuracy
- **embedding_score_generator.py:** Calculates similarity scores between user queries and hotel descriptions and stores them in the database

## 🧪 Evaluation

- Test cases are defined directly in the tester script (query + expected hotel ID)
- Output shows Top-3 recommendations and whether the expected hotel was found
- **Current Top-3-Accuracy:** **60%** (based on recent test runs)

> **Note:**  
> We recommend using the embedding-based recommendation system, as it consistently achieves much better results and higher accuracy compared to parameter-based and text-based models.

## 📦 Structure

```
modelling/
├── demo.py
├── models/
│   ├── hotel_recommender.py
│   ├── hotel_recommender_tester.py
│   ├── embedding_score_generator.py
│   └── ...
├── data_acquisition/
│   └── database/
│       └── travelhunters.db
└── requirements.txt
```

## 💡 Example: CLI Recommendation

```bash
python models/hotel_recommender.py
# Input: "I want a hotel in Paris with breakfast, max 200"
# Output: Top 3 hotels with description, price, rating, score
```

## 💡 Example: Evaluation

```bash
python models/hotel_recommender_tester.py
# Output: For each test query, Top-3 recommendations and accuracy
```

## 📈 Additional Info

- **Embedding model:** We use Alibaba-NLP/gte-multilingual-base, a powerful multilingual SentenceTransformer. This model converts both user queries and hotel descriptions into high-dimensional vector embeddings. By comparing these vectors using cosine similarity, the system can semantically match user preferences to the most relevant hotels—even if the wording is different or the query is complex.
- **Price filter:** Automatically extracted from user input and applied before semantic matching.
- **Output:** User-friendly, well formatted, and shows all relevant hotel details.
- **Extensibility:** Easily extendable for more models, test cases, and new recommendation logic.

---

**Status:**  
✅ Embedding-based recommendation and evaluation integrated  
✅ CLI tools and test scripts ready to use  
✅ Database and models up to date  
✅ Current Top-3-Accuracy: **60%**