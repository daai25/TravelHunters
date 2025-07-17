# Requires sentence-transformers>=3.0.0

from sentence_transformers import SentenceTransformer

input_texts = [
    "what is the capital of China?",
    "how to implement quick sort in python?",
    "北京",
    "快排算法介绍"
]

model_name_or_path="Alibaba-NLP/gte-multilingual-base"
model = SentenceTransformer(model_name_or_path, trust_remote_code=True)
embeddings = model.encode(input_texts, normalize_embeddings=True) # embeddings.shape (4, 768)

print(embeddings[0])
# sim scores
scores = model.similarity(embeddings[0], embeddings[1])

print(scores.tolist())
# [[0.301699697971344, 0.7503870129585266, 0.32030850648880005]]
