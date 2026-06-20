import pandas as pd
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

# Load FAQ data
faq_data = pd.read_csv("faqs.csv")

# Questions and Answers
questions = faq_data["Question"].tolist()
answers = faq_data["Answer"].tolist()

# Load AI model
print("Loading AI model...")

model = SentenceTransformer('all-MiniLM-L6-v2')

print("AI model loaded successfully!")

# Convert FAQ questions into embeddings
faq_embeddings = model.encode(questions)

# User question
user_question = input("Ask a question: ")

# Convert user question into embedding
user_embedding = model.encode([user_question])

# Calculate similarity
similarities = cosine_similarity(
    user_embedding,
    faq_embeddings
)

# Best match
best_match_index = similarities.argmax()

best_score = similarities[0][best_match_index]

print("\nSimilarity Score:", round(best_score, 3))

# Threshold
if best_score > 0.50:
    print("\nAnswer:")
    print(answers[best_match_index])
else:
    print("\nSorry, I couldn't find a relevant answer.")