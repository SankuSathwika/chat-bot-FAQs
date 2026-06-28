import streamlit as st
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

st.title("💬 AI FAQ Chatbot (TF-IDF Powered)")

# Load data
df = pd.read_csv("faq.csv")

questions = df["Question"].astype(str).tolist()
answers = df["Answer"].astype(str).tolist()

# Convert text into vectors
vectorizer = TfidfVectorizer()
question_vectors = vectorizer.fit_transform(questions)

user_input = st.text_input("Ask your question:")

if user_input:
    user_vector = vectorizer.transform([user_input])

    # similarity calculation
    similarity = cosine_similarity(user_vector, question_vectors)

    index = similarity.argmax()
    score = similarity[0][index]

    if score > 0.2:   # threshold
        st.success(answers[index])
        st.write(f"Confidence score: {score:.2f}")
    else:
        st.error("Sorry, I couldn't understand your question.")