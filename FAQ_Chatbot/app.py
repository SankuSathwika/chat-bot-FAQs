from flask import Flask, render_template, request, jsonify
import pandas as pd
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

app = Flask(__name__)

# --------------------------------------------------
# Load FAQ Data
# --------------------------------------------------

faq_data = pd.read_csv("faqs.csv")

questions = faq_data["Question"].tolist()
answers = faq_data["Answer"].tolist()

# --------------------------------------------------
# Load AI Model
# --------------------------------------------------

print("Loading AI Model...")

model = SentenceTransformer("all-MiniLM-L6-v2")

faq_embeddings = model.encode(questions)

print("Model Loaded Successfully!")

# --------------------------------------------------
# Home Page
# --------------------------------------------------

@app.route("/")
def home():
    return render_template("index.html")


# --------------------------------------------------
# Chat API
# --------------------------------------------------

@app.route("/chat", methods=["POST"])
def chat():

    data = request.get_json()

    user_question = data.get("message", "")

    greetings = [
        "hi",
        "hello",
        "hey",
        "good morning",
        "good afternoon",
        "good evening"
    ]

    if user_question.lower().strip() in greetings:

        bot_response = (
            "👋 Hello! Welcome to the Smart FAQ Chatbot. "
            "How can I help you today?"
        )

    else:

        user_embedding = model.encode([user_question])

        similarities = cosine_similarity(
            user_embedding,
            faq_embeddings
        )

        best_match_index = similarities.argmax()

        best_score = similarities[0][best_match_index]

        if best_score > 0.50:

            bot_response = answers[best_match_index]

        else:

            bot_response = (
                "❌ Sorry, I couldn't find a relevant answer. "
                "Please try rephrasing your question."
            )

    return jsonify({
        "response": bot_response
    })


# --------------------------------------------------
# Run Flask App
# --------------------------------------------------

if __name__ == "__main__":
    app.run(debug=True)