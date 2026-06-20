import pandas as pd
import nltk
import string

from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Download NLTK resources
nltk.download('punkt')
nltk.download('punkt_tab')
nltk.download('stopwords')

# Load FAQ data
faq_data = pd.read_csv("faqs.csv")


# Text preprocessing function
def preprocess_text(text):

    text = str(text).lower()

    tokens = word_tokenize(text)

    tokens = [
        word for word in tokens
        if word not in stopwords.words('english')
        and word not in string.punctuation
    ]

    return " ".join(tokens)


# Preprocess FAQ questions
faq_data["Processed_Question"] = faq_data["Question"].apply(preprocess_text)

# TF-IDF Vectorization
vectorizer = TfidfVectorizer()

question_vectors = vectorizer.fit_transform(
    faq_data["Processed_Question"]
)

# User Input
user_question = input("Ask a question: ")

processed_user_question = preprocess_text(user_question)

user_vector = vectorizer.transform(
    [processed_user_question]
)

# Similarity Calculation
similarities = cosine_similarity(
    user_vector,
    question_vectors
)

# Best Match
best_match_index = similarities.argmax()

best_score = similarities[0][best_match_index]

print("\nSimilarity Score:", best_score)

# Confidence Threshold
if best_score > 0.50:
    print("\nAnswer:")
    print(faq_data.iloc[best_match_index]["Answer"])
else:
    print("\nSorry, I couldn't find a relevant answer.")