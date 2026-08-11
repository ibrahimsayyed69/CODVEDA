import os
import pandas as pd
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import classification_report, accuracy_score

# Download necessary NLTK data files (run once if needed)
nltk.download('punkt')
nltk.download('stopwords')

# Automatically set the working directory and load the dataset safely
script_dir = os.path.dirname(os.path.abspath(__file__))
file_path = os.path.join(script_dir, 'SMSSpamCollection.csv')

# Load dataset using tab separation and assign columns since SMSSpamCollection lacks headers
df = pd.read_csv(file_path, sep='\t', header=None, names=['label', 'text'], encoding='latin-1')

# 2. Preprocess text using NLTK (Tokenization and Stopword Removal)
stop_words = set(stopwords.words('english'))

def preprocess_text(text):
    # Tokenize the text into words
    tokens = word_tokenize(str(text).lower())
    # Remove stopwords and non-alphabetic tokens
    filtered_tokens = [word for word in tokens if word.isalnum() and word not in stop_words]
    # Join tokens back into a single string
    return " ".join(filtered_tokens)

# Apply NLTK preprocessing to the dataset text column
df['cleaned_text'] = df['text'].apply(preprocess_text)

# 3. Split data into Training and Testing sets
X_train, X_test, y_train, y_test = train_test_split(
    df['cleaned_text'], df['label'], test_size=0.2, random_state=42
)

# 4. Convert text into numerical representation using TF-IDF
vectorizer = TfidfVectorizer()
X_train_tfidf = vectorizer.fit_transform(X_train)
X_test_tfidf = vectorizer.transform(X_test)

# 5. Train a Classification Model
model = MultinomialNB()
model.fit(X_train_tfidf, y_train)

# 6. Evaluate the Model
y_pred = model.predict(X_test_tfidf)

print(f"Accuracy: {accuracy_score(y_test, y_pred):.4f}\n")
print("Classification Report:")
print(classification_report(y_test, y_pred))