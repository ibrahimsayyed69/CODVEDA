# Level 3 Task 2: NLP Text Classification (SMS Spam Detector):-

This project implements a machine learning model to classify text messages as either "ham" (legitimate) or "spam". The project fulfills advanced requirements for Natural Language Processing (NLP) by covering the full pipeline from data preprocessing to model evaluation.

### Project Objectives:

Preprocessing: Utilize NLTK for text tokenization and stopword removal.  
Feature Extraction: Convert text data into numerical vectors using TF-IDF (Term Frequency-Inverse Document Frequency).
Classification: Train a Naive Bayes model to predict the class of incoming text messages.
Evaluation: Assess model performance using Accuracy, Precision, Recall, and F1-score.

### Technologies Used:

Python: Core programming language.
Pandas: Data manipulation and loading.
NLTK (Natural Language Toolkit): Text preprocessing (tokenization, stopword filtering).
Scikit-learn: TF-IDF vectorization, model training, and performance metrics.

### Dataset:

The project uses the SMS Spam Collection dataset, which consists of 5,574 English messages labeled as 'ham' or 'spam'. The data is expected to be in a tab-separated CSV format named

### Project Structure:

Data Loading: Reads the dataset from the local directory and assigns appropriate labels.
Text Cleaning: Cleans text by converting to lowercase, removing punctuation/non-alphabetic characters, and filtering out common English stopwords.  
Vectorization: Transforms cleaned text into a TF-IDF matrix, which represents the importance of words in the corpus.
Model Training: Uses a MultinomialNB (Multinomial Naive Bayes) classifier, which is highly effective for text classification tasks.  
Evaluation: Outputs a classification report summarizing the model's predictive performance.

## How to Run:

1. Ensure you have the required libraries installed:  
   Bash
   pip install pandas nltk scikit-learn.
2. Place the SMSSpamCollection.csv file in the same directory as the script.
3. Run the script:
   Bash
   python Lv.3,Task2-NLP.py
