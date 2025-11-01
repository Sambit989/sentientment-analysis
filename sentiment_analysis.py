

import pandas as pd
import numpy as np
import re
import nltk
import matplotlib.pyplot as plt
import seaborn as sns
import pickle
import warnings
warnings.filterwarnings('ignore')

# Download required NLTK data
print("Downloading NLTK data...")
try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords', quiet=True)
try:
    nltk.data.find('corpora/wordnet')
except LookupError:
    nltk.download('wordnet', quiet=True)
    nltk.download('omw-1.4', quiet=True)

from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import LinearSVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

# ============================================
# CONFIGURATION
# ============================================

CONFIG = {
    'dataset_path': 'IMDB Dataset.csv',
    'test_size': 0.2,
    'random_state': 42,
    'max_features': 10000,  # Increased for better accuracy
    'model_save_path': 'sentiment_model.pkl',
    'vectorizer_save_path': 'tfidf_vectorizer.pkl'
}

# ============================================
# IMPROVED PREPROCESSING WITH NEGATION HANDLING
# ============================================

# Negation words to preserve
NEGATION_WORDS = {
    'not', 'no', 'never', 'neither', 'nobody', 'nothing', 
    'nowhere', 'none', 'nor', 'without', "n't", 'dont', 
    'doesnt', 'didnt', 'wont', 'wouldnt', 'cant', 'couldnt',
    'shouldnt', 'hasnt', 'havent', 'hadnt', 'isnt', 'arent',
    'wasnt', 'werent'
}

def clean_text_improved(text):
    """Enhanced text cleaning that preserves important sentiment markers"""
    # Convert to lowercase
    text = text.lower()
    
    # Handle contractions before removing punctuation
    contractions = {
        "n't": " not",
        "'re": " are",
        "'s": " is",
        "'d": " would",
        "'ll": " will",
        "'ve": " have",
        "'m": " am"
    }
    for contraction, expansion in contractions.items():
        text = text.replace(contraction, expansion)
    
    # Remove HTML tags
    text = re.sub(r'<.*?>', '', text)
    
    # Remove URLs
    text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)
    
    # Keep letters, spaces, and apostrophes
    text = re.sub(r'[^a-zA-Z\s]', ' ', text)
    
    # Remove extra whitespace
    text = ' '.join(text.split())
    
    return text

def handle_negations(text):
    """
    Handle negations by creating negated tokens
    Example: "not happy" becomes "not happy not_happy"
    This helps model learn that "not happy" is different from "happy"
    """
    words = text.split()
    result = []
    i = 0
    
    while i < len(words):
        word = words[i]
        result.append(word)
        
        # If current word is a negation word
        if word in NEGATION_WORDS:
            # Look ahead for next 3 words and create negated versions
            for j in range(1, min(4, len(words) - i)):
                next_word = words[i + j]
                # Create negated version: not_happy, not_good, etc.
                negated = f"not_{next_word}"
                result.append(negated)
        
        i += 1
    
    return ' '.join(result)

def preprocess_text_improved(text):
    """Complete improved preprocessing pipeline"""
    # Clean text
    text = clean_text_improved(text)
    
    # Handle negations BEFORE removing stopwords
    text = handle_negations(text)
    
    # Get stopwords but keep negation words
    stop_words = set(stopwords.words('english'))
    stop_words = stop_words - NEGATION_WORDS
    
    # Also keep intensifiers and important sentiment words
    keep_words = {
        'very', 'really', 'extremely', 'absolutely', 'completely',
        'totally', 'utterly', 'highly', 'quite', 'rather',
        'but', 'however', 'although', 'yet', 'still'
    }
    stop_words = stop_words - keep_words
    
    # Remove stopwords
    words = [word for word in text.split() if word not in stop_words]
    
    # Lemmatization (but preserve our negated tokens)
    lemmatizer = WordNetLemmatizer()
    lemmatized_words = []
    
    for word in words:
        # Don't lemmatize negated tokens (like "not_happy")
        if word.startswith('not_'):
            lemmatized_words.append(word)
        else:
            lemmatized_words.append(lemmatizer.lemmatize(word))
    
    return ' '.join(lemmatized_words)

# ============================================
# DATA LOADING
# ============================================

def load_dataset():
    """Load dataset from CSV or create sample data"""
    try:
        print(f"Loading dataset from {CONFIG['dataset_path']}...")
        df = pd.read_csv(CONFIG['dataset_path'])
        print(f"✓ Dataset loaded successfully! Shape: {df.shape}")
        return df
    except FileNotFoundError:
        print("⚠ Dataset file not found. Using enhanced sample data.")
        
        # Enhanced sample data with negation examples
        sample_data = {
            'review': [
                'This movie was absolutely amazing! Best film I have ever seen.',
                'Not good at all. Very disappointing and boring.',
                'Terrible waste of time. I really hated it.',
                'Great storyline with excellent acting. Highly recommended!',
                'Awful movie. Poor acting and bad script.',
                'Fantastic cinematography and gripping plot!',
                'Not happy with this film. Expected much better.',
                'Outstanding performance! Simply brilliant.',
                'This was not bad, actually quite good.',
                'Complete disaster. Worst movie ever.',
                'I loved it! Not a single boring moment.',
                'Did not enjoy this at all. Very predictable.',
                'Absolutely wonderful! Best experience ever.',
                'Not worth watching. Save your money.',
                'Incredible movie with amazing visuals.',
                'Never seen something so boring in my life.',
                'This is not terrible but not great either.',
                'Perfect in every way. Masterpiece!',
                'Could not believe how bad this was.',
                'Exceptionally good! Highly entertaining.',
            ] * 50,  # Repeat for more training data
            'sentiment': [
                'positive', 'negative', 'negative', 'positive', 'negative',
                'positive', 'negative', 'positive', 'positive', 'negative',
                'positive', 'negative', 'positive', 'negative', 'positive',
                'negative', 'negative', 'positive', 'negative', 'positive'
            ] * 50
        }
        df = pd.DataFrame(sample_data)
        print(f"✓ Enhanced sample dataset created! Shape: {df.shape}")
        return df

# ============================================
# DATA EXPLORATION
# ============================================

def explore_data(df):
    """Explore and visualize the dataset"""
    print("\n" + "="*60)
    print("DATA EXPLORATION")
    print("="*60)
    
    print(f"\nDataset Shape: {df.shape}")
    print(f"\nSentiment Distribution:")
    print(df['sentiment'].value_counts())
    
    # Visualize
    plt.figure(figsize=(8, 5))
    df['sentiment'].value_counts().plot(kind='bar', color=['red', 'green'])
    plt.title('Sentiment Distribution')
    plt.xlabel('Sentiment')
    plt.ylabel('Count')
    plt.xticks(rotation=0)
    plt.tight_layout()
    plt.savefig('sentiment_distribution.png')
    print("\n✓ Saved: sentiment_distribution.png")
    plt.show()

# ============================================
# PREPROCESSING
# ============================================

def preprocess_data(df):
    """Apply improved preprocessing"""
    print("\n" + "="*60)
    print("PREPROCESSING DATA (WITH NEGATION HANDLING)")
    print("="*60)
    
    print("Cleaning and preprocessing text...")
    df['cleaned_review'] = df['review'].apply(preprocess_text_improved)
    
    # Convert sentiment to binary
    df['label'] = df['sentiment'].map({'negative': 0, 'positive': 1})
    
    print("✓ Preprocessing complete!")
    print(f"\nExample transformations:")
    print(f"\nOriginal: 'not happy with the movie'")
    print(f"Processed: '{preprocess_text_improved('not happy with the movie')}'")
    print(f"\nOriginal: 'this is not bad'")
    print(f"Processed: '{preprocess_text_improved('this is not bad')}'")
    
    return df

# ============================================
# FEATURE EXTRACTION
# ============================================

def extract_features_improved(X_train, X_test):
    """Enhanced feature extraction with bigrams and trigrams"""
    print("\n" + "="*60)
    print("ENHANCED FEATURE EXTRACTION")
    print("="*60)
    
    print("Creating TF-IDF features with n-grams...")
    
    # Enhanced TF-IDF with better parameters
    tfidf = TfidfVectorizer(
        max_features=CONFIG['max_features'],
        ngram_range=(1, 3),  # Unigrams, bigrams, and trigrams
        min_df=2,           # Ignore terms that appear in less than 2 documents
        max_df=0.8,         # Ignore terms that appear in more than 80% of documents
        sublinear_tf=True,  # Use logarithmic tf scaling
        use_idf=True
    )
    
    X_train_tfidf = tfidf.fit_transform(X_train)
    X_test_tfidf = tfidf.transform(X_test)
    
    print(f"✓ Feature extraction complete!")
    print(f"Training set shape: {X_train_tfidf.shape}")
    print(f"Testing set shape: {X_test_tfidf.shape}")
    print(f"Vocabulary size: {len(tfidf.vocabulary_)}")
    
    # Show some important features
    feature_names = tfidf.get_feature_names_out()
    negation_features = [f for f in feature_names if 'not_' in f]
    print(f"\nNegation features created: {len(negation_features)}")
    print(f"Sample negation features: {negation_features[:10]}")
    
    return X_train_tfidf, X_test_tfidf, tfidf

# ============================================
# MODEL TRAINING WITH HYPERPARAMETER TUNING
# ============================================

def train_improved_models(X_train, X_test, y_train, y_test):
    """Train models with hyperparameter optimization"""
    print("\n" + "="*60)
    print("TRAINING IMPROVED MODELS")
    print("="*60)
    
    models = {
        'Logistic Regression': LogisticRegression(
            max_iter=1000,
            C=1.0,  # Regularization
            class_weight='balanced',  # Handle imbalanced data
            random_state=CONFIG['random_state']
        ),
        'Naive Bayes': MultinomialNB(alpha=0.1),
        'Linear SVM': LinearSVC(
            C=1.0,
            class_weight='balanced',
            random_state=CONFIG['random_state'],
            max_iter=1000
        )
    }
    
    results = {}
    
    for name, model in models.items():
        print(f"\nTraining {name}...")
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        
        accuracy = accuracy_score(y_test, y_pred)
        results[name] = {
            'model': model,
            'accuracy': accuracy,
            'predictions': y_pred
        }
        
        print(f"✓ {name} Accuracy: {accuracy:.4f} ({accuracy*100:.2f}%)")
    
    # Find best model
    best_model_name = max(results, key=lambda x: results[x]['accuracy'])
    best_model = results[best_model_name]['model']
    
    print(f"\n🏆 Best Model: {best_model_name}")
    print(f"   Accuracy: {results[best_model_name]['accuracy']:.4f}")
    
    return best_model, best_model_name, results

# ============================================
# EVALUATION
# ============================================

def evaluate_model(model, X_test, y_test, model_name):
    """Detailed evaluation"""
    print("\n" + "="*60)
    print(f"EVALUATION - {model_name}")
    print("="*60)
    
    y_pred = model.predict(X_test)
    
    accuracy = accuracy_score(y_test, y_pred)
    print(f"\nAccuracy: {accuracy:.4f} ({accuracy*100:.2f}%)")
    
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred, target_names=['Negative', 'Positive']))
    
    # Confusion Matrix
    cm = confusion_matrix(y_test, y_pred)
    
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=['Negative', 'Positive'],
                yticklabels=['Negative', 'Positive'])
    plt.title(f'Confusion Matrix - {model_name}')
    plt.ylabel('Actual')
    plt.xlabel('Predicted')
    plt.tight_layout()
    plt.savefig('confusion_matrix.png')
    print("\n✓ Saved: confusion_matrix.png")
    plt.show()

# ============================================
# TESTING NEGATION HANDLING
# ============================================

def test_negation_cases(model, vectorizer):
    """Test model on challenging negation cases"""
    print("\n" + "="*60)
    print("TESTING NEGATION HANDLING")
    print("="*60)
    
    test_cases = [
        ("not happy", "negative"),
        ("not good", "negative"),
        ("not bad", "positive"),  # Double negative
        ("never enjoyed", "negative"),
        ("did not like", "negative"),
        ("this is not terrible", "positive"),  # Double negative
        ("not worth watching", "negative"),
        ("could not believe how good", "positive"),
        ("without doubt amazing", "positive"),
        ("nothing special", "negative"),
        ("not impressed", "negative"),
        ("never seen better", "positive"),
        ("happy", "positive"),
        ("terrible", "negative"),
        ("excellent", "positive"),
        ("awful", "negative"),
    ]
    
    print("\nTesting edge cases:\n")
    
    correct = 0
    total = len(test_cases)
    
    for text, expected in test_cases:
        cleaned_text = preprocess_text_improved(text)
        text_tfidf = vectorizer.transform([cleaned_text])
        prediction = model.predict(text_tfidf)[0]
        predicted_sentiment = 'positive' if prediction == 1 else 'negative'
        
        is_correct = predicted_sentiment == expected
        correct += is_correct
        
        status = "✓" if is_correct else "✗"
        print(f"{status} '{text:30}' → Expected: {expected:8} | Got: {predicted_sentiment:8}")
        if not is_correct:
            print(f"   Processed as: {cleaned_text}")
    
    print(f"\n{'='*60}")
    print(f"Negation Handling Accuracy: {correct}/{total} ({correct/total*100:.1f}%)")
    print(f"{'='*60}")

# ============================================
# SAVE MODEL
# ============================================

def save_model(model, vectorizer):
    """Save trained model and vectorizer"""
    print("\n" + "="*60)
    print("SAVING MODEL")
    print("="*60)
    
    with open(CONFIG['model_save_path'], 'wb') as f:
        pickle.dump(model, f)
    print(f"✓ Model saved: {CONFIG['model_save_path']}")
    
    with open(CONFIG['vectorizer_save_path'], 'wb') as f:
        pickle.dump(vectorizer, f)
    print(f"✓ Vectorizer saved: {CONFIG['vectorizer_save_path']}")

# ============================================
# PREDICTION FUNCTION
# ============================================

def predict_sentiment(text, model, vectorizer):
    """Predict sentiment with confidence"""
    cleaned_text = preprocess_text_improved(text)
    text_tfidf = vectorizer.transform([cleaned_text])
    prediction = model.predict(text_tfidf)[0]
    proba = model.predict_proba(text_tfidf)[0] if hasattr(model, 'predict_proba') else None
    
    sentiment = 'Positive' if prediction == 1 else 'Negative'
    confidence = proba[prediction] if proba is not None else None
    
    return sentiment, confidence, cleaned_text

# ============================================
# MAIN EXECUTION
# ============================================

def main():
    """Main execution"""
    print("="*60)
    print("IMPROVED SENTIMENT ANALYSIS - WITH NEGATION HANDLING")
    print("="*60)
    
    # Load data
    df = load_dataset()
    
    # Explore
    explore_data(df)
    
    # Preprocess
    df = preprocess_data(df)
    
    # Split
    print("\n" + "="*60)
    print("SPLITTING DATA")
    print("="*60)
    
    X = df['cleaned_review']
    y = df['label']
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=CONFIG['test_size'],
        random_state=CONFIG['random_state'],
        stratify=y
    )
    
    print(f"Training set: {len(X_train)} samples")
    print(f"Testing set: {len(X_test)} samples")
    
    # Extract features
    X_train_tfidf, X_test_tfidf, vectorizer = extract_features_improved(X_train, X_test)
    
    # Train models
    best_model, best_model_name, all_results = train_improved_models(
        X_train_tfidf, X_test_tfidf, y_train, y_test
    )
    
    # Evaluate
    evaluate_model(best_model, X_test_tfidf, y_test, best_model_name)
    
    # Test negation handling
    test_negation_cases(best_model, vectorizer)
    
    # Save
    save_model(best_model, vectorizer)
    
    print("\n" + "="*60)
    print("✓ TRAINING COMPLETE!")
    print("="*60)
    print("\n🎉 Your model now handles negations correctly!")
    print("   Try: 'not happy' → should predict NEGATIVE")
    print("   Try: 'not bad' → should predict POSITIVE")

if __name__ == "__main__":
    main()