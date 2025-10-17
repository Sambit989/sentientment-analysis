"""
Quick test script to verify negation handling
Run this after training your improved model

USAGE:
python test_model.py
"""

import pickle
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

# Download NLTK data if needed
try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords', quiet=True)
    nltk.download('wordnet', quiet=True)
    nltk.download('omw-1.4', quiet=True)

# Negation words
NEGATION_WORDS = {
    'not', 'no', 'never', 'neither', 'nobody', 'nothing', 
    'nowhere', 'none', 'nor', 'without', "n't", 'dont', 
    'doesnt', 'didnt', 'wont', 'wouldnt', 'cant', 'couldnt',
    'shouldnt', 'hasnt', 'havent', 'hadnt', 'isnt', 'arent',
    'wasnt', 'werent'
}

def clean_text(text):
    text = text.lower()
    contractions = {
        "n't": " not", "'re": " are", "'s": " is",
        "'d": " would", "'ll": " will", "'ve": " have", "'m": " am"
    }
    for contraction, expansion in contractions.items():
        text = text.replace(contraction, expansion)
    text = re.sub(r'<.*?>', '', text)
    text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)
    text = re.sub(r'[^a-zA-Z\s]', ' ', text)
    text = ' '.join(text.split())
    return text

def handle_negations(text):
    words = text.split()
    result = []
    i = 0
    while i < len(words):
        word = words[i]
        result.append(word)
        if word in NEGATION_WORDS:
            for j in range(1, min(4, len(words) - i)):
                next_word = words[i + j]
                negated = f"not_{next_word}"
                result.append(negated)
        i += 1
    return ' '.join(result)

def preprocess_text(text):
    text = clean_text(text)
    text = handle_negations(text)
    stop_words = set(stopwords.words('english'))
    stop_words = stop_words - NEGATION_WORDS
    keep_words = {'very', 'really', 'extremely', 'absolutely', 'completely',
                  'but', 'however', 'although', 'yet', 'still'}
    stop_words = stop_words - keep_words
    words = [word for word in text.split() if word not in stop_words]
    lemmatizer = WordNetLemmatizer()
    lemmatized_words = []
    for word in words:
        if word.startswith('not_'):
            lemmatized_words.append(word)
        else:
            lemmatized_words.append(lemmatizer.lemmatize(word))
    return ' '.join(lemmatized_words)

def test_model():
    """Test the trained model on various inputs"""
    
    print("="*60)
    print("TESTING SENTIMENT ANALYSIS MODEL")
    print("="*60)
    
    # Load model
    try:
        with open('sentiment_model.pkl', 'rb') as f:
            model = pickle.load(f)
        with open('tfidf_vectorizer.pkl', 'rb') as f:
            vectorizer = pickle.load(f)
        print("✓ Model loaded successfully!\n")
    except FileNotFoundError:
        print("❌ Error: Model files not found!")
        print("Please run 'sentiment_analysis.py' first to train the model.")
        return
    
    # Test cases with expected results
    test_cases = [
        # Negation cases
        ("not happy", "NEGATIVE", "Should be negative - negation of positive word"),
        ("not good", "NEGATIVE", "Should be negative - negation of positive word"),
        ("not bad", "POSITIVE", "Should be positive - double negative"),
        ("not terrible", "POSITIVE", "Should be positive - negation of negative word"),
        ("not impressed", "NEGATIVE", "Should be negative"),
        ("never enjoyed", "NEGATIVE", "Should be negative"),
        ("didn't like", "NEGATIVE", "Should be negative"),
        ("couldn't be better", "POSITIVE", "Should be positive"),
        ("without doubt amazing", "POSITIVE", "Should be positive"),
        
        # Clear positive cases
        ("amazing movie", "POSITIVE", "Clear positive"),
        ("excellent performance", "POSITIVE", "Clear positive"),
        ("loved it", "POSITIVE", "Clear positive"),
        ("wonderful experience", "POSITIVE", "Clear positive"),
        
        # Clear negative cases
        ("terrible film", "NEGATIVE", "Clear negative"),
        ("awful acting", "NEGATIVE", "Clear negative"),
        ("waste of time", "NEGATIVE", "Clear negative"),
        ("boring plot", "NEGATIVE", "Clear negative"),
        
        # Complex cases
        ("the movie was not worth watching", "NEGATIVE", "Negation phrase"),
        ("I would not recommend this", "NEGATIVE", "Negation phrase"),
        ("this is not the best but it's okay", "NEGATIVE", "Mixed sentiment"),
        ("absolutely fantastic", "POSITIVE", "Intensifier + positive"),
        ("really disappointing", "NEGATIVE", "Intensifier + negative")]