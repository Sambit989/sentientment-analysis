"""
SENTIMENT ANALYSIS WEB APP
A user-friendly interface for sentiment prediction

INSTALLATION:
pip install streamlit

RUN:
streamlit run app.py
"""

import streamlit as st
import pickle
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import pandas as pd
import plotly.graph_objects as go
import time

# Download NLTK data
@st.cache_resource
def download_nltk_data():
    try:
        nltk.data.find('corpora/stopwords')
    except LookupError:
        nltk.download('stopwords', quiet=True)
    try:
        nltk.data.find('corpora/wordnet')
    except LookupError:
        nltk.download('wordnet', quiet=True)
        nltk.download('omw-1.4', quiet=True)

download_nltk_data()

# Page configuration
st.set_page_config(
    page_title="Sentiment Analyzer",
    page_icon="🎭",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
    <style>
    .main {
        background-color: #f5f7fa;
    }
    .stTextArea textarea {
        font-size: 16px;
    }
    .positive-sentiment {
        background-color: #d4edda;
        border-left: 5px solid #28a745;
        padding: 20px;
        border-radius: 5px;
        margin: 10px 0;
    }
    .negative-sentiment {
        background-color: #f8d7da;
        border-left: 5px solid #dc3545;
        padding: 20px;
        border-radius: 5px;
        margin: 10px 0;
    }
    .neutral-sentiment {
        background-color: #fff3cd;
        border-left: 5px solid #ffc107;
        padding: 20px;
        border-radius: 5px;
        margin: 10px 0;
    }
    </style>
""", unsafe_allow_html=True)

# Load model and vectorizer
@st.cache_resource
def load_model_and_vectorizer():
    try:
        with open('sentiment_model.pkl', 'rb') as f:
            model = pickle.load(f)
        with open('tfidf_vectorizer.pkl', 'rb') as f:
            vectorizer = pickle.load(f)
        return model, vectorizer
    except FileNotFoundError:
        return None, None

# Text preprocessing functions
# Negation words to preserve
NEGATION_WORDS = {
    'not', 'no', 'never', 'neither', 'nobody', 'nothing', 
    'nowhere', 'none', 'nor', 'without', "n't", 'dont', 
    'doesnt', 'didnt', 'wont', 'wouldnt', 'cant', 'couldnt',
    'shouldnt', 'hasnt', 'havent', 'hadnt', 'isnt', 'arent',
    'wasnt', 'werent'
}

def clean_text(text):
    text = text.lower()
    
    # Handle contractions
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
    """Handle negations by creating negated tokens"""
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

def predict_sentiment(text, model, vectorizer):
    cleaned_text = preprocess_text(text)
    text_tfidf = vectorizer.transform([cleaned_text])
    prediction = model.predict(text_tfidf)[0]
    
    if hasattr(model, 'predict_proba'):
        proba = model.predict_proba(text_tfidf)[0]
        confidence = proba[prediction]
    else:
        confidence = None
    
    sentiment = 'Positive' if prediction == 1 else 'Negative'
    return sentiment, confidence, cleaned_text

# Sidebar
st.sidebar.title("🎭 Sentiment Analyzer")
st.sidebar.markdown("---")
st.sidebar.info("""
    **About this App:**
    
    This application uses Machine Learning to analyze the sentiment of text reviews.
    
    **How it works:**
    1. Enter your text
    2. Click 'Analyze Sentiment'
    3. Get instant results!
    
    **Model Details:**
    - Algorithm: Logistic Regression
    - Features: TF-IDF
    - Training Data: IMDB Reviews
    - Accuracy: ~88%
""")

st.sidebar.markdown("---")
st.sidebar.markdown("**Sample Texts to Try:**")

sample_texts = {
    "Positive Review": "This movie was absolutely fantastic! The acting was superb and the storyline kept me engaged throughout. Highly recommended!",
    "Negative Review": "Terrible experience. Poor acting, bad script, and boring plot. Complete waste of time and money.",
    "Mixed Review": "The movie had great visuals but the story was quite predictable and the ending was disappointing."
}

selected_sample = st.sidebar.selectbox("Select a sample:", [""] + list(sample_texts.keys()))

# Main content
st.title("🎭 Sentiment Analysis System")
st.markdown("### Analyze the sentiment of your text using AI")

# Check if model is loaded
model, vectorizer = load_model_and_vectorizer()

if model is None or vectorizer is None:
    st.error("⚠️ Model files not found! Please train the model first by running 'sentiment_analysis.py'")
    st.info("After training, make sure 'sentiment_model.pkl' and 'tfidf_vectorizer.pkl' are in the same directory as this app.")
    st.stop()

# Create two columns
col1, col2 = st.columns([2, 1])

with col1:
    st.markdown("#### Enter your text below:")
    
    # Use sample text if selected
    if selected_sample and selected_sample in sample_texts:
        default_text = sample_texts[selected_sample]
    else:
        default_text = ""
    
    user_input = st.text_area(
        "",
        value=default_text,
        height=200,
        placeholder="Type or paste your review, comment, or any text here...",
        key="text_input"
    )
    
    col_btn1, col_btn2, col_btn3 = st.columns([1, 1, 2])
    
    with col_btn1:
        analyze_button = st.button("🔍 Analyze Sentiment", type="primary", use_container_width=True)
    
    with col_btn2:
        clear_button = st.button("🗑️ Clear", use_container_width=True)

with col2:
    st.markdown("#### Quick Stats")
    if user_input:
        word_count = len(user_input.split())
        char_count = len(user_input)
        
        st.metric("Word Count", word_count)
        st.metric("Character Count", char_count)
    else:
        st.info("Enter text to see statistics")

# Clear button functionality
if clear_button:
    st.rerun()

# Analysis
if analyze_button and user_input.strip():
    with st.spinner("Analyzing sentiment..."):
        time.sleep(0.5)  # Small delay for better UX
        
        sentiment, confidence, cleaned_text = predict_sentiment(user_input, model, vectorizer)
        
        # Display results
        st.markdown("---")
        st.markdown("## 📊 Analysis Results")
        
        # Create three columns for results
        res_col1, res_col2, res_col3 = st.columns(3)
        
        with res_col1:
            if sentiment == "Positive":
                st.markdown("""
                    <div class='positive-sentiment'>
                        <h2 style='color: #28a745; margin: 0;'>😊 Positive</h2>
                        <p style='margin: 5px 0 0 0; color: #155724;'>This text expresses positive sentiment</p>
                    </div>
                """, unsafe_allow_html=True)
            else:
                st.markdown("""
                    <div class='negative-sentiment'>
                        <h2 style='color: #dc3545; margin: 0;'>😞 Negative</h2>
                        <p style='margin: 5px 0 0 0; color: #721c24;'>This text expresses negative sentiment</p>
                    </div>
                """, unsafe_allow_html=True)
        
        with res_col2:
            if confidence:
                st.metric(
                    "Confidence Score",
                    f"{confidence * 100:.1f}%",
                    delta=None
                )
                
                # Confidence interpretation
                if confidence >= 0.9:
                    conf_text = "Very High"
                    conf_color = "green"
                elif confidence >= 0.7:
                    conf_text = "High"
                    conf_color = "lightgreen"
                elif confidence >= 0.6:
                    conf_text = "Moderate"
                    conf_color = "orange"
                else:
                    conf_text = "Low"
                    conf_color = "red"
                
                st.markdown(f"**Confidence Level:** :{conf_color}[{conf_text}]")
        
        with res_col3:
            # Create a gauge chart
            if confidence:
                fig = go.Figure(go.Indicator(
                    mode="gauge+number",
                    value=confidence * 100,
                    domain={'x': [0, 1], 'y': [0, 1]},
                    gauge={
                        'axis': {'range': [None, 100]},
                        'bar': {'color': "#28a745" if sentiment == "Positive" else "#dc3545"},
                        'steps': [
                            {'range': [0, 60], 'color': "lightgray"},
                            {'range': [60, 80], 'color': "gray"},
                            {'range': [80, 100], 'color': "darkgray"}
                        ],
                        'threshold': {
                            'line': {'color': "red", 'width': 4},
                            'thickness': 0.75,
                            'value': 90
                        }
                    }
                ))
                fig.update_layout(height=200, margin=dict(l=10, r=10, t=10, b=10))
                st.plotly_chart(fig, use_container_width=True)
        
        # Detailed analysis
        st.markdown("---")
        st.markdown("### 📝 Detailed Analysis")
        
        detail_col1, detail_col2 = st.columns(2)
        
        with detail_col1:
            st.markdown("**Original Text:**")
            st.text_area("", value=user_input, height=150, disabled=True, key="original")
        
        with detail_col2:
            st.markdown("**Preprocessed Text:**")
            st.text_area("", value=cleaned_text, height=150, disabled=True, key="cleaned")
        
        # Additional insights
        st.markdown("---")
        st.markdown("### 💡 Insights")
        
        insight_col1, insight_col2, insight_col3 = st.columns(3)
        
        with insight_col1:
            words = user_input.split()
            avg_word_length = sum(len(word) for word in words) / len(words) if words else 0
            st.metric("Average Word Length", f"{avg_word_length:.1f}")
        
        with insight_col2:
            sentences = user_input.count('.') + user_input.count('!') + user_input.count('?')
            st.metric("Number of Sentences", max(sentences, 1))
        
        with insight_col3:
            cleaned_words = cleaned_text.split()
            st.metric("Words After Preprocessing", len(cleaned_words))

elif analyze_button and not user_input.strip():
    st.warning("⚠️ Please enter some text to analyze!")

# Batch Analysis Section
st.markdown("---")
st.markdown("## 📋 Batch Analysis")
st.markdown("Analyze multiple texts at once by uploading a CSV file")

uploaded_file = st.file_uploader("Upload CSV file (must have a 'text' column)", type=['csv'])

if uploaded_file is not None:
    try:
        df = pd.read_csv(uploaded_file)
        
        if 'text' not in df.columns:
            st.error("❌ CSV file must contain a 'text' column!")
        else:
            st.success(f"✅ Loaded {len(df)} rows")
            
            if st.button("Analyze All Texts", type="primary"):
                with st.spinner("Analyzing all texts..."):
                    progress_bar = st.progress(0)
                    
                    results = []
                    for idx, row in df.iterrows():
                        sentiment, confidence, _ = predict_sentiment(row['text'], model, vectorizer)
                        results.append({
                            'text': row['text'][:100] + '...' if len(row['text']) > 100 else row['text'],
                            'sentiment': sentiment,
                            'confidence': f"{confidence * 100:.1f}%" if confidence else "N/A"
                        })
                        progress_bar.progress((idx + 1) / len(df))
                    
                    results_df = pd.DataFrame(results)
                    
                    st.markdown("### Results")
                    st.dataframe(results_df, use_container_width=True)
                    
                    # Summary
                    positive_count = sum(1 for r in results if r['sentiment'] == 'Positive')
                    negative_count = len(results) - positive_count
                    
                    sum_col1, sum_col2, sum_col3 = st.columns(3)
                    
                    with sum_col1:
                        st.metric("Total Analyzed", len(results))
                    with sum_col2:
                        st.metric("Positive", positive_count, delta=f"{positive_count/len(results)*100:.1f}%")
                    with sum_col3:
                        st.metric("Negative", negative_count, delta=f"{negative_count/len(results)*100:.1f}%")
                    
                    # Download results
                    csv = results_df.to_csv(index=False)
                    st.download_button(
                        "📥 Download Results",
                        csv,
                        "sentiment_analysis_results.csv",
                        "text/csv",
                        key='download-csv'
                    )
    
    except Exception as e:
        st.error(f"❌ Error processing file: {str(e)}")

# Footer
st.markdown("---")
st.markdown("""
    <div style='text-align: center; color: gray; padding: 20px;'>
        <p>Built with ❤️ using Streamlit and Scikit-learn</p>
        <p>Model trained on IMDB Movie Reviews Dataset</p>
    </div>
""", unsafe_allow_html=True)