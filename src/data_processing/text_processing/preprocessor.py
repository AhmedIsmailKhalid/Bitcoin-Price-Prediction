import re
import string
from typing import List

import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize


class TextPreprocessor:
    """Text preprocessing for news articles"""
    
    def __init__(self):
        self.download_nltk_data()
        self.lemmatizer = WordNetLemmatizer()
        self.stop_words = set(stopwords.words('english'))
        
        # Add crypto-specific stopwords that don't add sentiment value
        crypto_stopwords = {
            'cryptocurrency', 'cryptocurrencies', 'crypto', 'blockchain', 
            'digital', 'currency', 'token', 'coins', 'trading', 'market',
            'price', 'value', 'usd', 'dollar', 'exchange'
        }
        self.stop_words.update(crypto_stopwords)
    
    def download_nltk_data(self):
        """Download required NLTK data"""
        required_data = ['punkt', 'stopwords', 'wordnet', 'averaged_perceptron_tagger']
        
        for data in required_data:
            try:
                nltk.data.find(f'tokenizers/{data}')
            except LookupError:
                nltk.download(data, quiet=True)
    
    def clean_text(self, text: str) -> str:
        """Basic text cleaning"""
        
        if not text:
            return ""
        
        # Convert to lowercase
        text = text.lower()
        
        # Remove URLs
        text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)
        
        # Remove email addresses
        text = re.sub(r'\S+@\S+', '', text)
        
        # Remove mentions and hashtags (social media artifacts)
        text = re.sub(r'@\w+|#\w+', '', text)
        
        # Remove extra whitespace
        text = re.sub(r'\s+', ' ', text)
        
        # Remove leading/trailing whitespace
        text = text.strip()
        
        return text
    
    def remove_punctuation(self, text: str, keep_sentiment: bool = True) -> str:
        """Remove punctuation, optionally keeping sentiment-relevant marks"""
        
        if keep_sentiment:
            # Keep exclamation marks and question marks for sentiment
            text = re.sub(r'[^\w\s!?]', ' ', text)
        else:
            # Remove all punctuation
            text = text.translate(str.maketrans('', '', string.punctuation))
        
        # Clean up extra spaces
        text = re.sub(r'\s+', ' ', text).strip()
        
        return text
    
    def tokenize(self, text: str) -> List[str]:
        """Tokenize text into words"""
        
        try:
            tokens = word_tokenize(text)
            return tokens
        except Exception:
            # Fallback to simple split if NLTK fails
            return text.split()
    
    def remove_stopwords(self, tokens: List[str]) -> List[str]:
        """Remove stopwords from token list"""
        
        return [token for token in tokens if token.lower() not in self.stop_words]
    
    def lemmatize(self, tokens: List[str]) -> List[str]:
        """Lemmatize tokens to base forms"""
        
        try:
            return [self.lemmatizer.lemmatize(token) for token in tokens]
        except Exception:
            # Return original tokens if lemmatization fails
            return tokens
    
    def filter_tokens(self, tokens: List[str], min_length: int = 2) -> List[str]:
        """Filter tokens by length and remove numbers"""
        
        filtered = []
        for token in tokens:
            # Skip if too short
            if len(token) < min_length:
                continue
            
            # Skip if it's just numbers
            if token.isdigit():
                continue
            
            # Skip if it's only punctuation
            if all(c in string.punctuation for c in token):
                continue
            
            filtered.append(token)
        
        return filtered
    
    def preprocess_for_sentiment(self, text: str) -> str:
        """Preprocess text specifically for sentiment analysis"""
        
        # Light preprocessing to preserve sentiment signals
        text = self.clean_text(text)
        
        # Keep punctuation for sentiment analysis
        text = self.remove_punctuation(text, keep_sentiment=True)
        
        return text
    
    def preprocess_for_features(self, text: str) -> List[str]:
        """Full preprocessing for feature extraction"""
        
        # Complete preprocessing pipeline
        text = self.clean_text(text)
        text = self.remove_punctuation(text, keep_sentiment=False)
        
        tokens = self.tokenize(text)
        tokens = self.remove_stopwords(tokens)
        tokens = self.lemmatize(tokens)
        tokens = self.filter_tokens(tokens)
        
        return tokens
    
    def get_text_stats(self, text: str) -> dict:
        """Get basic statistics about text"""
        
        if not text:
            return {
                'char_count': 0,
                'word_count': 0,
                'sentence_count': 0,
                'avg_word_length': 0
            }
        
        words = text.split()
        sentences = re.split(r'[.!?]+', text)
        
        return {
            'char_count': len(text),
            'word_count': len(words),
            'sentence_count': len([s for s in sentences if s.strip()]),
            'avg_word_length': sum(len(word) for word in words) / len(words) if words else 0
        }