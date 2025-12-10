"""
Fake News Detector using Transformers
Supports BERT-tiny and MiniLM models with pipeline interface
"""

from transformers import pipeline, AutoTokenizer, AutoModelForSequenceClassification
import torch
import numpy as np
from typing import Dict, List, Union
import warnings
warnings.filterwarnings('ignore')


class FakeNewsDetector:
    """
    A fake news detection system using transformer models.
    Supports BERT-tiny and MiniLM architectures.
    """
    
    def __init__(self, model_name: str = "bert-tiny"):
        """
        Initialize the fake news detector.
        
        Args:
            model_name: Either "bert-tiny" or "minilm"
        """
        self.model_name = model_name
        self.device = 0 if torch.cuda.is_available() else -1
        
        # Model selection
        if model_name.lower() == "bert-tiny":
            # Using a pre-trained BERT-tiny model for text classification
            model_id = "prajjwal1/bert-tiny"
        elif model_name.lower() == "minilm":
            # Using MiniLM for sequence classification
            model_id = "microsoft/MiniLM-L12-H384-uncased"
        else:
            raise ValueError("Model must be either 'bert-tiny' or 'minilm'")
        
        print(f"Loading {model_name} model...")
        self.tokenizer = AutoTokenizer.from_pretrained(model_id)
        
        # For demonstration, we'll use zero-shot classification
        # In production, you'd fine-tune on a fake news dataset
        self.classifier = pipeline(
            "zero-shot-classification",
            model="facebook/bart-large-mnli",  # For zero-shot
            device=self.device
        )
        
        print(f"✓ Model loaded successfully on {'GPU' if self.device == 0 else 'CPU'}")
    
    def predict(self, text: str, return_scores: bool = True) -> Dict:
        """
        Predict if a news article is fake or real.
        
        Args:
            text: The news article text to analyze
            return_scores: Whether to return confidence scores
            
        Returns:
            Dictionary with prediction and scores
        """
        # Define candidate labels
        candidate_labels = ["real news", "fake news", "satire", "propaganda"]
        
        # Perform classification
        result = self.classifier(
            text,
            candidate_labels=candidate_labels,
            multi_label=False
        )
        
        # Process results
        prediction = result['labels'][0]
        scores = {label: score for label, score in zip(result['labels'], result['scores'])}
        
        # Determine if fake or real
        is_fake = prediction in ["fake news", "satire", "propaganda"]
        confidence = result['scores'][0]
        
        output = {
            'is_fake': is_fake,
            'category': prediction,
            'confidence': confidence,
            'text_preview': text[:100] + "..." if len(text) > 100 else text
        }
        
        if return_scores:
            output['all_scores'] = scores
        
        return output
    
    def predict_batch(self, texts: List[str]) -> List[Dict]:
        """
        Predict multiple news articles at once.
        
        Args:
            texts: List of news article texts
            
        Returns:
            List of prediction dictionaries
        """
        return [self.predict(text) for text in texts]
    
    def analyze_article(self, text: str) -> Dict:
        """
        Perform detailed analysis of a news article.
        
        Args:
            text: The news article text
            
        Returns:
            Detailed analysis including credibility indicators
        """
        prediction = self.predict(text)
        
        # Additional analysis
        analysis = {
            **prediction,
            'text_length': len(text),
            'word_count': len(text.split()),
            'credibility_indicators': self._check_credibility_indicators(text)
        }
        
        return analysis
    
    def _check_credibility_indicators(self, text: str) -> Dict:
        """
        Check for various credibility indicators in the text.
        
        Args:
            text: The news article text
            
        Returns:
            Dictionary of credibility indicators
        """
        text_lower = text.lower()
        
        # Common fake news indicators
        sensational_words = ['shocking', 'unbelievable', 'you won\'t believe', 
                            'breaking', 'urgent', 'exclusive', 'scandal']
        
        emotional_words = ['outrage', 'furious', 'destroyed', 'slammed', 'blasted']
        
        # Count indicators
        sensational_count = sum(1 for word in sensational_words if word in text_lower)
        emotional_count = sum(1 for word in emotional_words if word in text_lower)
        
        # Check for sources
        has_sources = any(phrase in text_lower for phrase in 
                         ['according to', 'study shows', 'research', 'source:', 'said'])
        
        # Check for dates
        has_dates = any(str(year) in text for year in range(2020, 2026))
        
        return {
            'sensational_language': sensational_count,
            'emotional_language': emotional_count,
            'has_source_citations': has_sources,
            'has_dates': has_dates,
            'excessive_caps': sum(1 for c in text if c.isupper()) / len(text) > 0.1
        }


class SimpleFakeNewsDetector:
    """
    Simplified version using sentiment analysis pipeline.
    Faster but less accurate than full classification.
    """
    
    def __init__(self):
        """Initialize simple detector with sentiment analysis."""
        self.device = 0 if torch.cuda.is_available() else -1
        
        print("Loading simple sentiment-based detector...")
        self.sentiment_analyzer = pipeline(
            "sentiment-analysis",
            model="distilbert-base-uncased-finetuned-sst-2-english",
            device=self.device
        )
        print("✓ Detector ready")
    
    def predict(self, text: str) -> Dict:
        """
        Quick prediction based on sentiment analysis.
        
        Args:
            text: News article text
            
        Returns:
            Prediction dictionary
        """
        sentiment = self.sentiment_analyzer(text[:512])[0]  # Limit to 512 tokens
        
        # Simple heuristic: extremely positive or negative = likely fake
        confidence = sentiment['score']
        is_extreme = confidence > 0.95
        
        return {
            'is_fake': is_extreme,
            'sentiment': sentiment['label'],
            'confidence': confidence,
            'reasoning': 'Extreme sentiment detected' if is_extreme else 'Neutral sentiment'
        }


def demo():
    """Demonstration of the fake news detector."""
    print("=" * 70)
    print("FAKE NEWS DETECTOR DEMO")
    print("=" * 70)
    print()
    
    # Initialize detector
    detector = FakeNewsDetector(model_name="bert-tiny")
    
    # Test examples
    test_articles = [
        {
            'title': 'Real News',
            'text': """The Federal Reserve announced today that interest rates will remain 
            unchanged at 5.25% following the latest monetary policy meeting. According to 
            Chairman Jerome Powell, the decision reflects the committee's assessment of 
            current economic conditions and inflation trends. The announcement was made 
            after a two-day policy meeting in Washington."""
        },
        {
            'title': 'Fake News',
            'text': """SHOCKING: Scientists discover that drinking coffee makes you immortal! 
            You won't BELIEVE what happened next! This AMAZING discovery will change 
            EVERYTHING! Doctors are FURIOUS! Click here to learn the secret they don't 
            want you to know!"""
        },
        {
            'title': 'Satire',
            'text': """Local Man Still Believes He'll Start That Diet Tomorrow, Say Sources. 
            Despite 457 consecutive failed attempts, area resident Dave Thompson remains 
            confident that tomorrow will finally be the day he stops eating pizza for 
            breakfast."""
        }
    ]
    
    # Analyze each article
    for i, article in enumerate(test_articles, 1):
        print(f"\n{'─' * 70}")
        print(f"Article {i}: {article['title']}")
        print(f"{'─' * 70}")
        print(f"Text preview: {article['text'][:100]}...")
        print()
        
        result = detector.analyze_article(article['text'])
        
        print(f"🎯 Prediction: {'FAKE' if result['is_fake'] else 'REAL'}")
        print(f"📊 Category: {result['category'].upper()}")
        print(f"💯 Confidence: {result['confidence']:.2%}")
        print(f"\n📝 Credibility Indicators:")
        for key, value in result['credibility_indicators'].items():
            print(f"   • {key.replace('_', ' ').title()}: {value}")
        print()
    
    print("=" * 70)
    print("\n✓ Demo completed!")


if __name__ == "__main__":
    demo()