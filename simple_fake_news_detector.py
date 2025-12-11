"""
Simple Fake News Detector using Sentiment Analysis
Faster but less accurate than full classification
"""

from transformers import pipeline
import torch
from typing import Dict
import warnings
warnings.filterwarnings('ignore')


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
        print("[OK] Detector ready")

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
