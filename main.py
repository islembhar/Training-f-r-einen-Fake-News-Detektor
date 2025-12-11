"""
Fake News Detector - Main Demo Script
Imports detector classes from separate modules
"""

from fake_news_detector import FakeNewsDetector
from simple_fake_news_detector import SimpleFakeNewsDetector


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
        print(f"\n{'-' * 70}")
        print(f"Article {i}: {article['title']}")
        print(f"{'-' * 70}")
        print(f"Text preview: {article['text'][:100]}...")
        print()

        result = detector.analyze_article(article['text'])

        print(f"Prediction: {'FAKE' if result['is_fake'] else 'REAL'}")
        print(f"Category: {result['category'].upper()}")
        print(f"Confidence: {result['confidence']:.2%}")
        print(f"\nCredibility Indicators:")
        for key, value in result['credibility_indicators'].items():
            print(f"   - {key.replace('_', ' ').title()}: {value}")
        print()

    print("=" * 70)
    print("\n[OK] Demo completed!")


if __name__ == "__main__":
    demo()