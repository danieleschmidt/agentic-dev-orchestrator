#!/usr/bin/env python3
"""
Sentiment Analysis Examples
Comprehensive examples demonstrating various use cases of the sentiment analysis system
"""

import asyncio
import json
import time
import statistics
from typing import List, Dict, Any
from pathlib import Path

# Add parent directory to path for imports
import sys
sys.path.append(str(Path(__file__).parent.parent))

from src.sentiment import (
    SentimentAnalyzer, AsyncSentimentAnalyzer, SentimentCache,
    performance_monitor, SentimentResult
)


def example_basic_usage():
    """Basic sentiment analysis example"""
    print("🎯 Example 1: Basic Sentiment Analysis")
    print("=" * 50)
    
    # Initialize analyzer
    analyzer = SentimentAnalyzer()
    
    # Sample texts
    texts = [
        "I absolutely love this product! It's amazing!",
        "This is the worst experience I've ever had.",
        "The service was okay, nothing special.",
        "Fantastic quality and great customer support!",
        "Completely disappointed with the delivery time."
    ]
    
    print("Analyzing individual texts:")
    for i, text in enumerate(texts, 1):
        result = analyzer.analyze(text)
        print(f"\n{i}. Text: '{text}'")
        print(f"   Sentiment: {result.label.value.upper()}")
        print(f"   Confidence: {result.confidence:.1%}")
        print(f"   Compound Score: {result.scores.compound:.3f}")


def example_batch_processing():
    """Batch processing example"""
    print("\n\n🚀 Example 2: Batch Processing")
    print("=" * 50)
    
    analyzer = SentimentAnalyzer()
    
    # Generate sample customer reviews
    reviews = [
        "Excellent product, highly recommended!",
        "Poor quality, money wasted.",
        "Average product, meets expectations.",
        "Outstanding service, will buy again!",
        "Terrible shipping experience.",
        "Good value for the price.",
        "Not what I expected, disappointed.",
        "Perfect! Everything works great.",
        "Could be better, has some issues.",
        "Amazing experience from start to finish!"
    ]
    
    print(f"Processing {len(reviews)} reviews in batch...")
    
    # Measure performance
    start_time = time.time()
    results = analyzer.analyze_batch(reviews)
    end_time = time.time()
    
    # Analyze results
    positive_count = sum(1 for r in results if r.label.value == 'positive')
    negative_count = sum(1 for r in results if r.label.value == 'negative')
    neutral_count = sum(1 for r in results if r.label.value == 'neutral')
    
    avg_confidence = statistics.mean(r.confidence for r in results)
    processing_time = end_time - start_time
    throughput = len(reviews) / processing_time
    
    print(f"\n📊 Batch Processing Results:")
    print(f"Total reviews: {len(reviews)}")
    print(f"Processing time: {processing_time:.3f} seconds")
    print(f"Throughput: {throughput:.1f} reviews/second")
    print(f"Average confidence: {avg_confidence:.1%}")
    print(f"\n📈 Sentiment Distribution:")
    print(f"  Positive: {positive_count} ({positive_count/len(reviews)*100:.1f}%)")
    print(f"  Negative: {negative_count} ({negative_count/len(reviews)*100:.1f}%)")
    print(f"  Neutral: {neutral_count} ({neutral_count/len(reviews)*100:.1f}%)")


async def example_async_processing():
    """Async processing example"""
    print("\n\n⚡ Example 3: Async Processing")
    print("=" * 50)
    
    analyzer = AsyncSentimentAnalyzer(max_workers=4, batch_size=20)
    
    try:
        # Generate larger dataset
        texts = []
        categories = [
            ("Excellent service", "positive"),
            ("Poor quality", "negative"), 
            ("Average experience", "neutral")
        ]
        
        for i in range(100):
            template, sentiment = categories[i % len(categories)]
            texts.append(f"{template} - comment {i+1}")
        
        print(f"Processing {len(texts)} texts asynchronously...")
        
        # Track progress
        progress_updates = []
        def progress_callback(completed, total):
            progress_updates.append((completed, total))
            if completed % 20 == 0 or completed == total:
                print(f"Progress: {completed}/{total} ({completed/total*100:.1f}%)")
        
        # Process asynchronously
        start_time = time.time()
        results = await analyzer.analyze_batch_async(texts, progress_callback)
        end_time = time.time()
        
        # Performance metrics
        processing_time = end_time - start_time
        throughput = len(texts) / processing_time
        
        print(f"\n⚡ Async Processing Results:")
        print(f"Total texts: {len(texts)}")
        print(f"Processing time: {processing_time:.3f} seconds")
        print(f"Throughput: {throughput:.1f} texts/second")
        print(f"Progress updates: {len(progress_updates)}")
        
        # Get performance stats
        stats = analyzer.get_performance_stats()
        cache_stats = stats['cache_stats']
        print(f"Cache hit rate: {cache_stats['hit_rate_percent']:.1f}%")
        
    finally:
        analyzer.cleanup()


def example_caching_performance():
    """Caching performance example"""
    print("\n\n💾 Example 4: Caching Performance")
    print("=" * 50)
    
    # Create analyzer with cache
    cache = SentimentCache(ttl=300)  # 5 minutes
    analyzer = SentimentAnalyzer()
    
    # Test texts (some duplicates to demonstrate caching)
    test_texts = [
        "Great product, very satisfied!",
        "Poor service, not recommended.",
        "Great product, very satisfied!",  # Duplicate
        "Average quality, nothing special.",
        "Poor service, not recommended.",  # Duplicate
        "Excellent customer support!",
        "Great product, very satisfied!",  # Duplicate
    ]
    
    print("Testing cache performance...")
    
    # First pass (populate cache)
    print("\n🔄 First pass (cache population):")
    start_time = time.time()
    for i, text in enumerate(test_texts):
        result = analyzer.analyze(text)
        cached = cache.set(text, result)
        print(f"Text {i+1}: {result.label.value} (cached: {'✅' if cached else '❌'})")
    first_pass_time = time.time() - start_time
    
    # Second pass (cache hits)
    print("\n🎯 Second pass (cache retrieval):")
    start_time = time.time()
    cache_hits = 0
    for i, text in enumerate(test_texts):
        cached_result = cache.get(text)
        if cached_result:
            cache_hits += 1
            print(f"Text {i+1}: {cached_result.label.value} (cache hit: ✅)")
        else:
            result = analyzer.analyze(text)
            print(f"Text {i+1}: {result.label.value} (cache miss: ❌)")
    second_pass_time = time.time() - start_time
    
    # Performance comparison
    print(f"\n📊 Caching Performance Results:")
    print(f"First pass time: {first_pass_time:.3f} seconds")
    print(f"Second pass time: {second_pass_time:.3f} seconds")
    print(f"Speed improvement: {first_pass_time/second_pass_time:.1f}x faster")
    print(f"Cache hits: {cache_hits}/{len(test_texts)} ({cache_hits/len(test_texts)*100:.1f}%)")
    
    # Cache statistics
    stats = cache.get_stats()
    print(f"Cache efficiency: {stats['cache_efficiency']}")


def example_real_time_monitoring():
    """Real-time sentiment monitoring example"""
    print("\n\n📊 Example 5: Real-time Sentiment Monitoring")
    print("=" * 50)
    
    analyzer = SentimentAnalyzer()
    
    # Simulate real-time data stream
    sentiment_stream = [
        "Just bought the new iPhone - loving it!",
        "Delivery was delayed again, very frustrating",
        "Customer service helped solve my issue quickly",
        "Product quality is declining lately",
        "Best purchase I've made this year!",
        "Website is confusing and slow",
        "Great value for money, recommended",
        "Return process was a nightmare",
        "Perfect condition, fast shipping",
        "App keeps crashing, need better testing"
    ]
    
    print("Monitoring sentiment stream...")
    
    # Real-time analysis with running statistics
    running_stats = {
        'total_processed': 0,
        'positive_count': 0,
        'negative_count': 0,
        'neutral_count': 0,
        'confidence_sum': 0,
        'compound_scores': []
    }
    
    for i, message in enumerate(sentiment_stream, 1):
        # Simulate real-time delay
        time.sleep(0.5)
        
        # Analyze sentiment
        result = analyzer.analyze(message)
        
        # Update running statistics
        running_stats['total_processed'] = i
        running_stats['confidence_sum'] += result.confidence
        running_stats['compound_scores'].append(result.scores.compound)
        
        if result.label.value == 'positive':
            running_stats['positive_count'] += 1
        elif result.label.value == 'negative':
            running_stats['negative_count'] += 1
        else:
            running_stats['neutral_count'] += 1
        
        # Calculate current metrics
        avg_confidence = running_stats['confidence_sum'] / i
        avg_compound = statistics.mean(running_stats['compound_scores'])
        
        # Display real-time update
        print(f"\n📨 Message {i}: '{message[:50]}{'...' if len(message) > 50 else ''}'")
        print(f"   Sentiment: {result.label.value.upper()} ({result.confidence:.1%})")
        print(f"   Running avg confidence: {avg_confidence:.1%}")
        print(f"   Running avg compound: {avg_compound:.3f}")
        print(f"   Distribution: P:{running_stats['positive_count']} "
              f"N:{running_stats['negative_count']} "
              f"U:{running_stats['neutral_count']}")


def example_business_intelligence():
    """Business intelligence example"""
    print("\n\n💼 Example 6: Business Intelligence Dashboard")
    print("=" * 50)
    
    analyzer = SentimentAnalyzer()
    
    # Simulate business data from different sources
    business_data = {
        "customer_reviews": [
            "Amazing product quality, exceeded expectations!",
            "Delivery was late but product is good",
            "Poor customer service experience",
            "Best value for money in the market",
            "Product arrived damaged, very disappointed"
        ],
        "social_media": [
            "@company loving the new update! 🚀",
            "Why is your app so slow? Fix it please",
            "Great customer support, problem solved quickly",
            "Overpriced for what you get",
            "Highly recommend this to everyone!"
        ],
        "support_tickets": [
            "Issue resolved quickly, satisfied with help",
            "Still waiting for response, very frustrated",
            "Technical support was knowledgeable and helpful",
            "Multiple issues with the same product",
            "Excellent follow-up service"
        ],
        "employee_feedback": [
            "Love working here, great team culture",
            "Management needs better communication",
            "Good work-life balance and benefits",
            "Career growth opportunities are limited",
            "Proud to be part of this company"
        ]
    }
    
    print("Analyzing business sentiment across channels...")
    
    channel_analysis = {}
    overall_insights = []
    
    for channel, texts in business_data.items():
        print(f"\n📈 Analyzing {channel.replace('_', ' ').title()}...")
        
        # Batch analyze channel data
        results = analyzer.analyze_batch(texts)
        
        # Calculate channel metrics
        sentiments = [r.label.value for r in results]
        confidences = [r.confidence for r in results]
        compounds = [r.scores.compound for r in results]
        
        positive_pct = sentiments.count('positive') / len(sentiments) * 100
        negative_pct = sentiments.count('negative') / len(sentiments) * 100
        neutral_pct = sentiments.count('neutral') / len(sentiments) * 100
        
        avg_confidence = statistics.mean(confidences)
        avg_compound = statistics.mean(compounds)
        
        # Store channel analysis
        channel_analysis[channel] = {
            'total_messages': len(texts),
            'positive_percentage': positive_pct,
            'negative_percentage': negative_pct,
            'neutral_percentage': neutral_pct,
            'average_confidence': avg_confidence,
            'average_compound': avg_compound,
            'sentiment_trend': 'positive' if avg_compound > 0.1 else 'negative' if avg_compound < -0.1 else 'neutral'
        }
        
        print(f"   Total messages: {len(texts)}")
        print(f"   Positive: {positive_pct:.1f}%")
        print(f"   Negative: {negative_pct:.1f}%")
        print(f"   Neutral: {neutral_pct:.1f}%")
        print(f"   Overall trend: {channel_analysis[channel]['sentiment_trend'].upper()}")
        print(f"   Confidence: {avg_confidence:.1%}")
        
        # Generate insights
        if negative_pct > 40:
            overall_insights.append(f"⚠️  High negative sentiment in {channel} ({negative_pct:.1f}%)")
        if positive_pct > 70:
            overall_insights.append(f"🎉 Strong positive sentiment in {channel} ({positive_pct:.1f}%)")
        if avg_confidence < 0.6:
            overall_insights.append(f"📊 Low confidence scores in {channel} - review data quality")
    
    # Overall business sentiment summary
    print(f"\n🎯 Business Intelligence Summary:")
    print("=" * 30)
    
    all_compounds = []
    all_messages = 0
    
    for channel, analysis in channel_analysis.items():
        all_messages += analysis['total_messages']
        # Weight by message count
        weighted_compound = analysis['average_compound'] * analysis['total_messages']
        all_compounds.extend([weighted_compound])
    
    overall_compound = sum(all_compounds) / all_messages if all_messages > 0 else 0
    overall_trend = 'POSITIVE' if overall_compound > 0.1 else 'NEGATIVE' if overall_compound < -0.1 else 'NEUTRAL'
    
    print(f"Total messages analyzed: {all_messages}")
    print(f"Overall business sentiment: {overall_trend}")
    print(f"Overall compound score: {overall_compound:.3f}")
    
    print(f"\n🔍 Key Insights:")
    for insight in overall_insights:
        print(f"  {insight}")
    
    if not overall_insights:
        print("  ✅ Sentiment levels are balanced across all channels")
    
    # Export results
    export_data = {
        'analysis_timestamp': time.time(),
        'channel_analysis': channel_analysis,
        'overall_metrics': {
            'total_messages': all_messages,
            'overall_sentiment': overall_trend,
            'overall_compound': overall_compound
        },
        'insights': overall_insights
    }
    
    with open('business_sentiment_report.json', 'w') as f:
        json.dump(export_data, f, indent=2)
    
    print(f"\n💾 Report exported to business_sentiment_report.json")


def example_custom_preprocessing():
    """Custom text preprocessing example"""
    print("\n\n🔧 Example 7: Custom Text Preprocessing")
    print("=" * 50)
    
    import re
    
    def advanced_preprocessing(text):
        """Advanced text preprocessing for better sentiment analysis"""
        if not text:
            return ""
        
        # Convert to string if needed
        text = str(text)
        
        # Remove URLs
        text = re.sub(r'http[s]?://\S+', ' ', text)
        
        # Remove email addresses
        text = re.sub(r'\S+@\S+', ' ', text)
        
        # Handle emoticons and emojis (expand common ones)
        emoticon_map = {
            ':)': ' positive_emotion ',
            ':-)': ' positive_emotion ',
            ':(': ' negative_emotion ',
            ':-(': ' negative_emotion ',
            ':D': ' very_positive_emotion ',
            ':/': ' mixed_emotion ',
            ':P': ' playful_emotion '
        }
        
        for emoticon, replacement in emoticon_map.items():
            text = text.replace(emoticon, replacement)
        
        # Handle repeated characters (e.g., "sooooo good" -> "so good")
        text = re.sub(r'(.)\1{2,}', r'\1\1', text)
        
        # Handle all caps (might indicate strong emotion)
        if text.isupper() and len(text) > 10:
            text = text.lower() + ' strong_emphasis'
        
        # Remove excessive punctuation
        text = re.sub(r'[!]{2,}', '!', text)
        text = re.sub(r'[?]{2,}', '?', text)
        
        # Clean whitespace
        text = re.sub(r'\s+', ' ', text.strip())
        
        return text
    
    # Test preprocessing with challenging texts
    test_texts = [
        "This is ABSOLUTELY AMAZING!!!! https://example.com",
        "Sooooo disappointed :( contact@support.com didn't help",
        "Average product :/ nothing special",
        "WORST    EXPERIENCE    EVER!!!!",
        "Great!!! Will definitely buy again :D"
    ]
    
    analyzer = SentimentAnalyzer()
    
    print("Comparing sentiment analysis with and without preprocessing:")
    
    for i, original_text in enumerate(test_texts, 1):
        # Analysis without preprocessing
        result_original = analyzer.analyze(original_text)
        
        # Analysis with preprocessing
        preprocessed_text = advanced_preprocessing(original_text)
        result_preprocessed = analyzer.analyze(preprocessed_text)
        
        print(f"\n{i}. Original: '{original_text}'")
        print(f"   Preprocessed: '{preprocessed_text}'")
        print(f"   Original sentiment: {result_original.label.value} ({result_original.confidence:.1%})")
        print(f"   Preprocessed sentiment: {result_preprocessed.label.value} ({result_preprocessed.confidence:.1%})")
        
        # Highlight significant changes
        if result_original.label != result_preprocessed.label:
            print(f"   🔄 Sentiment changed: {result_original.label.value} → {result_preprocessed.label.value}")
        
        confidence_diff = abs(result_original.confidence - result_preprocessed.confidence)
        if confidence_diff > 0.1:
            print(f"   📊 Confidence change: {confidence_diff:.1%}")


async def example_performance_comparison():
    """Performance comparison example"""
    print("\n\n⚡ Example 8: Performance Comparison")
    print("=" * 50)
    
    # Generate test dataset
    test_sizes = [10, 100, 500, 1000]
    base_texts = [
        "Excellent product quality!",
        "Poor customer service.",
        "Average experience overall.",
        "Outstanding value for money!",
        "Disappointing delivery time."
    ]
    
    print("Comparing synchronous vs asynchronous performance...")
    
    for size in test_sizes:
        # Generate test data
        texts = [f"{base_texts[i % len(base_texts)]} - item {i}" for i in range(size)]
        
        print(f"\n📊 Testing with {size} texts:")
        
        # Synchronous analysis
        sync_analyzer = SentimentAnalyzer()
        start_time = time.time()
        sync_results = sync_analyzer.analyze_batch(texts)
        sync_time = time.time() - start_time
        sync_throughput = size / sync_time
        
        # Asynchronous analysis
        async_analyzer = AsyncSentimentAnalyzer(max_workers=4)
        try:
            start_time = time.time()
            async_results = await async_analyzer.analyze_batch_async(texts)
            async_time = time.time() - start_time
            async_throughput = size / async_time
            
            # Performance comparison
            speedup = sync_time / async_time if async_time > 0 else 1
            
            print(f"   Sync time: {sync_time:.3f}s ({sync_throughput:.1f} texts/sec)")
            print(f"   Async time: {async_time:.3f}s ({async_throughput:.1f} texts/sec)")
            print(f"   Speedup: {speedup:.1f}x {'faster' if speedup > 1 else 'slower'}")
            
            # Verify results consistency
            sync_sentiments = [r.label.value for r in sync_results]
            async_sentiments = [r.label.value for r in async_results]
            
            consistency = sum(1 for s, a in zip(sync_sentiments, async_sentiments) if s == a)
            consistency_pct = consistency / len(sync_sentiments) * 100
            
            print(f"   Result consistency: {consistency_pct:.1f}%")
            
        finally:
            async_analyzer.cleanup()


def main():
    """Run all sentiment analysis examples"""
    print("🎯 Sentiment Analysis Examples")
    print("=" * 60)
    print("This script demonstrates various use cases and features")
    print("of the sentiment analysis system.\n")
    
    # Run synchronous examples
    example_basic_usage()
    example_batch_processing()
    example_caching_performance()
    example_real_time_monitoring()
    example_business_intelligence()
    example_custom_preprocessing()
    
    # Run asynchronous examples
    print("\n🔄 Running asynchronous examples...")
    asyncio.run(example_async_processing())
    asyncio.run(example_performance_comparison())
    
    print("\n\n✅ All examples completed successfully!")
    print("Check the generated files:")
    print("  - business_sentiment_report.json")
    print("\nFor more examples, see the documentation at docs/SENTIMENT_ANALYSIS_GUIDE.md")


if __name__ == "__main__":
    main()