# 🎯 Sentiment Analysis Guide

Complete guide for using the sentiment analysis features in the Agentic Dev Orchestrator.

## 📋 Table of Contents

- [Overview](#overview)
- [Quick Start](#quick-start)
- [CLI Usage](#cli-usage)
- [API Reference](#api-reference)
- [Python SDK](#python-sdk)
- [Advanced Features](#advanced-features)
- [Performance Tuning](#performance-tuning)
- [Examples](#examples)
- [Best Practices](#best-practices)

## 🔍 Overview

The sentiment analysis module provides real-time sentiment classification for text data using advanced NLP techniques. It supports:

- **Real-time analysis**: Single text sentiment analysis
- **Batch processing**: Efficient bulk analysis
- **Async processing**: Concurrent analysis for high throughput
- **Smart caching**: Intelligent result caching
- **Multi-language support**: Unicode and international text
- **Performance monitoring**: Built-in metrics and profiling

## 🚀 Quick Start

### Installation

```bash
# Install the package
pip install agentic-dev-orchestrator[sentiment]

# Or install from source
git clone <repository-url>
cd agentic-dev-orchestrator
pip install -e .[sentiment]
```

### Basic Usage

```python
from src.sentiment import SentimentAnalyzer

# Create analyzer
analyzer = SentimentAnalyzer()

# Analyze single text
result = analyzer.analyze("This product is absolutely amazing!")

print(f"Sentiment: {result.label.value}")
print(f"Confidence: {result.confidence:.2%}")
print(f"Scores: {result.scores}")
```

## 💻 CLI Usage

### Single Text Analysis

```bash
# Analyze text directly
python ado.py sentiment "This is fantastic work!"

# Analyze file content
python ado.py sentiment --file README.md

# Get help
python ado.py sentiment --help
```

### Batch Analysis

```bash
# Analyze all backlog items
python ado.py sentiment-backlog
```

### Example Output

```
💭 Sentiment Analysis
📝 Text: 'This is fantastic work!'
📊 Sentiment: POSITIVE
🎯 Confidence: 82.5%
📈 Scores:
  Positive: 0.750
  Negative: 0.100
  Neutral:  0.150
  Compound: 0.625
```

## 🔌 API Reference

### REST Endpoints

#### Analyze Single Text

```http
POST /api/v1/sentiment/analyze
Content-Type: application/json

{
  "text": "Your text here",
  "metadata": {
    "source": "user_feedback",
    "user_id": "123"
  }
}
```

Response:
```json
{
  "text": "Your text here",
  "label": "positive",
  "confidence": 0.85,
  "scores": {
    "positive": 0.75,
    "negative": 0.10,
    "neutral": 0.15,
    "compound": 0.65
  },
  "metadata": {
    "source": "user_feedback",
    "user_id": "123"
  }
}
```

#### Batch Analysis

```http
POST /api/v1/sentiment/batch
Content-Type: application/json

{
  "texts": [
    "Great product!",
    "Poor quality.",
    "Average experience."
  ]
}
```

Response:
```json
{
  "results": [
    {
      "text": "Great product!",
      "label": "positive",
      "confidence": 0.90,
      "scores": {...}
    },
    {
      "text": "Poor quality.",
      "label": "negative", 
      "confidence": 0.85,
      "scores": {...}
    },
    {
      "text": "Average experience.",
      "label": "neutral",
      "confidence": 0.70,
      "scores": {...}
    }
  ],
  "total": 3,
  "timestamp": "2024-01-15T10:30:00Z"
}
```

#### Backlog Item Analysis

```http
POST /api/v1/sentiment/backlog/{item_id}
```

Response:
```json
{
  "item_id": "TASK-123",
  "title_sentiment": {
    "text": "Fix critical bug",
    "label": "negative",
    "confidence": 0.75,
    "metadata": {"field": "title"}
  },
  "description_sentiment": {
    "text": "Users are experiencing crashes...",
    "label": "negative",
    "confidence": 0.80,
    "metadata": {"field": "description"}
  },
  "timestamp": "2024-01-15T10:30:00Z"
}
```

### Error Handling

```json
{
  "error": "Text required",
  "code": 400,
  "timestamp": "2024-01-15T10:30:00Z"
}
```

Common error codes:
- `400`: Bad request (missing text, invalid format)
- `429`: Rate limit exceeded
- `500`: Internal server error

## 🐍 Python SDK

### Basic Usage

```python
from src.sentiment import SentimentAnalyzer, SentimentCache

# Initialize with caching
cache = SentimentCache(ttl=3600)  # 1 hour cache
analyzer = SentimentAnalyzer(cache=cache)

# Single analysis
result = analyzer.analyze("Amazing product!")
print(f"Sentiment: {result.label.value}")

# Batch analysis
texts = ["Great!", "Terrible!", "Okay."]
results = analyzer.analyze_batch(texts)

for result in results:
    print(f"{result.text} -> {result.label.value}")
```

### Async Processing

```python
import asyncio
from src.sentiment import AsyncSentimentAnalyzer

async def analyze_large_batch():
    analyzer = AsyncSentimentAnalyzer(max_workers=8)
    
    texts = [f"Sample text {i}" for i in range(1000)]
    
    # Progress tracking
    def progress_callback(completed, total):
        print(f"Progress: {completed}/{total} ({completed/total*100:.1f}%)")
    
    results = await analyzer.analyze_batch_async(
        texts, 
        progress_callback=progress_callback
    )
    
    # Stream processing
    async for result in analyzer.analyze_stream(texts[:10]):
        print(f"Streaming: {result.label.value}")
    
    analyzer.cleanup()

# Run async analysis
asyncio.run(analyze_large_batch())
```

### Custom Validation

```python
from src.sentiment import SentimentAnalyzer, InputValidator

# Custom validator
validator = InputValidator(max_length=5000, min_length=5)
analyzer = SentimentAnalyzer(validator=validator)

try:
    result = analyzer.analyze("Short")  # Will raise ValidationError
except ValidationError as e:
    print(f"Validation failed: {e}")
```

### Performance Monitoring

```python
from src.sentiment import performance_monitor

# Monitor analysis performance
with performance_monitor.monitor_operation("batch_analysis", 100) as metrics:
    results = analyzer.analyze_batch(texts)
    metrics['cache_hit_rate'] = 85.0  # Custom metric

# Get performance summary
summary = performance_monitor.get_performance_summary()
print(f"Average throughput: {summary['overall_throughput_per_sec']:.1f} texts/sec")
```

## 🔬 Advanced Features

### Sentiment Trends

```python
def analyze_sentiment_trends(texts_by_date):
    analyzer = SentimentAnalyzer()
    trends = {}
    
    for date, texts in texts_by_date.items():
        results = analyzer.analyze_batch(texts)
        
        # Calculate sentiment distribution
        sentiment_counts = {
            'positive': sum(1 for r in results if r.label.value == 'positive'),
            'negative': sum(1 for r in results if r.label.value == 'negative'),
            'neutral': sum(1 for r in results if r.label.value == 'neutral')
        }
        
        # Calculate average confidence
        avg_confidence = sum(r.confidence for r in results) / len(results)
        
        trends[date] = {
            'sentiment_distribution': sentiment_counts,
            'average_confidence': avg_confidence,
            'total_texts': len(texts)
        }
    
    return trends
```

### Custom Sentiment Categories

```python
from src.sentiment import SentimentAnalyzer

def categorize_feedback(feedback_texts):
    analyzer = SentimentAnalyzer()
    categories = {
        'promoters': [],      # Very positive (compound > 0.5)
        'passives': [],       # Neutral (-0.1 < compound < 0.5)
        'detractors': []      # Negative (compound <= -0.1)
    }
    
    for text in feedback_texts:
        result = analyzer.analyze(text)
        
        if result.scores.compound > 0.5:
            categories['promoters'].append({
                'text': text,
                'score': result.scores.compound
            })
        elif result.scores.compound <= -0.1:
            categories['detractors'].append({
                'text': text,
                'score': result.scores.compound
            })
        else:
            categories['passives'].append({
                'text': text,
                'score': result.scores.compound
            })
    
    return categories
```

### Integration with Data Processing

```python
import pandas as pd
from src.sentiment import SentimentAnalyzer

def process_dataframe(df, text_column='text'):
    analyzer = SentimentAnalyzer()
    
    # Batch process all texts
    texts = df[text_column].tolist()
    results = analyzer.analyze_batch(texts)
    
    # Add sentiment columns
    df['sentiment_label'] = [r.label.value for r in results]
    df['sentiment_confidence'] = [r.confidence for r in results]
    df['sentiment_compound'] = [r.scores.compound for r in results]
    
    return df

# Usage
df = pd.DataFrame({
    'id': [1, 2, 3],
    'text': ['Great product!', 'Poor quality', 'Average service']
})

df_with_sentiment = process_dataframe(df)
print(df_with_sentiment)
```

## ⚡ Performance Tuning

### Optimal Configuration

```python
# High-throughput configuration
analyzer = AsyncSentimentAnalyzer(
    max_workers=8,           # 2x CPU cores
    batch_size=50,           # Larger batches
    cache_ttl=7200           # 2 hour cache
)

# Memory-optimized configuration
analyzer = AsyncSentimentAnalyzer(
    max_workers=2,           # Fewer workers
    batch_size=10,           # Smaller batches
    cache_ttl=1800           # 30 minute cache
)
```

### Caching Strategies

```python
from src.sentiment import SentimentCache

# Long-term cache for static content
static_cache = SentimentCache(ttl=86400)  # 24 hours

# Short-term cache for dynamic content  
dynamic_cache = SentimentCache(ttl=300)   # 5 minutes

# Use appropriate cache based on content type
def analyze_with_smart_caching(text, is_static=False):
    cache = static_cache if is_static else dynamic_cache
    analyzer = SentimentAnalyzer(cache=cache)
    return analyzer.analyze(text)
```

### Performance Benchmarking

```python
import time
from src.sentiment import SentimentAnalyzer

def benchmark_analysis(texts, iterations=5):
    analyzer = SentimentAnalyzer()
    
    times = []
    for i in range(iterations):
        start_time = time.time()
        results = analyzer.analyze_batch(texts)
        end_time = time.time()
        
        times.append(end_time - start_time)
    
    avg_time = sum(times) / len(times)
    throughput = len(texts) / avg_time
    
    print(f"Average time: {avg_time:.3f}s")
    print(f"Throughput: {throughput:.1f} texts/sec")
    print(f"Per-text time: {avg_time/len(texts)*1000:.1f}ms")

# Benchmark with sample data
sample_texts = ["Sample text"] * 100
benchmark_analysis(sample_texts)
```

## 📊 Examples

### Customer Feedback Analysis

```python
def analyze_customer_feedback():
    from src.sentiment import SentimentAnalyzer
    
    # Sample customer feedback
    feedback = [
        "Excellent customer service, very helpful!",
        "Product arrived damaged, disappointed.",
        "Average quality, nothing special.",
        "Amazing experience, will buy again!",
        "Terrible shipping, took forever to arrive."
    ]
    
    analyzer = SentimentAnalyzer()
    results = analyzer.analyze_batch(feedback)
    
    # Analyze results
    sentiment_counts = {}
    total_confidence = 0
    
    print("🔍 Customer Feedback Analysis")
    print("=" * 50)
    
    for i, result in enumerate(results):
        print(f"\nFeedback {i+1}: {result.text}")
        print(f"Sentiment: {result.label.value.upper()}")
        print(f"Confidence: {result.confidence:.1%}")
        print(f"Compound Score: {result.scores.compound:.3f}")
        
        # Count sentiments
        label = result.label.value
        sentiment_counts[label] = sentiment_counts.get(label, 0) + 1
        total_confidence += result.confidence
    
    # Summary statistics
    print(f"\n📊 Summary:")
    print(f"Total feedback: {len(results)}")
    print(f"Average confidence: {total_confidence/len(results):.1%}")
    print(f"Sentiment distribution:")
    
    for sentiment, count in sentiment_counts.items():
        percentage = count / len(results) * 100
        print(f"  {sentiment.capitalize()}: {count} ({percentage:.1f}%)")

if __name__ == "__main__":
    analyze_customer_feedback()
```

### Social Media Monitoring

```python
async def monitor_social_mentions():
    from src.sentiment import AsyncSentimentAnalyzer
    import json
    
    # Simulated social media data
    mentions = [
        {"id": 1, "text": "Love the new update! 🚀", "platform": "twitter"},
        {"id": 2, "text": "App keeps crashing, very frustrated", "platform": "facebook"},
        {"id": 3, "text": "Good features but UI needs work", "platform": "reddit"},
        {"id": 4, "text": "Best app ever! Highly recommend!", "platform": "instagram"},
        {"id": 5, "text": "Meh, it's okay I guess", "platform": "twitter"},
    ]
    
    analyzer = AsyncSentimentAnalyzer()
    
    try:
        # Extract texts for analysis
        texts = [mention["text"] for mention in mentions]
        
        # Batch analyze
        results = await analyzer.analyze_batch_async(texts)
        
        # Combine results with original data
        analyzed_mentions = []
        for mention, result in zip(mentions, results):
            analyzed_mentions.append({
                **mention,
                "sentiment": {
                    "label": result.label.value,
                    "confidence": result.confidence,
                    "scores": {
                        "positive": result.scores.positive,
                        "negative": result.scores.negative,
                        "neutral": result.scores.neutral,
                        "compound": result.scores.compound
                    }
                }
            })
        
        # Print analysis results
        print("📱 Social Media Sentiment Analysis")
        print("=" * 50)
        
        platform_sentiment = {}
        
        for mention in analyzed_mentions:
            platform = mention["platform"]
            sentiment = mention["sentiment"]
            
            print(f"\n{mention['platform'].upper()}: {mention['text']}")
            print(f"Sentiment: {sentiment['label'].upper()} ({sentiment['confidence']:.1%})")
            
            # Track platform sentiment
            if platform not in platform_sentiment:
                platform_sentiment[platform] = []
            platform_sentiment[platform].append(sentiment["compound"])
        
        # Platform summary
        print(f"\n📊 Platform Sentiment Summary:")
        for platform, scores in platform_sentiment.items():
            avg_score = sum(scores) / len(scores)
            print(f"{platform.capitalize()}: {avg_score:.3f} (avg)")
        
        # Save results
        with open("social_sentiment_analysis.json", "w") as f:
            json.dump(analyzed_mentions, f, indent=2)
        
        print(f"\n💾 Results saved to social_sentiment_analysis.json")
        
    finally:
        analyzer.cleanup()

# Run the analysis
import asyncio
asyncio.run(monitor_social_mentions())
```

### Product Review Classification

```python
def classify_product_reviews():
    from src.sentiment import SentimentAnalyzer
    import statistics
    
    # Sample product reviews
    reviews = {
        "iPhone 15": [
            "Amazing camera quality and battery life!",
            "Expensive but worth every penny",
            "Screen is gorgeous, very satisfied",
            "Some apps crash occasionally",
            "Best iPhone yet, highly recommend"
        ],
        "Samsung Galaxy": [
            "Good value for money",
            "Battery drains too quickly", 
            "Nice display but camera is average",
            "Solid performance, no complaints",
            "Better than expected, impressed"
        ],
        "Google Pixel": [
            "Pure Android experience is fantastic",
            "Camera is absolutely incredible",
            "Some build quality issues",
            "Software updates are timely",
            "Great for photography enthusiasts"
        ]
    }
    
    analyzer = SentimentAnalyzer()
    
    product_analysis = {}
    
    print("🛍️ Product Review Classification")
    print("=" * 50)
    
    for product, product_reviews in reviews.items():
        # Analyze all reviews for this product
        results = analyzer.analyze_batch(product_reviews)
        
        # Calculate metrics
        sentiments = [r.label.value for r in results]
        confidences = [r.confidence for r in results]
        compounds = [r.scores.compound for r in results]
        
        positive_count = sentiments.count('positive')
        negative_count = sentiments.count('negative') 
        neutral_count = sentiments.count('neutral')
        
        avg_confidence = statistics.mean(confidences)
        avg_compound = statistics.mean(compounds)
        
        # Determine overall product sentiment
        if avg_compound > 0.1:
            overall_sentiment = "POSITIVE"
        elif avg_compound < -0.1:
            overall_sentiment = "NEGATIVE"
        else:
            overall_sentiment = "NEUTRAL"
        
        product_analysis[product] = {
            'total_reviews': len(results),
            'sentiment_distribution': {
                'positive': positive_count,
                'negative': negative_count, 
                'neutral': neutral_count
            },
            'avg_confidence': avg_confidence,
            'avg_compound_score': avg_compound,
            'overall_sentiment': overall_sentiment
        }
        
        print(f"\n📱 {product}")
        print(f"Reviews analyzed: {len(results)}")
        print(f"Overall sentiment: {overall_sentiment}")
        print(f"Average confidence: {avg_confidence:.1%}")
        print(f"Sentiment breakdown:")
        print(f"  ✅ Positive: {positive_count} ({positive_count/len(results)*100:.1f}%)")
        print(f"  ❌ Negative: {negative_count} ({negative_count/len(results)*100:.1f}%)")
        print(f"  ⚪ Neutral: {neutral_count} ({neutral_count/len(results)*100:.1f}%)")
        print(f"Compound score: {avg_compound:.3f}")
    
    # Overall comparison
    print(f"\n🏆 Product Comparison:")
    sorted_products = sorted(
        product_analysis.items(), 
        key=lambda x: x[1]['avg_compound_score'], 
        reverse=True
    )
    
    for i, (product, analysis) in enumerate(sorted_products, 1):
        print(f"{i}. {product}: {analysis['avg_compound_score']:.3f}")

if __name__ == "__main__":
    classify_product_reviews()
```

## 🎯 Best Practices

### 1. Input Preprocessing

```python
def preprocess_text(text):
    """Clean and prepare text for sentiment analysis"""
    import re
    
    if not text:
        return ""
    
    # Remove URLs
    text = re.sub(r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+', '', text)
    
    # Remove excessive whitespace
    text = re.sub(r'\s+', ' ', text.strip())
    
    # Handle encoding issues
    text = text.encode('utf-8', errors='ignore').decode('utf-8')
    
    return text

# Usage
analyzer = SentimentAnalyzer()
clean_text = preprocess_text("Check out this link: https://example.com   Amazing product!!!")
result = analyzer.analyze(clean_text)
```

### 2. Batch Size Optimization

```python
def optimal_batch_size(total_texts, available_memory_gb=4):
    """Calculate optimal batch size based on available memory"""
    # Rough estimate: 1MB per 1000 texts
    max_batch_size = int(available_memory_gb * 1000)
    
    if total_texts <= max_batch_size:
        return total_texts
    
    # Use smaller batches for very large datasets
    if total_texts > 100000:
        return min(1000, max_batch_size)
    
    return min(max_batch_size, total_texts // 10)

# Usage
texts = load_large_dataset()  # 50,000 texts
batch_size = optimal_batch_size(len(texts))
analyzer = AsyncSentimentAnalyzer(batch_size=batch_size)
```

### 3. Error Handling

```python
def robust_sentiment_analysis(texts):
    """Perform sentiment analysis with comprehensive error handling"""
    from src.sentiment import SentimentAnalyzer, SentimentAnalysisError
    import logging
    
    analyzer = SentimentAnalyzer()
    results = []
    failed_analyses = []
    
    for i, text in enumerate(texts):
        try:
            result = analyzer.analyze(text)
            results.append(result)
        except SentimentAnalysisError as e:
            logging.error(f"Sentiment analysis failed for text {i}: {e}")
            failed_analyses.append({'index': i, 'text': text, 'error': str(e)})
            
            # Add neutral result as fallback
            from src.sentiment import SentimentResult, SentimentScore, SentimentLabel
            fallback_result = SentimentResult(
                text=text,
                scores=SentimentScore(0.0, 0.0, 1.0, 0.0),
                label=SentimentLabel.NEUTRAL,
                confidence=0.0,
                metadata={'error': str(e)}
            )
            results.append(fallback_result)
        
        except Exception as e:
            logging.error(f"Unexpected error for text {i}: {e}")
            failed_analyses.append({'index': i, 'text': text, 'error': str(e)})
            # Continue processing other texts
    
    return results, failed_analyses
```

### 4. Monitoring and Alerting

```python
def setup_sentiment_monitoring():
    """Set up monitoring for sentiment analysis system"""
    from src.sentiment import performance_monitor
    import time
    
    # Monitor key metrics
    def check_performance_metrics():
        stats = performance_monitor.get_performance_summary()
        
        # Alert conditions
        if stats['overall_throughput_per_sec'] < 10:
            print("⚠️  ALERT: Low throughput detected")
        
        if stats['error_rate_percent'] > 5:
            print("⚠️  ALERT: High error rate detected")
        
        if stats['average_memory_mb'] > 1000:
            print("⚠️  ALERT: High memory usage detected")
        
        return stats
    
    # Periodic monitoring
    def monitor_loop():
        while True:
            try:
                stats = check_performance_metrics()
                print(f"📊 Throughput: {stats['overall_throughput_per_sec']:.1f} texts/sec")
                time.sleep(60)  # Check every minute
            except KeyboardInterrupt:
                break
    
    return monitor_loop

# Usage
# monitor = setup_sentiment_monitoring()
# monitor()  # Run in background thread
```

### 5. Data Pipeline Integration

```python
def create_sentiment_pipeline():
    """Create a data processing pipeline with sentiment analysis"""
    from src.sentiment import AsyncSentimentAnalyzer
    import asyncio
    
    async def process_data_stream(data_stream):
        analyzer = AsyncSentimentAnalyzer()
        
        try:
            batch = []
            batch_size = 100
            
            async for data_item in data_stream:
                # Extract text from data item
                text = data_item.get('content', '')
                
                if text:
                    batch.append((data_item['id'], text))
                
                # Process batch when full
                if len(batch) >= batch_size:
                    await process_batch(analyzer, batch)
                    batch = []
            
            # Process remaining items
            if batch:
                await process_batch(analyzer, batch)
                
        finally:
            analyzer.cleanup()
    
    async def process_batch(analyzer, batch):
        texts = [item[1] for item in batch]
        ids = [item[0] for item in batch]
        
        results = await analyzer.analyze_batch_async(texts)
        
        # Store results
        for id_, result in zip(ids, results):
            await store_sentiment_result(id_, result)
    
    async def store_sentiment_result(id_, result):
        # Store in database, cache, or message queue
        print(f"Storing result for {id_}: {result.label.value}")
    
    return process_data_stream
```

---

## 🔗 Additional Resources

- **API Documentation**: Complete REST API reference
- **Performance Guide**: Detailed performance tuning recommendations  
- **Deployment Guide**: Production deployment instructions
- **Contributing Guide**: How to contribute to the project
- **Examples Repository**: Additional code examples and use cases

---

**💡 Need help?** Check the troubleshooting section or open an issue on GitHub!