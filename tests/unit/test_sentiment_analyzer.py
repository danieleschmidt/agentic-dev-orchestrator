"""
Unit tests for sentiment analysis module
"""
import pytest
from src.sentiment import (
    SentimentAnalyzer, SentimentResult, SentimentScore, SentimentLabel,
    SentimentAnalysisError, ValidationError, InputValidator
)


class TestSentimentAnalyzer:
    """Test suite for SentimentAnalyzer"""
    
    def setup_method(self):
        """Setup test fixtures"""
        self.analyzer = SentimentAnalyzer()
    
    def test_analyzer_initialization(self):
        """Test analyzer initialization"""
        assert isinstance(self.analyzer, SentimentAnalyzer)
        assert hasattr(self.analyzer, 'positive_words')
        assert hasattr(self.analyzer, 'negative_words')
        assert len(self.analyzer.positive_words) > 0
        assert len(self.analyzer.negative_words) > 0
    
    def test_positive_sentiment(self):
        """Test positive sentiment detection"""
        result = self.analyzer.analyze("This is absolutely amazing and wonderful!")
        
        assert isinstance(result, SentimentResult)
        assert result.label == SentimentLabel.POSITIVE
        assert result.scores.positive > result.scores.negative
        assert result.scores.compound > 0
        assert result.confidence > 0.5
    
    def test_negative_sentiment(self):
        """Test negative sentiment detection"""
        result = self.analyzer.analyze("This is terrible and awful!")
        
        assert result.label == SentimentLabel.NEGATIVE
        assert result.scores.negative > result.scores.positive
        assert result.scores.compound < 0
        assert result.confidence > 0.5
    
    def test_neutral_sentiment(self):
        """Test neutral sentiment detection"""
        result = self.analyzer.analyze("The sky is blue today.")
        
        assert result.label == SentimentLabel.NEUTRAL
        assert abs(result.scores.compound) < 0.05
    
    def test_empty_text(self):
        """Test empty text handling"""
        result = self.analyzer.analyze("")
        
        assert result.label == SentimentLabel.NEUTRAL
        assert result.scores.neutral == 1.0
        assert result.confidence == 1.0
    
    def test_none_text(self):
        """Test None text handling"""
        result = self.analyzer.analyze(None)
        
        assert result.label == SentimentLabel.NEUTRAL
        assert result.scores.neutral == 1.0
    
    def test_intensifiers(self):
        """Test sentiment intensification"""
        basic_result = self.analyzer.analyze("good")
        intense_result = self.analyzer.analyze("very good")
        
        assert intense_result.scores.compound > basic_result.scores.compound
    
    def test_negation(self):
        """Test negation handling"""
        positive_result = self.analyzer.analyze("good")
        negated_result = self.analyzer.analyze("not good")
        
        assert positive_result.scores.compound > 0
        assert negated_result.scores.compound < positive_result.scores.compound
    
    def test_metadata_inclusion(self):
        """Test metadata inclusion in results"""
        metadata = {'source': 'test', 'user_id': 123}
        result = self.analyzer.analyze("test text", metadata)
        
        assert result.metadata == metadata
    
    def test_batch_analysis(self):
        """Test batch text analysis"""
        texts = [
            "This is great!",
            "This is terrible!",
            "This is okay."
        ]
        
        results = self.analyzer.analyze_batch(texts)
        
        assert len(results) == 3
        assert all(isinstance(r, SentimentResult) for r in results)
        assert results[0].label == SentimentLabel.POSITIVE
        assert results[1].label == SentimentLabel.NEGATIVE
    
    def test_preprocess_text(self):
        """Test text preprocessing"""
        text = "Hello, WORLD! 123 #special"
        words = self.analyzer.preprocess_text(text)
        
        assert isinstance(words, list)
        assert all(isinstance(word, str) for word in words)
        assert 'hello' in words
        assert 'world' in words
    
    def test_sentiment_scores_calculation(self):
        """Test sentiment scores calculation"""
        words = ['good', 'great', 'excellent']
        scores = self.analyzer.calculate_sentiment_scores(words)
        
        assert isinstance(scores, SentimentScore)
        assert scores.positive > 0
        assert scores.compound > 0
        assert abs(scores.positive + scores.negative + scores.neutral - 1.0) < 0.01


class TestInputValidator:
    """Test suite for InputValidator"""
    
    def setup_method(self):
        """Setup test fixtures"""
        self.validator = InputValidator()
    
    def test_valid_text(self):
        """Test validation of valid text"""
        text = "This is a valid text."
        result = self.validator.validate_text(text)
        
        assert result == text.strip()
    
    def test_none_text(self):
        """Test None text validation"""
        with pytest.raises(ValidationError):
            self.validator.validate_text(None)
    
    def test_empty_text(self):
        """Test empty text validation"""
        with pytest.raises(ValidationError):
            self.validator.validate_text("")
    
    def test_text_too_long(self):
        """Test text length validation"""
        long_text = "x" * 20000
        
        with pytest.raises(ValidationError):
            self.validator.validate_text(long_text)
    
    def test_text_sanitization(self):
        """Test text sanitization"""
        dirty_text = "  Hello\n\nWorld  \t\t  "
        cleaned = self.validator.validate_text(dirty_text)
        
        assert cleaned == "Hello World"
    
    def test_batch_validation(self):
        """Test batch text validation"""
        texts = ["text1", "text2", "text3"]
        result = self.validator.validate_batch(texts)
        
        assert len(result) == 3
        assert all(isinstance(text, str) for text in result)
    
    def test_empty_batch(self):
        """Test empty batch validation"""
        with pytest.raises(ValidationError):
            self.validator.validate_batch([])
    
    def test_large_batch(self):
        """Test large batch validation"""
        large_batch = ["text"] * 150
        
        with pytest.raises(ValidationError):
            self.validator.validate_batch(large_batch)
    
    def test_metadata_validation(self):
        """Test metadata validation"""
        metadata = {'key': 'value', 'number': 42}
        result = self.validator.validate_metadata(metadata)
        
        assert isinstance(result, dict)
        assert 'key' in result
        assert 'number' in result
    
    def test_none_metadata(self):
        """Test None metadata validation"""
        result = self.validator.validate_metadata(None)
        
        assert result == {}
    
    def test_metadata_sanitization(self):
        """Test metadata sanitization"""
        metadata = {
            'key' * 50: 'value' * 200,  # Long key and value
            'normal_key': 'normal_value'
        }
        
        result = self.validator.validate_metadata(metadata)
        
        assert len(list(result.keys())[0]) <= 100  # Key truncated
        assert len(list(result.values())[0]) <= 500  # Value truncated


class TestSentimentModels:
    """Test suite for sentiment models"""
    
    def test_sentiment_score_dominant(self):
        """Test dominant sentiment calculation"""
        # Positive dominant
        score = SentimentScore(0.7, 0.2, 0.1, 0.5)
        assert score.dominant_sentiment == SentimentLabel.POSITIVE
        
        # Negative dominant
        score = SentimentScore(0.1, 0.7, 0.2, -0.6)
        assert score.dominant_sentiment == SentimentLabel.NEGATIVE
        
        # Neutral
        score = SentimentScore(0.3, 0.3, 0.4, 0.0)
        assert score.dominant_sentiment == SentimentLabel.NEUTRAL
    
    def test_sentiment_result_to_dict(self):
        """Test SentimentResult to_dict conversion"""
        score = SentimentScore(0.6, 0.2, 0.2, 0.4)
        result = SentimentResult(
            text="test",
            scores=score,
            label=SentimentLabel.POSITIVE,
            confidence=0.8,
            metadata={'test': True}
        )
        
        data = result.to_dict()
        
        assert isinstance(data, dict)
        assert data['text'] == 'test'
        assert data['label'] == 'positive'
        assert data['confidence'] == 0.8
        assert 'scores' in data
        assert data['metadata']['test'] is True


@pytest.fixture
def sample_texts():
    """Fixture providing sample texts for testing"""
    return [
        ("I love this product! It's amazing!", SentimentLabel.POSITIVE),
        ("This is the worst thing ever. Hate it!", SentimentLabel.NEGATIVE),
        ("The weather is cloudy today.", SentimentLabel.NEUTRAL),
        ("", SentimentLabel.NEUTRAL),
        ("Great job! Well done! Excellent work!", SentimentLabel.POSITIVE),
        ("Terrible service. Very disappointed. Bad experience.", SentimentLabel.NEGATIVE),
    ]


class TestSentimentIntegration:
    """Integration tests for sentiment analysis"""
    
    def test_end_to_end_analysis(self, sample_texts):
        """Test end-to-end sentiment analysis"""
        analyzer = SentimentAnalyzer()
        
        for text, expected_label in sample_texts:
            result = analyzer.analyze(text)
            
            # Check that result is properly formatted
            assert isinstance(result, SentimentResult)
            assert isinstance(result.scores, SentimentScore)
            assert isinstance(result.label, SentimentLabel)
            assert 0 <= result.confidence <= 1
            
            # For non-empty texts, check expected sentiment
            if text.strip():
                if expected_label != SentimentLabel.NEUTRAL:
                    assert result.label == expected_label, f"Expected {expected_label}, got {result.label} for '{text}'"
    
    def test_batch_consistency(self, sample_texts):
        """Test that batch and individual analysis are consistent"""
        analyzer = SentimentAnalyzer()
        
        texts = [text for text, _ in sample_texts]
        
        # Individual analysis
        individual_results = [analyzer.analyze(text) for text in texts]
        
        # Batch analysis
        batch_results = analyzer.analyze_batch(texts)
        
        assert len(individual_results) == len(batch_results)
        
        # Results should be similar (allowing for small floating point differences)
        for ind, batch in zip(individual_results, batch_results):
            assert ind.label == batch.label
            assert abs(ind.confidence - batch.confidence) < 0.01
            assert abs(ind.scores.compound - batch.scores.compound) < 0.01