"""
Core sentiment analysis implementation
"""
import re
import math
import logging
from typing import List, Dict, Optional
from .models import SentimentResult, SentimentScore, SentimentLabel
from .validator import InputValidator
from .exceptions import SentimentAnalysisError, ProcessingError, ModelError

logger = logging.getLogger(__name__)


class SentimentAnalyzer:
    """
    Simple rule-based sentiment analyzer with lexicon approach
    """
    
    def __init__(self, validator: Optional[InputValidator] = None):
        """Initialize analyzer with sentiment lexicons"""
        self.validator = validator or InputValidator()
        logger.info("SentimentAnalyzer initialized")
        # Basic sentiment lexicons
        self.positive_words = {
            'good', 'great', 'excellent', 'amazing', 'awesome', 'fantastic', 
            'wonderful', 'perfect', 'love', 'like', 'enjoy', 'happy', 'pleased',
            'satisfied', 'brilliant', 'outstanding', 'superb', 'magnificent',
            'impressive', 'remarkable', 'exceptional', 'terrific', 'marvelous',
            'fabulous', 'delightful', 'splendid', 'incredible', 'phenomenal'
        }
        
        self.negative_words = {
            'bad', 'terrible', 'awful', 'horrible', 'hate', 'dislike', 'disgusting',
            'disappointing', 'frustrated', 'angry', 'upset', 'annoying', 'irritating',
            'pathetic', 'useless', 'worthless', 'dreadful', 'appalling', 'atrocious',
            'abysmal', 'deplorable', 'ghastly', 'hideous', 'revolting', 'repulsive',
            'detestable', 'loathsome', 'despicable', 'contemptible'
        }
        
        # Intensifiers and negations
        self.intensifiers = {
            'very': 1.5, 'extremely': 2.0, 'incredibly': 1.8, 'absolutely': 1.7,
            'completely': 1.6, 'totally': 1.5, 'really': 1.3, 'quite': 1.2,
            'rather': 1.1, 'somewhat': 0.8, 'slightly': 0.7
        }
        
        self.negations = {
            'not', 'no', 'never', 'neither', 'nothing', 'nowhere', 'nobody',
            'none', 'cannot', 'cant', 'couldnt', 'shouldnt', 'wouldnt',
            'dont', 'doesnt', 'didnt', 'isnt', 'arent', 'wasnt', 'werent'
        }

    def preprocess_text(self, text: str) -> List[str]:
        """Clean and tokenize text"""
        try:
            # Convert to lowercase and remove special characters
            text = re.sub(r'[^a-zA-Z0-9\s]', '', text.lower())
            
            # Simple tokenization
            words = text.split()
            
            logger.debug(f"Preprocessed text: {len(words)} words")
            return words
            
        except Exception as e:
            logger.error(f"Error preprocessing text: {e}")
            raise ProcessingError(f"Failed to preprocess text: {e}") from e

    def calculate_sentiment_scores(self, words: List[str]) -> SentimentScore:
        """Calculate sentiment scores from word list"""
        try:
            positive_score = 0.0
            negative_score = 0.0
            word_count = len(words)
            
            if word_count == 0:
                logger.debug("Empty word list, returning neutral sentiment")
                return SentimentScore(0.0, 0.0, 1.0, 0.0)
            
            i = 0
            while i < len(words):
                word = words[i]
                
                # Check for intensifiers
                intensity = 1.0
                if i > 0 and words[i-1] in self.intensifiers:
                    intensity = self.intensifiers[words[i-1]]
                
                # Check for negations (flip sentiment in next 2-3 words)
                negated = False
                if i > 0 and words[i-1] in self.negations:
                    negated = True
                elif i > 1 and words[i-2] in self.negations:
                    negated = True
                
                # Calculate sentiment contribution
                if word in self.positive_words:
                    score = intensity
                    if negated:
                        negative_score += score
                    else:
                        positive_score += score
                        
                elif word in self.negative_words:
                    score = intensity
                    if negated:
                        positive_score += score
                    else:
                        negative_score += score
                
                i += 1
        
            # Normalize scores
            total_sentiment_words = positive_score + negative_score
            if total_sentiment_words > 0:
                positive_norm = positive_score / total_sentiment_words
                negative_norm = negative_score / total_sentiment_words
                neutral_norm = max(0, 1 - positive_norm - negative_norm)
            else:
                positive_norm = 0.0
                negative_norm = 0.0
                neutral_norm = 1.0
            
            # Calculate compound score (similar to VADER)
            compound = (positive_score - negative_score) / (word_count + 1)
            compound = max(-1.0, min(1.0, compound))  # Clamp to [-1, 1]
            
            logger.debug(f"Sentiment scores calculated: pos={positive_norm:.3f}, neg={negative_norm:.3f}, neu={neutral_norm:.3f}, compound={compound:.3f}")
            
            return SentimentScore(
                positive=positive_norm,
                negative=negative_norm,
                neutral=neutral_norm,
                compound=compound
            )
            
        except Exception as e:
            logger.error(f"Error calculating sentiment scores: {e}")
            raise ModelError(f"Failed to calculate sentiment scores: {e}") from e

    def analyze(self, text: str, metadata: Optional[Dict] = None) -> SentimentResult:
        """
        Analyze sentiment of given text
        
        Args:
            text: Input text to analyze
            metadata: Optional metadata to include in result
            
        Returns:
            SentimentResult with scores and classification
            
        Raises:
            SentimentAnalysisError: If analysis fails
        """
        try:
            # Validate and sanitize input
            validated_text = self.validator.validate_text(text) if text else ""
            validated_metadata = self.validator.validate_metadata(metadata)
            
            logger.info(f"Analyzing sentiment for text of length {len(validated_text)}")
            
            if not validated_text or not validated_text.strip():
                logger.debug("Empty text provided, returning neutral sentiment")
                return SentimentResult(
                    text=validated_text,
                    scores=SentimentScore(0.0, 0.0, 1.0, 0.0),
                    label=SentimentLabel.NEUTRAL,
                    confidence=1.0,
                    metadata=validated_metadata
                )
            
            # Preprocess text
            words = self.preprocess_text(validated_text)
            
            # Calculate scores
            scores = self.calculate_sentiment_scores(words)
            
            # Determine label and confidence
            label = scores.dominant_sentiment
            
            # Calculate confidence based on how decisive the compound score is
            confidence = min(1.0, abs(scores.compound) + 0.3)
            
            result = SentimentResult(
                text=validated_text,
                scores=scores,
                label=label,
                confidence=confidence,
                metadata=validated_metadata
            )
            
            logger.info(f"Sentiment analysis completed: {label.value} (confidence: {confidence:.2%})")
            return result
            
        except Exception as e:
            logger.error(f"Sentiment analysis failed: {e}")
            if isinstance(e, SentimentAnalysisError):
                raise
            else:
                raise SentimentAnalysisError(f"Analysis failed: {e}") from e

    def analyze_batch(self, texts: List[str]) -> List[SentimentResult]:
        """
        Analyze multiple texts efficiently
        
        Args:
            texts: List of texts to analyze
            
        Returns:
            List of SentimentResult objects
            
        Raises:
            SentimentAnalysisError: If batch analysis fails
        """
        try:
            # Validate batch input
            validated_texts = self.validator.validate_batch(texts)
            
            logger.info(f"Starting batch sentiment analysis for {len(validated_texts)} texts")
            
            results = []
            failed_count = 0
            
            for i, text in enumerate(validated_texts):
                try:
                    result = self.analyze(text, {'batch_index': i})
                    results.append(result)
                except Exception as e:
                    failed_count += 1
                    logger.warning(f"Failed to analyze text at index {i}: {e}")
                    # Return neutral result for failed analysis
                    results.append(SentimentResult(
                        text=text,
                        scores=SentimentScore(0.0, 0.0, 1.0, 0.0),
                        label=SentimentLabel.NEUTRAL,
                        confidence=0.0,
                        metadata={'batch_index': i, 'error': str(e)}
                    ))
            
            if failed_count > 0:
                logger.warning(f"Batch analysis completed with {failed_count} failures out of {len(texts)} texts")
            else:
                logger.info(f"Batch analysis completed successfully for {len(texts)} texts")
            
            return results
            
        except Exception as e:
            logger.error(f"Batch sentiment analysis failed: {e}")
            raise SentimentAnalysisError(f"Batch analysis failed: {e}") from e