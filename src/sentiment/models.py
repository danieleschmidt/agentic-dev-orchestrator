"""
Data models for sentiment analysis
"""
from enum import Enum
from typing import Dict, List, Optional
from dataclasses import dataclass


class SentimentLabel(str, Enum):
    """Sentiment classification labels"""
    POSITIVE = "positive"
    NEGATIVE = "negative" 
    NEUTRAL = "neutral"


@dataclass
class SentimentScore:
    """Sentiment analysis scores"""
    positive: float
    negative: float
    neutral: float
    compound: float
    
    @property
    def dominant_sentiment(self) -> SentimentLabel:
        """Get the dominant sentiment based on scores"""
        if self.compound >= 0.05:
            return SentimentLabel.POSITIVE
        elif self.compound <= -0.05:
            return SentimentLabel.NEGATIVE
        else:
            return SentimentLabel.NEUTRAL


@dataclass
class SentimentResult:
    """Complete sentiment analysis result"""
    text: str
    scores: SentimentScore
    label: SentimentLabel
    confidence: float
    metadata: Optional[Dict] = None
    
    def to_dict(self) -> Dict:
        """Convert to dictionary for JSON serialization"""
        return {
            "text": self.text,
            "scores": {
                "positive": self.scores.positive,
                "negative": self.scores.negative,
                "neutral": self.scores.neutral,
                "compound": self.scores.compound
            },
            "label": self.label.value,
            "confidence": self.confidence,
            "metadata": self.metadata or {}
        }