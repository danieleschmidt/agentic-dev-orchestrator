"""
Custom exceptions for sentiment analysis module
"""


class SentimentAnalysisError(Exception):
    """Base exception for sentiment analysis errors"""
    pass


class InvalidInputError(SentimentAnalysisError):
    """Raised when input text is invalid"""
    pass


class ValidationError(SentimentAnalysisError):
    """Raised when validation fails"""
    pass


class ProcessingError(SentimentAnalysisError):
    """Raised when text processing fails"""
    pass


class ModelError(SentimentAnalysisError):
    """Raised when there's an error with the sentiment model"""
    pass