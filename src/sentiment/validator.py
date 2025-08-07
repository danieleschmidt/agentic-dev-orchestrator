"""
Input validation and sanitization for sentiment analysis
"""
import re
import logging
from typing import Optional, List
from .exceptions import ValidationError, InvalidInputError

logger = logging.getLogger(__name__)


class InputValidator:
    """Validates and sanitizes input for sentiment analysis"""
    
    # Security patterns to detect potentially malicious input
    SUSPICIOUS_PATTERNS = [
        r'<script.*?>.*?</script>',  # Script tags
        r'javascript:',             # JavaScript URIs
        r'on\w+\s*=',              # Event handlers
        r'eval\s*\(',              # eval() calls
        r'import\s+\w+',           # Import statements
        r'__\w+__',                # Python dunder methods
        r'exec\s*\(',              # exec() calls
        r'system\s*\(',            # system() calls
    ]
    
    def __init__(self, max_length: int = 10000, min_length: int = 1):
        """
        Initialize validator
        
        Args:
            max_length: Maximum allowed text length
            min_length: Minimum allowed text length
        """
        self.max_length = max_length
        self.min_length = min_length
    
    def validate_text(self, text: str) -> str:
        """
        Validate and sanitize input text
        
        Args:
            text: Input text to validate
            
        Returns:
            Sanitized text
            
        Raises:
            InvalidInputError: If input is invalid
            ValidationError: If validation fails
        """
        if text is None:
            raise InvalidInputError("Text cannot be None")
        
        if not isinstance(text, str):
            try:
                text = str(text)
            except Exception as e:
                raise InvalidInputError(f"Cannot convert input to string: {e}")
        
        # Check length constraints
        if len(text) < self.min_length:
            raise ValidationError(f"Text too short (minimum {self.min_length} characters)")
        
        if len(text) > self.max_length:
            raise ValidationError(f"Text too long (maximum {self.max_length} characters)")
        
        # Security validation
        self._check_security(text)
        
        # Sanitize text
        sanitized = self._sanitize_text(text)
        
        logger.debug(f"Validated text: {len(sanitized)} characters")
        return sanitized
    
    def validate_batch(self, texts: List[str]) -> List[str]:
        """
        Validate batch of texts
        
        Args:
            texts: List of texts to validate
            
        Returns:
            List of sanitized texts
        """
        if not isinstance(texts, list):
            raise InvalidInputError("Texts must be a list")
        
        if len(texts) == 0:
            raise ValidationError("Empty text list provided")
        
        if len(texts) > 100:  # Prevent batch abuse
            raise ValidationError("Too many texts in batch (maximum 100)")
        
        validated_texts = []
        for i, text in enumerate(texts):
            try:
                validated = self.validate_text(text)
                validated_texts.append(validated)
            except Exception as e:
                raise ValidationError(f"Invalid text at index {i}: {e}")
        
        return validated_texts
    
    def _check_security(self, text: str) -> None:
        """Check for suspicious patterns that might indicate malicious input"""
        text_lower = text.lower()
        
        for pattern in self.SUSPICIOUS_PATTERNS:
            if re.search(pattern, text_lower, re.IGNORECASE | re.DOTALL):
                logger.warning(f"Suspicious pattern detected: {pattern}")
                # Don't raise exception - just log for monitoring
                # In production, you might want to implement rate limiting
                break
    
    def _sanitize_text(self, text: str) -> str:
        """
        Sanitize input text
        
        Args:
            text: Raw input text
            
        Returns:
            Sanitized text
        """
        # Remove null bytes and control characters (except common whitespace)
        sanitized = ''.join(char for char in text 
                          if ord(char) >= 32 or char in '\t\n\r')
        
        # Remove excessive whitespace
        sanitized = re.sub(r'\s+', ' ', sanitized)
        
        # Trim whitespace
        sanitized = sanitized.strip()
        
        return sanitized
    
    def validate_metadata(self, metadata: Optional[dict]) -> dict:
        """
        Validate metadata dictionary
        
        Args:
            metadata: Optional metadata dictionary
            
        Returns:
            Validated metadata dictionary
        """
        if metadata is None:
            return {}
        
        if not isinstance(metadata, dict):
            raise ValidationError("Metadata must be a dictionary")
        
        # Limit metadata size and depth
        if len(str(metadata)) > 1000:
            raise ValidationError("Metadata too large")
        
        # Sanitize metadata keys and values
        sanitized_metadata = {}
        for key, value in metadata.items():
            if not isinstance(key, str):
                key = str(key)
            
            # Limit key length
            if len(key) > 100:
                key = key[:100]
            
            # Sanitize value
            if isinstance(value, str):
                if len(value) > 500:
                    value = value[:500]
                value = self._sanitize_text(value)
            elif isinstance(value, (int, float, bool)):
                pass  # These are safe
            else:
                value = str(value)[:500]  # Convert and limit
            
            sanitized_metadata[key] = value
        
        return sanitized_metadata