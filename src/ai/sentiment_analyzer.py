#!/usr/bin/env python3
"""
Advanced Sentiment Analysis Engine for ADO
Integrates with backlog analysis and team dynamics monitoring
"""

import json
import logging
import re
from typing import Dict, List, Optional, Tuple, Any
from datetime import datetime
from pathlib import Path
from dataclasses import dataclass, asdict
from enum import Enum


class SentimentLabel(Enum):
    """Sentiment classification labels"""
    VERY_POSITIVE = "very_positive"
    POSITIVE = "positive" 
    NEUTRAL = "neutral"
    NEGATIVE = "negative"
    VERY_NEGATIVE = "very_negative"


@dataclass
class SentimentResult:
    """Sentiment analysis result with confidence scores"""
    label: SentimentLabel
    confidence: float
    positive_score: float
    negative_score: float
    neutral_score: float
    keywords: List[str]
    emotions: Dict[str, float]
    timestamp: str
    text_length: int
    

class SentimentAnalyzer:
    """Advanced sentiment analysis with team dynamics insights"""
    
    def __init__(self):
        self.logger = logging.getLogger("sentiment_analyzer")
        self._load_lexicons()
        
    def _load_lexicons(self):
        """Load sentiment lexicons and patterns"""
        # Positive sentiment indicators
        self.positive_patterns = {
            'achievement': ['completed', 'finished', 'done', 'success', 'accomplished', 'resolved'],
            'enthusiasm': ['excited', 'love', 'awesome', 'fantastic', 'great', 'excellent'],
            'collaboration': ['team', 'together', 'collaborate', 'shared', 'unified', 'synergy'],
            'progress': ['improved', 'better', 'enhanced', 'optimized', 'upgraded', 'advanced']
        }
        
        # Negative sentiment indicators  
        self.negative_patterns = {
            'frustration': ['frustrated', 'annoyed', 'blocked', 'stuck', 'difficult', 'hard'],
            'concern': ['worried', 'concerned', 'issue', 'problem', 'trouble', 'challenge'],
            'conflict': ['disagree', 'conflict', 'tension', 'argue', 'dispute', 'clash'],
            'delay': ['delayed', 'late', 'behind', 'slow', 'waiting', 'overdue']
        }
        
        # Neutral indicators
        self.neutral_patterns = ['update', 'status', 'report', 'meeting', 'review', 'discussion']
        
    def analyze_text(self, text: str, context: Optional[Dict] = None) -> SentimentResult:
        """Analyze sentiment of text with contextual awareness"""
        if not text or not text.strip():
            return self._create_neutral_result(text)
            
        text_clean = self._preprocess_text(text)
        
        # Calculate base sentiment scores
        scores = self._calculate_sentiment_scores(text_clean)
        
        # Apply contextual adjustments
        if context:
            scores = self._apply_context(scores, context)
            
        # Determine final label
        label = self._determine_label(scores)
        
        # Extract keywords and emotions
        keywords = self._extract_keywords(text_clean)
        emotions = self._analyze_emotions(text_clean)
        
        return SentimentResult(
            label=label,
            confidence=max(scores.values()),
            positive_score=scores['positive'],
            negative_score=scores['negative'],
            neutral_score=scores['neutral'],
            keywords=keywords,
            emotions=emotions,
            timestamp=datetime.now().isoformat(),
            text_length=len(text)
        )
        
    def _preprocess_text(self, text: str) -> str:
        """Clean and normalize text for analysis"""
        # Convert to lowercase
        text = text.lower()
        
        # Remove extra whitespace
        text = re.sub(r'\s+', ' ', text).strip()
        
        # Handle common abbreviations and informal language
        text = re.sub(r'\b(can\'t|cannot)\b', 'can not', text)
        text = re.sub(r'\b(won\'t|will not)\b', 'will not', text)
        text = re.sub(r'\b(don\'t|do not)\b', 'do not', text)
        
        return text
        
    def _calculate_sentiment_scores(self, text: str) -> Dict[str, float]:
        """Calculate sentiment scores using lexicon matching"""
        words = text.split()
        positive_score = 0.0
        negative_score = 0.0
        total_matches = 0
        
        # Check positive patterns
        for category, patterns in self.positive_patterns.items():
            for pattern in patterns:
                if pattern in text:
                    weight = 1.5 if category == 'achievement' else 1.0
                    positive_score += weight
                    total_matches += 1
                    
        # Check negative patterns
        for category, patterns in self.negative_patterns.items():
            for pattern in patterns:
                if pattern in text:
                    weight = 1.5 if category == 'frustration' else 1.0
                    negative_score += weight
                    total_matches += 1
                    
        # Normalize scores
        if total_matches > 0:
            positive_score = min(positive_score / total_matches, 1.0)
            negative_score = min(negative_score / total_matches, 1.0)
        else:
            # No clear sentiment indicators - check neutral patterns
            neutral_matches = sum(1 for pattern in self.neutral_patterns if pattern in text)
            if neutral_matches > 0:
                return {'positive': 0.2, 'negative': 0.2, 'neutral': 0.6}
                
        # Calculate neutral as inverse of positive + negative
        sentiment_total = positive_score + negative_score
        if sentiment_total > 0:
            neutral_score = max(0.0, 1.0 - sentiment_total)
        else:
            neutral_score = 0.8
            
        return {
            'positive': positive_score,
            'negative': negative_score, 
            'neutral': neutral_score
        }
        
    def _apply_context(self, scores: Dict[str, float], context: Dict) -> Dict[str, float]:
        """Apply contextual adjustments to sentiment scores"""
        # Adjust for backlog item priority
        if context.get('priority') == 'high':
            scores['negative'] *= 1.2  # High priority items may carry more stress
            
        # Adjust for item age
        age_days = context.get('age_days', 0)
        if age_days > 30:
            scores['negative'] *= 1.1  # Older items may be more frustrating
            
        # Adjust for team size
        team_size = context.get('team_size', 1)
        if team_size > 5:
            scores['positive'] *= 1.1  # Larger teams may have more positive dynamics
            
        return scores
        
    def _determine_label(self, scores: Dict[str, float]) -> SentimentLabel:
        """Determine sentiment label from scores"""
        pos_score = scores['positive']
        neg_score = scores['negative']
        neu_score = scores['neutral']
        
        if pos_score > neg_score and pos_score > neu_score:
            return SentimentLabel.VERY_POSITIVE if pos_score > 0.7 else SentimentLabel.POSITIVE
        elif neg_score > pos_score and neg_score > neu_score:
            return SentimentLabel.VERY_NEGATIVE if neg_score > 0.7 else SentimentLabel.NEGATIVE
        else:
            return SentimentLabel.NEUTRAL
            
    def _extract_keywords(self, text: str) -> List[str]:
        """Extract sentiment-relevant keywords"""
        keywords = []
        
        # Extract achievement keywords
        for category, patterns in self.positive_patterns.items():
            for pattern in patterns:
                if pattern in text:
                    keywords.append(f"{category}:{pattern}")
                    
        # Extract concern keywords
        for category, patterns in self.negative_patterns.items():
            for pattern in patterns:
                if pattern in text:
                    keywords.append(f"{category}:{pattern}")
                    
        return keywords[:10]  # Limit to top 10 keywords
        
    def _analyze_emotions(self, text: str) -> Dict[str, float]:
        """Analyze emotional content beyond basic sentiment"""
        emotions = {
            'joy': 0.0,
            'anger': 0.0,
            'fear': 0.0,
            'sadness': 0.0,
            'surprise': 0.0,
            'trust': 0.0
        }
        
        # Joy indicators
        joy_words = ['happy', 'excited', 'celebration', 'success', 'achieve', 'win']
        emotions['joy'] = sum(0.2 for word in joy_words if word in text)
        
        # Anger indicators
        anger_words = ['angry', 'frustrated', 'annoyed', 'furious', 'mad']
        emotions['anger'] = sum(0.3 for word in anger_words if word in text)
        
        # Fear indicators
        fear_words = ['worried', 'concerned', 'afraid', 'anxious', 'nervous']
        emotions['fear'] = sum(0.25 for word in fear_words if word in text)
        
        # Trust indicators
        trust_words = ['reliable', 'confident', 'trust', 'believe', 'faith']
        emotions['trust'] = sum(0.2 for word in trust_words if word in text)
        
        # Normalize emotions to 0-1 scale
        for emotion in emotions:
            emotions[emotion] = min(emotions[emotion], 1.0)
            
        return emotions
        
    def _create_neutral_result(self, text: str) -> SentimentResult:
        """Create neutral sentiment result for empty/invalid text"""
        return SentimentResult(
            label=SentimentLabel.NEUTRAL,
            confidence=0.5,
            positive_score=0.0,
            negative_score=0.0,
            neutral_score=1.0,
            keywords=[],
            emotions={},
            timestamp=datetime.now().isoformat(),
            text_length=len(text) if text else 0
        )
        
    def analyze_backlog_item(self, item_data: Dict) -> SentimentResult:
        """Analyze sentiment of a backlog item"""
        text_parts = []
        
        # Combine title and description
        if 'title' in item_data:
            text_parts.append(item_data['title'])
        if 'description' in item_data:
            text_parts.append(item_data['description'])
            
        # Add acceptance criteria
        if 'acceptance_criteria' in item_data:
            if isinstance(item_data['acceptance_criteria'], list):
                text_parts.extend(item_data['acceptance_criteria'])
            else:
                text_parts.append(str(item_data['acceptance_criteria']))
                
        combined_text = ' '.join(text_parts)
        
        # Create context from item metadata
        context = {
            'priority': self._extract_priority(item_data),
            'age_days': self._calculate_age_days(item_data),
            'team_size': item_data.get('team_size', 1)
        }
        
        return self.analyze_text(combined_text, context)
        
    def _extract_priority(self, item_data: Dict) -> str:
        """Extract priority level from item data"""
        wsjf = item_data.get('wsjf', {})
        if isinstance(wsjf, dict):
            total_score = sum([
                wsjf.get('user_business_value', 0),
                wsjf.get('time_criticality', 0),
                wsjf.get('risk_reduction_opportunity_enablement', 0)
            ])
            job_size = wsjf.get('job_size', 1)
            wsjf_score = total_score / job_size if job_size > 0 else 0
            
            if wsjf_score > 15:
                return 'high'
            elif wsjf_score > 8:
                return 'medium'
            else:
                return 'low'
        return 'unknown'
        
    def _calculate_age_days(self, item_data: Dict) -> int:
        """Calculate item age in days"""
        created_at = item_data.get('created_at')
        if created_at:
            try:
                created_date = datetime.fromisoformat(created_at.replace('Z', '+00:00'))
                age = (datetime.now() - created_date).days
                return max(0, age)
            except (ValueError, AttributeError):
                pass
        return 0
        
    def generate_team_sentiment_report(self, backlog_items: List[Dict]) -> Dict[str, Any]:
        """Generate team sentiment analysis report from backlog items"""
        if not backlog_items:
            return {'error': 'No backlog items provided'}
            
        results = []
        sentiment_distribution = {'positive': 0, 'negative': 0, 'neutral': 0}
        total_confidence = 0.0
        emotion_totals = {'joy': 0, 'anger': 0, 'fear': 0, 'sadness': 0, 'surprise': 0, 'trust': 0}
        
        for item in backlog_items:
            result = self.analyze_backlog_item(item)
            results.append({
                'item_id': item.get('id', 'unknown'),
                'title': item.get('title', 'No title'),
                'sentiment': result.label.value,
                'confidence': result.confidence,
                'keywords': result.keywords
            })
            
            # Aggregate statistics
            if result.label in [SentimentLabel.POSITIVE, SentimentLabel.VERY_POSITIVE]:
                sentiment_distribution['positive'] += 1
            elif result.label in [SentimentLabel.NEGATIVE, SentimentLabel.VERY_NEGATIVE]:
                sentiment_distribution['negative'] += 1
            else:
                sentiment_distribution['neutral'] += 1
                
            total_confidence += result.confidence
            
            for emotion, score in result.emotions.items():
                emotion_totals[emotion] += score
                
        # Calculate averages
        total_items = len(backlog_items)
        avg_confidence = total_confidence / total_items if total_items > 0 else 0.0
        
        for emotion in emotion_totals:
            emotion_totals[emotion] = emotion_totals[emotion] / total_items if total_items > 0 else 0.0
            
        # Generate insights
        insights = self._generate_insights(sentiment_distribution, emotion_totals, total_items)
        
        return {
            'timestamp': datetime.now().isoformat(),
            'total_items_analyzed': total_items,
            'sentiment_distribution': sentiment_distribution,
            'average_confidence': round(avg_confidence, 3),
            'emotion_analysis': emotion_totals,
            'insights': insights,
            'individual_results': results
        }
        
    def _generate_insights(self, sentiment_dist: Dict, emotions: Dict, total_items: int) -> List[str]:
        """Generate actionable insights from sentiment analysis"""
        insights = []
        
        # Overall sentiment insights
        positive_ratio = sentiment_dist['positive'] / total_items if total_items > 0 else 0
        negative_ratio = sentiment_dist['negative'] / total_items if total_items > 0 else 0
        
        if positive_ratio > 0.6:
            insights.append("Team shows high positive sentiment - good momentum and morale")
        elif negative_ratio > 0.4:
            insights.append("High negative sentiment detected - consider addressing team concerns")
        elif sentiment_dist['neutral'] / total_items > 0.7:
            insights.append("Neutral sentiment dominates - team may need more engagement")
            
        # Emotion-based insights
        if emotions.get('anger', 0) > 0.3:
            insights.append("Elevated anger levels - review blockers and process frustrations")
        if emotions.get('fear', 0) > 0.3:
            insights.append("High concern levels - provide more clarity and support")
        if emotions.get('joy', 0) > 0.4:
            insights.append("Strong positive emotions - team is engaged and achieving goals")
        if emotions.get('trust', 0) > 0.4:
            insights.append("High trust levels - team collaboration is strong")
            
        return insights


def main():
    """CLI entry point for sentiment analysis"""
    import sys
    
    if len(sys.argv) < 2:
        print("Usage: python sentiment_analyzer.py <command> [args]")
        print("Commands:")
        print("  analyze <text> - Analyze sentiment of text")
        print("  backlog <file> - Analyze sentiment of backlog items")
        return
        
    analyzer = SentimentAnalyzer()
    command = sys.argv[1]
    
    if command == "analyze" and len(sys.argv) > 2:
        text = " ".join(sys.argv[2:])
        result = analyzer.analyze_text(text)
        print(json.dumps(asdict(result), indent=2))
        
    elif command == "backlog" and len(sys.argv) > 2:
        backlog_file = sys.argv[2]
        try:
            with open(backlog_file, 'r') as f:
                items = json.load(f)
            if not isinstance(items, list):
                items = [items]
            report = analyzer.generate_team_sentiment_report(items)
            print(json.dumps(report, indent=2))
        except Exception as e:
            print(f"Error analyzing backlog: {e}")
    else:
        print("Invalid command or missing arguments")


if __name__ == "__main__":
    main()
