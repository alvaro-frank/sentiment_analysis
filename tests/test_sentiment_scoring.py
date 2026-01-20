"""
Unit Tests for Sentiment Scoring Logic (Knowledge Distillation)
-----------------------------------------------------
Tests the scoring logic used to convert classification probs to continuous scores.
"""
import pytest

def calculate_sentiment_score(results):
    """
    Given the output from the FinBERT pipeline, calculate the sentiment score.
    Logic: Score = (Positive Probability - Negative Probability)
    """
    pos_score = next(item['score'] for item in results if item['label'] == 'positive')
    neg_score = next(item['score'] for item in results if item['label'] == 'negative')
    return pos_score - neg_score

class TestSentimentScoring:

    def test_positive_dominance(self):
        """If positive prob is high, score should be close to 1.0"""
        mock_output = [
            {'label': 'positive', 'score': 0.9},
            {'label': 'negative', 'score': 0.05},
            {'label': 'neutral', 'score': 0.05}
        ]
        score = calculate_sentiment_score(mock_output)
        # 0.9 - 0.05 = 0.85
        assert score == pytest.approx(0.85)

    def test_negative_dominance(self):
        """If negative prob is high, score should be close to -1.0"""
        mock_output = [
            {'label': 'positive', 'score': 0.1},
            {'label': 'negative', 'score': 0.8},
            {'label': 'neutral', 'score': 0.1}
        ]
        score = calculate_sentiment_score(mock_output)
        # 0.1 - 0.8 = -0.7
        assert score == pytest.approx(-0.7)

    def test_neutral_case(self):
        """If neutral dominates, score should be balanced (close to 0)"""
        mock_output = [
            {'label': 'positive', 'score': 0.1},
            {'label': 'negative', 'score': 0.1},
            {'label': 'neutral', 'score': 0.8}
        ]
        score = calculate_sentiment_score(mock_output)
        # 0.1 - 0.1 = 0.0
        assert score == pytest.approx(0.0)