"""Unit tests for the EmotionDetection package."""

import json
import unittest
from unittest.mock import Mock, patch

from EmotionDetection import emotion_detector


class TestEmotionDetector(unittest.TestCase):
    """Test dominant emotion detection."""

    @staticmethod
    def build_response(emotion_scores):
        """Create a mocked Watson NLP response."""
        response = Mock()
        response.status_code = 200
        response.text = json.dumps({
            "emotionPredictions": [
                {
                    "emotion": emotion_scores
                }
            ]
        })
        response.raise_for_status = Mock()
        return response

    def run_emotion_test(self, text, scores, expected_emotion):
        """Run one mocked emotion detection test."""
        response = self.build_response(scores)

        with patch(
            "EmotionDetection.emotion_detection.requests.post",
            return_value=response
        ):
            result = emotion_detector(text)

        self.assertEqual(
            result["dominant_emotion"],
            expected_emotion
        )

    def test_joy(self):
        """Test joy detection."""
        self.run_emotion_test(
            "I am glad this happened",
            {
                "anger": 0.01,
                "disgust": 0.01,
                "fear": 0.02,
                "joy": 0.95,
                "sadness": 0.01
            },
            "joy"
        )

    def test_anger(self):
        """Test anger detection."""
        self.run_emotion_test(
            "I am really mad about this",
            {
                "anger": 0.92,
                "disgust": 0.02,
                "fear": 0.02,
                "joy": 0.01,
                "sadness": 0.03
            },
            "anger"
        )

    def test_disgust(self):
        """Test disgust detection."""
        self.run_emotion_test(
            "I feel disgusted just hearing about this",
            {
                "anger": 0.03,
                "disgust": 0.90,
                "fear": 0.02,
                "joy": 0.01,
                "sadness": 0.04
            },
            "disgust"
        )

    def test_sadness(self):
        """Test sadness detection."""
        self.run_emotion_test(
            "I am so sad about this",
            {
                "anger": 0.01,
                "disgust": 0.01,
                "fear": 0.02,
                "joy": 0.01,
                "sadness": 0.95
            },
            "sadness"
        )

    def test_fear(self):
        """Test fear detection."""
        self.run_emotion_test(
            "I am really afraid that this will happen",
            {
                "anger": 0.01,
                "disgust": 0.01,
                "fear": 0.94,
                "joy": 0.01,
                "sadness": 0.03
            },
            "fear"
        )


if __name__ == "__main__":
    unittest.main()
