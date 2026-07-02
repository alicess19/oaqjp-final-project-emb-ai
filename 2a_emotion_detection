"""Emotion detection module using Watson NLP."""

import json

import requests


def emotion_detector(text_to_analyze):
    """Send text to Watson NLP and return the emotion analysis."""
    url = (
        "https://sn-watson-emotion.labs.skills.network/"
        "v1/watson.runtime.nlp.v1/NlpService/EmotionPredict"
    )

    input_json = {
        "raw_document": {
            "text": text_to_analyze
        }
    }

    headers = {
        "grpc-metadata-mm-model-id":
        "emotion_aggregated-workflow_lang_en_stock"
    }

    response = requests.post(
        url,
        json=input_json,
        headers=headers,
        timeout=30
    )

    formatted_response = json.loads(response.text)

    return formatted_response
