import pytest
import logging
import json
import os
import joblib
from prediction_service.prediction import form_response, api_response
import prediction_service

input_data = {
    "incorrect_range":
        {
            "T": 7,
            "TM": 35,
            "Tm": 28,
            "SLP": 1025,
            "H": 69,
            "VV": 4,
            "V": 20,
            "VM": 35
        },

    "correct_range":
        {
            "T": 7,
            "TM": 35,
            "Tm": 28,
            "SLP": 1000,
            "H": 69,
            "VV": 4,
            "V": 20,
            "VM": 35
        },

    "incorrect_col":
        {
            "T": 7,
            "tM": 35,
            "Tm": 28,
            "SLP": 1000,
            "H": 69,
            "VV": 4,
            "V": 20,
            "VM": 35
        },
}

TARGET_range = {
    "min": 0.0,
    "max": 404.5
}


def test_form_response_correct_range(data=input_data["correct_range"]):
    res = form_response(data)
    assert TARGET_range["min"] <= res <= TARGET_range["max"]


def test_api_response_correct_range(data=input_data["correct_range"]):
    res = api_response(data)
    assert TARGET_range["min"] <= res["response"] <= TARGET_range["max"]


def test_form_response_incorrect_range(data=input_data["incorrect_range"]):
    with pytest.raises(prediction_service.prediction.NotInRange):
        res = form_response(data)


def test_api_response_incorrect_range(data=input_data["incorrect_range"]):
    res = api_response(data)
    assert res["response"] == prediction_service.prediction.NotInRange().message


def test_api_response_incorrect_col(data=input_data["incorrect_col"]):
    res = api_response(data)
    assert res["response"] == prediction_service.prediction.NotInCols().message
