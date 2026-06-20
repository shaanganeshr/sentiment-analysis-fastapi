import pytest
from fastapi.testclient import TestClient
from app import app

client=TestClient(app)

# Valid text prediction and expected output
def test_pred():
    with client as c:
        payload={"text": "This framework is great"}
        response=c.post("/predict", json=payload, auth=("admin", "123"))

        assert response.status_code==200
        data=response.json()
        assert "label" in data
        assert "confidence" in data
        assert data["label"]=="positive"

# Test response validation: Short
def test_tooshort():
    payload={"text": "hi"}
    response=client.post("/predict", json=payload,auth=("admin", "123"))
    assert response.status_code==422

# Test response validation: White
def test_blank():
    payload={"text": "       "}
    response=client.post("/predict", json=payload, auth=("admin", "123"))
    assert response.status_code==400
    assert response.json()["detail"]=="Input string payload cannot be empty"
