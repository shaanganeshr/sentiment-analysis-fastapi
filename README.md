# 📊 ML Model Sentiment Analysis API Service

![CI-CD Status](https://github.com/shaanganeshr/sentiment-analysis-fastapi/actions/workflows/main.yml/badge.svg)

A production-grade, containerized REST API built with **FastAPI** that wraps a fine-tuned **Hugging Face DistilBERT** transformer pipeline to serve multi-lingual sentiment predictions. This service includes built-in HTTP Basic Authentication, asynchronous background logging tasks, input data validation via Pydantic, a robust unit testing suite via Pytest, and an automated **CI/CD pipeline** via GitHub Actions.

---

## 🛠️ Tech Stack & Ecosystem

* **Core Framework:** FastAPI (Asynchronous Python Web Framework)
* **ML Inference Engine:** Hugging Face Transformers (`lxyuan/distilbert-base-multilingual-cased-sentiments-student`)
* **Data Validation Engine:** Pydantic v2
* **Containerization:** Docker (Isolated Virtual Sandboxing)
* **Testing Framework:** Pytest + HTTPX
* **Automation:** GitHub Actions (Automated CI Testing Loop)

---

## 🚀 Architectural Design Flow

```text
[Client Request] 
       │
       ▼ (HTTP Basic Auth Validation)
[FastAPI Router]
       │
       ▼ (Pydantic Input & String Cleansing Verification)
[ML Lifespan Context Memory Cache] ──► [Hugging Face DistilBERT Pipeline]
       │
       ├──► (Response Payload) ─────► [Client Output JSON]
       │
       └──► (Async Background Task) ──► [Thread Safe System Metadata Logging]
