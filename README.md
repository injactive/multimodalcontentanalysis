# 🚀 Multi-Modal Content Analysis API

[![CI/CD Pipeline](https://github.com/injactive/multimodalcontentanalysis/workflows/CI/CD%20Pipeline/badge.svg)](https://github.com/username/multimodal-content-analysis/actions)
[![Python 3.13.2](https://img.shields.io/badge/python-3.13.2-blue.svg)](https://www.python.org/downloads/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.115.14-009688.svg)](https://fastapi.tiangolo.com)
[![JinaCLIP v2](https://img.shields.io/badge/JinaCLIP-v2-orange.svg)](https://huggingface.co/jinaai/jina-clip-v2)
[![MLflow](https://img.shields.io/badge/MLflow-3.1.1-blue.svg)](https://mlflow.org)

## 📋 Overview

The Multi-Modal Content Analysis API solution for analyzing text and image content to predict engagement scores with confidence intervals. Built with **JinaCLIP v2**, the latest advancement in multimodal AI, this API provides superior performance compared to traditional CLIP models.

### 🎯 Key Features

- **🧠 JinaCLIP v2 Integration**: State-of-the-art multimodal AI model
- **📊 Engagement Prediction**: ML-powered engagement scoring (0-100) with confidence intervals
- **🌍 Multilingual Support**: Excellent performance in German, Spanish, French, and more
- **📈 MLflow Integration**: Comprehensive experiment tracking and ML observability
- **🔒 Production Security**: SSRF protection, input validation, rate limiting
- **⚡ High Performance**: Async processing, caching, optimized for scale
- **🛠️ Enterprise CI/CD**: Automated testing, linting, security scanning, deployment
- **📖 Comprehensive Docs**: OpenAPI/Swagger documentation with examples

### Installation

```bash
# Clone the repository
git clone https://github.com/injactive/multimodalcontentanalysis.git
cd multimodalcontentanalysis

# Create virtual environment
python -m venv .venv
.venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

```

### Demo Run

```bash
# Start the API
python main.py

# Wait until the following can be seen
INFO:     Application startup complete.
INFO:     Uvicorn running on http://0.0.0.0:8000 (Press CTRL+C to quit)

# API will be available at:
# - API: http://localhost:8000
# - Docs: http://localhost:8000/docs
# - Health: http://localhost:8000/health

# Run Demonstrator (in separate terminal with same virtual enviroment)
python demo.py

# Run MLFlow for validating results (in separate terminal with same virtual enviroment)
mlflow ui
# MLFlow will be available at http://localhost:5000
```

### Expected Response on Demo Run
```bash
🚀 Multi-Modal Content Analysis API Test
==================================================
🔍 Testing health check endpoint...
✅ Health check passed!
   Status: healthy
   Version: 0.0.1
   Models loaded: True

🧪 TEST CASE 1: Positive Travel Post
------------------------------

🔍 Testing post analysis...
   Text: Just had the most amazing sunset experience! 🌅 The colors were absolutely breathtaking and I feel so...
   Image URL: https://images.unsplash.com/photo-1506905925346-21bda4d32df4?w=800&h=600&fit=crop
```