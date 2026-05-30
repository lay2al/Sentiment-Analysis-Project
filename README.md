# Sentiment Analysis Web Application

A university graduation project for sentiment analysis using Python, FastAPI, and Transformers.

## Project Structure

```
Sentiment_Analysis_Project/
├── main.py              # FastAPI application entry point
├── requirements.txt     # Python dependencies
├── README.md           # Project documentation
└── venv/               # Virtual environment (to be created)
```

## Setup Instructions

### 1. Create Virtual Environment

```bash
python3 -m venv venv
```

### 2. Activate Virtual Environment

**On macOS/Linux:**
```bash
source venv/bin/activate
```

**On Windows:**
```bash
venv\Scripts\activate
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

### 4. Run the Application

```bash
python main.py
```

Or using uvicorn directly:
```bash
uvicorn main:app --reload --host 0.0.0.0 --port 8000
```

The API will be available at `http://localhost:8000`

### 5. Access API Documentation

- Swagger UI: `http://localhost:8000/docs`
- ReDoc: `http://localhost:8000/redoc`

## Dependencies

- **FastAPI**: Modern, fast web framework for building APIs
- **Uvicorn**: ASGI server for running FastAPI
- **Transformers**: Hugging Face library for NLP tasks
- **PyTorch**: Deep learning framework
- **Pydantic**: Data validation using Python type annotations

## Next Steps

- Implement sentiment analysis endpoint
- Add model loading and prediction logic
- Create frontend interface
- Add authentication and rate limiting
- Implement database integration for storing results
