from fastapi import FastAPI, HTTPException, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from transformers import pipeline
import logging
import pandas as pd
import io
from typing import List, Optional

app = FastAPI(
    title="Sentiment Analysis API",
    description="A FastAPI backend for sentiment analysis with single text and file upload",
    version="2.0.0"
)

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize sentiment analysis pipeline
try:
    sentiment_pipeline = pipeline(
        "sentiment-analysis",
        model="distilbert-base-uncased-finetuned-sst-2-english"
    )
    logger.info("Sentiment analysis pipeline loaded successfully")
except Exception as e:
    logger.error(f"Failed to load sentiment pipeline: {str(e)}")
    sentiment_pipeline = None

# Allow CORS for frontend integration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class TextInput(BaseModel):
    text: str = Field(..., example="I love this product!")

@app.post("/analyze")
async def analyze_sentiment(data: TextInput):
    if sentiment_pipeline is None:
        raise HTTPException(status_code=503, detail="Sentiment analysis model is not available")
    
    try:
        # الفحص هنا: الموديل بيرجع قائمة [dict]
        result = sentiment_pipeline(data.text)
        
        # الإصلاح السحري: نأخذ العنصر صفر
        label = result[0]['label']
        score = result[0]['score']
        
        logger.info(f"Analyzed single text: '{data.text[:30]}...' - Label: {label}, Score: {score}")
        
        return {
            "label": label,
            "score": score
        }
    except Exception as e:
        logger.error(f"Error during single sentiment analysis: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/analyze-file")
async def analyze_file(file: UploadFile = File(...)):
    if sentiment_pipeline is None:
        raise HTTPException(status_code=503, detail="Sentiment analysis model is not available")
        
    try:
        content = await file.read()
        file_extension = file.filename.split('.')[-1].lower()
        
        if file_extension == 'csv':
            df = pd.read_csv(io.BytesIO(content))
        elif file_extension in ['xlsx', 'xls']:
            df = pd.read_excel(io.BytesIO(content))
        else:
            raise HTTPException(status_code=400, detail="Unsupported file format. Please upload CSV or Excel.")
            
        # Find a text column
        text_column = None
        possible_names = ['text', 'review', 'comment', 'body', 'sentence', 'المحتوى', 'النص']
        
        for col in df.columns:
            if col.lower() in possible_names:
                text_column = col
                break
                
        if text_column is None:
            text_column = df.select_dtypes(include=['object']).columns[0]
            
        texts = df[text_column].dropna().astype(str).tolist()
        
        if not texts:
            raise HTTPException(status_code=400, detail="No valid text found in the file")
            
        results = []
        positive_count = 0
        negative_count = 0
        total_confidence = 0.0
        positive_confidence = 0.0
        negative_confidence = 0.0
        
        # Process in bulk
        pipeline_results = sentiment_pipeline(texts)
        
        for text, res in zip(texts, pipeline_results):
            label = res['label']
            score = res['score']
            
            total_confidence += score
            if label == 'POSITIVE':
                positive_count += 1
                positive_confidence += score
            else:
                negative_count += 1
                negative_confidence += score
                
            results.append({
                "text": text[:100] + "..." if len(text) > 100 else text,
                "label": label,
                "score": round(score, 4)
            })
            
        total_records = len(texts)
        
        return {
            "total_records": total_records,
            "positive_count": positive_count,
            "negative_count": negative_count,
            "average_confidence": round(total_confidence / total_records, 4) if total_records else 0,
            "positive_average_confidence": round(positive_confidence / positive_count, 4) if positive_count else 0,
            "negative_average_confidence": round(negative_confidence / negative_count, 4) if negative_count else 0,
            "preview_data": results[:100]  # Return first 100 records for preview table
        }
    except HTTPException as he:
        raise he
    except Exception as e:
        logger.error(f"Error during file analysis: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="127.0.0.1", port=8000, reload=True)