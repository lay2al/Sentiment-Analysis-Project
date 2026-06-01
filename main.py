import os
import sqlite3
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import pandas as pd
from transformers import pipeline
from langdetect import detect, DetectorFactory

# لضمان استقرار نتائج كشف اللغة
DetectorFactory.seed = 0

app = FastAPI(title="Super AI Sentiment & Emotion Analytics System")

# تفعيل الـ CORS لتتصل الواجهة بالسيرفر بأمان
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- 1. إعداد قاعدة البيانات (SQLite History) ---
DB_PATH = "project_history.db"

def init_db():
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS history (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            text TEXT,
            language TEXT,
            sentiment TEXT,
            intensity_score REAL,
            dominant_emotion TEXT,
            timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
        )
    ''')
    conn.commit()
    conn.close()

init_db()

# --- 2. تحميل نماذج الذكاء الاصطناعي (AI Pipelines) ---
print("🔄 Loading Advanced NLP Pipelines...")
# موديل المشاعر الأساسي (الإنجليزي والسريع)
sentiment_pipeline = pipeline("sentiment-analysis", model="distilbert-base-uncased-finetuned-sst-2-english")
# موديل العواطف المتعددة (تحليل الأبعاد العاطفية الـ 4 الأساسية)
emotion_pipeline = pipeline("text-classification", model="bhadresh-savani/bert-base-uncased-emotion", top_k=None)
print("✅ All AI Pipelines Loaded Successfully!")

# --- 3. نماذج البيانات (Pydantic Models) ---
class TextInput(BaseModel):
    text: str

# --- 4. دالة معالجة النصوص الذكية حسابياً ---
def process_text_ai(text: str):
    # أ) كشف اللغة تلقائياً
    try:
        lang = detect(text)
        lang_label = "Arabic" if lang == 'ar' else "English"
    except:
        lang_label = "English"  # افتراضي في حال فشل الكشف
    
    # ب) تحليل المشاعر الأساسي وحساب معادلة المشرف (-5 إلى 5)
    sentiment_res = sentiment_pipeline(text)[0]
    label = sentiment_res['label']  # POSITIVE or NEGATIVE
    confidence = sentiment_res['score']
    
    # تطبيق معادلة المشرف الحسابية الدقيقة
    if label == "POSITIVE":
        # تحويل من (0.5 لـ 1) إلى مقياس (0 لـ 5)
        intensity_score = round((confidence - 0.5) * 2 * 5, 2)
        # ضمان عدم تجاوز الحدود برمجياً
        intensity_score = min(5.0, max(0.0, intensity_score))
        final_label = "Positive"
    else:
        # تحويل من (0.5 لـ 1) إلى مقياس (-5 لـ 0)
        intensity_score = round((confidence - 0.5) * -2 * 5, 2)
        intensity_score = max(-5.0, min(0.0, intensity_score))
        final_label = "Negative"
        
    # ج) تحليل العواطف المتعددة (Emotion Prototyping)
    emotion_res = emotion_pipeline(text)[0]
    # مخرجات الموديل بتعطي نسب لكل العواطف، بنقشط العواطف الـ 4 الأساسية اللي بدناياها
    allowed_emotions = ['joy', 'sadness', 'anger', 'fear']
    filtered_emotions = {e['label']: e['score'] for e in emotion_res if e['label'] in allowed_emotions}
    
    # تحديد العاطفة المسيطرة
    dominant_emotion = max(filtered_emotions, key=filtered_emotions.get).capitalize()
    
    # د) حفظ النتيجة في قاعدة البيانات كـ History للمشرف
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute('''
        INSERT INTO history (text, language, sentiment, intensity_score, dominant_emotion)
        VALUES (?, ?, ?, ?, ?)
    ''', (text, lang_label, final_label, intensity_score, dominant_emotion))
    conn.commit()
    conn.close()
    
    return {
        "text": text,
        "language": lang_label,
        "sentiment": final_label,
        "intensity_score": intensity_score,
        "dominant_emotion": dominant_emotion,
        "emotions_breakdown": {k.capitalize(): round(v * 100, 2) for k, v in filtered_emotions.items()}
    }

# --- 5. الـ Endpoints الأساسية والمحدثة ---

@app.post("/analyze")
async def analyze_single_text(input_data: TextInput):
    if not input_data.text.strip():
        raise HTTPException(status_code=400, detail="Text cannot be empty")
    return process_text_ai(input_data.text)

@app.post("/analyze-file")
async def analyze_bulk_file(file: UploadFile = File(...)):
    # دعم صيغ الـ Excel والـ CSV بالكامل
    ext = os.path.splitext(file.filename)[1].lower()
    try:
        if ext == ".csv":
            df = pd.read_csv(file.file)
        elif ext in [".xlsx", ".xls"]:
            df = pd.read_excel(file.file)
        else:
            raise HTTPException(status_code=400, detail="Unsupported file format. Please upload CSV or Excel.")
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Error reading file: {str(e)}")

    # البحث الذكي عن عمود النصوص (Smart Column Mapping)
    text_col = None
    possible_cols = ['text', 'content', 'tweet', 'review', 'النص', 'المحتوى', 'comment']
    for c in df.columns:
        if str(c).lower().strip() in possible_cols:
            text_col = c
            break
    if text_col is None:
        # إذا ما لقى الاسم، بياخد أول عمود نصوص تلقائياً
        text_col = df.select_dtypes(include=['object']).columns[0]

    results = []
    for index, row in df.iterrows():
        text_val = str(row[text_col])
        if text_val.strip() and text_val != "nan":
            res = process_text_ai(text_val)
            results.append(res)

    return {
        "filename": file.filename,
        "total_records": len(results),
        "data": results
    }

@app.get("/history")
async def get_history():
    # سحب سجل السيرفر بالكامل لعرضه بجدول متقدم
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    cursor = conn.cursor()
    cursor.execute("SELECT * FROM history ORDER BY timestamp DESC")
    rows = cursor.fetchall()
    conn.close()
    return [dict(r) for r in rows]

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="127.0.0.1", port=8000, reload=True)