# 🧠 AI-Based Sentiment Analysis & Statistics Dashboard

An enterprise-level software engineering graduation project featuring a full-stack **FastAPI** backend integrated with a state-of-the-art **Transformer-based (DistilBERT)** NLP pipeline and a dynamic responsive **TailwindCSS / Chart.js** analytics dashboard.

---

## 🚀 Key Features

* **Single Text Analytics:** Real-time sentiment prediction (Positive/Negative) with confidence scores.
* **Bulk File Processing:** Supports massive datasets via CSV and Excel (`.xlsx`, `.xls`) file uploads.
* **Smart Column Mapping:** Automatic text column detection supporting multi-language data inputs.
* **Interactive Dashboard:** Beautiful distribution graphs, total evaluated metrics, and split high-fidelity KPI counters.
* **Production-Ready Architecture:** Lightweight FastAPI server with fully configured asynchronous endpoints and CORS middleware.

---

## 🛠️ Tech Stack & Architecture

* **Backend Framework:** FastAPI (Python 3.9+)
* **AI/NLP Model:** Hugging Face Transformers (`distilbert-base-uncased-finetuned-sst-2-english`)
* **Data Processing:** Pandas, OpenPyXL
* **Frontend Interface:** HTML5, TailwindCSS, FontAwesome 6, Chart.js (via Content Delivery Networks)
* **Server Deployment:** Uvicorn (Asynchronous Server Gateway Interface)

---

## 📦 System Installation & Running Locally

Follow these steps to deploy and test the system locally on your environment:

### 1. Clone and Navigate to the Directory
```bash
git clone [https://github.com/lay2al/Sentiment-Analysis-Project.git](https://github.com/lay2al/Sentiment-Analysis-Project.git)
cd Sentiment-Analysis-Project