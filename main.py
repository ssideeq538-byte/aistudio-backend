import os
import psycopg2
from fastapi import FastAPI
from pydantic import BaseModel

app = FastAPI()

# قراءة رابط قاعدة البيانات تلقائياً من بيئة Railway
DATABASE_URL = os.environ.get("DATABASE_URL") or os.environ.get("DATABASE_PRIVATE_URL")

class CodeRequest(BaseModel):
    prompt: str
    language: str

def save_to_db(prompt: str, language: str, code: str):
    if not DATABASE_URL:
        return
    try:
        conn = psycopg2.connect(DATABASE_URL)
        cur = conn.cursor()
        cur.execute(
            "INSERT INTO generated_codes (prompt, language, code) VALUES (%s, %s, %s)",
            (prompt, language, code)
        )
        conn.commit()
        cur.close()
        conn.close()
    except Exception as e:
        print("Database Error:", e)

@app.get("/")
def home():
    return {"status": "AI Studio Backend is Running on Railway!"}

@app.post("/generate")
def generate_code(req: CodeRequest):
    p = req.prompt.lower()
    lang = req.language

    # محرك التوليد الذكي
    if "كره" in p or "كرة" in p or "قدم" in p:
        code = f"# لعبة كرة قدم سحابية ({lang})\nimport random\nprint('⚽ جووووول سحابي رائع!')"
    elif "حساب" in p or "حاسب" in p:
        code = f"# آلة حاسبة سحابية ({lang})\ndef calc(a, b): return a + b\nprint('الناتج:', calc(10, 20))"
    elif "نص" in p:
        code = f"# معالج نصوص سحابي ({lang})\nprint('تمت معالجة النص بنجاح في السيرفر!')"
    else:
        code = f"// كود تم توليده من سيرفر Railway\n// الطلب: {req.prompt}\n// اللغة: {lang}\nprint('أهلاً بك! تم التوليد بنجاح عبر السيرفر.')"

    # حفظ الطلب والكود المولد في قاعدة بيانات Postgres
    save_to_db(req.prompt, lang, code)

    return {"generated_code": code}

if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 8080))
    uvicorn.run(app, host="0.0.0.0", port=port)
