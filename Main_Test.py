import numpy as np
import io
from deepface import DeepFace
import os
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import JSONResponse
import uvicorn
from PIL import Image


app = FastAPI()

def resize_image(image_bytes: bytes, max_width: int = 500) -> bytes:
    """تصغير حجم الصورة مع الحفاظ على النسبة"""
    img = Image.open(io.BytesIO(image_bytes))
    if img.width > max_width:
        ratio = max_width / img.width
        new_height = int(img.height * ratio)
        img = img.resize((max_width, new_height), Image.LANCZOS)
    
    buffer = io.BytesIO()
    img.save(buffer, format="JPEG", quality=85)
    return buffer.getvalue()

def save_temp_file(file_bytes: bytes, prefix: str = "temp") -> str:
    """حفظ ملف مؤقت وإرجاع مساره"""
    temp_path = f"{prefix}_{os.urandom(4).hex()}.jpg"
    with open(temp_path, "wb") as f:
        f.write(file_bytes)
    return temp_path

@app.post("/verify-face")
async def verify_face(
    ID_image: UploadFile = File(...),
    reference_image: UploadFile = File(...),
    test_image: UploadFile = File(...)
):
    temp_files = []
    try:
        # 1. معالجة الصور
        id_bytes = resize_image(await ID_image.read())
        ref_bytes = resize_image(await reference_image.read())
        test_bytes = resize_image(await test_image.read())

        # 2. حفظ الصور مؤقتًا
        id_path = save_temp_file(id_bytes, "id")
        ref_path = save_temp_file(ref_bytes, "ref")
        test_path = save_temp_file(test_bytes, "test")
        temp_files.extend([id_path, ref_path, test_path])

        # 3. مقارنة الوجوه (مع كل من ID وreference)
        result_id = DeepFace.verify(img1_path=id_path, img2_path=test_path, enforce_detection=False)
        result_ref = DeepFace.verify(img1_path=ref_path, img2_path=test_path, enforce_detection=False)

        # 4. النتيجة (يكفي تطابق مع أي من الصورتين)
        final_result = result_id["verified"] or result_ref["verified"]

        return {
            "result": final_result,
            "details": {
                "ID_match": {
                    "verified": result_id["verified"],
                    "distance": float(result_id["distance"]),
                    "similarity": round((1 - result_id["distance"]) * 100, 2)
                },
                "REF_match": {
                    "verified": result_ref["verified"],
                    "distance": float(result_ref["distance"]),
                    "similarity": round((1 - result_ref["distance"]) * 100, 2)
                }
            }
        }

    except Exception as e:
        return {"result": False, "error": str(e)}
    
    finally:
        # 5. تنظيف الملفات المؤقتة
        for file_path in temp_files:
            if os.path.exists(file_path):
                os.remove(file_path)

@app.get("/")
async def health_check():
    return {"status": "running", "api": "face-verification"}
