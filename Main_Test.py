import numpy as np
import io
import face_recognition
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import JSONResponse
import uvicorn
from PIL import Image

app = FastAPI()

def resize_image(image_bytes: bytes, max_width: int = 500) -> bytes:
    img = Image.open(io.BytesIO(image_bytes))
    if img.width > max_width:
        ratio = max_width / img.width
        new_height = int(img.height * ratio)
        img = img.resize((max_width, new_height), Image.LANCZOS)
    buffer = io.BytesIO()
    img.save(buffer, format="JPEG", quality=85)
    return buffer.getvalue()

def match_percentage(face_distance) -> float:
    # تحويل المسافة إلى نسبة مئوية (0% = غير متطابق، 100% = متطابق تمامًا)
    return max(0, (1 - face_distance) * 100)

@app.post("/verify-face")
async def verify_face(
    ID_image: UploadFile = File(...),
    reference_image: UploadFile = File(...),
    test_image: UploadFile = File(...)
):
    try:
        # 1. قراءة وتصغير الصور
        id_bytes = await ID_image.read()
        ref_bytes = await reference_image.read()
        test_bytes = await test_image.read()

        id_bytes = resize_image(id_bytes)
        ref_bytes = resize_image(ref_bytes)
        test_bytes = resize_image(test_bytes)

        # 2. تحميل الصور وتشفير الوجوه
        id_img = face_recognition.load_image_file(io.BytesIO(id_bytes))
        ref_img = face_recognition.load_image_file(io.BytesIO(ref_bytes))
        test_img = face_recognition.load_image_file(io.BytesIO(test_bytes))

        id_enc = face_recognition.face_encodings(id_img)
        ref_enc = face_recognition.face_encodings(ref_img)
        test_enc = face_recognition.face_encodings(test_img)

        if not id_enc or not ref_enc or not test_enc:
            return {"result": False, "reason": "No face found in one or more images."}

        # 3. حساب المسافات ونسب التطابق
        id_distance = face_recognition.face_distance([id_enc[0]], test_enc[0])[0]
        ref_distance = face_recognition.face_distance([ref_enc[0]], test_enc[0])[0]

        id_percentage = match_percentage(id_distance)
        ref_percentage = match_percentage(ref_distance)

        # 4. تحديد إذا كان هناك تطابق (مثلاً إذا كانت النسبة > 60%)
        threshold = 60.0  # الحد الأدنى للتطابق
        is_id_match = id_percentage >= threshold
        is_ref_match = ref_percentage >= threshold

        return {
            "result": is_id_match or is_ref_match,
            "match_percentages": {
                "ID_image": round(id_percentage, 2),
                "reference_image": round(ref_percentage, 2)
            },
            "details": "Match if percentage >= 60%"
        }

    except Exception as e:
        return {"result": False, "error": str(e)}
