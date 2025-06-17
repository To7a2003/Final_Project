from fastapi import FastAPI, UploadFile, File
from fastapi.responses import JSONResponse
from deepface import DeepFace
import os
import uuid

app = FastAPI()

@app.post("/verify-face")
async def verify_face(
    ID_image: UploadFile = File(...),
    reference_image: UploadFile = File(...),
    test_image: UploadFile = File(...)
):
    try:
        # حفظ الملفات المؤقتة بأسماء فريدة
        id_path = f"temp_id_{uuid.uuid4()}.jpg"
        ref_path = f"temp_ref_{uuid.uuid4()}.jpg"
        test_path = f"temp_test_{uuid.uuid4()}.jpg"

        with open(id_path, "wb") as f:
            f.write(await ID_image.read())
        with open(ref_path, "wb") as f:
            f.write(await reference_image.read())
        with open(test_path, "wb") as f:
            f.write(await test_image.read())

        # المقارنة مع إلغاء الكشف الإجباري عن الوجوه
        result_id = DeepFace.verify(
            img1_path=id_path,
            img2_path=test_path,
            model_name="VGG-Face",
            detector_backend="opencv",
            enforce_detection=False
        )

        result_ref = DeepFace.verify(
            img1_path=ref_path,
            img2_path=test_path,
            model_name="VGG-Face",
            detector_backend="opencv",
            enforce_detection=False
        )

        # تنظيف الملفات المؤقتة
        os.remove(id_path)
        os.remove(ref_path)
        os.remove(test_path)

        return {
            "result": result_id["verified"] or result_ref["verified"],
            "details": {
                "ID_match": result_id["verified"],
                "REF_match": result_ref["verified"],
                "distance": {
                    "ID": float(result_id["distance"]),
                    "REF": float(result_ref["distance"])
                }
            }
        }

    except Exception as e:
        return JSONResponse(
            status_code=500,
            content={"error": str(e)}
        )
