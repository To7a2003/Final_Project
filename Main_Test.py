import numpy as np
import io
import face_recognition
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import JSONResponse
import uvicorn


app = FastAPI()

@app.post("/verify-face")
async def verify_face(
    ID_image: UploadFile = File(...),
    reference_image: UploadFile = File(...),
    test_image: UploadFile = File(...)
):
    try:
        # Read bytes
        id_bytes = await ID_image.read()
        ref_bytes = await reference_image.read()
        test_bytes = await test_image.read()

        # Load images from bytes
        id_img = face_recognition.load_image_file(io.BytesIO(id_bytes))
        ref_img = face_recognition.load_image_file(io.BytesIO(ref_bytes))
        test_img = face_recognition.load_image_file(io.BytesIO(test_bytes))

        # Encode faces
        id_enc = face_recognition.face_encodings(id_img)
        ref_enc = face_recognition.face_encodings(ref_img)
        test_enc = face_recognition.face_encodings(test_img)

        if not id_enc or not ref_enc or not test_enc:
            return {"result": False, "reason": "No face found in one or more images."}

        # Compare test face to ID and Reference
        match_id = face_recognition.compare_faces([id_enc[0]], test_enc[0])[0]
        match_ref = face_recognition.compare_faces([ref_enc[0]], test_enc[0])[0]

        return {"result": match_id and match_ref}

    except Exception as e:
        return {"result": False, "error": str(e)}
