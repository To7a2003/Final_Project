import numpy as np
import io
import face_recognition
from fastapi import FastAPI, UploadFile, File
from fastapi.responses import JSONResponse

app = FastAPI()

@app.post("/verify-face")
async def verify_face(
    ID_image: UploadFile = File(...),
    reference_image: UploadFile = File(...),
    test_image: UploadFile = File(...)
):
    try:
        # Read image bytes
        id_bytes = await ID_image.read()
        ref_bytes = await reference_image.read()
        test_bytes = await test_image.read()

        # Load and encode images
        id_img = face_recognition.load_image_file(io.BytesIO(id_bytes))
        ref_img = face_recognition.load_image_file(io.BytesIO(ref_bytes))
        test_img = face_recognition.load_image_file(io.BytesIO(test_bytes))

        id_enc = face_recognition.face_encodings(id_img)
        ref_enc = face_recognition.face_encodings(ref_img)
        test_enc = face_recognition.face_encodings(test_img)

        # Ensure faces were detected
        if not id_enc or not ref_enc or not test_enc:
            return {"result": False, "reason": "No face found in one or more images."}

        id_encoding = id_enc[0]
        ref_encoding = ref_enc[0]
        test_encoding = test_enc[0]

        match_id = face_recognition.compare_faces([id_encoding], test_encoding)[0]
        match_ref = face_recognition.compare_faces([ref_encoding], test_encoding)[0]

        return {"result": match_id and match_ref}

    except Exception as e:
        return {"result": False, "error": str(e)}
