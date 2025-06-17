import numpy as np
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
        # Load image bytes
        id_bytes = await ID_image.read()
        ref_bytes = await reference_image.read()
        test_bytes = await test_image.read()

        # Convert bytes to images
        id_np = face_recognition.load_image_file(np.frombuffer(id_bytes, dtype=np.uint8))
        ref_np = face_recognition.load_image_file(np.frombuffer(ref_bytes, dtype=np.uint8))
        test_np = face_recognition.load_image_file(np.frombuffer(test_bytes, dtype=np.uint8))

        # Encode faces
        id_enc = face_recognition.face_encodings(id_np)
        ref_enc = face_recognition.face_encodings(ref_np)
        test_enc = face_recognition.face_encodings(test_np)

        # Check if any image has no detected faces
        if not id_enc or not ref_enc or not test_enc:
            return JSONResponse(content={"verified": False, "detail": "No face found in one or more images."})

        # Use first face only
        id_encoding = id_enc[0]
        ref_encoding = ref_enc[0]
        test_encoding = test_enc[0]

        # Compare test image with both reference and ID
        is_match = (
            face_recognition.compare_faces([id_encoding], test_encoding, tolerance=0.6)[0] or
            face_recognition.compare_faces([ref_encoding], test_encoding, tolerance=0.6)[0]
        )

        return JSONResponse(content={"verified": is_match})

    except Exception as e:
        return JSONResponse(status_code=500, content={"detail": str(e)})
