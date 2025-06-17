import cv2
import numpy as np
import os
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
        # Read and process ID_image
        id_bytes = await ID_image.read()
        id_array = np.frombuffer(id_bytes, np.uint8)
        id_img = face_recognition.load_image_file(id_array)
        id_faces = face_recognition.face_locations(id_img)
        if not id_faces:
            raise HTTPException(status_code=400, detail="No face found in ID_image")
        id_encoding = face_recognition.face_encodings(id_img, known_face_locations=id_faces)[0]

        # Read and process reference_image
        ref_bytes = await reference_image.read()
        ref_array = np.frombuffer(ref_bytes, np.uint8)
        ref_img = face_recognition.load_image_file(ref_array)
        ref_faces = face_recognition.face_locations(ref_img)
        if not ref_faces:
            raise HTTPException(status_code=400, detail="No face found in reference_image")
        ref_encoding = face_recognition.face_encodings(ref_img, known_face_locations=ref_faces)[0]

        # Read and process test_image
        test_bytes = await test_image.read()
        test_array = np.frombuffer(test_bytes, np.uint8)
        test_img = face_recognition.load_image_file(test_array)
        test_faces = face_recognition.face_locations(test_img)
        if not test_faces:
            raise HTTPException(status_code=400, detail="No face found in test_image")
        test_encoding = face_recognition.face_encodings(test_img, known_face_locations=test_faces)[0]

        # Compare test image to both reference and ID
        threshold = 0.6
        is_verified = any(
            np.linalg.norm(enc - test_encoding) < threshold
            for enc in [id_encoding, ref_encoding]
        )

        return JSONResponse(content={"verified": is_verified})

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
