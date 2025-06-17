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
        # Read all 3 images
        id_bytes = await ID_image.read()
        ref_bytes = await reference_image.read()
        test_bytes = await test_image.read()

        # Convert to numpy arrays
        id_np = face_recognition.load_image_file(np.frombuffer(id_bytes, np.uint8))
        ref_np = face_recognition.load_image_file(np.frombuffer(ref_bytes, np.uint8))
        test_np = face_recognition.load_image_file(np.frombuffer(test_bytes, np.uint8))

        # Get encodings
        id_faces = face_recognition.face_encodings(id_np)
        ref_faces = face_recognition.face_encodings(ref_np)
        test_faces = face_recognition.face_encodings(test_np)

        if not id_faces or not ref_faces or not test_faces:
            return JSONResponse(content={"verified": False, "detail": "Face not found in one or more images."})

        id_encoding = id_faces[0]
        ref_encoding = ref_faces[0]
        test_encoding = test_faces[0]

        # Compare
        result = (
            face_recognition.compare_faces([id_encoding], test_encoding, tolerance=0.6)[0] or
            face_recognition.compare_faces([ref_encoding], test_encoding, tolerance=0.6)[0]
        )

        return JSONResponse(content={"verified": result})

    except Exception as e:
        return JSONResponse(status_code=500, content={"detail": str(e)})
