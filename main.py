from fastapi import FastAPI, UploadFile, File
import shutil
import os
from pipeline import process_images

app = FastAPI()

UPLOAD_FOLDER = "uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)


@app.get("/")
def home():
    return {"message": "PCB Defect Detection API with ROI Pipeline 🚀"}

from pipeline import process_images

@app.post("/predict")
async def predict_defect(template: UploadFile = File(...),
                         defect: UploadFile = File(...)):

    template_path = os.path.join(UPLOAD_FOLDER, "template_" + template.filename)
    defect_path = os.path.join(UPLOAD_FOLDER, "defect_" + defect.filename)

    with open(template_path, "wb") as buffer:
        shutil.copyfileobj(template.file, buffer)

    with open(defect_path, "wb") as buffer:
        shutil.copyfileobj(defect.file, buffer)

    results = process_images(template_path, defect_path)

    return {
        "results": results,
        "total_rois_detected": len(results)
    }