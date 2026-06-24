from fastapi import FastAPI, File, UploadFile
from ultralytics import YOLO
import shutil, os

app = FastAPI()
model = YOLO("best.pt")

@app.get("/")
def root():
    return {"status": "ShelfScan server is running"}

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    temp_path = f"/tmp/{file.filename}"
    with open(temp_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)
    
    results = model.predict(source=temp_path, conf=0.5)
    count = len(results[0].boxes)
    
    os.remove(temp_path)
    return {"count": count, "filename": file.filename}