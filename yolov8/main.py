from fastapi import FastAPI, File, UploadFile
import numpy as np
import cv2
import onnxruntime as ort
from ultralytics.utils import yaml_load
from ultralytics.utils.checks import check_yaml

app = FastAPI()

# ONNX modeli yükle
model_path = "D:/anaconda/Projeler/wyseye/yolov8/runs/detect/yolo_pawn_model/weights/best.onnx"
ort_session = ort.InferenceSession(model_path)
CLASSES = yaml_load(check_yaml("D:/anaconda/Projeler/wyseye/yolov8/data.yaml"))["names"]

# Model giriş adı
input_name = ort_session.get_inputs()[0].name

def preprocess_image(image_bytes: bytes) -> np.ndarray:
    image = cv2.imdecode(np.frombuffer(image_bytes, np.uint8), cv2.IMREAD_COLOR)
    image = cv2.resize(image, (640, 640))
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = image.transpose(2, 0, 1)
    image = image.astype("float32") / 255.0
    image = np.expand_dims(image, axis=0)
    return image

def process_output(output: np.ndarray, conf_threshold=0.5):
    outputs = np.array([cv2.transpose(output[0])])
    rows = outputs.shape[1]

    boxes, scores, class_ids = [], [], []

    for i in range(rows):
        classes_scores = outputs[0][i][4:]
        _, maxScore, _, (x, maxClassIndex) = cv2.minMaxLoc(classes_scores)
        if maxScore >= conf_threshold:
            box = [
                float(outputs[0][i][0]),
                float(outputs[0][i][1]),
                float(outputs[0][i][2]),
                float(outputs[0][i][3]),
            ]
            boxes.append(box)
            scores.append(float(maxScore))
            class_ids.append(int(maxClassIndex))

    result_boxes = cv2.dnn.NMSBoxes(boxes, scores, conf_threshold, 0.45)

    detections = []
    for i in range(len(result_boxes)):
        index = result_boxes[i]
        detections.append({
            "class_id": class_ids[index],
            "class_name": CLASSES[class_ids[index]],
            "confidence": scores[index],
            "box": {
                "x": boxes[index][0],
                "y": boxes[index][1],
                "width": boxes[index][2],
                "height": boxes[index][3]
            }
        })
    return detections

@app.post("/detect/")
async def detect_objects(file: UploadFile = File(..., media_type="image/jpeg")):
    image_bytes = await file.read()
    image_array = preprocess_image(image_bytes)

    ort_inputs = {input_name: image_array}
    ort_output = ort_session.run(None, ort_inputs)[0]

    detections = process_output(ort_output)

    return {"detections": detections}

@app.get("/pawns/{pawn_id}")
def get_pawn_info(pawn_id: int):
    pawn_classes = {0: "white-pawn", 1: "black-pawn"}
    if pawn_id not in pawn_classes:
        return {"error": "Invalid pawn ID. Use 0 for white-pawn or 1 for black-pawn."}
    return {"pawn_id": pawn_id, "pawn_type": pawn_classes[pawn_id]}

@app.put("/pawns/{pawn_id}")
def update_pawn_info(pawn_id: int, pawn_type: str):
    return {"message": f"Pawn {pawn_id} updated to {pawn_type}."}

@app.delete("/pawns/{pawn_id}")
def delete_pawn_info(pawn_id: int):
    return {"message": f"Pawn {pawn_id} deleted."}
