from ultralytics import YOLO

model = YOLO("yolo11n.pt") 
model.predict(source="./temp/masked_video.mp4", show=True, save=False, conf=0.2, classes=[8], tracker='botsort.yaml', iou=0.5)  # class 8 is boat