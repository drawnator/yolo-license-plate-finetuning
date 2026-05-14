from ultralytics import YOLO

model = YOLO('yolov12s.pt')
model.export(format='coreml', nms=True)
# model.export(format='mlpackage', imgsz=640, batch=1, device='cpu')