from ultralytics import YOLO

# 加载一个预训练模型 (如果本地没有，会自动下载)
model = YOLO('yolov8m_obb.pt')

# 使用模型进行预测
results = model.predict(source='your_image.jpg')