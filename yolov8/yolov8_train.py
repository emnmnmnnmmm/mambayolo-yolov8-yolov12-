from ultralytics import YOLO

# 1. 加载官方 YOLOv8n 预训练模型（原生 CNN 架构，无 Mamba 依赖）
model = YOLO("yolov8n.pt")  # 官方模型，自动从 Ultralytics 服务器下载

# 2. 数据集配置（使用 COCO 官方配置，或你的自定义数据集）
# data_config = "coco.yaml"  # 官方库会自动识别内置的 COCO 配置
# 若使用自定义数据集，替换为你的 data.yaml 路径，例如：
data_config = "/home/undergraduate/yolov8/ultralytics/cfg/datasets/coco.yaml"

# 3. 训练模型（官方参数，无 Mamba 相关依赖）
results = model.train(
    data=data_config,         # 数据集配置
    epochs=100,               # 训练轮次
    imgsz=640,                # 输入图像尺寸
    batch=16,                 # 批次大小（根据 GPU 显存调整）
    workers=8,                # 数据加载线程数
    optimizer="SGD",         # 优化器
    device='0',               # 使用 GPU 0（若用 CPU 则设为 'cpu'）
    amp=True,                 # 启用自动混合精度训练
    project="/home/undergraduate/yolov8/runs/coco",  # 输出项目目录
    name="train4",  # 实验名称
    # resume=True,  # 如需从断点续训，取消注释
)

# 4. （可选）训练完成后在验证集上评估
metrics = model.val()
print("验证集指标：", metrics)