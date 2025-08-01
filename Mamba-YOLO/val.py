import os
import argparse
from ultralytics import YOLO
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
import json
from tqdm import tqdm

def parse_args():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(description='使用 YOLOv8 模型对 COCO 验证集进行测试并评估')
    parser.add_argument('--model', type=str, default='yolov8n.pt', help='模型路径，默认为 yolov8n.pt')
    parser.add_argument('--data_dir', type=str, required=True, help='COCO 数据集根目录')
    parser.add_argument('--save_dir', type=str, default='./results', help='结果保存目录')
    parser.add_argument('--conf', type=float, default=0.001, help='置信度阈值，默认为 0.001')
    parser.add_argument('--iou', type=float, default=0.6, help='IOU 阈值，默认为 0.6')
    return parser.parse_args()

def main():
    """主函数：运行模型推理并评估性能"""
    args = parse_args()
    
    # 创建保存目录
    os.makedirs(args.save_dir, exist_ok=True)
    
    # 加载模型
    model = YOLO(args.model)
    
    # COCO 验证集路径
    val_images_dir = os.path.join(args.data_dir, 'val2017')
    annotations_file = os.path.join(args.data_dir, 'annotations', 'instances_val2017.json')
    
    # 检查文件是否存在
    if not os.path.exists(val_images_dir):
        print(f"错误：验证集图像目录 {val_images_dir} 不存在!")
        return
    
    if not os.path.exists(annotations_file):
        print(f"错误：标注文件 {annotations_file} 不存在!")
        return
    
    # 加载 COCO 验证集标注
    coco_gt = COCO(annotations_file)
    
    # 获取所有图像 ID
    img_ids = list(coco_gt.imgs.keys())
    print(f"找到 {len(img_ids)} 张验证集图像")
    
    # COCO 类别映射（YOLOv8 的类别 ID 到 COCO 类别 ID 的映射）
    coco_category_map = {
        0: 1,    # 人
        1: 2,    # 自行车
        2: 3,    # 汽车
        3: 4,    # 摩托车
        4: 5,    # 飞机
        5: 6,    # 公交车
        6: 7,    # 火车
        7: 8,    # 卡车
        8: 9,    # 船
        9: 10,   # 交通灯
        10: 11,  # 消防栓
        11: 13,  # 停止标志
        12: 14,  # 停车计费器
        13: 15,  # 长凳
        14: 16,  # 鸟
        15: 17,  # 猫
        16: 18,  # 狗
        17: 19,  # 马
        18: 20,  # 羊
        19: 21,  # 牛
        20: 22,  # 大象
        21: 23,  # 熊
        22: 24,  # 斑马
        23: 25,  # 长颈鹿
        24: 27,  # 背包
        25: 28,  # 雨伞
        26: 31,  # 手提包
        27: 32,  # 领带
        28: 33,  # 行李箱
        29: 34,  # 飞盘
        30: 35,  # 滑雪板
        31: 36,  # 滑雪板
        32: 37,  # 球
        33: 38,  # 风筝
        34: 39,  # 棒球棒
        35: 40,  # 棒球手套
        36: 41,  # 滑板
        37: 42,  # 冲浪板
        38: 43,  # 网球拍
        39: 44,  # 瓶子
        40: 46,  # 红酒杯
        41: 47,  # 杯子
        42: 48,  # 叉子
        43: 49,  # 刀
        44: 50,  # 勺子
        45: 51,  # 碗
        46: 52,  # 香蕉
        47: 53,  # 苹果
        48: 54,  # 三明治
        49: 55,  # 橙子
        50: 56,  # 西兰花
        51: 57,  # 胡萝卜
        52: 58,  # 热狗
        53: 59,  # 披萨
        54: 60,  # 甜甜圈
        55: 61,  # 蛋糕
        56: 62,  # 椅子
        57: 63,  # 沙发
        58: 64,  # 盆栽植物
        59: 65,  # 床
        60: 67,  # 餐桌
        61: 70,  # 厕所
        62: 72,  # 电视
        63: 73,  # 笔记本电脑
        64: 74,  # 鼠标
        65: 75,  # 遥控器
        66: 76,  # 键盘
        67: 77,  # 手机
        68: 78,  # 微波炉
        69: 79,  # 烤箱
        70: 80,  # 烤面包机
        71: 81,  # 水槽
        72: 82,  # 冰箱
        73: 84,  # 书
        74: 85,  # 时钟
        75: 86,  # 花瓶
        76: 87,  # 剪刀
        77: 88,  # 泰迪熊
        78: 89,  # 吹风机
        79: 90   # 牙刷
    }
    
    # 存储预测结果
    results = []
    
    # 对每张图像进行推理
    for img_id in tqdm(img_ids, desc="正在推理"):
        img_info = coco_gt.loadImgs(img_id)[0]
        img_path = os.path.join(val_images_dir, img_info['file_name'])
        
        # 运行推理
        predictions = model.predict(
            source=img_path,
            conf=args.conf,
            iou=args.iou,
            save=False,
            verbose=False
        )
        
        # 处理预测结果
        for pred in predictions:
            boxes = pred.boxes.cpu().numpy()
            
            for box in boxes:
                x1, y1, x2, y2 = box.xyxy[0].tolist()  # 边界框坐标 [x1, y1, x2, y2]
                conf = float(box.conf)  # 置信度
                cls_id = int(box.cls)   # 类别 ID（YOLOv8 格式）
                
                # 转换为 COCO 格式的边界框 [x, y, width, height]
                width = x2 - x1
                height = y2 - y1
                
                # 构建 COCO 格式的结果项
                result = {
                    "image_id": img_id,
                    "category_id": coco_category_map[cls_id],
                    "bbox": [round(x1, 2), round(y1, 2), round(width, 2), round(height, 2)],
                    "score": round(conf, 3)
                }
                
                results.append(result)
    
    # 保存预测结果到 JSON 文件
    results_file = os.path.join(args.save_dir, 'predictions.json')
    with open(results_file, 'w') as f:
        json.dump(results, f)
    
    print(f"推理完成! 预测结果已保存到 {results_file}")
    
    # 加载预测结果
    coco_dt = coco_gt.loadRes(results_file)
    
    # 创建评估对象并运行评估
    coco_eval = COCOeval(coco_gt, coco_dt, 'bbox')  # 'bbox' 用于目标检测评估
    coco_eval.evaluate()
    coco_eval.accumulate()
    
    # 打印评估结果
    print("\n评估结果:")
    coco_eval.summarize()
    
    # 保存评估结果到文本文件
    metrics_file = os.path.join(args.save_dir, 'evaluation_metrics.txt')
    with open(metrics_file, 'w') as f:
        # 重定向标准输出到文件
        import sys
        old_stdout = sys.stdout
        sys.stdout = f
        coco_eval.summarize()
        sys.stdout = old_stdout
    
    print(f"评估指标已保存到 {metrics_file}")

if __name__ == "__main__":
    main()    