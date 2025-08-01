from ultralytics.data.converter import convert_coco

# 转换COCO训练集标签为YOLO格式
convert_coco(labels_dir="/home/undergraduate/coco2017/annotations", 
             save_dir="/home/undergraduate/coco2017/labels", 
             use_keypoints=False)  # False表示只转换检测标签