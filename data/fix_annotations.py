import json
import os
from datetime import datetime

def add_coco_metadata(file_path, roboflow_metadata):
    """为标注文件添加完整的COCO格式元数据"""
    with open(file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    # 添加info字段
    if "info" not in data:
        data["info"] = {
            "description": roboflow_metadata.get("project", "Street View Dataset"),
            "url": roboflow_metadata.get("url", ""),
            "version": str(roboflow_metadata.get("version", "1.0")),
            "year": datetime.now().year,
            "contributor": "Roboflow User: " + roboflow_metadata.get("workspace", "fsmvu"),
            "date_created": datetime.now().strftime("%Y-%m-%d")
        }
    
    # 添加licenses字段
    if "licenses" not in data:
        data["licenses"] = [{
            "url": "https://creativecommons.org/licenses/by/4.0/",
            "id": 1,
            "name": "CC BY 4.0" if roboflow_metadata.get("license") == "CC BY 4.0" else "Unknown License"
        }]
    
    # 确保categories中的ID是连续的整数
    if "categories" in data:
        for idx, cat in enumerate(data["categories"]):
            cat["id"] = idx  # 强制重写ID为0-based连续整数
    
    # 保存修复后的文件
    with open(file_path, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
    print(f"Successfully fixed: {file_path}")

if __name__ == "__main__":
    # Roboflow数据集元数据
    ROBOFLOW_METADATA = {
        "workspace": "fsmvu",
        "project": "street-view-gdogo",
        "version": 1,
        "license": "CC BY 4.0",
        "url": "https://universe.roboflow.com/fsmvu/street-view-gdogo/dataset/1"
    }
    
    # 修复所有标注文件
    DATA_ROOT = 'data/traffic/annotations'
    for file_name in ['train.json', 'val.json']:
        file_path = os.path.join(DATA_ROOT, file_name)
        if os.path.exists(file_path):
            add_coco_metadata(file_path, ROBOFLOW_METADATA)
        else:
            print(f"Warning: File not found - {file_path}")

    print("\n修复完成！请重新运行训练脚本。")