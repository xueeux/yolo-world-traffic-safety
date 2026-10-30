from tensorboard.backend.event_processing import event_accumulator
import pandas as pd

# 设置日志文件路径
log_path = "D:/subject/semester3.2/dl/YOLO-World-master/work_dirs/tb_logs/events.out.tfevents.1751182076.XUEEUX.32724.0"

# 加载事件文件
ea = event_accumulator.EventAccumulator(log_path)
ea.Reload()

# 获取所有指标标签
metrics = ea.Tags()['scalars']
print("可用指标:", metrics)

# 提取特定指标数据
def extract_metric(metric_name):
    events = ea.Scalars(metric_name)
    return [(e.step, e.wall_time, e.value) for e in events]

# 提取并打印关键指标
mAP_data = extract_metric('coco/bbox_mAP')
precision_data = {cls: extract_metric(f'coco/{cls}_precision') for cls in ['bicycle', 'bus', 'car', 'motorbike', 'person']}

# 打印结果
print("\n==== 整体mAP ====")
for step, time, value in mAP_data:
    print(f"Step {step}: mAP={value:.3f}")

print("\n==== 各类别精度 ====")
for cls, data in precision_data.items():
    last_value = data[-1][2] if data else None
    print(f"{cls}: {last_value:.3f}")

max_step = max([e.step for e in ea.Scalars('coco/bbox_mAP')])

# 提取特定步数的指标
def get_metric_at_step(metric_name, target_step):
    events = ea.Scalars(metric_name)
    for e in events:
        if e.step == target_step:
            return e.value
    return None

# 获取最终指标
final_mAP = get_metric_at_step('coco/bbox_mAP', max_step)
final_precision = {
    cls: get_metric_at_step(f'coco/{cls}_precision', max_step)
    for cls in ['bicycle', 'bus', 'car', 'motorbike', 'person']
}

# 打印最终结果
print(f"\n==== 最终训练结果 (Step {max_step}) ====")
print(f"整体mAP: {final_mAP:.3f}")
print("各类别精度:")
for cls, value in final_precision.items():
    print(f"  {cls}: {value:.3f}")

# 额外获取其他关键指标
mAP_50 = get_metric_at_step('coco/bbox_mAP_50', max_step)
mAP_75 = get_metric_at_step('coco/bbox_mAP_75', max_step)
mAP_large = get_metric_at_step('coco/bbox_mAP_l', max_step)
mAP_middle = get_metric_at_step('coco/bbox_mAP_m', max_step)
mAP_small = get_metric_at_step('coco/bbox_mAP_s', max_step)
print(f"\n附加指标:")
print(f"  mAP_50: {mAP_50:.3f}")
print(f"  mAP_75: {mAP_75:.3f}")
print(f"  小目标mAP: {mAP_small:.3f}")
print(f"  中等目标mAP: {mAP_middle:.3f}")
print(f"  大目标mAP: {mAP_large:.3f}")
print(f"\n效率指标:")
data_time = get_metric_at_step('data_time', max_step)
time = get_metric_at_step('time', max_step)
print(f"  数据加载时间（秒）: {data_time:.3f}")
print(f"  数据推理时间（秒）: {time:.3f}")