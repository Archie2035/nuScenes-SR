# nuScenes-SR Dataset

标签数据已经准备好，但是目前未完全通过审批流程，所以先将数据放到了 .gitignore 中，等完成审批后再push，敬请谅解。

## 数据集简介

nuScenes-SR（nuScenes Scene Recognition）是基于著名的nuScenes自动驾驶数据集构建的场景识别数据集。该数据集专门针对自动驾驶场景理解任务进行了精心标注，涵盖了自动驾驶中最常见和最具挑战性的10种典型场景。

## 场景分类

数据集包含两大类场景，共10种具体场景类型：

### 动态场景（5种）

动态场景主要关注涉及移动物体和驾驶决策的复杂情况：

| 场景类别 | 英文标识 | 判断标准 |
|---------|---------|---------|
| 行人穿越 | PED_CROSSING | 行人在车辆前方横向移动，可能阻碍车辆正常行驶 |
| 左转场景 | LEFT_TURN | 自车执行左转操作，需要处理对向来车和行人 |
| 右转场景 | RIGHT_TURN | 自车执行右转操作，需要注意右侧盲区 |
| 施工车辆 | CONSTRUCTION_VEHICLE | 场景中存在施工车辆或工程设备 |
| 避让静止车辆 | AVOID_STATIONARY | 前方存在静止车辆，自车需要变道绕行 |

### 静态场景（5种）

静态场景主要关注环境条件和道路基础设施：

| 场景类别 | 英文标识 | 判断标准 |
|---------|---------|---------|
| 交叉路口 | INTERSECTION | 车辆通过交叉路口区域 |
| 停车场 | PARKING_LOT | 场景发生在停车场内部 |
| 交通信号灯 | TRAFFIC_LIGHT | 场景中存在交通信号灯控制 |
| 雨天天气 | RAINY_WEATHER | 在降雨天气条件下的驾驶场景 |
| 施工区域 | CONSTRUCTION_ZONE | 车辆经过道路施工区域 |

## 数据集特点

- **基于nuScenes**: 构建在业界知名的nuScenes数据集基础上，保证了数据质量和真实性
- **场景多样性**: 涵盖动态和静态两大类共10种典型自动驾驶场景
- **实用性强**: 选择的场景都是自动驾驶系统中最常遇到且最具挑战性的情况
- **标注精确**: 每个场景都有明确的判断标准，确保标注的一致性和准确性

## 应用场景

该数据集适用于以下研究和应用：

- **场景识别算法开发**: 训练和测试自动驾驶场景识别模型
- **驾驶行为分析**: 分析不同场景下的驾驶模式和决策策略
- **安全性评估**: 评估自动驾驶系统在不同场景下的安全性能
- **算法验证**: 验证场景理解算法的准确性和鲁棒性

## 数据结构

```
nuScenes-SR/
├── dataset/
│   ├── merged_final_labels_reviewed.json  # 场景标注文件
│   └── readme.md                          # 数据说明文档
├── dataloader.py                          # Python数据加载器
├── example_usage.py                       # 使用示例脚本
├── LICENSE                                # 许可证文件
└── readme.md                             # 项目说明文档
```

## 数据格式说明

标注文件 `merged_final_labels_reviewed.json` 采用JSON格式，每个样本包含以下字段：

```json
{
  "0053e9c440a94c1b84bd9c4223efc4b0": {
    "labels": [
      "PED_CROSSING",
      "RIGHT_TURN",
      "INTERSECTION",
      "TRAFFIC_LIGHT"
    ],
    "description": "Truck, light turns green, peds, parked cars, right turn"
  }
}
```

### 字段说明

- **键值（Key）**: `"0053e9c440a94c1b84bd9c4223efc4b0"`
  - nuScenes数据集中的场景token，用于唯一标识每个场景
  
- **labels**: `["PED_CROSSING", "RIGHT_TURN", "INTERSECTION", "TRAFFIC_LIGHT"]`
  - 场景标签列表，包含该场景中出现的所有场景类型
  - 每个标签对应上述10种场景类型中的一种或多种
  - 一个场景可以同时包含多个场景类型（如示例中同时包含行人穿越、右转、交叉路口和交通信号灯）
  
- **description**: `"Truck, light turns green, peds, parked cars, right turn"`
  - 场景的文字描述，来自nuScenes数据集原有的场景描述
  - 提供了场景的额外上下文信息，有助于理解场景内容

## 使用方法

### 基本使用

1. 下载完整的nuScenes数据集
2. 使用本数据集提供的标注文件进行场景识别训练
3. 根据场景标识和判断标准进行模型验证

### 使用数据加载器

我们提供了便捷的Python数据加载器 `dataloader.py`，支持多种数据操作：

```python
from dataloader import NuScenesSRDataloader

# 初始化数据加载器
loader = NuScenesSRDataloader()

# 打印数据集统计信息
loader.print_statistics()

# 获取特定标签的场景
ped_scenes = loader.get_scenes_by_label("PED_CROSSING")
print(f"找到 {len(ped_scenes)} 个行人穿越场景")

# 获取包含多个标签的场景
complex_scenes = loader.get_scenes_by_labels(
    ["INTERSECTION", "TRAFFIC_LIGHT"], mode="all"
)

# 随机采样场景
sample_scenes = loader.sample_scenes(n=10, random_seed=42)

# 数据集划分
train_tokens, val_tokens, test_tokens = loader.split_dataset(
    train_ratio=0.7, val_ratio=0.15, test_ratio=0.15, random_seed=42
)

# 导出特定场景的标签
loader.export_labels_only("train_labels.json", train_tokens)
```

### 主要功能

- **场景筛选**: 根据单个或多个标签筛选场景
- **统计分析**: 提供详细的数据集统计信息
- **数据划分**: 支持训练/验证/测试集划分
- **随机采样**: 支持随机采样和种子设置
- **标签导出**: 导出纯标签数据用于机器学习训练

## 致谢

Most thanks to nuScenes dataset:

[nuScenes](https://www.nuscenes.org/)

## 许可证

请参考LICENSE文件了解数据集的使用许可。

## 贡献者

如有问题或建议，欢迎提交Issue或Pull Request。