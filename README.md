# AttentionVRP
原论文：<Attention, Learn to Solve Routing Problems!>

## 重构了整个项目
- 新增Time Window
硬时间窗
- 新增capacity
一样的车型，所有车都为相同容量

## next week plan(预计2021年11月28日)
1、对应Solomon数据集，预处理数据
- 考虑将所有时间归一化至1
（ready time 或due time 除以service time）

- 考虑将所有demand归一化至1
（demand 除以 capacity）

- 考虑将所有坐标归一化至1
（XCOORD 或 YCOORD除以100）

2、更改loss函数
- 车辆配载不均衡
