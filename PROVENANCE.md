# 代码来源与公开适配

快照日期：2026-10-16。所有模块来自既有项目中的实际实现，未用一个新建演示系统替代原项目。

| 公开文件 | 原源码位置 | 适配范围 |
| --- | --- | --- |
| `reservoir_workbench/resampling.py` | `scripts/personal_reservoir/resampling.py` | 保留完整采样实现；调整标识工具导入 |
| `reservoir_workbench/sample_library.py` | `scripts/personal_reservoir/sample_library.py` | 保留样本关系、唯一键和资格校验；把现场井名单及目标排除名单改为显式参数 |
| `reservoir_workbench/model_compare.py` | `scripts/personal_reservoir/model_compare.py` | 保留前部通用模型、变换、指标、同队列整井留出及拟合函数；不复制后部现场模型阶段编排、路径和制品逻辑 |
| `reservoir_workbench/identifiers.py` | `scripts/personal_reservoir/segments.py` | 抽取通用标识标准化函数，移除现场特定别名规则 |
| `tests/test_resampling.py` | `tests/test_personal_reservoir_resampling.py` | 复用原合成测试，标识改为虚构值 |
| `tests/test_sample_library.py` | `tests/test_personal_reservoir_sample_library.py` | 复用原合成测试，显式传入虚构名单 |
| `tests/test_model_compare.py` | `tests/test_personal_reservoir_model_compare.py` | 选取通用模型比较相关测试和原合成样本生成器；不复制依赖真实制品的阶段测试 |

模型公开适配同时移除了未使用的现场模块导入，将审查井排除名单参数化，将缺少训练资格的默认值收紧为 False，并在整井留出入口拒绝少于两口合格井的输入。原有模型算法、折内预处理和误差计算逻辑保留。

19 个测试覆盖大缺口与层界限制、标签来源、非冻结状态拒绝、重复键、审核井资格、字符串布尔值、目标级排除、同队列留出比较、特征缺失和类别审核。测试与示例输出仅是软件验证，均不构成现场精度报告。

公开资料未包含原始测井/实验数据、真实井标识、地理信息、合作方材料、报告/PPT、实际预测结果、模型权重、个人本机路径或原 Git 历史。
