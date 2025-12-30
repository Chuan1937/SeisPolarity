# SeisPolarity

SeisPolarity 是一个极性拾取框架，目标是提供统一的数据接口、模型封装和评测工具，帮助快速试验和比较极性判读模型。

## 设计思路
- 采用模块化划分：`data` 负责波形读取，`models` 提供统一模型 API，`annotations`/`PickList` 统一拾取结果结构。
- 核心保持轻量，模型可以自由扩展或替换，便于持续迭代。
- 缓存与远端地址均可通过环境变量/配置切换，方便在本地或集群运行。

## 项目结构
- `pyproject.toml`：依赖与打包配置。
- `src/seispolarity/`：核心包代码。
  - `config.py`：缓存与模型注册配置。
  - `annotations.py`：`Pick`/`PickList`/`PolarityLabel` 基础类型。
  - `data/`：波形读取（目前提供本地目录扫描）。
- `models/`：模型基类与可扩展的模型注册机制。

## 快速开始
1. 安装依赖（建议虚拟环境）：
	```bash
	pip install -e .
	```
2. 使用任意实现的模型做极性拾取：
	```python
	import obspy
	from seispolarity.annotations import PickList
	from seispolarity.models.base import BasePolarityModel

	# 加载本地 miniSEED/Stream
	st = obspy.read("example.mseed")

	   # 示例：自定义模型应继承 BasePolarityModel 并实现 forward_tensor/build_picks
	   class DummyPolarityModel(BasePolarityModel):
		   def forward_tensor(self, tensor, **kwargs):
			   return tensor.mean(dim=-1)

		   def build_picks(self, raw_output, **kwargs):
			   return PickList()

	   model = DummyPolarityModel(name="dummy")
	   print(model.annotate(st).picks.to_dataframe())
	```

## 计划与 TODO
- 数据模块：支持远端数据仓库、标准化台站元信息。
- 模型模块：提供纯极性模型、训练脚本与基准；添加阈值/后处理策略。
- 评测模块：极性精度指标、混淆矩阵与可视化。

## 许可与注意
- 本项目主体代码使用 BSD-3-Clause 许可发布。
- 本项目主体代码使用 BSD-3-Clause 许可发布。
