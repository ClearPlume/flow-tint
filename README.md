# Flow Tint - 自然语言颜色调整系统

让AI理解"深一点的森林绿"。

## 这个项目证明了什么

**我的工作层级是系统架构设计，而非底层算法实现。**

具体来说：

- 使用PyTorch标准组件（TransformerEncoderLayer/DecoderLayer）
- 设计完整的技术路径：需求分析 → 架构设计 → 创新方案 → 实现
- 在标准方案上叠加任务特定的创新

这个项目展现了三个核心能力：

1. **需求到架构的映射能力** - 分析"让AI理解颜色调整"需要什么，设计完整实现路径
2. **基于原理的工程判断能力** - 理解Transformer原理，知道什么该复用、什么该自己设计
3. **创新方案设计能力** - DSL系统、×1000取整、DropoutScheduler

---

## 核心设计决策

### 架构选择：Encoder-Decoder

**推理链条：**

```
任务特性：seq2seq（自然语言 → 结构化DSL）
  ↓
需要分离理解和生成
  ↓
encoder 理解输入语义 + decoder 生成结构化输出
  ↓
选择 Transformer encoder-decoder（8层，512维，60.6M参数）
```

**不选择 decoder-only 的理由：**

- 这是明确的seq2seq任务（输入和输出是两个独立序列）
- Encoder-Decoder架构上分离输入理解和输出生成
- Encoder可以双向理解完整输入，再由decoder基于此生成
- Decoder-only需要把输入输出拼接在同一序列，对这个任务而言，语义不够清晰

---

### 色彩空间：OKLCh

**推理链条：**

```
需求：符合人类感知的颜色调整

评估：
  RGB
    ✗ 不符合人类直觉（"亮一些" ≠ R+10, G+10, B+10）
    ✗ 通道耦合（调亮度会影响色相）
  
  HSV / HSL
    ✗ 存在非线性问题
    ✗ 饱和度和明度不是感知均匀的
  
  OKLCh
    ✓ 感知均匀（ΔE ≈ 欧氏距离）
    ✓ L/C/h 独立可调
    ✓ L: 亮度, C: 色度, h: 色相

选择：OKLCh
```

**数值处理：**

```
OKLCh 坐标（感知空间）: (0.558, 0.169, 142.910)
  ↓ ×1000取整
中间 表示（离散空间）: 558 169 142910
  ↓ 执行时还原
最终 RGB 输出: (58, 124, 95)
```

---

### 输出表示：DSL而非直接OKLCh

**推理链条：**

**第一步：不直接输出坐标**

```
问题：模型应该输出什么？

方案A：直接输出最终OKLCh坐标
  问题：无法表达调整的推理过程
  
方案B：输出操作序列
  优势：符合人类思维（"深一点" = 降低亮度的操作）

选择：方案B
```

**第二步：不直接输出小数坐标**

```
OKLCh坐标是小数：(0.558, 0.169, 142.910)

方案A：直接输出小数
  代价：连续空间，学习难度大
  
方案B：×1000取整为整数
  优势：离散空间，降低难度
  代价：精度损失（1/1000颗粒度）

选择：方案B（在颜色调整场景下，1/1000精度足够）
```

**第三步：借鉴CoT（思维链）概念**

```
启发：让模型输出"推理过程"，而非直接结果

类比：
  数学题：不只输出答案，而是输出解题步骤
  颜色调整：不只输出最终坐标，而是输出调整步骤
```

**第四步：不输出自然语言，而用结构化**

```
CoT通常用自然语言：
  "首先设置基准色为蓝色，然后降低亮度0.02"
  
问题：
  - 自然语言冗余（token数量大）
  - 难以精确解析执行
  
改进：用DSL（领域特定语言）
  color 452 313 264059
  lightness $p -0.02
  
优势：
  - 精确、可执行
  - token数量少
```

**第五步：极致精简输出要素**

```
DSL设计原则：只保留必要信息

最终设计：
  操作符 + 参数
  color 452 313 264059        # 设置颜色
  lightness $p -0.021         # 调整亮度
  chroma $p 0.041             # 调整色度
  
去除冗余：
  ✗ 不用自然语言
  ✗ 不用括号、逗号等分隔符
  ✓ 空格分隔，简洁高效
```

**DSL语法：**

```bnf
<dsl> ::= <instruction>+

# ($p: 上一步操作的结果)
<instruction> ::= 
    | "color" <int> <int> <int>           # 设置颜色 (OKLCh坐标，L C h)
    | "lightness" "$p" <float>            # 调整亮度 (相对值)
    | "chroma" "$p" <float>               # 调整色度 (相对值)
    | "hue" "$p" <float>                  # 调整色相 (相对值)
    | "mix" "$p" "$p" <float>             # 混合颜色 (比例)
    | "temperature" "$p" <float>          # 调整色温 (冷白/暖白混合，相对值)

<int> ::= [0-9]+                          # OKLCh坐标×1000取整的整数
<float> ::= [-+]?[0-9]+\.[0-9]+           # 调整量
```

---

### 动态正则化：DropoutScheduler

**设计理念：**

```
监测过拟合信号（训练损失下降的同时，验证损失上升）
  ↓
自动调整 dropout 强度
  ↓
减少手动调参
```

**实现逻辑：**

- 连续N轮，训练损失下降的同时，验证损失上升 → 提高dropout
- 验证损失改善 → 降低dropout
- 动态范围：0.1 ~ 0.5

**不选择固定dropout的理由：**

- 训练初期：过拟合风险低，低dropout加速学习
- 训练后期：过拟合风险高，高dropout防止过拟合
- DropoutScheduler自动适应训练阶段

---

## 训练结果

### 最终指标（Epoch 14/50，early stopping）

```
训练损失: 0.66851
验证损失: 0.66767
Token准确率: 93.06%
序列准确率: 11.46%
```

**关于序列准确率的说明：**

序列准确率低（11.46%）**不代表模型质量差**，原因在于：

1. **训练数据的数值随机性**：
    - 同一个修饰词（如"稍微亮一些"），对应的调整量从预定义集合中随机选择
    - 选中后还会叠加 ±10% 的随机波动
    - 示例：[0.02, 0.03, 0.04, 0.05, 0.06] → 选中0.05 → 波动后0.054

2. **评估标准的严苛性**：
    - 序列准确率要求每个token完全匹配
    - 数值稍有偏差即判定为错误

3. **实际质量的验证方式**：
    - 能生成完全正确的序列（序列准确率 > 0%）✓
    - 生成的DSL语义正确（操作符、参数类型正确）✓
    - Token准确率高（93%）✓

**示例：**

```
输入："深一些的蓝"
标注DSL：  lightness $p -0.021
模型生成：  lightness $p -0.019

→ 序列不匹配（数值不同）
→ 但语义正确（都是"降低亮度"且幅度合理）
→ 实际颜色差异人眼无法分辨
```

对于当前阶段，设计考虑数值接近度的soft metric投入产出比不高。**Token准确率（93%）已足够反映模型质量**。

---

## 推理示例

### 基础颜色

```
输入: "蓝"
DSL:  color 452 313 264059
RGB:  (0, 112, 255)
```

### 简单修饰

```
输入: "淡蓝"
DSL:  color 975 13 244379
RGB:  (233, 244, 255)

输入: "深蓝"
DSL:  color 288 199 264059
RGB:  (0, 51, 153)
```

### 相对调整（带上下文）

```
上下文: "蓝" → color 452 313 264059
输入:   "深一些"
DSL:    color 452 313 264059
        lightness $p -0.021
RGB:    (0, 96, 230)
```

### 连续修饰（带上下文）

```
上下文: "蓝" → color 452 313 264059
       "深一些" → lightness $p -0.021
输入:   "艳一点"
DSL:    color 452 313 264059
        lightness $p -0.01
        chroma $p 0.041
RGB:    (0, 108, 255)
```

---

## 技术栈

- **模型架构**: Transformer encoder-decoder (8层, 512维, 60.6M参数)
- **框架**: PyTorch 2.x + transformers
- **包管理**: [uv](https://github.com/astral-sh/uv)
- **色彩空间**: OKLCh（感知均匀）
- **训练策略**: AdamW + DropoutScheduler + Early Stopping
- **输出格式**: 自定义DSL

---

## 项目结构

```
flow-tint/
├── .gitignore                          # Git忽略规则
├── pyproject.toml                      # 项目配置（uv）
├── uv.lock                             # 依赖锁定文件
├── README.md                           # 项目文档
│
├── src/flow_tint/                      # 源代码
│   ├── __init__.py
│   ├── path.py                         # 路径管理工具
│   │
│   ├── core/                           # 核心工具模块
│   │   ├── constants.py                # 常量定义
│   │   └── utils.py                    # 工具函数
│   │
│   ├── model/                          # 模型模块
│   │   ├── trainer.py                  # 训练入口
│   │   ├── tester.py                   # 推理入口
│   │   │
│   │   └── component/                  # 模型组件
│   │       ├── batch_sampler.py        # 批次采样器
│   │       ├── color_dataset.py        # 数据集类
│   │       ├── color_model.py          # Transformer模型
│   │       ├── dropout_scheduler.py    # 动态dropout调度器
│   │       ├── history.py              # 训练历史记录
│   │       ├── model_config.py         # 模型配置
│   │       ├── positional_encoding.py  # 位置编码
│   │       ├── tokenizer.py            # Tokenizer
│   │       └── trainer.py              # 训练器
│   │
│   └── dsl/                            # DSL系统（待实现）
│       ├── spec.py                     # DSL语法规范
│       ├── parser.py                   # DSL解析器
│       ├── executor.py                 # DSL执行器
│       └── color_ops.py                # 颜色空间转换
│
├── data/                               # 数据目录
│   ├── seed/                           # 种子数据
│   │   └── stage1-data.json            # 1321条颜色名称→RGB映射
│   │
│   ├── samples/                        # 生成的语料（未纳入git）
│   │   ├── stage1_samples.txt          # 阶段1：基础颜色
│   │   ├── stage2_easy_samples.txt     # 阶段2：简单修饰
│   │   ├── stage2_complex_samples.txt  # 阶段2：复杂修饰
│   │   ├── stage2_samples.txt          # 阶段2：合并
│   │   ├── stage3_samples.txt          # 阶段3：上下文相对调整
│   │   └── samples.txt                 # 全部语料合并
│   │
│   └── vocab/                          # 词表（未纳入git）
│       ├── color_semantic.model        # SentencePiece模型
│       └── color_semantic.vocab        # 词表文件
│
├── scripts/                            # 脚本工具
│   ├── samples/                        # 语料生成脚本
│   │   ├── stage1_generator.py         # 生成阶段1语料
│   │   ├── stage1_report.py            # 阶段1统计报告
│   │   ├── stage2_simple_generator.py  # 生成阶段2简单修饰
│   │   ├── stage2_complex_generator.py # 生成阶段2复杂修饰
│   │   ├── stage2_generator.py         # 合并阶段2语料
│   │   └── stage3_generator.py         # 生成阶段3语料
│   │
│   └── vocab/                          # 词表构建脚本
│       └── build_vocabulary.py         # 构建SentencePiece词表
│
├── models/                             # 训练好的模型
│   ├── color_semantic_0.1.0.pt         # 版本模型
│   └── checkpoints/                    # 训练检查点
│       └── color_semantic_*.pt         # 训练过程中保存的最佳模型
│
├── logs/                               # 训练日志（未纳入git）
│   ├── training_*.log                  # 实时训练日志
│   ├── training_*_history.json         # 每次训练指标
│   └── training_*_report.txt           # 每次训练总结报告
│
├── examples/                           # 推理示例（待实现）
│   ├── inference.py                    # 单次推理示例
│   └── batch_inference.py              # 批量推理示例
│
└── tests/                              # 单元测试（待实现）
    ├── test_dsl_parser.py              # DSL解析测试
    ├── test_dsl_executor.py            # DSL执行测试
    ├── test_color_ops.py               # 颜色转换测试
    ├── test_dropout_scheduler.py       # Dropout调度器测试
    ├── test_tokenizer.py               # Tokenizer测试
    └── test_positional_encoding.py     # 位置编码测试
```

---

## 快速开始

### 安装依赖

本项目使用 [uv](https://github.com/astral-sh/uv) 管理依赖。

```bash
uv sync
```

### 准备数据

> **注意**：如果只想使用已训练的模型进行推理，可以跳过此步骤。

项目仅提供种子数据：

- [stage1-data.json](data/seed/stage1-data.json) - 1321条颜色名称到RGB坐标的映射

#### 生成语料

```bash
# 阶段1：基础颜色
python scripts/samples/stage1_generator.py

# 阶段2：修饰颜色
python scripts/samples/stage2_simple_generator.py
python scripts/samples/stage2_complex_generator.py
python scripts/samples/stage2_generator.py

# 阶段3：上下文相对调整
python scripts/samples/stage3_generator.py
```

生成的语料保存在[samples](data/samples)下，未纳入git。

#### 构建词表

```bash
# 生成SentencePiece词表
python scripts/vocab/build_vocabulary.py
```

生成的词表保存在[vocab](data/vocab)下，未纳入git。

### 推理

```python
import torch

from src.flow_tint.model.tester import Tester
from src.flow_tint.path import get_models_dir, get_data_dir

tester = Tester(
    model_path=get_models_dir() / 'checkpoints' / 'color_semantic_20251006_194128.pt',
    tokenizer_path=f"{get_data_dir()}vocab/color_semantic.model",
    device='cuda' if torch.cuda.is_available() else 'cpu'
)
result = tester.infer("深一点的森林绿")
# → DSL: color 558 169 142910<sep>lightness $p -0.046
# → RGB: (58, 124, 95)
```

---

## 待完善清单

### 🔴 核心功能

- [ ] **DSL系统实现**
    - [spec.py](src/flow_tint/dsl/spec.py) - DSL语法定义和常量（操作符、坐标范围）
    - [parser.py](src/flow_tint/dsl/parser.py) - 解析DSL文本为指令对象
    - [executor.py](src/flow_tint/dsl/executor.py) - 执行DSL指令链
    - [color_ops.py](src/flow_tint/dsl/color_ops.py) - 颜色空间转换（OKLCh ↔ RGB）
    - [\_\_init__.py](src/flow_tint/dsl/__init__.py) - 暴露公开API
- [ ] **推理脚本**
    - [inference.py](examples/inference.py) - 单次推理示例，依赖DSL
    - [batch_inference.py](examples/batch_inference.py) - 批量测试验证

### 🟡 质量提升

- [ ] **单元测试**
    - [test_dsl_parser](tests/test_dsl_parser.py) - DSL解析（有效/无效/边界）
    - [test_dsl_executor](tests/test_dsl_executor.py) - DSL执行（单步/链式/$p传递）
    - [test_color_ops](tests/test_color_ops.py) - 颜色转换精度
    - [test_dropout_scheduler](tests/test_dropout_scheduler.py) - 过拟合检测和调整逻辑
    - [test_tokenizer](tests/test_tokenizer.py) - 编码/解码一致性
    - [test_positional_encoding](tests/test_positional_encoding.py) - 位置编码生成
- [ ] **训练器文件管理优化**
    - [trainer.py](src/flow_tint/model/component/trainer.py)
        - History管理：一次训练应只有一个history.json文件，每次epoch后更新内容，而非每个最佳epoch创建新文件
        - Checkpoint管理：保存新的最佳checkpoint时，自动删除同一训练中的旧checkpoint
- [ ] **训练报告图表优化**
    - [history.py](src/flow_tint/model/component/history.py)
        - `generate_ascii_chart`：修复文本图表绘制逻辑（当前为堆叠图，应改为趋势折线图）
        - 添加X轴标注（epoch数），使图表能清晰展示不同epoch的变化趋势
        - 当前所有`█`堆在左侧，无法看出指标随epoch的变化
        - `generate_event_timeline:219`：修复丢弃率变化循环索引错误（应从1开始，避免i=0时访问dropouts[-1]）
- [ ] **代码清理**
    - 移除调试用print语句
    - 运行 `ruff format` 统一代码风格
    - 补全核心函数docstring（ColorModel、Trainer、Tokenizer）
    - 添加类型标注（公开API）

### 🟢 文档完善

- [ ] **DSL文档**
    - [dsl_spec.md](docs/dsl_spec.md) - DSL语法规范、操作符定义、示例
- [ ] **数据文档**
    - [data_generation.md](docs/data_generation.md) - stage1-3生成逻辑、随机性说明
- [ ] **项目文档**
    - 补全README推理示例截图
    - 添加训练loss曲线图

---

## 后续探索

这个项目解决了"让AI理解基础颜色和简单修饰"的问题，但留下了新的疑问：

- **外置程序真的必要吗？** 模型能否直接学会精确的数值调整？
- **DSL是约束还是枷锁？** 它提供了可控性，但也限制了表达自由度
- **如何理解更柔和的表达？** "比刚刚亮一些"、"柔和一点"

这些问题会在后续版本中继续探索。

---

## 相关博客

- [心潮涌链：它真的懂我说的话吗？](https://fallingangel.net/archives/xin-chao-yong-lian-ta-zhen-de-dong-wo-shuo-de-hua-ma) -
  LLM推理链路完整梳理
- [韵染流光·其一：亮一些的蓝](https://fallingangel.net/categories/su-yuan-qiu-suo-du-xing) - 本项目的完整设计实践过程

---

**这个项目是《溯源·求索·笃行》系列的第一个实践。**

从理解语言模型推理链路，到用它解决实际问题。

路还很长，但至少现在，它已经能理解"亮一些的蓝"了。