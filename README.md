
# Web 信息处理与应用 · 实验 1

## 环境

本实验基于 Python 3.13，使用独立虚拟环境 `.venv`。

精确依赖版本记录在 `requirements.txt` 中。

支持的运行环境：
- Linux
- Windows 10 / Windows 11

实验使用 `Qilin-IR-50K` 数据集，完整实验处理全部 50,000 篇笔记，不将抽样结果作为全库实验结果。

---

## 一、Linux 环境配置与运行

### 1. 克隆仓库

```bash
git clone https://github.com/yunleeeee/web_lab1.git
cd web_lab1
```

### 2. 创建 Python 虚拟环境

确保已经安装 Python 3.13。

```bash
python3.13 -m venv .venv
```

### 3. 安装项目依赖

```bash
.venv/bin/python -m pip install -r requirements.txt
```

### 4. 准备数据集

将 `Qilin-IR-50K` 数据集放在项目根目录下。

数据集目录应包含：

```text
Qilin-IR-50K/
├── documents.jsonl
├── queries.jsonl
└── search_logs.jsonl
```

### 5. 下载预训练模型

首次运行语义表征实验，需要下载中文 BGE 模型。

模型保存在：

```text
models/bge-small-zh-v1.5/
```

执行：

```bash
.venv/bin/python scripts/download_model.py
```

如果 Hugging Face 官方源无法连接，可以指定镜像：

```bash
.venv/bin/python scripts/download_model.py --endpoint https://hf-mirror.com
```

下载程序使用固定模型 revision，并对模型权重进行 SHA-256 校验。

如果仅运行分词和传统检索实验，可以暂时跳过模型下载。

### 6. 运行实验

#### 6.1 分词与预处理

运行三种分词方法和两种停用词设置：

```bash
.venv/bin/python run_lab.py prepare
```

#### 6.2 传统检索实验

运行倒排索引、压缩、布尔检索、短语检索、跳表和 TF-IDF 实验：

```bash
.venv/bin/python run_lab.py lexical
```

#### 6.3 语义表征实验

在相同预训练模型下运行四种文本表征设置：

```bash
.venv/bin/python run_lab.py semantic
```

#### 6.4 运行全部实验

```bash
.venv/bin/python run_lab.py all
```

运行前需要准备完整数据集和预训练模型。

---

## 二、Windows 环境配置与运行

推荐使用 Windows 10 / Windows 11、Python 3.13 和 PowerShell。

### 1. 克隆仓库

打开 PowerShell，执行：

```powershell
git clone https://github.com/yunleeeee/web_lab1.git
cd web_lab1
```

也可以通过 GitHub 下载 ZIP 压缩包，然后解压并使用 PowerShell 进入项目目录。

### 2. 创建 Python 虚拟环境

确保已安装 Python 3.13。

检查 Python 版本：

```powershell
py -3.13 --version
```

创建虚拟环境：

```powershell
py -3.13 -m venv .venv
```

Windows 下虚拟环境中的 Python 路径为：

```text
.venv\Scripts\python.exe
```

后续命令直接使用此路径，无需手动激活虚拟环境。

### 3. 安装项目依赖

```powershell
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

项目依赖包括 Jieba、NumPy、scikit-learn、PyTorch、Transformers 等。

依赖版本固定在 `requirements.txt` 中。Windows 安装时需要确保对应版本提供兼容的安装包。

### 4. 准备数据集

将 `Qilin-IR-50K` 数据集放在项目根目录下。

目录结构：

```text
web_lab1/
├── Qilin-IR-50K/
│   ├── documents.jsonl
│   ├── queries.jsonl
│   └── search_logs.jsonl
├── lab_ir/
├── resources/
├── scripts/
├── run_lab.py
└── requirements.txt
```

注意：`Qilin-IR-50K` 数据集不包含在 GitHub 仓库中，需要单独准备。

### 5. 下载预训练模型

首次运行语义表征实验时执行：

```powershell
.\.venv\Scripts\python.exe scripts/download_model.py
```

如果官方 Hugging Face 下载失败，可以使用镜像：

```powershell
.\.venv\Scripts\python.exe scripts/download_model.py --endpoint https://hf-mirror.com
```

模型下载后保存在：

```text
models\bge-small-zh-v1.5\
```

如果仅进行分词、索引和传统检索实验，可以跳过此步骤。

### 6. 运行实验

#### 6.1 分词与预处理

```powershell
.\.venv\Scripts\python.exe run_lab.py prepare
```

#### 6.2 传统检索实验

```powershell
.\.venv\Scripts\python.exe run_lab.py lexical
```

#### 6.3 语义表征实验

```powershell
.\.venv\Scripts\python.exe run_lab.py semantic
```

#### 6.4 运行全部实验

```powershell
.\.venv\Scripts\python.exe run_lab.py all
```

运行前需要准备完整数据集和预训练模型。

### 7. Windows 注意事项

1. 建议使用 PowerShell 执行上述命令。
2. Windows 与 Linux 的虚拟环境 Python 路径不同。
3. 项目使用 `pathlib` 处理主要文件路径，通常无需手动修改路径分隔符。
4. 语义表征实验需要较多计算资源，运行时间与 CPU、GPU 和内存配置有关。
5. PyTorch 在 Windows 下可以使用 CPU；如需 CUDA 加速，需要安装兼容的 NVIDIA 驱动和 PyTorch 环境。
6. 如果安装依赖时出现兼容性问题，可考虑使用 WSL2 Ubuntu 运行 Linux 版本的命令。

---

## 三、项目文件

| 路径 | 用途 |
| --- | --- |
| `lab_ir/preprocessing.py` | 正向/逆向最大匹配、统计分词、规范化和位置保留 |
| `resources/stopwords.txt` | 自定义停用词表；保留否定词 |
| `lab_ir/index.py` | 字段位置索引和无损压缩 |
| `lab_ir/retrieval.py` | 布尔、短语、跳指针、TF-IDF |
| `lab_ir/lexical_experiments.py` | 传统检索的可复现实验 |
| `lab_ir/semantic.py` | 中文预训练编码器和词项表征聚合 |
| `lab_ir/semantic_experiments.py` | 全库四组表征及点击代理评估 |
| `scripts/download_model.py` | 下载并校验 BGE 预训练模型 |
| `run_lab.py` | 实验统一运行入口 |
| `requirements.txt` | Python 依赖及版本 |
| `outputs/` | 分词、索引、向量、计时及搜索结果 |

---

## 四、实验实现说明

### 1. 分词与预处理

标题和正文分别分词并存位置，排名时合并词频。

位置从规范化后的有效词项开始计数，去停用词后保留间隙。

匹配法包括：

- 正向最大匹配（FMM，`matching`）
- 逆向最大匹配（RMM，`reverse_matching`）
- 统计分词（`statistical`）

正向和逆向最大匹配使用相同的 Jieba 词典。

统计分词使用 Jieba 的词频 DAG 解码和未登录词 HMM。

`prepare` 分别生成：

- `matching_keep.jsonl`
- `matching_drop.jsonl`
- `reverse_matching_keep.jsonl`
- `reverse_matching_drop.jsonl`
- `tokens_keep.jsonl`
- `tokens_drop.jsonl`

共六份 JSONL 文件。

同时在 `preprocessing_results.json` 中保存：

- 三种方法的耗时
- 词项数
- 词表大小
- 三组两两差异字段数
- 真实案例

差异字段数表示切分结果不同的标题/正文数量，分词质量需结合具体语境判断。

### 2. 传统检索实验

传统检索实验包括：

- 倒排索引构建
- 索引无损压缩
- 布尔检索
- 短语检索
- 跳指针实验
- TF-IDF 排序检索

实验结果保存在 `outputs/` 目录下。

### 3. 语义表征实验

使用中文预训练 BGE 模型：

```text
BAAI/bge-small-zh-v1.5
```

在同一预训练模型下比较四组文本表征设置：

- 不预先分词，保留停用词
- 不预先分词，去除停用词
- 预先分词，保留停用词
- 预先分词，去除停用词

模型的“预先分词”实验将每个词项分别输入同一编码器，再聚合词向量；实现遵循实验 PDF 中的定义。

长文档分块覆盖全部文本，不静默丢弃后半部分。

### 4. 实验评估

评估使用搜索测试集的点击作为相关性代理，候选范围固定为全部 50,000 篇文档。

未点击和未曝光不直接标记为不相关。

统一的前 100 条测试请求用于传统检索和四组模型设置的直接比较。

完整测试集另外报告 TF-IDF 结果。

参数固定，不根据测试结果挑参数。

---

## 五、实验输出

实验运行后，结果默认保存在：

```text
outputs/
```

主要包括：

- 分词结果
- 分词统计与方法比较
- 倒排索引及压缩结果
- 布尔、短语和跳指针实验结果
- TF-IDF 检索与评估结果
- 语义表征向量及缓存
- 语义检索与评估结果

`outputs/`、`models/`、`.venv/` 和 `Qilin-IR-50K/` 已配置在 `.gitignore` 中，默认不会提交到 GitHub。

## 六、补充说明

完整实验基于全部 50,000 篇笔记进行。

不同操作系统和硬件环境下的实验运行时间可能存在差异。

对于涉及预训练模型的实验，应确保模型版本与仓库代码要求一致。

Windows 与 Linux 使用相同的实验代码、数据集和评价方法，主要区别在于虚拟环境的 Python 可执行文件路径及运行命令。

