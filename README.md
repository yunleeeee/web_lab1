# Web 信息处理与应用 · 实验 1


## 环境

本地实验使用 Python 3.13 和独立虚拟环境 `.venv`。精确依赖版本记录在 `requirements.txt`。新机器上：

```bash
python3 -m venv .venv
.venv/bin/python -m pip install -r requirements.txt
```

首次运行表征实验需要下载中文 BGE 模型，模型保存至 `models/bge-small-zh-v1.5`：

```bash
python3 scripts/download_model.py
# 官方源无法连接时，可指定镜像；仍按固定 revision 和官方 SHA-256 校验权重
python3 scripts/download_model.py --endpoint https://hf-mirror.com
```

完整实验使用全部 50,000 篇笔记，不把抽样结果作为全库实验。

## 运行

在本目录执行：

```bash
# 全库分词，保留三种分词方法 × 两种停用词设置
.venv/bin/python run_lab.py prepare

# 索引、压缩、布尔/短语/跳表实验和 TF-IDF
.venv/bin/python run_lab.py lexical

# 同一预训练模型下的四种文本表征设置
.venv/bin/python run_lab.py semantic


# 一次运行上述全部阶段
.venv/bin/python run_lab.py all



```


## 文件

| 路径 | 用途 |
| --- | --- |
| `lab_ir/preprocessing.py` | 正向/逆向最大匹配、统计分词、规范化和位置保留 |
| `resources/stopwords.txt` | 自定义停用词表；保留否定词 |
| `lab_ir/index.py` | 字段位置索引和无损压缩 |
| `lab_ir/retrieval.py` | 布尔、短语、跳指针、TF-IDF |
| `lab_ir/lexical_experiments.py` | 传统检索的可复现实验 |
| `lab_ir/semantic.py` | 中文预训练编码器和词项表征聚合 |
| `lab_ir/semantic_experiments.py` | 全库四组表征及点击代理评估 |
| `outputs/` | 分词、索引、向量、计时及搜索结果 |

标题和正文分别分词并存位置，排名时合并词频。位置从规范化后的有效词项开始计数，去停用词后保留间隙。匹配法包括正向最大匹配（FMM，`matching`）和逆向最大匹配（RMM，`reverse_matching`），二者使用相同的 Jieba 词典；统计分词（`statistical`）使用 Jieba 的词频 DAG 解码和未登录词 HMM。`prepare` 分别生成 `matching_*`、`reverse_matching_*`、`tokens_*` 的 keep/drop 六份 JSONL，并在 `preprocessing_results.json` 中保存三种方法的耗时、词项数、词表大小、三组两两差异字段数和真实案例。差异字段数表示切分结果不同的标题/正文数量，分词质量需结合具体语境判断。

模型的“预先分词”实验将每个词项分别输入同一编码器，再聚合词向量；实现遵循 PDF 中的定义。长文档分块覆盖全部文本，不静默丢弃后半部分。

评估使用搜索测试集的点击作为相关性代理，候选范围固定为全部 50K 文档。未点击和未曝光不直接标记为不相关。统一的前 100 条测试请求用于传统检索和四组模型设置的直接比较；完整测试集另外报告 TF-IDF 结果。参数固定，不根据测试结果挑参数。

