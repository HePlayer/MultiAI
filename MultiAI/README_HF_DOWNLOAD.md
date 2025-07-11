# 无需VPN下载Hugging Face模型

这个项目提供了一套工具，用于在没有VPN（梯子）的情况下下载和使用Hugging Face模型。通过使用国内镜像源、ModelScope替代和断点续传等技术，可以稳定高效地获取模型。

## 主要功能

- 使用国内镜像源下载Hugging Face模型
- 支持ModelScope作为替代方案
- 断点续传功能，避免下载中断
- 自动修补常用库（transformers、sentence-transformers）
- 支持命令行和API调用方式

## 快速开始

### 安装依赖

```bash
pip install requests tqdm numpy -i https://pypi.tuna.tsinghua.edu.cn/simple
# 可选依赖
pip install sentence-transformers transformers -i https://pypi.tuna.tsinghua.edu.cn/simple
pip install modelscope -i https://pypi.tuna.tsinghua.edu.cn/simple  # 如果需要使用ModelScope
```

### 命令行使用

下载模型：

```bash
python hf_model_downloader.py --model sentence-transformers/paraphrase-MiniLM-L6-v2 --mirror hf-mirror
```

设置库使用镜像并下载模型：

```bash
python hf_model_downloader.py --model bert-base-chinese --mirror hf-mirror --setup-libs
```

使用ModelScope替代：

```bash
python hf_model_downloader.py --model bert-base-chinese --use-modelscope
```

### 在代码中使用

```python
from hf_model_downloader import setup_transformers, setup_sentence_transformers, download_sentence_transformer

# 设置库使用镜像
setup_transformers()
setup_sentence_transformers()

# 下载并使用sentence-transformers模型
model_path = download_sentence_transformer("paraphrase-MiniLM-L6-v2")
print(f"模型下载成功: {model_path}")

# 现在可以正常导入和使用模型
from sentence_transformers import SentenceTransformer
model = SentenceTransformer(model_path)
```

## 可用镜像源

- `hf-mirror`: [HF Mirror](https://hf-mirror.com) (默认，推荐)
- `modelscope`: [ModelScope](https://modelscope.cn)
- `openi`: [OpenI](https://openi.pcl.ac.cn)
- `default`: 原始Hugging Face网站 (需要VPN)

## 示例脚本

项目包含一个完整的示例脚本 `use_hf_model_example.py`，演示了如何下载和使用模型：

```bash
python use_hf_model_example.py
```

## 与项目集成

已将下载工具集成到 `ContextEngineering.py` 中，使其能够自动使用镜像下载sentence-transformers模型。

## 常见问题

### 1. 为什么需要这个工具？

在中国大陆，直接访问Hugging Face可能会遇到连接问题。这个工具提供了多种解决方案，无需使用VPN即可下载模型。

### 2. 支持哪些类型的模型？

理论上支持所有Hugging Face上的模型，包括但不限于：
- BERT系列模型
- Sentence Transformers模型
- GPT系列模型
- CLIP模型
- 等等

### 3. 如何添加新的ModelScope映射？

如果你发现某个Hugging Face模型在ModelScope上有对应版本，可以通过以下方式添加映射：

```python
from hf_model_downloader import HFDownloader

downloader = HFDownloader()
downloader.add_modelscope_mapping("huggingface/model-name", "modelscope/model-name")
```

### 4. 下载速度慢怎么办？

- 尝试不同的镜像源
- 确保网络稳定
- 利用断点续传功能，即使下载中断也可以继续

## 技术细节

### 工作原理

1. **镜像下载**：通过替换Hugging Face的URL为国内镜像站点
2. **ModelScope替代**：使用ModelScope上的对应模型
3. **断点续传**：记录已下载的部分，支持断点继续
4. **库修补**：修改transformers和sentence-transformers库的下载行为

### 文件说明

- `hf_model_downloader.py`: 核心下载工具
- `use_hf_model_example.py`: 使用示例
- `modelscope_mapping.json`: HF模型到ModelScope模型的映射（自动生成）

## 贡献

欢迎提交问题和改进建议！

## 许可证

MIT
