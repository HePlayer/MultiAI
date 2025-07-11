"""
Hugging Face模型下载工具 - 无需VPN即可下载模型
支持多种下载方式：镜像源、ModelScope替代、修改URL等
"""

import os
import sys
import json
import time
import hashlib
import argparse
import requests
from pathlib import Path
from tqdm import tqdm
from typing import Optional, Dict, Any, List, Tuple
import logging
import shutil
import tarfile
import zipfile

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("HF_Downloader")

# 常量定义
DEFAULT_CACHE_DIR = os.path.expanduser("~/.cache/huggingface")
MIRRORS = {
    "default": "https://huggingface.co",
    "hf-mirror": "https://hf-mirror.com",
    "modelscope": "https://modelscope.cn/api/v1/models",
    "openi": "https://openi.pcl.ac.cn/api/v1/models"
}

class HFDownloader:
    """Hugging Face模型下载器"""
    
    def __init__(self, 
                 cache_dir: Optional[str] = None, 
                 mirror: str = "hf-mirror",
                 use_modelscope: bool = False,
                 chunk_size: int = 8192,
                 max_retries: int = 5,
                 timeout: int = 30):
        """
        初始化下载器
        
        Args:
            cache_dir: 缓存目录，默认为~/.cache/huggingface
            mirror: 使用的镜像源，可选值：default, hf-mirror, modelscope, openi
            use_modelscope: 是否优先使用ModelScope替代
            chunk_size: 下载时的块大小
            max_retries: 最大重试次数
            timeout: 连接超时时间（秒）
        """
        self.cache_dir = cache_dir or DEFAULT_CACHE_DIR
        self.mirror = mirror
        self.use_modelscope = use_modelscope
        self.chunk_size = chunk_size
        self.max_retries = max_retries
        self.timeout = timeout
        
        # 确保缓存目录存在
        os.makedirs(self.cache_dir, exist_ok=True)
        
        # 镜像基础URL
        self.base_url = MIRRORS.get(mirror, MIRRORS["hf-mirror"])
        
        # ModelScope映射表
        self.modelscope_mapping = self._load_modelscope_mapping()
        
        logger.info(f"初始化HF下载器，使用镜像: {self.mirror}, 缓存目录: {self.cache_dir}")
    
    def _load_modelscope_mapping(self) -> Dict[str, str]:
        """加载HF到ModelScope的映射表"""
        mapping_file = os.path.join(os.path.dirname(__file__), "modelscope_mapping.json")
        
        if os.path.exists(mapping_file):
            try:
                with open(mapping_file, 'r', encoding='utf-8') as f:
                    return json.load(f)
            except Exception as e:
                logger.error(f"加载ModelScope映射表失败: {str(e)}")
        
        # 返回一些常用模型的映射
        return {
            "sentence-transformers/paraphrase-MiniLM-L6-v2": "damo/nlp_corom_sentence-embedding_english-base",
            "bert-base-chinese": "damo/nlp_bert_base-chinese",
            "bert-base-uncased": "damo/nlp_bert_base-uncased",
            "gpt2": "ZhipuAI/gpt2",
            "facebook/bart-large-cnn": "damo/nlp_bart_summarization-large-cnn",
            "facebook/wav2vec2-base-960h": "damo/speech_wav2vec2_asr-zh-cn-16k-common",
            "microsoft/DialoGPT-medium": "ZhipuAI/dialogpt-medium",
            "clip-vit-base-patch32": "damo/multi-modal_clip-vit-base-patch32"
        }
    
    def _save_modelscope_mapping(self):
        """保存ModelScope映射表"""
        mapping_file = os.path.join(os.path.dirname(__file__), "modelscope_mapping.json")
        
        try:
            with open(mapping_file, 'w', encoding='utf-8') as f:
                json.dump(self.modelscope_mapping, f, ensure_ascii=False, indent=2)
            logger.info(f"ModelScope映射表已保存到 {mapping_file}")
        except Exception as e:
            logger.error(f"保存ModelScope映射表失败: {str(e)}")
    
    def download_model(self, model_name: str, revision: str = "main") -> str:
        """
        下载模型
        
        Args:
            model_name: 模型名称，如"bert-base-chinese"或"sentence-transformers/paraphrase-MiniLM-L6-v2"
            revision: 模型版本，默认为"main"
            
        Returns:
            str: 模型本地路径
        """
        logger.info(f"开始下载模型: {model_name}, 版本: {revision}")
        
        # 尝试使用ModelScope替代
        if self.use_modelscope:
            modelscope_path = self._try_modelscope(model_name)
            if modelscope_path:
                return modelscope_path
        
        # 构建模型缓存目录
        model_dir = os.path.join(self.cache_dir, model_name.replace("/", "--"))
        os.makedirs(model_dir, exist_ok=True)
        
        # 下载模型配置
        config_path = self._download_file(model_name, "config.json", revision, model_dir)
        
        # 解析配置，确定需要下载的文件
        model_files = self._get_model_files(config_path, model_name, revision)
        
        # 下载所有模型文件
        for file_name in model_files:
            self._download_file(model_name, file_name, revision, model_dir)
        
        logger.info(f"模型 {model_name} 下载完成，保存在: {model_dir}")
        return model_dir
    
    def _try_modelscope(self, model_name: str) -> Optional[str]:
        """尝试从ModelScope下载模型"""
        if model_name in self.modelscope_mapping:
            modelscope_name = self.modelscope_mapping[model_name]
            logger.info(f"找到ModelScope替代模型: {modelscope_name}")
            
            try:
                # 尝试导入modelscope
                from modelscope.hub.snapshot_download import snapshot_download
                
                # 下载模型
                model_dir = snapshot_download(modelscope_name)
                logger.info(f"成功从ModelScope下载模型: {modelscope_name}")
                
                # 创建符号链接到HF缓存目录
                hf_model_dir = os.path.join(self.cache_dir, model_name.replace("/", "--"))
                if os.path.exists(hf_model_dir):
                    shutil.rmtree(hf_model_dir)
                
                # 复制文件而不是创建符号链接，更可靠
                shutil.copytree(model_dir, hf_model_dir)
                
                return hf_model_dir
            except ImportError:
                logger.warning("ModelScope未安装，无法使用ModelScope替代")
                logger.info("可以通过 pip install modelscope -i https://pypi.tuna.tsinghua.edu.cn/simple 安装")
            except Exception as e:
                logger.error(f"从ModelScope下载失败: {str(e)}")
        
        return None
    
    def _get_model_files(self, config_path: str, model_name: str, revision: str) -> List[str]:
        """根据配置确定需要下载的文件"""
        files = ["config.json", "tokenizer_config.json", "vocab.txt", "tokenizer.json", 
                "special_tokens_map.json", "pytorch_model.bin", "model.safetensors"]
        
        # 读取配置，检查是否有分片模型文件
        try:
            with open(config_path, 'r', encoding='utf-8') as f:
                config = json.load(f)
            
            # 检查是否有分片信息
            if "sharded_ddp" in config or "_num_shards" in config:
                shard_count = config.get("_num_shards", 2)
                for i in range(shard_count):
                    files.append(f"pytorch_model-{i:05d}-of-{shard_count:05d}.bin")
                    files.append(f"model-{i:05d}-of-{shard_count:05d}.safetensors")
        except Exception as e:
            logger.warning(f"解析配置文件失败: {str(e)}，将尝试下载常见文件")
        
        return files
    
    def _download_file(self, model_name: str, file_name: str, revision: str, save_dir: str) -> str:
        """
        下载单个文件
        
        Args:
            model_name: 模型名称
            file_name: 文件名
            revision: 模型版本
            save_dir: 保存目录
            
        Returns:
            str: 文件本地路径
        """
        save_path = os.path.join(save_dir, file_name)
        
        # 如果文件已存在且完整，跳过下载
        if os.path.exists(save_path) and os.path.getsize(save_path) > 0:
            logger.info(f"文件已存在，跳过下载: {file_name}")
            return save_path
        
        # 构建下载URL
        if self.mirror == "default" or self.mirror == "hf-mirror":
            url = f"{self.base_url}/{model_name}/resolve/{revision}/{file_name}"
        else:
            # 其他镜像可能有不同的URL结构
            url = f"{self.base_url}/{model_name.replace('/', '--')}/{file_name}?revision={revision}"
        
        # 创建临时文件
        temp_path = f"{save_path}.tmp"
        
        # 获取已下载的文件大小，用于断点续传
        initial_size = 0
        if os.path.exists(temp_path):
            initial_size = os.path.getsize(temp_path)
            logger.info(f"找到未完成的下载，将从 {initial_size} 字节继续")
        
        # 设置HTTP头，支持断点续传
        headers = {}
        if initial_size > 0:
            headers['Range'] = f'bytes={initial_size}-'
        
        # 重试下载
        for attempt in range(self.max_retries):
            try:
                # 发送请求
                response = requests.get(url, headers=headers, stream=True, timeout=self.timeout)
                
                # 检查响应状态
                if response.status_code == 404:
                    logger.warning(f"文件不存在: {url}")
                    return save_path  # 返回空路径，表示文件不存在
                
                response.raise_for_status()
                
                # 获取文件总大小
                total_size = int(response.headers.get('content-length', 0)) + initial_size
                
                # 打开文件，追加模式
                with open(temp_path, 'ab') as f:
                    with tqdm(
                        total=total_size,
                        initial=initial_size,
                        unit='B',
                        unit_scale=True,
                        desc=f"下载 {file_name}"
                    ) as pbar:
                        for chunk in response.iter_content(chunk_size=self.chunk_size):
                            if chunk:
                                f.write(chunk)
                                pbar.update(len(chunk))
                
                # 下载完成，重命名文件
                os.replace(temp_path, save_path)
                logger.info(f"文件下载完成: {file_name}")
                return save_path
                
            except requests.exceptions.RequestException as e:
                logger.warning(f"下载失败 (尝试 {attempt+1}/{self.max_retries}): {str(e)}")
                time.sleep(2 ** attempt)  # 指数退避
                
            except Exception as e:
                logger.error(f"下载过程中发生错误: {str(e)}")
                if os.path.exists(temp_path):
                    os.remove(temp_path)
                raise
        
        # 所有重试都失败
        logger.error(f"下载失败，已达到最大重试次数: {file_name}")
        raise RuntimeError(f"下载失败: {file_name}")
    
    def download_from_snapshot(self, snapshot_url: str, save_dir: Optional[str] = None) -> str:
        """
        从快照URL下载模型
        
        Args:
            snapshot_url: 快照URL，通常是.tar.gz或.zip文件
            save_dir: 保存目录，默认为缓存目录
            
        Returns:
            str: 模型本地路径
        """
        if save_dir is None:
            # 从URL中提取模型名称
            model_name = snapshot_url.split('/')[-1].split('.')[0]
            save_dir = os.path.join(self.cache_dir, model_name)
        
        os.makedirs(save_dir, exist_ok=True)
        
        # 下载快照文件
        file_name = snapshot_url.split('/')[-1]
        snapshot_path = os.path.join(save_dir, file_name)
        
        # 如果快照已存在，跳过下载
        if not os.path.exists(snapshot_path):
            # 下载快照
            response = requests.get(snapshot_url, stream=True, timeout=self.timeout)
            response.raise_for_status()
            
            total_size = int(response.headers.get('content-length', 0))
            
            with open(snapshot_path, 'wb') as f:
                with tqdm(
                    total=total_size,
                    unit='B',
                    unit_scale=True,
                    desc=f"下载 {file_name}"
                ) as pbar:
                    for chunk in response.iter_content(chunk_size=self.chunk_size):
                        if chunk:
                            f.write(chunk)
                            pbar.update(len(chunk))
        
        # 解压快照
        if snapshot_path.endswith('.tar.gz') or snapshot_path.endswith('.tgz'):
            with tarfile.open(snapshot_path, 'r:gz') as tar:
                tar.extractall(path=save_dir)
        elif snapshot_path.endswith('.zip'):
            with zipfile.ZipFile(snapshot_path, 'r') as zip_ref:
                zip_ref.extractall(save_dir)
        else:
            logger.warning(f"未知的快照格式: {file_name}")
        
        return save_dir
    
    def add_modelscope_mapping(self, hf_name: str, modelscope_name: str):
        """添加HF到ModelScope的映射"""
        self.modelscope_mapping[hf_name] = modelscope_name
        self._save_modelscope_mapping()
        logger.info(f"添加映射: {hf_name} -> {modelscope_name}")


def download_sentence_transformer(model_name: str = "paraphrase-MiniLM-L6-v2", 
                                 mirror: str = "hf-mirror",
                                 cache_dir: Optional[str] = None) -> str:
    """
    下载Sentence Transformer模型的便捷函数
    
    Args:
        model_name: 模型名称，默认为"paraphrase-MiniLM-L6-v2"
        mirror: 使用的镜像源
        cache_dir: 缓存目录
        
    Returns:
        str: 模型本地路径
    """
    # 如果没有指定完整路径，添加前缀
    if "/" not in model_name:
        full_model_name = f"sentence-transformers/{model_name}"
    else:
        full_model_name = model_name
    
    downloader = HFDownloader(cache_dir=cache_dir, mirror=mirror, use_modelscope=True)
    return downloader.download_model(full_model_name)


def setup_sentence_transformers():
    """设置sentence-transformers以使用镜像"""
    try:
        import sentence_transformers
        
        # 修改SentenceTransformer类的__init__方法
        original_init = sentence_transformers.SentenceTransformer.__init__
        
        def patched_init(self, model_name_or_path, *args, **kwargs):
            # 如果是字符串路径且看起来像是Hugging Face模型ID
            if isinstance(model_name_or_path, str) and ('/' in model_name_or_path or not os.path.exists(model_name_or_path)):
                logger.info(f"拦截SentenceTransformer加载请求: {model_name_or_path}")
                try:
                    # 尝试从镜像下载
                    model_path = download_sentence_transformer(model_name_or_path)
                    model_name_or_path = model_path
                    logger.info(f"成功从镜像下载模型: {model_path}")
                except Exception as e:
                    logger.warning(f"从镜像下载失败，将尝试原始路径: {str(e)}")
            
            # 调用原始初始化方法
            original_init(self, model_name_or_path, *args, **kwargs)
        
        # 替换初始化方法
        sentence_transformers.SentenceTransformer.__init__ = patched_init
        logger.info("成功修补SentenceTransformer，现在将使用镜像下载模型")
        
        return True
    except ImportError:
        logger.warning("sentence-transformers未安装，无法设置镜像")
        return False
    except Exception as e:
        logger.error(f"设置sentence-transformers镜像失败: {str(e)}")
        return False


def setup_transformers():
    """设置transformers以使用镜像"""
    try:
        import transformers
        
        # 修改模型下载URL
        transformers.utils.hub.HUGGINGFACE_CO_URL_HOME = "https://hf-mirror.com"
        
        # 如果有环境变量，也设置它
        os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
        
        logger.info("成功设置transformers使用镜像: https://hf-mirror.com")
        return True
    except ImportError:
        logger.warning("transformers未安装，无法设置镜像")
        return False
    except Exception as e:
        logger.error(f"设置transformers镜像失败: {str(e)}")
        return False


def main():
    """命令行入口"""
    parser = argparse.ArgumentParser(description="Hugging Face模型下载工具 - 无需VPN")
    
    parser.add_argument("--model", "-m", type=str, required=True,
                      help="要下载的模型名称，如bert-base-chinese或sentence-transformers/paraphrase-MiniLM-L6-v2")
    
    parser.add_argument("--revision", "-r", type=str, default="main",
                      help="模型版本，默认为main")
    
    parser.add_argument("--mirror", type=str, default="hf-mirror", 
                      choices=["default", "hf-mirror", "modelscope", "openi"],
                      help="使用的镜像源")
    
    parser.add_argument("--cache-dir", type=str, default=None,
                      help="缓存目录，默认为~/.cache/huggingface")
    
    parser.add_argument("--use-modelscope", action="store_true",
                      help="是否优先使用ModelScope替代")
    
    parser.add_argument("--setup-libs", action="store_true",
                      help="设置常用库(transformers, sentence-transformers)使用镜像")
    
    args = parser.parse_args()
    
    # 设置库使用镜像
    if args.setup_libs:
        setup_transformers()
        setup_sentence_transformers()
    
    # 下载模型
    downloader = HFDownloader(
        cache_dir=args.cache_dir,
        mirror=args.mirror,
        use_modelscope=args.use_modelscope
    )
    
    try:
        model_path = downloader.download_model(args.model, args.revision)
        print(f"模型下载成功: {model_path}")
    except Exception as e:
        print(f"下载失败: {str(e)}")
        sys.exit(1)


if __name__ == "__main__":
    main()
