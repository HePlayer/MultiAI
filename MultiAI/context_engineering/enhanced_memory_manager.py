"""
增强版记忆管理器 - 使用hf_model_downloader工具自动下载模型
"""

import os
import logging
from typing import Dict, List, Optional, Any

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("EnhancedMemoryManager")

# 导入原始MemoryManager
from ContextEngineering import MemoryManager

class EnhancedMemoryManager(MemoryManager):
    """
    增强版记忆管理器 - 自动使用镜像下载模型
    
    特点:
    - 自动使用hf_model_downloader下载模型
    - 支持多种镜像源
    - 与原始MemoryManager完全兼容
    """
    
    def __init__(self, agent_id: str = None, mirror: str = "hf-mirror", use_modelscope: bool = True):
        """
        初始化增强版记忆管理器
        
        Args:
            agent_id: Agent ID
            mirror: 使用的镜像源，可选值：default, hf-mirror, modelscope, openi
            use_modelscope: 是否优先使用ModelScope替代
        """
        self.mirror = mirror
        self.use_modelscope = use_modelscope
        
        # 设置库使用镜像
        self._setup_mirrors()
        
        # 调用父类初始化
        super().__init__(agent_id)
        
        logger.info(f"增强版记忆管理器初始化完成，使用镜像: {mirror}")
    
    def _setup_mirrors(self):
        """设置库使用镜像"""
        try:
            # 导入下载工具
            from hf_model_downloader import setup_transformers, setup_sentence_transformers
            
            # 设置transformers使用镜像
            setup_transformers()
            logger.info("成功设置transformers使用镜像")
            
            # 设置sentence-transformers使用镜像
            setup_sentence_transformers()
            logger.info("成功设置sentence-transformers使用镜像")
            
        except ImportError:
            logger.warning("hf_model_downloader未安装，无法设置镜像")
        except Exception as e:
            logger.error(f"设置镜像失败: {str(e)}")
    
    def _load_embedding_model(self, model_name: str = 'paraphrase-MiniLM-L6-v2'):
        """
        加载嵌入模型，使用镜像下载
        
        Args:
            model_name: 模型名称
        """
        try:
            # 尝试使用下载工具
            try:
                from hf_model_downloader import download_sentence_transformer
                
                # 下载模型
                logger.info(f"使用镜像下载模型: {model_name}")
                model_path = download_sentence_transformer(
                    model_name, 
                    mirror=self.mirror, 
                    cache_dir=None
                )
                
                # 加载模型
                from sentence_transformers import SentenceTransformer
                self.embedding_model = SentenceTransformer(model_path)
                logger.info(f"成功从镜像加载模型: {model_name}")
                
            except ImportError:
                # 如果下载工具不可用，尝试直接加载
                logger.warning("hf_model_downloader未安装，尝试直接加载模型")
                from sentence_transformers import SentenceTransformer
                self.embedding_model = SentenceTransformer(model_name)
                
        except Exception as e:
            logger.error(f"加载嵌入模型失败: {str(e)}")
            self.embedding_model = None


# 使用示例
if __name__ == "__main__":
    # 初始化增强版记忆管理器
    memory_manager = EnhancedMemoryManager(agent_id="test_agent", mirror="hf-mirror")
    
    # 添加记忆
    memory_manager.add_memory("facts", "地球是太阳系中的第三颗行星")
    
    # 检索记忆
    results = memory_manager.retrieve_memory("太阳系 行星")
    
    # 打印结果
    for result in results:
        print(f"找到记忆: {result['content']}")
