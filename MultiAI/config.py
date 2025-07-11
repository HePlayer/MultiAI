"""
项目配置文件
存储所有可配置的参数和常量，支持环境变量覆盖
保持向后兼容性，确保原有功能不变
"""

import os
from typing import Dict, Any, Optional

class AppConfig:
    """应用程序配置类 - 集中管理所有配置项"""
    
    def __init__(self):
        """初始化配置，支持环境变量覆盖"""
        
        # API密钥配置 - 保留原有默认值确保兼容性
        self.SPARK_API_KEY = os.getenv('SPARK_API_KEY', 'BcSOEhzuOxuqbZVBjZWI:lLBQCKkaFGdmPdWNRICk')
        self.ZHIPU_API_KEY = os.getenv('ZHIPU_API_KEY', '3615e68f81697dc2db853f13cd8ae37d.gpGd71TLyTTkCepS')
        
        # 服务器配置
        self.SERVER_HOST = os.getenv('SERVER_HOST', '127.0.0.1')
        self.SERVER_PORT = int(os.getenv('SERVER_PORT', '5000'))
        self.DEBUG_MODE = os.getenv('DEBUG_MODE', 'True').lower() == 'true'
        
        # 模型配置
        self.SPARK_MODEL = os.getenv('SPARK_MODEL', 'lite')
        self.ZHIPU_MODEL = os.getenv('ZHIPU_MODEL', 'glm-4-flash-250414')
        
        # API URL配置
        self.SPARK_API_URL = os.getenv('SPARK_API_URL', 'https://spark-api-open.xf-yun.com/v1/chat/completions')
        
        # 记忆系统配置
        self.MEMORY_DIR = os.getenv('MEMORY_DIR', 'agent_memory')
        self.CONTEXT_DATA_DIR = os.getenv('CONTEXT_DATA_DIR', 'context_data')
        
        # 聊天配置
        self.MAX_TOKENS = int(os.getenv('MAX_TOKENS', '2048'))
        self.TEMPERATURE = float(os.getenv('TEMPERATURE', '0.7'))
        self.MAX_CONTEXT_LENGTH = int(os.getenv('MAX_CONTEXT_LENGTH', '11000'))
        
        # 讨论系统配置
        self.DEFAULT_DISCUSSION_ROUNDS = int(os.getenv('DEFAULT_DISCUSSION_ROUNDS', '3'))
        self.MAX_DISCUSSION_ROUNDS = int(os.getenv('MAX_DISCUSSION_ROUNDS', '10'))
        
        # 流式输出配置
        self.STREAM_TIMEOUT = int(os.getenv('STREAM_TIMEOUT', '30'))
        self.CHUNK_BUFFER_SIZE = int(os.getenv('CHUNK_BUFFER_SIZE', '1024'))
        
        # Web搜索配置
        self.SEARCH_RESULT_COUNT = int(os.getenv('SEARCH_RESULT_COUNT', '5'))
        self.SEARCH_CONTENT_SIZE = os.getenv('SEARCH_CONTENT_SIZE', 'medium')
        
        # 有效的记忆类型（用于验证）
        self.VALID_MEMORY_TYPES = ["preferences", "conclusions", "requirements", "facts"]
        
        # CORS配置
        self.CORS_ORIGINS = os.getenv('CORS_ORIGINS', '*').split(',')
        
        # 日志配置
        self.LOG_LEVEL = os.getenv('LOG_LEVEL', 'INFO')
        self.LOG_FILE = os.getenv('LOG_FILE', None)  # None表示不记录到文件
        
    def get_api_key(self, model_type: str) -> str:
        """获取指定模型的API密钥"""
        if model_type.lower() == 'spark':
            return self.SPARK_API_KEY
        elif model_type.lower() == 'zhipu':
            return self.ZHIPU_API_KEY
        else:
            raise ValueError(f"未知的模型类型: {model_type}")
    
    def get_model_config(self, model_type: str) -> Dict[str, Any]:
        """获取指定模型的配置"""
        if model_type.lower() == 'spark':
            return {
                'api_key': self.SPARK_API_KEY,
                'model': self.SPARK_MODEL,
                'api_url': self.SPARK_API_URL,
                'max_tokens': self.MAX_TOKENS,
                'temperature': self.TEMPERATURE
            }
        elif model_type.lower() == 'zhipu':
            return {
                'api_key': self.ZHIPU_API_KEY,
                'model': self.ZHIPU_MODEL,
                'max_tokens': self.MAX_TOKENS,
                'temperature': self.TEMPERATURE
            }
        else:
            raise ValueError(f"未知的模型类型: {model_type}")
    
    def get_discussion_config(self) -> Dict[str, Any]:
        """获取讨论系统配置"""
        return {
            'default_rounds': self.DEFAULT_DISCUSSION_ROUNDS,
            'max_rounds': self.MAX_DISCUSSION_ROUNDS,
            'timeout': self.STREAM_TIMEOUT
        }
    
    def validate_memory_type(self, memory_type: str) -> bool:
        """验证记忆类型是否有效"""
        return memory_type in self.VALID_MEMORY_TYPES
    
    def to_dict(self) -> Dict[str, Any]:
        """将配置转换为字典格式（不包含敏感信息）"""
        return {
            'server_host': self.SERVER_HOST,
            'server_port': self.SERVER_PORT,
            'debug_mode': self.DEBUG_MODE,
            'spark_model': self.SPARK_MODEL,
            'zhipu_model': self.ZHIPU_MODEL,
            'max_tokens': self.MAX_TOKENS,
            'temperature': self.TEMPERATURE,
            'max_context_length': self.MAX_CONTEXT_LENGTH,
            'default_discussion_rounds': self.DEFAULT_DISCUSSION_ROUNDS,
            'max_discussion_rounds': self.MAX_DISCUSSION_ROUNDS,
            'valid_memory_types': self.VALID_MEMORY_TYPES
        }

# 全局配置实例 - 保持单例模式
_config_instance: Optional[AppConfig] = None

def get_config() -> AppConfig:
    """获取配置实例 - 单例模式"""
    global _config_instance
    if _config_instance is None:
        _config_instance = AppConfig()
    return _config_instance

def reload_config() -> AppConfig:
    """重新加载配置 - 用于配置更新后的刷新"""
    global _config_instance
    _config_instance = AppConfig()
    return _config_instance

# 快捷访问函数 - 保持向后兼容性
def get_api_key(model_type: str) -> str:
    """获取API密钥的快捷函数"""
    return get_config().get_api_key(model_type)

def get_model_config(model_type: str) -> Dict[str, Any]:
    """获取模型配置的快捷函数"""
    return get_config().get_model_config(model_type)

# 导出常用配置常量 - 确保原有代码可以直接使用
config = get_config()
SPARK_API_KEY = config.SPARK_API_KEY
ZHIPU_API_KEY = config.ZHIPU_API_KEY
MEMORY_DIR = config.MEMORY_DIR
