"""
Agent记忆管理器 - 处理agent的上下文历史和记忆
"""

import os
import json
import pickle
import threading
import time
from typing import Dict, List, Any, Optional
from datetime import datetime
import logging

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("AgentMemoryManager")

# 上下文数据目录
CONTEXT_DIR = "context_data"
CONTEXT_FILE_DIR = os.path.join(CONTEXT_DIR, "contexts")

# 确保目录存在
os.makedirs(CONTEXT_FILE_DIR, exist_ok=True)

class AgentMemoryManager:
    """
    Agent记忆管理器 - 处理agent的上下文历史
    
    特点:
    - 持久化存储上下文历史
    - 线程安全的读写操作
    - 自动保存机制
    """
    
    def __init__(self, agent_id: str):
        """
        初始化Agent记忆管理器
        
        Args:
            agent_id: Agent ID
        """
        self.agent_id = agent_id
        self.context_file = os.path.join(CONTEXT_FILE_DIR, f"{agent_id}_context.pkl")
        self.lock = threading.RLock()
        
        # 上下文历史
        self.context = []
        
        # 上次保存时间
        self.last_save_time = time.time()
        self.save_interval = 10  # 10秒保存一次
        self.modified = False
        
        # 从磁盘加载上下文
        self._load_from_disk()
        
    def get_context(self) -> List[Dict]:
        """
        获取上下文历史
        
        Returns:
            List[Dict]: 上下文历史列表
        """
        with self.lock:
            return self.context.copy()
    
    def update_context(self, new_context: List[Dict]) -> bool:
        """
        更新上下文历史
        
        Args:
            new_context: 新的上下文历史
            
        Returns:
            bool: 是否成功更新
        """
        with self.lock:
            if new_context != self.context:
                self.context = new_context.copy()
                self.modified = True
                
                # 检查是否需要保存
                current_time = time.time()
                if current_time - self.last_save_time > self.save_interval:
                    self.save_to_disk()
                    self.last_save_time = current_time
                
                return True
            return False
    
    def add_to_context(self, message: Dict) -> bool:
        """
        添加消息到上下文历史
        
        Args:
            message: 要添加的消息
            
        Returns:
            bool: 是否成功添加
        """
        with self.lock:
            # 检查是否已存在相同消息
            if any(msg.get("role") == message.get("role") and 
                   msg.get("content") == message.get("content") 
                   for msg in self.context):
                return False
                
            self.context.append(message)
            self.modified = True
            
            # 检查是否需要保存
            current_time = time.time()
            if current_time - self.last_save_time > self.save_interval:
                self.save_to_disk()
                self.last_save_time = current_time
                
            return True
    
    def clear_context(self) -> bool:
        """
        清除上下文历史
        
        Returns:
            bool: 是否成功清除
        """
        with self.lock:
            if self.context:
                # 保留系统消息
                system_messages = [msg for msg in self.context if msg.get("role") == "system"]
                self.context = system_messages
                self.modified = True
                self.save_to_disk()
                return True
            return False
    
    def save_to_disk(self) -> bool:
        """
        将上下文历史保存到磁盘
        
        Returns:
            bool: 是否成功保存
        """
        if not self.modified:
            return True
            
        with self.lock:
            try:
                # 使用临时文件+重命名方式确保原子写入
                temp_path = f"{self.context_file}.tmp"
                
                with open(temp_path, 'wb') as f:
                    pickle.dump({
                        "agent_id": self.agent_id,
                        "context": self.context,
                        "last_updated": datetime.now().isoformat()
                    }, f)
                
                # 原子重命名
                os.replace(temp_path, self.context_file)
                self.modified = False
                logger.debug(f"上下文历史已保存到 {self.context_file}")
                return True
            except Exception as e:
                logger.error(f"保存上下文历史失败: {str(e)}")
                return False
    
    def _load_from_disk(self) -> bool:
        """
        从磁盘加载上下文历史
        
        Returns:
            bool: 是否成功加载
        """
        if not os.path.exists(self.context_file):
            return False
            
        try:
            with open(self.context_file, 'rb') as f:
                data = pickle.load(f)
                
            self.context = data.get("context", [])
            logger.info(f"从 {self.context_file} 加载了上下文历史")
            return True
        except Exception as e:
            logger.error(f"加载上下文历史失败: {str(e)}")
            return False

# 全局记忆管理器实例
_memory_managers = {}
_memory_managers_lock = threading.RLock()

def get_agent_memory_manager(agent_id: str) -> AgentMemoryManager:
    """
    获取Agent记忆管理器实例
    
    Args:
        agent_id: Agent ID
        
    Returns:
        AgentMemoryManager: Agent记忆管理器实例
    """
    with _memory_managers_lock:
        if agent_id not in _memory_managers:
            _memory_managers[agent_id] = AgentMemoryManager(agent_id)
        return _memory_managers[agent_id]

# 示例用法
if __name__ == "__main__":
    # 初始化记忆管理器
    memory_manager = get_agent_memory_manager("test_agent")
    
    # 添加消息
    memory_manager.add_to_context({
        "role": "system",
        "content": "你是一个智能助手"
    })
    
    memory_manager.add_to_context({
        "role": "user",
        "content": "你好"
    })
    
    memory_manager.add_to_context({
        "role": "assistant",
        "content": "你好，有什么可以帮助你的？"
    })
    
    # 获取上下文
    context = memory_manager.get_context()
    print(f"上下文长度: {len(context)}")
    
    # 保存到磁盘
    memory_manager.save_to_disk()
