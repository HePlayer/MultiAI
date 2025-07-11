"""
上下文安全管理模块 - 使用函数式编程确保上下文操作的安全性
防止Agent上下文丢失和意外修改
"""

import copy
import datetime
import logging
from typing import Dict, Any, List, Optional, Tuple
from dataclasses import dataclass, field
from contextlib import contextmanager


logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class ContextState:
    """不可变的上下文状态"""
    agent_id: str
    context: List[Dict[str, str]] = field(default_factory=list)
    context_length: int = 0
    last_updated: str = ""
    compression_applied: bool = False
    
    def __post_init__(self):
        if not self.last_updated:
            object.__setattr__(self, 'last_updated', datetime.datetime.now().isoformat())
        object.__setattr__(self, 'context_length', len(self.context))
    
    def add_message(self, role: str, content: str) -> 'ContextState':
        """添加消息到上下文，返回新状态"""
        new_message = {"role": role, "content": content}
        new_context = self.context + [new_message]
        
        return ContextState(
            agent_id=self.agent_id,
            context=new_context,
            last_updated=datetime.datetime.now().isoformat(),
            compression_applied=self.compression_applied
        )
    
    def apply_compression(self, compressed_context: List[Dict[str, str]]) -> 'ContextState':
        """应用上下文压缩，返回新状态"""
        return ContextState(
            agent_id=self.agent_id,
            context=compressed_context,
            last_updated=datetime.datetime.now().isoformat(),
            compression_applied=True
        )


class SafeContextManager:
    """安全的上下文管理器"""
    
    def __init__(self, agent_id: str, initial_context: List[Dict[str, str]] = None):
        self._agent_id = agent_id
        initial_context = initial_context or []
        self._current_state = ContextState(agent_id, copy.deepcopy(initial_context))
        self._backup_states = []
    
    @property
    def current_context(self) -> List[Dict[str, str]]:
        """获取当前上下文的只读副本"""
        return copy.deepcopy(self._current_state.context)
    
    @property
    def context_length(self) -> int:
        """获取上下文长度"""
        return self._current_state.context_length
    
    def add_user_message(self, content: str) -> bool:
        """安全添加用户消息"""
        try:
            self._backup_current_state()
            self._current_state = self._current_state.add_message("user", content)
            logger.info(f"Agent {self._agent_id} 添加用户消息，上下文长度: {self.context_length}")
            return True
        except Exception as e:
            logger.error(f"添加用户消息失败: {str(e)}")
            self._restore_backup_state()
            return False
    
    def add_assistant_message(self, content: str) -> bool:
        """安全添加助手消息"""
        try:
            self._backup_current_state()
            self._current_state = self._current_state.add_message("assistant", content)
            logger.info(f"Agent {self._agent_id} 添加助手消息，上下文长度: {self.context_length}")
            return True
        except Exception as e:
            logger.error(f"添加助手消息失败: {str(e)}")
            self._restore_backup_state()
            return False
    
    def apply_context_compression(self, compressed_context: List[Dict[str, str]]) -> bool:
        """安全应用上下文压缩"""
        try:
            if not validate_context_structure(compressed_context):
                logger.error("压缩后的上下文结构无效")
                return False
            
            self._backup_current_state()
            self._current_state = self._current_state.apply_compression(compressed_context)
            logger.info(f"Agent {self._agent_id} 应用上下文压缩，新长度: {self.context_length}")
            return True
        except Exception as e:
            logger.error(f"应用上下文压缩失败: {str(e)}")
            self._restore_backup_state()
            return False
    
    def sync_with_agent(self, agent) -> bool:
        """与agent对象同步上下文"""
        try:
            if hasattr(agent, 'public_context'):
                agent.public_context = self.current_context
                logger.info(f"Agent {self._agent_id} 上下文同步完成，长度: {self.context_length}")
                return True
            else:
                logger.error(f"Agent {self._agent_id} 没有public_context属性")
                return False
        except Exception as e:
            logger.error(f"上下文同步失败: {str(e)}")
            return False
    
    def _backup_current_state(self):
        """备份当前状态"""
        self._backup_states.append(self._current_state)
        # 只保留最近的3个备份
        if len(self._backup_states) > 3:
            self._backup_states.pop(0)
    
    def _restore_backup_state(self):
        """恢复备份状态"""
        if self._backup_states:
            self._current_state = self._backup_states.pop()
            logger.info(f"Agent {self._agent_id} 恢复到备份状态")


# 上下文相关的纯函数
def validate_context_structure(context: List[Dict[str, str]]) -> bool:
    """验证上下文结构的纯函数"""
    if not isinstance(context, list):
        return False
    
    required_fields = {"role", "content"}
    valid_roles = {"system", "user", "assistant"}
    
    for item in context:
        if not isinstance(item, dict):
            return False
        
        if not required_fields.issubset(item.keys()):
            return False
        
        if item["role"] not in valid_roles:
            return False
        
        if not isinstance(item["content"], str):
            return False
    
    return True


def create_system_context(system_prompt: str) -> List[Dict[str, str]]:
    """创建系统上下文的纯函数"""
    return [{"role": "system", "content": system_prompt}]


def merge_contexts_safe(context1: List[Dict[str, str]], context2: List[Dict[str, str]]) -> List[Dict[str, str]]:
    """安全合并两个上下文的纯函数"""
    if not validate_context_structure(context1) or not validate_context_structure(context2):
        logger.error("无效的上下文结构，无法合并")
        return context1  # 返回第一个上下文作为默认值
    
    merged = copy.deepcopy(context1)
    merged.extend(copy.deepcopy(context2))
    return merged


def compress_context_safe(context: List[Dict[str, str]], max_length: int = 10) -> List[Dict[str, str]]:
    """安全压缩上下文的纯函数"""
    if not validate_context_structure(context):
        logger.error("无效的上下文结构，无法压缩")
        return context
    
    if len(context) <= max_length:
        return copy.deepcopy(context)
    
    # 保留系统消息和最近的消息
    system_messages = [msg for msg in context if msg["role"] == "system"]
    non_system_messages = [msg for msg in context if msg["role"] != "system"]
    
    # 保留最近的消息
    recent_messages = non_system_messages[-max_length+len(system_messages):]
    
    compressed = copy.deepcopy(system_messages + recent_messages)
    logger.info(f"上下文压缩: {len(context)} -> {len(compressed)}")
    return compressed


def get_context_summary(context: List[Dict[str, str]]) -> Dict[str, Any]:
    """获取上下文摘要的纯函数"""
    if not validate_context_structure(context):
        return {"error": "Invalid context structure"}
    
    role_counts = {}
    total_length = 0
    
    for msg in context:
        role = msg["role"]
        content_length = len(msg["content"])
        
        if role not in role_counts:
            role_counts[role] = {"count": 0, "total_length": 0}
        
        role_counts[role]["count"] += 1
        role_counts[role]["total_length"] += content_length
        total_length += content_length
    
    return {
        "total_messages": len(context),
        "total_length": total_length,
        "role_breakdown": role_counts,
        "average_message_length": total_length / len(context) if context else 0
    }


@contextmanager
def safe_context_operation(agent_id: str, operation_name: str):
    """安全上下文操作的上下文管理器"""
    start_time = datetime.datetime.now()
    logger.info(f"开始上下文操作: {operation_name} for Agent {agent_id}")
    
    try:
        yield
        duration = (datetime.datetime.now() - start_time).total_seconds()
        logger.info(f"上下文操作完成: {operation_name} for Agent {agent_id}, 耗时: {duration:.3f}s")
    except Exception as e:
        duration = (datetime.datetime.now() - start_time).total_seconds()
        logger.error(f"上下文操作失败: {operation_name} for Agent {agent_id}, 耗时: {duration:.3f}s, 错误: {str(e)}")
        raise


# 全局上下文管理器实例
_context_managers: Dict[str, SafeContextManager] = {}


def get_context_manager(agent_id: str, initial_context: List[Dict[str, str]] = None) -> SafeContextManager:
    """获取或创建agent的上下文管理器"""
    if agent_id not in _context_managers:
        _context_managers[agent_id] = SafeContextManager(agent_id, initial_context)
    return _context_managers[agent_id]


def ensure_agent_context_safe(agent, agent_id: str, use_memory: bool = True) -> Tuple[bool, List[Dict[str, str]]]:
    """确保agent有安全的上下文"""
    try:
        with safe_context_operation(agent_id, "ensure_context"):
            # 获取上下文管理器
            context_manager = get_context_manager(agent_id)
            
            # 如果agent有现有上下文，同步到管理器
            if hasattr(agent, 'public_context') and agent.public_context:
                if validate_context_structure(agent.public_context):
                    # 创建新的管理器以同步现有上下文
                    _context_managers[agent_id] = SafeContextManager(agent_id, agent.public_context)
                    context_manager = _context_managers[agent_id]
                else:
                    logger.warning(f"Agent {agent_id} 的现有上下文结构无效，使用默认上下文")
            
            # 同步回agent
            success = context_manager.sync_with_agent(agent)
            return success, context_manager.current_context
            
    except Exception as e:
        logger.error(f"确保agent上下文安全失败: {str(e)}")
        return False, []


def update_agent_context_atomic(agent, user_message: str, assistant_response: str, agent_id: str) -> bool:
    """原子性更新agent上下文"""
    try:
        with safe_context_operation(agent_id, "atomic_update"):
            context_manager = get_context_manager(agent_id)
            
            # 添加用户消息
            if not context_manager.add_user_message(user_message):
                return False
            
            # 添加助手回复
            if not context_manager.add_assistant_message(assistant_response):
                return False
            
            # 同步到agent
            return context_manager.sync_with_agent(agent)
            
    except Exception as e:
        logger.error(f"原子性更新上下文失败: {str(e)}")
        return False
