"""
Context Engineering模块 - 智能体上下文和记忆管理

包含以下模块：
- ContextEngineering: 核心上下文工程类
- agent_memory_manager: 智能体记忆管理器
- context_safe_manager: 上下文安全管理器
- discussion_state_manager: 讨论状态管理器
- stream_manager: 流式输出管理器
- enhanced_memory_manager: 增强记忆管理器
"""

# 导入主要类和函数，方便外部使用
from .ContextEngineering import ContextEngineering, get_context_engineering, ScratchpadManager
from .agent_memory_manager import AgentMemoryManager, get_agent_memory_manager
from .context_safe_manager import SafeContextManager, ensure_agent_context_safe, update_agent_context_atomic
from .discussion_state_manager import SafeDiscussionManager, DiscussionStatus, UserInterventionType
from .stream_manager import SafeDiscussionStreamer, StreamOutputManager, StreamEventType

__version__ = "1.0.0"
__author__ = "MultiAI Team"
