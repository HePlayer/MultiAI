"""
流式输出管理模块 - 使用函数式编程和不可变状态
防止流式输出过程中的意外修改
"""

import json
import copy
import datetime
from typing import Dict, Any, Optional, Iterator, Tuple
from dataclasses import dataclass, replace
from enum import Enum


class StreamEventType(Enum):
    """流式事件类型枚举"""
    START = "start"
    CONTENT = "content"
    END = "end"
    ERROR = "error"
    AGENT_THINKING = "agent_thinking"
    AGENT_CONTENT_CHUNK = "agent_content_chunk"
    AGENT_CONTENT_COMPLETE = "agent_content_complete"
    AGENT_COMPLETE = "agent_complete"
    FRAMEWORK_START = "framework_start"
    FRAMEWORK_COMPLETE = "framework_complete"
    DISCUSSION_COMPLETE = "discussion_complete"
    DISCUSSION_STOPPED = "discussion_stopped"


@dataclass(frozen=True)
class StreamState:
    """不可变的流式状态"""
    discussion_id: str
    agent_id: str
    is_active: bool = False
    buffer: str = ""
    chunks_sent: int = 0
    error_count: int = 0
    start_time: Optional[str] = None
    end_time: Optional[str] = None
    
    def start_stream(self) -> 'StreamState':
        """开始流式输出，返回新状态"""
        return replace(
            self,
            is_active=True,
            start_time=datetime.datetime.now().isoformat(),
            buffer="",
            chunks_sent=0,
            error_count=0
        )
    
    def add_chunk(self, chunk: str) -> 'StreamState':
        """添加内容块，返回新状态"""
        return replace(
            self,
            buffer=self.buffer + chunk,
            chunks_sent=self.chunks_sent + 1
        )
    
    def add_error(self) -> 'StreamState':
        """记录错误，返回新状态"""
        return replace(self, error_count=self.error_count + 1)
    
    def end_stream(self) -> 'StreamState':
        """结束流式输出，返回新状态"""
        return replace(
            self,
            is_active=False,
            end_time=datetime.datetime.now().isoformat()
        )


@dataclass(frozen=True)
class StreamEvent:
    """不可变的流式事件"""
    event_type: StreamEventType
    agent_id: str
    content: str = ""
    model: str = ""
    discussion_id: str = ""
    timestamp: str = ""
    metadata: Dict[str, Any] = None
    
    def __post_init__(self):
        if not self.timestamp:
            object.__setattr__(self, 'timestamp', datetime.datetime.now().isoformat())
        if self.metadata is None:
            object.__setattr__(self, 'metadata', {})


class StreamOutputManager:
    """流式输出管理器 - 封装流式输出逻辑"""
    
    def __init__(self, discussion_id: str, agent_id: str):
        self._initial_state = StreamState(discussion_id, agent_id)
        self._current_state = self._initial_state
        self._event_log = []
    
    @property
    def current_state(self) -> StreamState:
        """获取当前状态的只读副本"""
        return self._current_state
    
    def start_stream(self) -> StreamEvent:
        """开始流式输出"""
        self._current_state = self._current_state.start_stream()
        event = StreamEvent(
            event_type=StreamEventType.START,
            agent_id=self._current_state.agent_id,
            discussion_id=self._current_state.discussion_id
        )
        self._event_log.append(event)
        return event
    
    def add_content_chunk(self, chunk: str, model: str = "") -> StreamEvent:
        """添加内容块"""
        # 验证chunk大小
        if len(chunk) > 2048:  # 限制单个chunk大小
            chunk = chunk[:2048]
        
        # 清理内容
        cleaned_chunk = self._sanitize_content(chunk)
        
        self._current_state = self._current_state.add_chunk(cleaned_chunk)
        event = StreamEvent(
            event_type=StreamEventType.CONTENT,
            agent_id=self._current_state.agent_id,
            content=cleaned_chunk,
            model=model,
            discussion_id=self._current_state.discussion_id
        )
        self._event_log.append(event)
        return event
    
    def end_stream(self, final_content: str = "") -> StreamEvent:
        """结束流式输出"""
        self._current_state = self._current_state.end_stream()
        
        # 使用buffer内容作为最终内容（如果没有提供）
        content = final_content or self._current_state.buffer
        
        event = StreamEvent(
            event_type=StreamEventType.END,
            agent_id=self._current_state.agent_id,
            content=content,
            discussion_id=self._current_state.discussion_id,
            metadata={
                'total_chunks': self._current_state.chunks_sent,
                'total_length': len(self._current_state.buffer),
                'duration': self._calculate_duration()
            }
        )
        self._event_log.append(event)
        return event
    
    def add_error(self, error_message: str) -> StreamEvent:
        """添加错误事件"""
        self._current_state = self._current_state.add_error()
        event = StreamEvent(
            event_type=StreamEventType.ERROR,
            agent_id=self._current_state.agent_id,
            content=f"[流式输出错误: {error_message}]",
            discussion_id=self._current_state.discussion_id
        )
        self._event_log.append(event)
        return event
    
    def _sanitize_content(self, content: str) -> str:
        """清理内容，移除敏感信息"""
        if not content:
            return ""
        
        # 移除可能的内部调试信息
        sensitive_patterns = [
            'debug:', 'DEBUG:', 'trace:', 'TRACE:',
            'complexity_score:', 'internal_state:',
            'is_complex_question:', 'framework_analysis:'
        ]
        
        cleaned = content
        for pattern in sensitive_patterns:
            if pattern in cleaned:
                # 移除包含敏感信息的行
                lines = cleaned.split('\n')
                cleaned_lines = [line for line in lines if pattern not in line]
                cleaned = '\n'.join(cleaned_lines)
        
        return cleaned
    
    def _calculate_duration(self) -> float:
        """计算流式输出持续时间"""
        if not self._current_state.start_time or not self._current_state.end_time:
            return 0.0
        
        try:
            start = datetime.datetime.fromisoformat(self._current_state.start_time)
            end = datetime.datetime.fromisoformat(self._current_state.end_time)
            return (end - start).total_seconds()
        except:
            return 0.0
    
    def get_event_log(self) -> list:
        """获取事件日志的只读副本"""
        return copy.deepcopy(self._event_log)


class SafeDiscussionStreamer:
    """安全的讨论流式输出器"""
    
    def __init__(self, discussion_data: Dict[str, Any]):
        # 深拷贝讨论数据，防止意外修改
        self._discussion_data = copy.deepcopy(discussion_data)
        self._stream_managers = {}
        self._event_sequence = []
    
    def get_stream_manager(self, agent_id: str) -> StreamOutputManager:
        """获取或创建agent的流式管理器"""
        if agent_id not in self._stream_managers:
            self._stream_managers[agent_id] = StreamOutputManager(
                self._discussion_data['id'],
                agent_id
            )
        return self._stream_managers[agent_id]
    
    def stream_agent_thinking(self, agent_id: str, question: str, model: str = "") -> StreamEvent:
        """流式输出agent思考状态"""
        event = StreamEvent(
            event_type=StreamEventType.AGENT_THINKING,
            agent_id=agent_id,
            content=question,
            model=model,
            discussion_id=self._discussion_data['id']
        )
        self._event_sequence.append(event)
        return event
    
    def stream_agent_response_start(self, agent_id: str, model: str = "") -> StreamEvent:
        """开始agent回复流式输出"""
        manager = self.get_stream_manager(agent_id)
        return manager.start_stream()
    
    def stream_agent_content_chunk(self, agent_id: str, chunk: str, model: str = "") -> StreamEvent:
        """流式输出agent内容块"""
        manager = self.get_stream_manager(agent_id)
        return manager.add_content_chunk(chunk, model)
    
    def stream_agent_response_end(self, agent_id: str, final_content: str = "") -> StreamEvent:
        """结束agent回复流式输出"""
        manager = self.get_stream_manager(agent_id)
        return manager.end_stream(final_content)
    
    def stream_framework_event(self, event_type: StreamEventType, content: str = "") -> StreamEvent:
        """流式输出框架事件"""
        event = StreamEvent(
            event_type=event_type,
            agent_id="system",
            content=content,
            discussion_id=self._discussion_data['id']
        )
        self._event_sequence.append(event)
        return event
    
    def get_discussion_summary(self) -> Dict[str, Any]:
        """获取讨论摘要（只读）"""
        return {
            'discussion_id': self._discussion_data['id'],
            'total_events': len(self._event_sequence),
            'agents_participated': list(self._stream_managers.keys()),
            'status': self._discussion_data.get('status', 'unknown'),
            'stream_summary': {
                agent_id: {
                    'chunks_sent': manager.current_state.chunks_sent,
                    'total_content_length': len(manager.current_state.buffer),
                    'error_count': manager.current_state.error_count
                }
                for agent_id, manager in self._stream_managers.items()
            }
        }


# 流式输出相关的纯函数
def create_stream_event_data(event: StreamEvent) -> Dict[str, Any]:
    """创建流式事件数据的纯函数"""
    return {
        'type': event.event_type.value,
        'agent_id': event.agent_id,
        'content': event.content,
        'model': event.model,
        'discussion_id': event.discussion_id,
        'timestamp': event.timestamp,
        **event.metadata
    }


def format_stream_response(event: StreamEvent) -> str:
    """格式化流式响应的纯函数"""
    data = create_stream_event_data(event)
    return f"data: {json.dumps(data, ensure_ascii=False)}\n\n"


def validate_stream_chunk(chunk: str, max_size: int = 1024) -> Tuple[bool, str]:
    """验证流式块的纯函数"""
    if not isinstance(chunk, str):
        return False, "Chunk must be a string"
    
    if len(chunk) > max_size:
        return False, f"Chunk size {len(chunk)} exceeds maximum {max_size}"
    
    # 检查是否包含控制字符
    if any(ord(c) < 32 and c not in '\n\r\t' for c in chunk):
        return False, "Chunk contains invalid control characters"
    
    return True, "Valid"


def sanitize_for_frontend(data: Dict[str, Any]) -> Dict[str, Any]:
    """清理发送到前端的数据，移除内部字段"""
    # 定义前端允许的字段白名单
    FRONTEND_ALLOWED_FIELDS = {
        'type', 'agent_id', 'content', 'model', 'discussion_id', 
        'timestamp', 'status', 'mode', 'question'
    }
    
    # 定义需要移除的内部字段
    INTERNAL_FIELDS = {
        'framework', 'sub_questions', 'agent_positions', 
        'complexity_analysis', 'debug_info', 'is_complex_question',
        'framework_analysis', 'internal_state', 'eval_targets'
    }
    
    cleaned_data = {}
    for key, value in data.items():
        if key in FRONTEND_ALLOWED_FIELDS and key not in INTERNAL_FIELDS:
            cleaned_data[key] = value
    
    return cleaned_data
