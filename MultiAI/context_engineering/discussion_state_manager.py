"""
讨论状态管理模块 - 使用函数式编程管理讨论状态
包括暂停/继续、用户插入发言等功能
"""

import copy
import datetime
import uuid
from typing import Dict, Any, List, Optional, Tuple
from dataclasses import dataclass, replace
from enum import Enum


class DiscussionStatus(Enum):
    """讨论状态枚举"""
    STARTING = "starting"
    RUNNING = "running"
    PAUSED = "paused"
    WAITING_USER = "waiting_user"
    STOPPING = "stopping"
    COMPLETED = "completed"
    ERROR = "error"


class UserInterventionType(Enum):
    """用户插入发言类型"""
    FRAMEWORK_CHANGE = "framework_change"
    QUESTION_TO_AGENT = "question_to_agent"
    SUPPORT_VIEWPOINT = "support_viewpoint"
    OPPOSE_VIEWPOINT = "oppose_viewpoint"
    ADD_CONSTRAINT = "add_constraint"
    GENERAL_COMMENT = "general_comment"


@dataclass(frozen=True)
class DiscussionState:
    """不可变的讨论状态"""
    discussion_id: str
    user_message: str
    agent_ids: List[str]
    status: DiscussionStatus = DiscussionStatus.STARTING
    current_agent_idx: int = 0
    current_question_idx: int = 0
    is_debate: bool = False
    paused: bool = False
    stop_requested: bool = False
    created_time: str = ""
    last_updated: str = ""
    
    def __post_init__(self):
        if not self.created_time:
            now = datetime.datetime.now().isoformat()
            object.__setattr__(self, 'created_time', now)
            object.__setattr__(self, 'last_updated', now)
    
    def update_status(self, new_status: DiscussionStatus) -> 'DiscussionState':
        """更新状态，返回新实例"""
        return replace(
            self,
            status=new_status,
            last_updated=datetime.datetime.now().isoformat()
        )
    
    def pause_discussion(self) -> 'DiscussionState':
        """暂停讨论，返回新实例"""
        return replace(
            self,
            status=DiscussionStatus.PAUSED,
            paused=True,
            last_updated=datetime.datetime.now().isoformat()
        )
    
    def resume_discussion(self) -> 'DiscussionState':
        """恢复讨论，返回新实例"""
        return replace(
            self,
            status=DiscussionStatus.RUNNING,
            paused=False,
            last_updated=datetime.datetime.now().isoformat()
        )
    
    def request_stop(self) -> 'DiscussionState':
        """请求停止讨论，返回新实例"""
        return replace(
            self,
            status=DiscussionStatus.STOPPING,
            stop_requested=True,
            last_updated=datetime.datetime.now().isoformat()
        )
    
    def advance_to_next_agent(self) -> 'DiscussionState':
        """推进到下一个agent，返回新实例"""
        next_idx = (self.current_agent_idx + 1) % len(self.agent_ids)
        return replace(
            self,
            current_agent_idx=next_idx,
            last_updated=datetime.datetime.now().isoformat()
        )
    
    def advance_to_next_question(self) -> 'DiscussionState':
        """推进到下一个问题，返回新实例"""
        return replace(
            self,
            current_question_idx=self.current_question_idx + 1,
            last_updated=datetime.datetime.now().isoformat()
        )


@dataclass(frozen=True)
class DiscussionMessage:
    """不可变的讨论消息"""
    agent_id: str
    content: str
    message_type: str = "agent_response"
    model: str = ""
    position: str = ""  # 用于辩论模式的立场
    timestamp: str = ""
    metadata: Dict[str, Any] = None
    
    def __post_init__(self):
        if not self.timestamp:
            object.__setattr__(self, 'timestamp', datetime.datetime.now().isoformat())
        if self.metadata is None:
            object.__setattr__(self, 'metadata', {})


@dataclass(frozen=True)
class UserIntervention:
    """用户插入发言数据"""
    content: str
    intervention_type: UserInterventionType
    target_agent: str = ""
    timestamp: str = ""
    analysis_result: Dict[str, Any] = None
    
    def __post_init__(self):
        if not self.timestamp:
            object.__setattr__(self, 'timestamp', datetime.datetime.now().isoformat())
        if self.analysis_result is None:
            object.__setattr__(self, 'analysis_result', {})


class SafeDiscussionManager:
    """安全的讨论管理器"""
    
    def __init__(self, user_message: str, agent_ids: List[str]):
        self._discussion_id = str(uuid.uuid4())
        self._state = DiscussionState(self._discussion_id, user_message, agent_ids.copy())
        self._messages: List[DiscussionMessage] = []
        self._framework = ""
        self._sub_questions: List[str] = []
        self._agent_positions: Dict[str, str] = {}  # 辩论立场
        self._user_interventions: List[UserIntervention] = []
    
    @property
    def discussion_id(self) -> str:
        """获取讨论ID"""
        return self._discussion_id
    
    @property
    def current_state(self) -> DiscussionState:
        """获取当前状态的只读副本"""
        return self._state
    
    @property
    def messages(self) -> List[DiscussionMessage]:
        """获取消息列表的只读副本"""
        return copy.deepcopy(self._messages)
    
    @property
    def sub_questions(self) -> List[str]:
        """获取子问题列表的只读副本"""
        return self._sub_questions.copy()
    
    def set_framework(self, framework: str, sub_questions: List[str], is_debate: bool = False) -> bool:
        """设置讨论框架"""
        try:
            self._framework = framework
            self._sub_questions = sub_questions.copy()
            self._state = replace(self._state, is_debate=is_debate)
            return True
        except Exception:
            return False
    
    def set_agent_positions(self, positions: Dict[str, str]) -> bool:
        """设置agent立场（辩论模式）"""
        try:
            self._agent_positions = positions.copy()
            return True
        except Exception:
            return False
    
    def add_message(self, agent_id: str, content: str, message_type: str = "agent_response", 
                   model: str = "", position: str = "") -> bool:
        """添加讨论消息"""
        try:
            message = DiscussionMessage(
                agent_id=agent_id,
                content=content,
                message_type=message_type,
                model=model,
                position=position
            )
            self._messages.append(message)
            return True
        except Exception:
            return False
    
    def pause_discussion(self) -> bool:
        """暂停讨论"""
        try:
            self._state = self._state.pause_discussion()
            return True
        except Exception:
            return False
    
    def resume_discussion(self) -> bool:
        """恢复讨论"""
        try:
            self._state = self._state.resume_discussion()
            return True
        except Exception:
            return False
    
    def request_stop(self) -> bool:
        """请求停止讨论"""
        try:
            self._state = self._state.request_stop()
            return True
        except Exception:
            return False
    
    def add_user_intervention(self, content: str, intervention_type: UserInterventionType, 
                            target_agent: str = "") -> bool:
        """添加用户插入发言"""
        try:
            intervention = UserIntervention(
                content=content,
                intervention_type=intervention_type,
                target_agent=target_agent
            )
            self._user_interventions.append(intervention)
            
            # 添加用户消息到讨论历史
            self.add_message("user", content, "user_intervention")
            return True
        except Exception:
            return False
    
    def advance_progress(self) -> bool:
        """推进讨论进度"""
        try:
            if self._state.current_question_idx < len(self._sub_questions) - 1:
                self._state = self._state.advance_to_next_question()
            self._state = self._state.advance_to_next_agent()
            return True
        except Exception:
            return False
    
    def complete_discussion(self) -> bool:
        """完成讨论"""
        try:
            self._state = self._state.update_status(DiscussionStatus.COMPLETED)
            return True
        except Exception:
            return False
    
    def get_frontend_safe_data(self) -> Dict[str, Any]:
        """获取前端安全的数据（过滤内部字段）"""
        # 只返回前端需要的基本信息
        safe_messages = []
        for msg in self._messages:
            safe_message = {
                'agent_id': msg.agent_id,
                'content': msg.content,
                'type': msg.message_type,
                'model': msg.model,
                'timestamp': msg.timestamp
            }
            # 只在辩论模式下包含立场信息
            if self._state.is_debate and msg.position:
                safe_message['position'] = msg.position
                
            safe_messages.append(safe_message)
        
        return {
            'discussion_id': self._discussion_id,
            'status': self._state.status.value,
            'messages': safe_messages,
            'is_paused': self._state.paused,
            'current_agent_id': self._state.agent_ids[self._state.current_agent_idx] if self._state.agent_ids else None,
            'total_questions': len(self._sub_questions),
            'current_question_index': self._state.current_question_idx
        }
    
    def get_full_state_for_backend(self) -> Dict[str, Any]:
        """获取后端完整状态（包含内部字段）"""
        return {
            'id': self._discussion_id,
            'state': self._state,
            'messages': self._messages,
            'framework': self._framework,
            'sub_questions': self._sub_questions,
            'agent_positions': self._agent_positions,
            'user_interventions': self._user_interventions
        }


# 讨论相关的纯函数
def is_complex_question_safe(message: str, agent_count: int) -> Tuple[bool, Dict[str, Any]]:
    """安全的问题复杂度判断纯函数"""
    # 单agent时不需要讨论
    if agent_count <= 1:
        return False, {
            'reason': 'single_agent',
            'complexity_score': 0,
            'keywords_found': []
        }
    
    try:
        message_lower = message.lower().strip()
        
        # 简单问题的关键词和模式
        simple_patterns = [
            r'^(你好|hi|hello|早上好|晚上好|晚安|再见|bye)[\s\?？。！!]*$',
            r'^(谢谢|感谢|不客气|没关系)[\s\?？。！!]*$',
            r'^(今天几号|现在几点|什么时候|怎么样)[\s\?？。！!]*$',
            r'^(你是谁|介绍.*自己|你叫什么)[\s\?？。！!]*$',
            r'^(测试|试试|hello|test)[\s\?？。！!]*$',
            r'^(你.*记住|还记得|上次.*说)[\s\?？。！!]*$'
        ]
        
        # 检查是否匹配简单模式
        import re
        for pattern in simple_patterns:
            if re.match(pattern, message_lower):
                return False, {
                    'reason': 'simple_pattern_match',
                    'matched_pattern': pattern,
                    'complexity_score': 0,
                    'keywords_found': []
                }
        
        # 问题长度判断
        if len(message.strip()) <= 10:
            return False, {
                'reason': 'too_short',
                'length': len(message),
                'complexity_score': 0,
                'keywords_found': []
            }
        
        # 复杂问题的关键词
        complex_keywords = [
            '更重要', '还是', 'vs', '对比', '比较', '哪个好', '选择', '辩论',
            '支持', '反对', '观点', '立场', '争议', '分歧',
            '分析', '探讨', '研究', '论证', '评价', '评估', '深入',
            '多角度', '全面', '综合', '系统',
            '设计', '规划', '方案', '策略', '计划', '建议', '解决方案',
            '如何实现', '怎么做', '步骤', '流程'
        ]
        
        # 统计复杂关键词
        found_keywords = [keyword for keyword in complex_keywords if keyword in message]
        complexity_score = len(found_keywords)
        
        # 需要多个复杂关键词才判断为复杂问题
        is_complex = complexity_score >= 2
        
        return is_complex, {
            'reason': 'complex_keywords' if is_complex else 'insufficient_complexity',
            'complexity_score': complexity_score,
            'keywords_found': found_keywords,
            'threshold': 2
        }
        
    except Exception as e:
        return False, {
            'reason': 'analysis_error',
            'error': str(e),
            'complexity_score': 0,
            'keywords_found': []
        }


def should_show_discussion_panel(complexity_result: Tuple[bool, Dict[str, Any]], 
                               agent_count: int) -> bool:
    """判断是否显示讨论悬浮窗的纯函数"""
    is_complex, analysis = complexity_result
    
    # 只有在多agent且问题复杂时才显示
    if agent_count <= 1:
        return False
    
    if not is_complex:
        return False
    
    # 额外检查：确保分析结果合理
    if analysis.get('complexity_score', 0) < 2:
        return False
    
    return True


def analyze_user_intervention_safe(user_message: str, discussion_context: str) -> UserInterventionType:
    """安全分析用户插入发言类型的纯函数"""
    try:
        message_lower = user_message.lower()
        
        # 框架改变关键词
        framework_keywords = ['换个角度', '重新组织', '重新分析', '这样分析', '改变思路']
        if any(keyword in message_lower for keyword in framework_keywords):
            return UserInterventionType.FRAMEWORK_CHANGE
        
        # 针对特定agent的疑问
        agent_question_patterns = ['agent', '你刚才说', '刚才的观点', '我对.*疑问']
        if any(pattern in message_lower for pattern in agent_question_patterns):
            return UserInterventionType.QUESTION_TO_AGENT
        
        # 支持观点
        support_keywords = ['同意', '支持', '我觉得对', '有道理', '正确']
        if any(keyword in message_lower for keyword in support_keywords):
            return UserInterventionType.SUPPORT_VIEWPOINT
        
        # 反对观点
        oppose_keywords = ['不同意', '反对', '不对', '有问题', '错误']
        if any(keyword in message_lower for keyword in oppose_keywords):
            return UserInterventionType.OPPOSE_VIEWPOINT
        
        # 添加约束
        constraint_keywords = ['还要考虑', '需要考虑', '加上', '另外', '限制', '条件']
        if any(keyword in message_lower for keyword in constraint_keywords):
            return UserInterventionType.ADD_CONSTRAINT
        
        # 默认为一般评论
        return UserInterventionType.GENERAL_COMMENT
        
    except Exception:
        return UserInterventionType.GENERAL_COMMENT


def filter_discussion_data_for_frontend(discussion_data: Dict[str, Any]) -> Dict[str, Any]:
    """过滤讨论数据，移除内部字段的纯函数"""
    # 前端允许的字段白名单
    FRONTEND_ALLOWED_FIELDS = {
        'discussion_id', 'status', 'messages', 'is_paused', 
        'current_agent_id', 'total_questions', 'current_question_index',
        'mode', 'stream_events'
    }
    
    # 需要移除的内部字段
    INTERNAL_FIELDS = {
        'framework', 'sub_questions', 'agent_positions', 
        'complexity_analysis', 'debug_info', 'is_complex_question',
        'user_interventions', 'state', 'full_context'
    }
    
    filtered_data = {}
    for key, value in discussion_data.items():
        if key in FRONTEND_ALLOWED_FIELDS and key not in INTERNAL_FIELDS:
            if key == 'messages' and isinstance(value, list):
                # 过滤消息中的内部字段
                filtered_messages = []
                for msg in value:
                    if isinstance(msg, dict):
                        filtered_msg = {
                            k: v for k, v in msg.items() 
                            if k in {'agent_id', 'content', 'type', 'model', 'timestamp', 'position'}
                        }
                        filtered_messages.append(filtered_msg)
                filtered_data[key] = filtered_messages
            else:
                filtered_data[key] = value
    
    return filtered_data


# 全局讨论管理器实例
_discussion_managers: Dict[str, SafeDiscussionManager] = {}


def get_discussion_manager(discussion_id: str) -> Optional[SafeDiscussionManager]:
    """获取讨论管理器"""
    return _discussion_managers.get(discussion_id)


def create_discussion_manager(user_message: str, agent_ids: List[str]) -> SafeDiscussionManager:
    """创建新的讨论管理器"""
    manager = SafeDiscussionManager(user_message, agent_ids)
    _discussion_managers[manager.discussion_id] = manager
    return manager


def remove_discussion_manager(discussion_id: str) -> bool:
    """移除讨论管理器"""
    if discussion_id in _discussion_managers:
        del _discussion_managers[discussion_id]
        return True
    return False
