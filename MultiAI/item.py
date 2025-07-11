"""
item.py - 从 server.py 提取的较长函数模块
包含复杂的业务逻辑函数，保持功能不变
"""

import random
import datetime
import uuid
import threading
import time
import json
from typing import Dict, List, Any, Optional, Tuple

# 导入必要的模块
from context_engineering.ContextEngineering import ContextEngineering, get_context_engineering, ScratchpadManager
from context_engineering.discussion_state_manager import (
    is_complex_question_safe, analyze_user_intervention_safe, 
    UserInterventionType
)

class StructuredDiscussionFramework:
    """结构化讨论框架 - 清晰的阶段控制"""
    
    def __init__(self, user_question, agent_count):
        self.user_question = user_question
        self.agent_count = agent_count
        
        # 明确的讨论阶段
        self.phases = [
            {
                'name': 'analysis',
                'description': '问题分析阶段',
                'rounds_per_agent': 1,  # 每个agent只说一轮
                'prompt_template': self.get_analysis_prompt
            },
            {
                'name': 'discussion', 
                'description': '深入讨论阶段',
                'rounds_per_agent': 1,
                'prompt_template': self.get_discussion_prompt
            },
            {
                'name': 'conclusion',
                'description': '结论整合阶段', 
                'rounds_per_agent': 1,
                'prompt_template': self.get_conclusion_prompt
            }
        ]
        
        self.current_phase = 0
        self.current_round = 0
    
    def get_analysis_prompt(self, agent_id, agent_position, discussion_history):
        """分析阶段提示 - 专注于问题分解"""
        return f"""
【讨论阶段】问题分析阶段 (第1阶段/共3阶段)
【你的任务】分析问题的核心要素，提出你的初步观点

【原始问题】{self.user_question}

【要求】
1. 简洁分析问题的关键点（不超过3个要点）
2. 提出你的初步观点（1-2句话）
3. 不要展开详细论述，等待下一阶段
4. 不要回应其他agent，专注于分析问题本身

请直接开始分析，不要说"感谢"之类的开场白。
"""
    
    def get_discussion_prompt(self, agent_id, agent_position, discussion_history):
        """讨论阶段提示 - 基于前一阶段的分析展开"""
        return f"""
【讨论阶段】深入讨论阶段 (第2阶段/共3阶段)  
【你的任务】基于分析阶段的观点，展开详细论述

【原始问题】{self.user_question}

【分析阶段总结】
{self._get_phase_summary('analysis', discussion_history)}

【要求】
1. 基于你在分析阶段的观点，提供详细论证
2. 可以适当回应其他agent的分析，但要保持简洁
3. 专注于论据和例证，不要重复分析阶段的内容
4. 控制篇幅，重点突出你的核心论据

请直接开始论述，避免重复之前说过的内容。
"""
    
    def get_conclusion_prompt(self, agent_id, agent_position, discussion_history):
        """结论阶段提示 - 整合观点，给出最终结论"""
        return f"""
【讨论阶段】结论整合阶段 (第3阶段/共3阶段)
【你的任务】整合讨论内容，给出最终结论

【原始问题】{self.user_question}

【前期讨论总结】
{self._get_phase_summary('analysis', discussion_history)}
{self._get_phase_summary('discussion', discussion_history)}

【要求】
1. 整合前两阶段的讨论，给出你的最终结论
2. 承认其他agent合理的观点
3. 明确回答用户的原始问题
4. 保持简洁，不要重复之前的论述

这是最后阶段，请给出明确的结论性回答。
"""
    
    def _get_phase_summary(self, phase_name, discussion_history):
        """获取指定阶段的总结"""
        phase_messages = [msg for msg in discussion_history if msg.get('phase') == phase_name]
        if not phase_messages:
            return f"【{phase_name}阶段】暂无内容"
        
        summary_lines = []
        for msg in phase_messages[-self.agent_count:]:  # 只取最近一轮
            summary_lines.append(f"Agent {msg['agent_id']}: {msg['content'][:150]}...")
        
        return f"【{phase_name}阶段】\n" + "\n".join(summary_lines)
    
    def get_current_prompt(self, agent_id, agent_position, discussion_history):
        """获取当前阶段的提示"""
        current_phase_info = self.phases[self.current_phase]
        return current_phase_info['prompt_template'](agent_id, agent_position, discussion_history)
    
    def should_advance_phase(self):
        """判断是否应该进入下一阶段"""
        current_phase_info = self.phases[self.current_phase]
        expected_rounds = current_phase_info['rounds_per_agent'] * self.agent_count
        
        return self.current_round >= expected_rounds
    
    def advance_phase(self):
        """进入下一阶段"""
        if self.current_phase < len(self.phases) - 1:
            self.current_phase += 1
            self.current_round = 0
            return True
        return False
    
    def is_discussion_complete(self):
        """判断讨论是否完成"""
        return (self.current_phase >= len(self.phases) - 1 and 
                self.current_round >= self.phases[self.current_phase]['rounds_per_agent'] * self.agent_count)

class MessageContext:
    """消息上下文管理 - 清晰区分不同类型的消息"""
    
    MESSAGE_TYPES = {
        'USER_ORIGINAL': 'user_original_question',     # 用户原始问题
        'USER_INTERVENTION': 'user_intervention',      # 用户插入发言
        'AGENT_ANALYSIS': 'agent_analysis',           # agent分析阶段发言
        'AGENT_DISCUSSION': 'agent_discussion',       # agent讨论阶段发言  
        'AGENT_CONCLUSION': 'agent_conclusion',       # agent结论阶段发言
        'AGENT_USER_RESPONSE': 'agent_user_response', # agent回应用户插入发言
        'SYSTEM_SUMMARY': 'system_summary'            # 系统总结
    }
    
    def __init__(self):
        self.message_history = []
        self.user_interventions = []
        
    def add_message(self, content, agent_id, message_type, phase=None, responding_to=None):
        """添加消息到上下文"""
        message = {
            'content': content,
            'agent_id': agent_id,
            'type': message_type,
            'phase': phase,
            'responding_to': responding_to,
            'timestamp': time.time(),
            'id': len(self.message_history)
        }
        
        self.message_history.append(message)
        
        # 单独跟踪用户插入发言
        if message_type == self.MESSAGE_TYPES['USER_INTERVENTION']:
            self.user_interventions.append(message)
        
        return message
    
    def get_phase_context(self, phase_name):
        """获取特定阶段的上下文"""
        return [msg for msg in self.message_history if msg.get('phase') == phase_name]
    
    def get_agent_history(self, agent_id, max_messages=3):
        """获取特定agent的历史发言"""
        agent_messages = [msg for msg in self.message_history if msg['agent_id'] == agent_id]
        return agent_messages[-max_messages:]
    
    def get_discussion_context(self, exclude_user_interventions=True):
        """获取纯讨论上下文，可选择排除用户插入发言"""
        if exclude_user_interventions:
            return [msg for msg in self.message_history 
                   if msg['type'] not in [self.MESSAGE_TYPES['USER_INTERVENTION'], 
                                        self.MESSAGE_TYPES['AGENT_USER_RESPONSE']]]
        return self.message_history
    
    def identify_message_source(self, content, source_hint=None):
        """智能识别消息来源"""
        
        # 明确的源头标识
        if source_hint:
            if 'user' in str(source_hint).lower():
                return self.MESSAGE_TYPES['USER_INTERVENTION']
            elif 'agent' in str(source_hint).lower():
                return 'agent_message'
        
        # 内容特征分析
        user_indicators = [
            # 第一人称表达
            '我认为', '我觉得', '我想问', '我的看法',
            # 疑问表达
            '为什么', '怎么', '请问', '能否',
            # 对话指向
            '你们觉得', '你们认为', '@', '#'
        ]
        
        agent_indicators = [
            # 分析性语言
            '基于以上分析', '从专业角度', '综合考虑',
            # 结构化表达
            '首先', '其次', '最后', '总的来说',
            # 引用性语言
            '根据讨论', '如前所述', '正如提到的'
        ]
        
        user_score = sum(1 for indicator in user_indicators if indicator in content)
        agent_score = sum(1 for indicator in agent_indicators if indicator in content)
        
        if user_score > agent_score and user_score > 0:
            return self.MESSAGE_TYPES['USER_INTERVENTION']
        else:
            return 'agent_message'

# 全局停止事件管理
stop_events = {}  # discussion_id -> threading.Event()

def create_stop_event(discussion_id):
    """为讨论创建停止事件"""
    stop_events[discussion_id] = threading.Event()
    return stop_events[discussion_id]

def signal_stop(discussion_id):
    """发送停止信号"""
    if discussion_id in stop_events:
        stop_events[discussion_id].set()

def should_stop(discussion_id):
    """检查是否应该停止"""
    return stop_events.get(discussion_id, threading.Event()).is_set()

def cleanup_stop_event(discussion_id):
    """清理停止事件"""
    if discussion_id in stop_events:
        del stop_events[discussion_id]

class AutoStopController:
    """自动停止控制器 - 极简版本：只检查轮次限制"""
    
    def __init__(self, discussion_id, agent_count, config=None):
        self.discussion_id = discussion_id
        self.agent_count = agent_count
        self.config = config or {
            'max_discussion_rounds': agent_count * 3  # 唯一的配置项：最大讨论轮次
        }
        
        # 只保留轮次跟踪
        self.discussion_rounds = 0      # 正常讨论轮次计数
        self.user_response_count = 0    # 用户发言回复计数（不计入讨论轮次）
    
    def should_auto_stop(self, current_content, discussion_messages, is_user_response=False):
        """极简的自动停止判断 - 只检查轮次限制"""
        
        # 更新轮次计数（用户回复不计入讨论轮次）
        if is_user_response:
            self.user_response_count += 1
        else:
            self.discussion_rounds += 1
        
        # 唯一检查：轮次限制
        if self.discussion_rounds > self.config['max_discussion_rounds']:
            return True, f"讨论轮次已达到上限（{self.discussion_rounds}/{self.config['max_discussion_rounds']}轮）"
        
        return False, None
    
    def get_progress_info(self):
        """获取进度信息 - 只显示轮次进度"""
        return {
            'discussion_rounds': self.discussion_rounds,
            'max_discussion_rounds': self.config['max_discussion_rounds'],
            'user_response_count': self.user_response_count,
            'round_progress': min(self.discussion_rounds / self.config['max_discussion_rounds'], 1.0)
        }

def structured_stream_output(agent_response, agent_id, phase_info, max_length=800):
    """基于阶段的结构化流式输出"""
    
    phase_name = phase_info['name']
    expected_length = {
        'analysis': 200,    # 分析阶段简洁
        'discussion': 600,  # 讨论阶段详细  
        'conclusion': 400   # 结论阶段明确
    }
    
    target_length = expected_length.get(phase_name, 400)
    accumulated_content = ""
    
    # 阶段特定的结束标志
    end_markers = {
        'analysis': ['。', '！', '？'],
        'discussion': ['。', '总之', '因此', '所以'],
        'conclusion': ['总结', '结论', '因此', '最终']
    }
    
    markers = end_markers.get(phase_name, ['。'])
    sentence_count = 0
    
    for chunk in agent_response:
        if chunk:
            accumulated_content += chunk
            
            # 计算句子数量
            for marker in markers:
                sentence_count += chunk.count(marker)
            
            # 阶段特定的停止条件
            if phase_name == 'analysis' and sentence_count >= 3:  # 分析阶段最多3句
                break
            elif phase_name == 'discussion' and len(accumulated_content) > target_length:
                # 讨论阶段在下一个句子结束时停止
                if any(marker in chunk for marker in markers):
                    yield chunk
                    break
            elif phase_name == 'conclusion' and sentence_count >= 5:  # 结论阶段最多5句
                break
            
            yield chunk
    
    return accumulated_content

def safe_determine_primary_responder(user_intervention, discussion_messages, agent_id_map, message_context, app_logger):
    """安全的主要回应者确定，避免误判"""
    
    # 首先验证这确实是用户消息
    message_type = message_context.identify_message_source(
        user_intervention, 'user_intervention'
    )
    
    if message_type != message_context.MESSAGE_TYPES['USER_INTERVENTION']:
        app_logger.warning(f"疑似非用户消息被误判为用户插入发言: {user_intervention[:50]}...")
        return None
    
    # 原有的确定逻辑
    return determine_primary_responder(user_intervention, discussion_messages, agent_id_map, app_logger)

def select_best_agent_for_query(query, agents, agent_id_map, context_engineering_instances, app_logger):
    """
    智能选择最适合回答问题的agent
    
    Args:
        query: 用户查询
        agents: agent列表
        agent_id_map: agent索引映射
        context_engineering_instances: 上下文工程实例字典
        app_logger: 应用日志记录器
        
    Returns:
        tuple: (选中的agent索引, agent_id)
    """
    try:
        # 计算每个agent与查询的相关性
        relevance_scores = {}
        
        for idx, agent in enumerate(agents):
            agent_id = agent_id_map[idx]
            
            # 检查该agent是否有相关记忆/上下文
            if agent_id in context_engineering_instances:
                ce = context_engineering_instances[agent_id]
                # 获取与查询相关的记忆
                relevant_memories = ce.retrieve_from_memory(query, limit=3)
                
                # 计算相关性得分
                relevance_score = 0
                for memory in relevant_memories:
                    relevance_score += memory.get('relevance_score', 0)
                
                # 如果有相关记忆，给予额外加分
                if relevant_memories:
                    relevance_score += len(relevant_memories) * 0.1
                
                relevance_scores[idx] = relevance_score
            else:
                relevance_scores[idx] = 0
        
        # 选择相关性最高的agent
        if max(relevance_scores.values()) > 0:
            best_idx = max(relevance_scores, key=relevance_scores.get)
            app_logger.info(f"基于记忆相关性选择Agent {agent_id_map[best_idx]}，得分: {relevance_scores[best_idx]}")
            return best_idx, agent_id_map[best_idx]
        else:
            # 如果都没有相关记忆，随机选择
            random_idx = random.randint(0, len(agents) - 1)
            app_logger.info(f"随机选择Agent {agent_id_map[random_idx]}")
            return random_idx, agent_id_map[random_idx]
            
    except Exception as e:
        app_logger.error(f"智能Agent选择失败: {str(e)}")
        # 默认选择第一个agent
        return 0, agent_id_map[0]

def calculate_semantic_similarity(text1, text2):
    """计算两个文本的语义相似性（简化版）"""
    import re
    
    # 提取关键词
    words1 = set(re.findall(r'\b\w{2,}\b', text1.lower()))
    words2 = set(re.findall(r'\b\w{2,}\b', text2.lower()))
    
    if not words1 or not words2:
        return 0.0
    
    # 计算交集和并集
    intersection = len(words1.intersection(words2))
    union = len(words1.union(words2))
    
    # Jaccard相似度
    return intersection / union if union > 0 else 0.0

def select_summary_agent(discussion_messages, agents, agent_id_map, app_logger):
    """智能选择最适合做总结的agent"""
    
    # 策略1：选择发言最平衡的agent（非极端立场）
    if len(discussion_messages) >= 4:
        agent_message_counts = {}
        for msg in discussion_messages:
            agent_id = msg['agent_id']
            agent_message_counts[agent_id] = agent_message_counts.get(agent_id, 0) + 1
        
        # 选择发言次数适中的agent
        sorted_agents = sorted(agent_message_counts.items(), key=lambda x: x[1])
        if len(sorted_agents) >= 2:
            middle_agent = sorted_agents[len(sorted_agents)//2][0]
            app_logger.info(f"选择发言适中的Agent {middle_agent} 作为总结者")
            
            # 返回对应的agent索引
            for idx, agent_id in agent_id_map.items():
                if agent_id == middle_agent:
                    return idx, agents[idx]
    
    # 策略2：默认选择第一个agent
    app_logger.info("使用默认Agent 1 作为总结者")
    return 0, agents[0]

def generate_discussion_summary(user_message, discussion_messages, summarizer_agent, 
                              agent_idx, agent_id_map, app_logger, stop_reason=None):
    """生成讨论总结的独立函数 - 支持流式输出"""
    
    if not discussion_messages:
        yield {'type': 'summary_complete', 'content': '没有讨论内容需要总结', 'message': None}
        return
    
    # 构建总结提示
    stop_context = f"\n\n【讨论结束原因】{stop_reason}" if stop_reason else ""
    
    summary_prompt = f"""
【原始问题】{user_message}

【完整讨论记录】
{chr(10).join([f"Agent {msg['agent_id']}: {msg['content']}" for msg in discussion_messages])}
{stop_context}

请基于以上讨论内容，生成一个全面的总结回答用户的原始问题。

总结应该包含：
1. **问题回顾**：简要重述用户的原始问题
2. **主要观点整合**：整合各个Agent的核心观点
3. **关键分歧与共识**：指出讨论中的分歧点和达成的共识
4. **结论与建议**：基于讨论给出明确的结论或建议
5. **重要洞察**：指出讨论中产生的关键洞察和价值

请提供结构化、逻辑清晰的总结回答。
"""
    
    try:
        # 开始总结
        yield {'type': 'summary_start', 'content': '正在生成讨论总结...'}
        
        # 流式生成总结
        summary_response = summarizer_agent.chat(
            summary_prompt, 
            agent_id=agent_idx, 
            eval_targets=[], 
            use_memory=False,
            stream=True  # 启用流式输出
        )
        
        # 处理流式响应
        if hasattr(summary_response, '__iter__') and not isinstance(summary_response, str):
            accumulated_summary = ""
            for chunk in summary_response:
                if chunk:
                    accumulated_summary += chunk
                    yield {
                        'type': 'summary_content_chunk',
                        'agent_id': agent_id_map[agent_idx],
                        'content': chunk
                    }
            summary_content = accumulated_summary
        else:
            summary_content = str(summary_response)
            yield {
                'type': 'summary_content_complete',
                'agent_id': agent_id_map[agent_idx],
                'content': summary_content
            }
        
        # 构建总结消息
        summary_message = {
            'agent_id': agent_id_map[agent_idx],
            'name': f'Agent{agent_id_map[agent_idx]}',
            'avatar': f'img/Zhipu.png',
            'content': summary_content,
            'type': 'summary'
        }
        
        # 完成总结
        yield {
            'type': 'summary_complete',
            'content': summary_content,
            'message': summary_message,
            'summarizer_agent': agent_id_map[agent_idx]
        }
        
        # 添加到讨论记录
        discussion_messages.append({
            'agent_id': agent_id_map[agent_idx],
            'content': summary_content,
            'type': 'summary'
        })
        
    except Exception as e:
        app_logger.error(f"生成总结失败: {str(e)}")
        yield {
            'type': 'summary_complete', 
            'content': '总结生成失败', 
            'message': None,
            'error': str(e)
        }

def determine_primary_responder(user_intervention, discussion_messages, agent_id_map, app_logger):
    """基于上下文动态确定主要回应者，不依赖关键词"""
    try:
        # 策略1：明确指向识别（数字、ID等）
        user_message_lower = user_intervention.lower()
        for agent_id in agent_id_map.values():
            patterns = [
                f"agent {agent_id}",
                f"agent{agent_id}", 
                f"第{agent_id}个",
                f"@{agent_id}",
                f"#{agent_id}"
            ]
            if any(pattern in user_message_lower for pattern in patterns):
                app_logger.info(f"明确指向识别: Agent {agent_id}")
                return agent_id
        
        # 策略2：语义相关性分析
        if discussion_messages:
            # 分析与最近几条消息的相关性
            relevance_scores = []
            
            for msg in discussion_messages[-3:]:  # 检查最近3条消息
                similarity = calculate_semantic_similarity(user_intervention, msg['content'])
                relevance_scores.append({
                    'agent_id': msg['agent_id'],
                    'score': similarity,
                    'recency': len(discussion_messages) - discussion_messages.index(msg)  # 越近权重越高
                })
            
            # 综合评分（相似度 + 时近性）
            if relevance_scores:
                best_match = max(relevance_scores, key=lambda x: x['score'] * 0.7 + x['recency'] * 0.3)
                
                if best_match['score'] > 0.3:  # 相似度阈值
                    app_logger.info(f"语义相关性分析选择: Agent {best_match['agent_id']}, 相似度: {best_match['score']:.3f}")
                    return best_match['agent_id']
        
        # 策略3：问句指向分析
        if '?' in user_intervention or any(word in user_intervention for word in ['吗', '呢', '吧', '如何', '怎么', '为什么']):
            # 问句类型，选择最近发言的agent（可能是针对其发言的问题）
            if discussion_messages:
                last_speaker = discussion_messages[-1]['agent_id']
                app_logger.info(f"问句指向分析选择最近发言者: Agent {last_speaker}")
                return last_speaker
        
        # 策略4：默认选择最近发言的agent
        if discussion_messages:
            default_agent = discussion_messages[-1]['agent_id']
            app_logger.info(f"默认选择最近发言者: Agent {default_agent}")
            return default_agent
        
        # 最后的备选：选择第一个agent
        if agent_id_map:
            fallback_agent = list(agent_id_map.values())[0]
            app_logger.info(f"备选方案选择: Agent {fallback_agent}")
            return fallback_agent
        
        return None
        
    except Exception as e:
        app_logger.error(f"确定主要回应者失败: {str(e)}")
        if discussion_messages:
            return discussion_messages[-1]['agent_id']
        return list(agent_id_map.values())[0] if agent_id_map else None

def should_provide_active_response(agent_id, intervention_id, response_tracking):
    """判断agent是否应该主动回应，而非仅仅作为上下文理解"""
    
    intervention_record = response_tracking.get(intervention_id, {})
    
    # 如果已有主要回应者且不是当前agent，则只能被动理解，不主动回应
    if (intervention_record.get('has_primary_response') and 
        intervention_record.get('primary_responder') != agent_id):
        return False, 'passive_context'  # 作为上下文，不主动回应
    
    # 如果是指定的主要回应者，允许主动回应
    if intervention_record.get('primary_responder') == agent_id:
        return True, 'active_response'
    
    # 如果还没有主要回应者，可以成为主要回应者
    return True, 'become_primary'

def identify_target_agent(user_message, discussion_messages, agent_id_map, app_logger):
    """识别用户发言针对的agent - 兼容旧接口"""
    primary_responder = determine_primary_responder(user_message, discussion_messages, agent_id_map, app_logger)
    return [primary_responder] if primary_responder else []

def build_targeted_response_prompt(user_message, intervention_analysis, response_reason, original_question, discussion_history, agent_position=None, agent_id=None, already_responded=False):
    """构建针对性回应提示 - 支持回应去重"""
    
    # 如果该agent已经回应过这个用户发言，构建不重复回应的提示
    if already_responded:
        prompt = f"【注意】你之前已经回应过用户的这个发言：\"{user_message}\"\n\n"
        prompt += "【当前任务】请继续讨论原定话题，不要再次回应用户的这个发言。\n"
        prompt += "避免重复说\"感谢您的提问\"、\"感谢您的意见\"等已经说过的话。\n\n"
        prompt += f"【讨论历史】\n{discussion_history}\n\n"
        prompt += f"【原定子问题】{original_question}\n\n"
        prompt += "请专注于当前的讨论子问题，提供新的观点和分析。"
        return prompt
    
    prompt = f"【用户针对你的发言】用户刚才说：\"{user_message}\"\n\n"
    prompt += f"【回应原因】{response_reason}\n\n"
    prompt += f"【重要提醒】这是你第一次回应这个用户发言，请充分回应，但之后不要再重复回应同一发言。\n\n"
    
    intervention_type = intervention_analysis.get('type', 'general_comment')
    
    if intervention_type == 'question_to_agent':
        prompt += "【重要】用户对你提出了质疑或疑问，你必须：\n"
        prompt += "1. 直接回应用户的质疑\n"
        prompt += "2. 解释你之前观点的依据\n"
        prompt += "3. 澄清任何可能的误解\n"
        prompt += "4. 如果用户说得对，承认并调整观点\n"
        prompt += "5. 保持谦逊和开放的态度\n\n"
        
    elif intervention_type == 'oppose_viewpoint':
        prompt += "【重要】用户反对你的观点，你必须：\n"
        prompt += "1. 理解用户的反对理由\n"
        prompt += "2. 为你的观点进行有力但谦逊的辩护\n"
        prompt += "3. 提供更多支持证据\n"
        prompt += "4. 或者承认用户观点的合理性并调整立场\n"
        prompt += "5. 寻求与用户观点的共同点\n\n"
        
    elif intervention_type == 'support_viewpoint':
        prompt += "【重要】用户支持你的观点，你可以：\n"
        prompt += "1. 感谢用户的支持\n"
        prompt += "2. 进一步深化和拓展你的观点\n"
        prompt += "3. 提供更多细节和例证\n"
        prompt += "4. 与用户的观点形成呼应\n\n"
        
    elif intervention_type == 'add_constraint':
        prompt += "【重要】用户添加了新的约束条件，你必须：\n"
        prompt += "1. 理解用户提出的新约束\n"
        prompt += "2. 在新约束下重新考虑问题\n"
        prompt += "3. 调整你的观点以符合新约束\n"
        prompt += "4. 说明约束如何影响你的分析\n\n"
    
    else:  # general_comment
        prompt += "【重要】用户发表了意见，你应该：\n"
        prompt += "1. 认真考虑用户的观点\n"
        prompt += "2. 回应用户关心的要点\n"
        prompt += "3. 将用户的意见融入你的思考\n"
        prompt += "4. 表达对用户参与的感谢\n\n"
    
    # 添加立场信息（如果是辩论模式）
    if agent_position:
        prompt += f"【你的立场】请记住你在辩论中的立场是：{agent_position}\n"
        prompt += "在回应用户的同时，也要维护你的立场。\n\n"
    
    prompt += f"【讨论历史】\n{discussion_history}\n\n"
    prompt += f"【原定子问题】{original_question}\n\n"
    prompt += "请首先充分回应用户的发言，然后继续原定的讨论。确保你的回应真诚、有针对性，并推动讨论向前发展。"
    
    return prompt

def analyze_user_intervention(user_intervention, original_question, discussion_messages, main_agent, app_logger):
    """分析用户插入发言的类型和意图 - 使用新的函数式模块"""
    try:
        # 构建讨论上下文
        discussion_context = f"原始问题: {original_question}\n\n讨论历史:\n"
        for msg in discussion_messages:
            discussion_context += f"Agent {msg['agent_id']}: {msg['content']}\n"
        
        # 使用新的函数式分析函数
        intervention_type = analyze_user_intervention_safe(user_intervention, discussion_context)
        
        # 转换为旧版API格式
        return {
            'type': intervention_type.value,
            'target_agent': None,  # 简化版本，后续可以扩展
            'original_analysis': f'Safe analysis result: {intervention_type.value}'
        }
        
    except Exception as e:
        app_logger.error(f"用户发言分析失败: {str(e)}")
        return {
            'type': 'general_comment',
            'target_agent': None,
            'original_analysis': 'analysis_failed'
        }

def adjust_discussion_framework(original_question, user_intervention, original_framework, main_agent, app_logger):
    """根据用户插入发言调整讨论框架"""
    
    adjustment_prompt = f"""
用户原始问题：{original_question}

原始讨论框架：
{original_framework}

用户插入发言：{user_intervention}

用户提出了框架调整建议。请根据用户的建议重新设计讨论框架，生成新的子问题列表。

新的子问题应该：
1. 采纳用户的建议和观点
2. 保持与原始问题的相关性
3. 确保逻辑清晰、结构合理
4. 适合多个AI智能体讨论

输出格式：
1. [调整后的子问题1]
2. [调整后的子问题2]
3. [调整后的子问题3]
...

只输出子问题列表，不要有其他内容。
"""
    
    try:
        adjusted_framework = main_agent.chat(adjustment_prompt, agent_id=0, eval_targets=[], use_memory=False)
        
        # 解析新的子问题
        new_sub_questions = []
        for line in adjusted_framework.strip().split('\n'):
            line = line.strip()
            if line and any(line.startswith(f"{i}.") for i in range(1, 10)):
                question = line[line.find('.')+1:].strip()
                new_sub_questions.append(question)
        
        return new_sub_questions
        
    except Exception as e:
        app_logger.error(f"讨论框架调整失败: {str(e)}")
        # 返回原始框架的子问题
        original_sub_questions = []
        for line in original_framework.strip().split('\n'):
            line = line.strip()
            if line and any(line.startswith(f"{i}.") for i in range(1, 10)):
                question = line[line.find('.')+1:].strip()
                original_sub_questions.append(question)
        return original_sub_questions

def run_discussion_logic(discussion_data, agent_instances, context_engineering_instances, app_logger):
    """异步讨论处理逻辑"""
    agents = discussion_data['agents']
    agent_id_map = discussion_data['agent_id_map']
    current_agent_idx = discussion_data['current_agent_idx']
    user_message = discussion_data['user_message']
    
    try:
        # 第一步：由主agent创建讨论框架
        main_agent = agents[current_agent_idx]
        
        # 判断是否为辩论类型的问题
        is_debate = any(keyword in user_message.lower() for keyword in [
            '更重要', '还是', 'vs', '对比', '比较', '哪个好', '选择', '辩论', 
            '支持', '反对', '观点', '立场', '争议', '分歧'
        ])
        
        if is_debate:
            # 辩论模式框架 - 确保观点对立
            framework_prompt = (
                f"你是一个多智能体辩论系统的主持人。用户问题是：{user_message}\n\n"
                "这是一个辩论类型的问题。你必须确保两个智能体持有明确对立的观点。\n\n"
                "请分析这个问题，找出两个可以辩论的对立观点，然后设计辩论框架：\n\n"
                "首先分析问题的两个对立面：\n"
                "- 观点A：[明确的立场A及其核心论据]\n"
                "- 观点B：[明确的立场B及其核心论据，必须与观点A形成对立]\n\n"
                "然后设计辩论流程：\n"
                "1. [让第一个智能体支持观点A并提供论据]\n"
                "2. [让第二个智能体支持观点B并提供论据]\n"
                "3. [第一个智能体反驳观点B并强化观点A]\n"
                "4. [第二个智能体反驳观点A并强化观点B]\n"
                "5. [中立评判者评判双方表现]\n\n"
                "重要要求：\n"
                "- 观点A和观点B必须是明确对立的，不能有重叠或相似性\n"
                "- 每个观点都要有充分的论据支持\n"
                "- 确保两个观点都有合理性，避免一边倒的情况\n\n"
                "只输出上述格式的内容，不要有其他内容。"
            )
        else:
            # 普通讨论框架
            framework_prompt = (
                f"你是一个多智能体系统中的主持人。用户问题是：{user_message}\n\n"
                "请为这个问题设计一个讨论框架，将问题分解为几个子问题或讨论点，以便多个AI智能体进行讨论。"
                "每个子问题应该是用户原始问题的一个方面，所有子问题合起来应该能够全面回答用户的问题。"
                "输出格式：\n"
                "1. [子问题1]\n"
                "2. [子问题2]\n"
                "3. [子问题3]\n"
                "...\n"
                "只输出子问题列表，不要有其他内容。"
            )
        
        framework = main_agent.chat(framework_prompt, agent_id=current_agent_idx, eval_targets=[], use_memory=False)
        discussion_data['framework'] = framework
        
        # 解析子问题和观点
        sub_questions = []
        debate_viewpoints = {'A': '', 'B': ''}
        
        if is_debate:
            # 解析辩论框架，提取对立观点
            lines = framework.strip().split('\n')
            for line in lines:
                line = line.strip()
                if line.startswith('- 观点A：'):
                    debate_viewpoints['A'] = line[6:].strip()
                elif line.startswith('- 观点B：'):
                    debate_viewpoints['B'] = line[6:].strip()
                elif line and any(line.startswith(f"{i}.") for i in range(1, 10)):
                    question = line[line.find('.')+1:].strip()
                    sub_questions.append(question)
        else:
            # 普通讨论模式
            for line in framework.strip().split('\n'):
                line = line.strip()
                if line and any(line.startswith(f"{i}.") for i in range(1, 10)):
                    question = line[line.find('.')+1:].strip()
                    sub_questions.append(question)
        
        discussion_data['sub_questions'] = sub_questions
        discussion_data['is_debate'] = is_debate
        discussion_data['debate_viewpoints'] = debate_viewpoints
        
        # 辩论模式预先分配立场
        if is_debate:
            discussion_data['agent_positions'] = {}
            # 第一个agent支持观点A，第二个agent支持观点B
            agent_ids = list(agent_id_map.values())
            if len(agent_ids) >= 2:
                discussion_data['agent_positions'][agent_ids[0]] = 'A'
                discussion_data['agent_positions'][agent_ids[1]] = 'B'
        
        # 进行讨论的其余逻辑...
        # [继续实现讨论逻辑，包括轮流回答、总结等]
        
        # 标记讨论完成
        discussion_data['status'] = 'completed'
        
    except Exception as e:
        app_logger.error(f"讨论处理异常: {str(e)}", exc_info=True)
        discussion_data['status'] = 'error'
        
        error_message = {
            'agent_id': agent_id_map[current_agent_idx],
            'name': f'Agent{agent_id_map[current_agent_idx]}',
            'avatar': f'img/Zhipu.png',
            'content': f"讨论过程中发生错误: {str(e)}",
            'type': 'error'
        }
        
        discussion_data['messages'].append(error_message)

def run_structured_discussion_stream(discussion_data, agent_instances, context_engineering_instances, app_logger):
    """结构化流式讨论处理逻辑 - 基于清晰的阶段框架"""
    agents = discussion_data['agents']
    agent_id_map = discussion_data['agent_id_map']
    user_message = discussion_data['user_message']
    discussion_id = discussion_data['id']
    
    try:
        # 初始化结构化框架和消息上下文
        framework = StructuredDiscussionFramework(user_message, len(agents))
        message_context = MessageContext()
        
        # 记录用户原始问题
        message_context.add_message(
            user_message, 'user', message_context.MESSAGE_TYPES['USER_ORIGINAL']
        )
        
        # 创建停止事件
        stop_event = create_stop_event(discussion_id)
        app_logger.info(f"开始结构化讨论: {len(agents)}个agent，3个阶段")
        
        # 阶段循环
        while not framework.is_discussion_complete():
            current_phase = framework.phases[framework.current_phase]
            phase_name = current_phase['name']
            
            # 发送阶段开始消息
            yield {
                'type': 'phase_start',
                'phase': phase_name,
                'description': current_phase['description'],
                'content': f"进入{current_phase['description']}..."
            }
            
            # Agent轮流发言
            for agent_idx in range(len(agents)):
                # 检查停止条件
                if discussion_data.get('stop_requested', False):
                    yield {'type': 'discussion_stopped', 'content': '讨论已停止'}
                    cleanup_stop_event(discussion_id)
                    return
                
                agent = agents[agent_idx]
                agent_id = agent_id_map[agent_idx]
                
                # 获取当前讨论历史
                discussion_history = message_context.get_discussion_context()
                
                # 生成阶段特定的提示
                prompt = framework.get_current_prompt(agent_id, None, discussion_history)
                
                # Agent开始思考
                yield {
                    'type': 'agent_thinking',
                    'agent_id': agent_id,
                    'model': 'Zhipu' if 'Zhipu' in str(type(agent)) else 'Spark',
                    'phase': phase_name,
                    'question': f"{current_phase['description']} - 第{framework.current_round + 1}轮"
                }
                
                try:
                    # Agent生成回应
                    agent_response = agent.chat(
                        prompt,
                        agent_id=agent_idx,
                        eval_targets=[],
                        use_memory=False,
                        stream=True
                    )
                    
                    # 使用结构化流式输出控制
                    if hasattr(agent_response, '__iter__') and not isinstance(agent_response, str):
                        accumulated_content = ""
                        phase_info = {'name': phase_name}
                        
                        for chunk in structured_stream_output(agent_response, agent_id, phase_info):
                            if chunk:
                                accumulated_content += chunk
                                yield {
                                    'type': 'agent_content_chunk',
                                    'agent_id': agent_id,
                                    'model': 'Zhipu' if 'Zhipu' in str(type(agent)) else 'Spark',
                                    'content': chunk,
                                    'phase': phase_name
                                }
                        
                        agent_response_text = accumulated_content
                    else:
                        agent_response_text = str(agent_response)
                        yield {
                            'type': 'agent_content_complete',
                            'agent_id': agent_id,
                            'model': 'Zhipu' if 'Zhipu' in str(type(agent)) else 'Spark',
                            'content': agent_response_text,
                            'phase': phase_name
                        }
                
                except Exception as e:
                    app_logger.error(f"Agent {agent_id} 在{phase_name}阶段回答失败: {str(e)}")
                    agent_response_text = f"在{current_phase['description']}中遇到问题: {str(e)}"
                    yield {
                        'type': 'agent_content_complete',
                        'agent_id': agent_id,
                        'model': 'Zhipu' if 'Zhipu' in str(type(agent)) else 'Spark',
                        'content': agent_response_text,
                        'phase': phase_name
                    }
                
                # 记录消息到上下文
                message_type = {
                    'analysis': message_context.MESSAGE_TYPES['AGENT_ANALYSIS'],
                    'discussion': message_context.MESSAGE_TYPES['AGENT_DISCUSSION'],
                    'conclusion': message_context.MESSAGE_TYPES['AGENT_CONCLUSION']
                }.get(phase_name, 'agent_message')
                
                message_context.add_message(
                    agent_response_text, agent_id, message_type, phase=phase_name
                )
                
                # 完成当前agent
                yield {
                    'type': 'agent_complete',
                    'agent_id': agent_id,
                    'model': 'Zhipu' if 'Zhipu' in str(type(agent)) else 'Spark',
                    'phase': phase_name
                }
                
                framework.current_round += 1
            
            # 阶段完成
            yield {
                'type': 'phase_complete',
                'phase': phase_name,
                'content': f"{current_phase['description']}完成"
            }
            
            # 进入下一阶段
            if not framework.advance_phase():
                break
        
        # 生成最终总结
        yield {'type': 'summary_start', 'content': '正在生成讨论总结...'}
        
        try:
            # 选择总结agent
            discussion_messages = message_context.get_discussion_context()
            summary_agent_idx, summary_agent = select_summary_agent(
                discussion_messages, agents, agent_id_map, app_logger
            )
            
            # 生成总结
            yield from generate_discussion_summary(
                user_message, discussion_messages, summary_agent,
                summary_agent_idx, agent_id_map, app_logger,
                stop_reason="完成所有讨论阶段"
            )
        
        except Exception as e:
            app_logger.error(f"生成总结失败: {str(e)}")
            yield {'type': 'summary_complete', 'content': '总结生成失败', 'message': None}
        
        # 讨论完成
        discussion_data['status'] = 'completed'
        yield {
            'type': 'discussion_complete',
            'status': 'completed',
            'phases_completed': len(framework.phases)
        }
        
    except Exception as e:
        app_logger.error(f"结构化讨论异常: {str(e)}", exc_info=True)
        discussion_data['status'] = 'error'
        yield {'type': 'error', 'content': f'讨论异常: {str(e)}'}
    
    finally:
        cleanup_stop_event(discussion_id)

def run_discussion_stream_logic(discussion_data, agent_instances, context_engineering_instances, app_logger):
    """流式讨论处理逻辑的生成器函数 - 支持用户插入发言"""
    agents = discussion_data['agents']
    agent_id_map = discussion_data['agent_id_map']
    current_agent_idx = discussion_data['current_agent_idx']
    user_message = discussion_data['user_message']
    discussion_id = discussion_data['id']
    
    # 创建便笺管理器，用于存储agent的推理过程
    scratchpad = ScratchpadManager(discussion_id)
    
    try:
        # 第一步：由主agent创建讨论框架
        main_agent = agents[current_agent_idx]
        
        # 发送框架创建开始消息
        yield {'type': 'framework_start', 'content': '正在创建讨论框架...'}
        
        # 判断是否为辩论类型的问题
        is_debate = any(keyword in user_message.lower() for keyword in [
            '更重要', '还是', 'vs', '对比', '比较', '哪个好', '选择', '辩论', 
            '支持', '反对', '观点', '立场', '争议', '分歧'
        ])
        
        if is_debate:
            # 辩论模式框架
            framework_prompt = (
                f"你是一个多智能体辩论系统的主持人。用户问题是：{user_message}\n\n"
                "这是一个辩论类型的问题。你必须确保两个智能体持有明确对立的观点。\n\n"
                "请分析这个问题，找出两个可以辩论的对立观点，然后设计辩论框架：\n\n"
                "首先分析问题的两个对立面：\n"
                "- 观点A：[明确的立场A及其核心论据]\n"
                "- 观点B：[明确的立场B及其核心论据，必须与观点A形成对立]\n\n"
                "然后设计辩论流程：\n"
                "1. [让第一个智能体支持观点A并提供论据]\n"
                "2. [让第二个智能体支持观点B并提供论据]\n"
                "3. [第一个智能体反驳观点B并强化观点A]\n"
                "4. [第二个智能体反驳观点A并强化观点B]\n"
                "5. [中立评判者评判双方表现]\n\n"
                "重要要求：\n"
                "- 观点A和观点B必须是明确对立的，不能有重叠或相似性\n"
                "- 每个观点都要有充分的论据支持\n"
                "- 确保两个观点都有合理性，避免一边倒的情况\n\n"
                "只输出上述格式的内容，不要有其他内容。"
            )
        else:
            # 普通讨论框架
            framework_prompt = (
                f"你是一个多智能体系统中的主持人。用户问题是：{user_message}\n\n"
                "请为这个问题设计一个讨论框架，将问题分解为几个子问题或讨论点，以便多个AI智能体进行讨论。"
                "每个子问题应该是用户原始问题的一个方面，所有子问题合起来应该能够全面回答用户的问题。"
                "输出格式：\n"
                "1. [子问题1]\n"
                "2. [子问题2]\n"
                "3. [子问题3]\n"
                "...\n"
                "只输出子问题列表，不要有其他内容。"
            )
        
        framework = main_agent.chat(framework_prompt, agent_id=current_agent_idx, eval_targets=[], use_memory=False)
        discussion_data['framework'] = framework
        
        # 发送框架创建完成消息
        yield {'type': 'framework_complete', 'content': framework}
        
        # 解析子问题和观点
        sub_questions = []
        debate_viewpoints = {'A': '', 'B': ''}
        
        if is_debate:
            # 解析辩论框架，提取对立观点
            lines = framework.strip().split('\n')
            for line in lines:
                line = line.strip()
                if line.startswith('- 观点A：'):
                    debate_viewpoints['A'] = line[6:].strip()
                elif line.startswith('- 观点B：'):
                    debate_viewpoints['B'] = line[6:].strip()
                elif line and any(line.startswith(f"{i}.") for i in range(1, 10)):
                    question = line[line.find('.')+1:].strip()
                    sub_questions.append(question)
        else:
            # 普通讨论模式
            for line in framework.strip().split('\n'):
                line = line.strip()
                if line and any(line.startswith(f"{i}.") for i in range(1, 10)):
                    question = line[line.find('.')+1:].strip()
                    sub_questions.append(question)
        
        discussion_data['sub_questions'] = sub_questions
        discussion_data['is_debate'] = is_debate
        discussion_data['debate_viewpoints'] = debate_viewpoints
        
        # 辩论模式预先分配立场
        if is_debate:
            discussion_data['agent_positions'] = {}
            # 第一个agent支持观点A，第二个agent支持观点B
            agent_ids = list(agent_id_map.values())
            if len(agent_ids) >= 2:
                discussion_data['agent_positions'][agent_ids[0]] = 'A'
                discussion_data['agent_positions'][agent_ids[1]] = 'B'
        
        # 主要讨论循环 - 实现完整的agent轮流回答和用户插入发言处理
        if not sub_questions:
            # 如果没有子问题，直接进行开放式讨论
            sub_questions = [user_message]
        
        discussion_messages = []
        current_question_idx = 0
        round_count = 0
        max_rounds = len(sub_questions) * len(agents)  # 每个子问题每个agent回答一次
        
        # 双层用户发言跟踪系统
        user_interventions_tracking = {}  # intervention_id -> {'content': str, 'responded_by': set, 'timestamp': float, 'round': int}
        active_response_tracking = {}     # intervention_id -> {'primary_responder': agent_id, 'has_primary_response': bool, 'response_round': int}
        intervention_counter = 0
        
        # 创建停止事件和自动停止控制器
        stop_event = create_stop_event(discussion_id)
        auto_stop_controller = AutoStopController(discussion_id, len(agents))
        
        app_logger.info(f"讨论开始，最大轮次: {auto_stop_controller.config['max_discussion_rounds']}轮（{len(agents)}个agent × 3）")
        
        while current_question_idx < len(sub_questions) and round_count < max_rounds:
            # 检查是否需要停止或暂停
            if discussion_data.get('stop_requested', False):
                yield {'type': 'discussion_stopped', 'content': '讨论已停止'}
                break
            
            # 主动检查用户插入发言（不仅限于暂停状态）
            user_intervention = discussion_data.get('user_intervention')
            if user_intervention:
                # 处理用户插入发言（无论是否暂停）
                app_logger.info(f"检测到用户插入发言: {user_intervention}")
                
                # 立即暂停当前讨论流程来处理用户发言
                discussion_data['paused'] = True
            
            if discussion_data.get('paused', False):
                # 暂停时检查用户插入发言
                user_intervention = discussion_data.get('user_intervention')
                if user_intervention:
                    # 处理用户插入发言
                    app_logger.info(f"检测到用户插入发言: {user_intervention}")
                    
                    # 改进的用户发言唯一性判断 - 按内容+轮次+时间窗口
                    intervention_signature = f"{user_intervention.strip()}_{round_count//2}"  # 每两轮为一个时间窗口
                    current_intervention_id = f"intervention_{intervention_signature}"
                    
                    # 检查是否是真正的新发言
                    is_new_intervention = True
                    current_time = time.time()
                    
                    # 检查最近是否有相似的发言（时间窗口5分钟）
                    for existing_id, existing_data in user_interventions_tracking.items():
                        if (existing_data['content'].strip() == user_intervention.strip() and 
                            current_time - existing_data['timestamp'] < 300):  # 5分钟内的相同发言
                            current_intervention_id = existing_id
                            is_new_intervention = False
                            break
                    
                    # 记录这个用户发言（如果是新发言）
                    if is_new_intervention and current_intervention_id not in user_interventions_tracking:
                        user_interventions_tracking[current_intervention_id] = {
                            'content': user_intervention,
                            'responded_by': set(),
                            'timestamp': current_time,
                            'round': round_count,
                            'debate_context': is_debate  # 记录是否在辩论模式
                        }
                        app_logger.info(f"新用户发言已记录: {current_intervention_id}")
                    elif not is_new_intervention:
                        app_logger.info(f"检测到重复用户发言，使用现有ID: {current_intervention_id}")
                    
                    # 分析用户发言
                    intervention_analysis = analyze_user_intervention(
                        user_intervention, user_message, discussion_messages, main_agent, app_logger
                    )
                    
                    # 确定主要回应者
                    primary_responder = determine_primary_responder(
                        user_intervention, discussion_messages, agent_id_map, app_logger
                    )
                    
                    # 如果用户建议调整框架
                    if '框架' in user_intervention or '重新' in user_intervention or '调整' in user_intervention:
                        # 调整讨论框架
                        new_sub_questions = adjust_discussion_framework(
                            user_message, user_intervention, framework, main_agent, app_logger
                        )
                        if new_sub_questions:
                            sub_questions = new_sub_questions
                            discussion_data['sub_questions'] = sub_questions
                            yield {'type': 'framework_adjusted', 'content': f'根据用户建议调整了讨论框架: {new_sub_questions}'}
                    
                    # 双层处理机制：检查是否应该主动回应
                    should_respond, response_type = should_provide_active_response(
                        primary_responder, current_intervention_id, active_response_tracking
                    )
                    
                    if not should_respond:
                        # 如果已有主要回应者，跳过处理，让所有agent继续以被动方式理解
                        app_logger.info(f"用户发言 {current_intervention_id} 已有主要回应者，跳过额外主动回应")
                        discussion_data['user_intervention'] = None
                        discussion_data['paused'] = False
                        continue
                    
                    # 设置回应agent
                    responding_agent_id = primary_responder
                    responding_agent_idx = None
                    responding_agent = None
                    
                    # 找到对应的agent实例
                    for idx, agent_id in agent_id_map.items():
                        if agent_id == responding_agent_id:
                            responding_agent_idx = idx
                            responding_agent = agents[idx]
                            break
                    
                    if responding_agent is None:
                        app_logger.error(f"无法找到Agent {responding_agent_id} 的实例")
                        discussion_data['user_intervention'] = None
                        discussion_data['paused'] = False
                        continue
                    
                    # 记录主要回应状态
                    active_response_tracking[current_intervention_id] = {
                        'primary_responder': responding_agent_id,
                        'has_primary_response': False,  # 将在回应后设置为True
                        'response_round': round_count,
                        'timestamp': current_time
                    }
                    app_logger.info(f"设置Agent {responding_agent_id} 为用户发言 {current_intervention_id} 的主要回应者")
                    
                    # 记录该agent将要回应这个用户发言
                    user_interventions_tracking[current_intervention_id]['responded_by'].add(responding_agent_id)
                    
                    # 构建针对性回应提示
                    current_question = sub_questions[current_question_idx] if current_question_idx < len(sub_questions) else user_message
                    discussion_history = "\n".join([f"Agent {msg['agent_id']}: {msg['content']}" for msg in discussion_messages])
                    
                    response_reason = f"用户对你的发言提出了意见，你需要回应用户的观点"
                    agent_position = discussion_data.get('agent_positions', {}).get(responding_agent_id)
                    
                    targeted_prompt = build_targeted_response_prompt(
                        user_intervention, intervention_analysis, response_reason, 
                        current_question, discussion_history, agent_position, 
                        responding_agent_id, False  # 主要回应者第一次回应，不是重复回应
                    )
                    
                    # Agent回应用户发言
                    yield {'type': 'agent_thinking', 'agent_id': responding_agent_id, 'model': 'Zhipu' if 'Zhipu' in str(type(responding_agent)) else 'Spark', 'question': '回应用户发言'}
                    
                    # 获取上下文（如果启用）
                    context = ""
                    if responding_agent_id in context_engineering_instances:
                        ce = context_engineering_instances[responding_agent_id]
                        context = ce.get_compressed_context(user_intervention)
                    
                    try:
                        # 尝试流式回应
                        if hasattr(responding_agent, 'chat') and hasattr(responding_agent.chat, '__call__'):
                            agent_response = responding_agent.chat(
                                targeted_prompt, 
                                agent_id=responding_agent_idx, 
                                eval_targets=[], 
                                use_memory=False,
                                stream=True,
                                context=context
                            )
                            
                            # 检查是否是生成器（流式）
                            if hasattr(agent_response, '__iter__') and not isinstance(agent_response, str):
                                # 流式响应
                                accumulated_content = ""
                                for chunk in agent_response:
                                    # 在每个chunk输出前检查停止条件
                                    if discussion_data.get('stop_requested', False) or should_stop(discussion_id):
                                        app_logger.info("检测到停止请求，中断流式输出")
                                        yield {'type': 'discussion_stopped', 'content': '讨论已被用户停止'}
                                        cleanup_stop_event(discussion_id)
                                        return
                                    
                                    # 检查自动停止条件
                                    if accumulated_content:  # 有内容时才检查
                                        should_auto_stop_flag, auto_stop_reason = auto_stop_controller.should_auto_stop(
                                            accumulated_content, discussion_messages, is_user_response=True  # 用户发言回复
                                        )
                                        if should_auto_stop_flag:
                                            app_logger.info(f"触发自动停止: {auto_stop_reason}")
                                            
                                            # 先完成当前agent输出
                                            yield {
                                                'type': 'agent_complete',
                                                'agent_id': responding_agent_id,
                                                'model': 'Zhipu' if 'Zhipu' in str(type(responding_agent)) else 'Spark',
                                                'auto_stopped': True
                                            }
                                            
                                            # 选择总结agent并生成总结
                                            summary_agent_idx, summary_agent = select_summary_agent(
                                                discussion_messages, agents, agent_id_map, app_logger
                                            )
                                            
                                            # 生成总结
                                            yield from generate_discussion_summary(
                                                user_message, discussion_messages, summary_agent,
                                                summary_agent_idx, agent_id_map, app_logger,
                                                stop_reason=auto_stop_reason
                                            )
                                            
                                            # 发送自动停止通知
                                            yield {'type': 'auto_stopped', 'content': f'讨论自动停止: {auto_stop_reason}'}
                                            cleanup_stop_event(discussion_id)
                                            return
                                    
                                    if chunk:
                                        accumulated_content += chunk
                                        yield {
                                            'type': 'agent_content_chunk',
                                            'agent_id': responding_agent_id,
                                            'model': 'Zhipu' if 'Zhipu' in str(type(responding_agent)) else 'Spark',
                                            'content': chunk
                                        }
                                
                                agent_response_text = accumulated_content
                            else:
                                # 非流式响应
                                agent_response_text = str(agent_response)
                                yield {
                                    'type': 'agent_content_complete',
                                    'agent_id': responding_agent_id,
                                    'model': 'Zhipu' if 'Zhipu' in str(type(responding_agent)) else 'Spark',
                                    'content': agent_response_text
                                }
                        else:
                            # 备用方式
                            agent_response_text = responding_agent.chat(
                                targeted_prompt, 
                                agent_id=responding_agent_idx, 
                                eval_targets=[], 
                                use_memory=False,
                                context=context
                            )
                            yield {
                                'type': 'agent_content_complete',
                                'agent_id': responding_agent_id,
                                'model': 'Zhipu' if 'Zhipu' in str(type(responding_agent)) else 'Spark',
                                'content': agent_response_text
                            }
                    
                    except Exception as e:
                        app_logger.error(f"Agent回应用户发言失败: {str(e)}")
                        agent_response_text = f"抱歉，我在回应你的发言时遇到了问题: {str(e)}"
                        yield {
                            'type': 'agent_content_complete',
                            'agent_id': responding_agent_id,
                            'model': 'Zhipu' if 'Zhipu' in str(type(responding_agent)) else 'Spark',
                            'content': agent_response_text
                        }
                    
                    # 添加到讨论历史
                    discussion_messages.append({
                        'agent_id': responding_agent_id,
                        'content': agent_response_text,
                        'type': 'user_response',
                        'responding_to_user': True
                    })
                    
                    # 标记主要回应已完成
                    if current_intervention_id in active_response_tracking:
                        active_response_tracking[current_intervention_id]['has_primary_response'] = True
                        app_logger.info(f"Agent {responding_agent_id} 已完成对用户发言 {current_intervention_id} 的主要回应")
                    
                    yield {
                        'type': 'agent_complete',
                        'agent_id': responding_agent_id,
                        'model': 'Zhipu' if 'Zhipu' in str(type(responding_agent)) else 'Spark'
                    }
                    
                    # 清除用户插入发言标记
                    discussion_data['user_intervention'] = None
                    discussion_data['paused'] = False
                
                # 如果仍在暂停状态，等待
                if discussion_data.get('paused', False):
                    time.sleep(0.5)
                    continue
            
            # 正常讨论流程
            current_question = sub_questions[current_question_idx]
            agent_idx = round_count % len(agents)
            agent = agents[agent_idx]
            agent_id = agent_id_map[agent_idx]
            
            # 构建针对当前子问题的提示
            discussion_history = "\n".join([f"Agent {msg['agent_id']}: {msg['content']}" for msg in discussion_messages])
            
            if is_debate and agent_id in discussion_data.get('agent_positions', {}):
                # 辩论模式
                position = discussion_data['agent_positions'][agent_id]
                viewpoint = debate_viewpoints.get(position, '')
                prompt = f"""
【辩论问题】{user_message}

【你的立场】你支持观点{position}: {viewpoint}

【当前讨论子问题】{current_question}

【讨论历史】
{discussion_history}

请基于你的立场回答当前子问题，提供有力的论据支持你的观点。
记住你的立场是{position}，要与对方观点形成鲜明对比。
"""
            else:
                # 普通讨论模式
                prompt = f"""
【原始问题】{user_message}

【当前子问题】{current_question}

【讨论历史】
{discussion_history}

请针对当前子问题提供你的分析和观点。考虑之前的讨论内容，提供新的见解或补充观点。
"""
            
            # Agent开始思考
            yield {
                'type': 'agent_thinking',
                'agent_id': agent_id,
                'model': 'Zhipu' if 'Zhipu' in str(type(agent)) else 'Spark',
                'question': current_question
            }
            
            # 获取上下文（如果启用）
            context = ""
            if agent_id in context_engineering_instances:
                ce = context_engineering_instances[agent_id]
                context = ce.get_compressed_context(prompt)
            
            try:
                # Agent回答子问题
                agent_response = agent.chat(
                    prompt, 
                    agent_id=agent_idx, 
                    eval_targets=[], 
                    use_memory=False,
                    stream=True,
                    context=context
                )
                
                # 检查是否是流式响应
                if hasattr(agent_response, '__iter__') and not isinstance(agent_response, str):
                    # 流式响应
                    accumulated_content = ""
                    for chunk in agent_response:
                        if chunk:
                            accumulated_content += chunk
                            yield {
                                'type': 'agent_content_chunk',
                                'agent_id': agent_id,
                                'model': 'Zhipu' if 'Zhipu' in str(type(agent)) else 'Spark',
                                'content': chunk
                            }
                    
                    agent_response_text = accumulated_content
                else:
                    # 非流式响应
                    agent_response_text = str(agent_response)
                    yield {
                        'type': 'agent_content_complete',
                        'agent_id': agent_id,
                        'model': 'Zhipu' if 'Zhipu' in str(type(agent)) else 'Spark',
                        'content': agent_response_text
                    }
            
            except Exception as e:
                app_logger.error(f"Agent回答失败: {str(e)}")
                agent_response_text = f"抱歉，我在回答这个问题时遇到了问题: {str(e)}"
                yield {
                    'type': 'agent_content_complete',
                    'agent_id': agent_id,
                    'model': 'Zhipu' if 'Zhipu' in str(type(agent)) else 'Spark',
                    'content': agent_response_text
                }
            
            # 检查自动停止条件（正常讨论）
            should_auto_stop_flag, auto_stop_reason = auto_stop_controller.should_auto_stop(
                agent_response_text, discussion_messages, is_user_response=False  # 正常讨论轮次
            )
            if should_auto_stop_flag:
                app_logger.info(f"触发自动停止: {auto_stop_reason}")
                
                # 先完成当前agent输出
                yield {
                    'type': 'agent_complete',
                    'agent_id': agent_id,
                    'model': 'Zhipu' if 'Zhipu' in str(type(agent)) else 'Spark',
                    'auto_stopped': True
                }
                
                # 添加当前回复到讨论历史（用于总结）
                discussion_messages.append({
                    'agent_id': agent_id,
                    'content': agent_response_text,
                    'question': current_question,
                    'round': round_count
                })
                
                # 选择总结agent并生成总结
                summary_agent_idx, summary_agent = select_summary_agent(
                    discussion_messages, agents, agent_id_map, app_logger
                )
                
                # 生成总结
                yield from generate_discussion_summary(
                    user_message, discussion_messages, summary_agent,
                    summary_agent_idx, agent_id_map, app_logger,
                    stop_reason=auto_stop_reason
                )
                
                # 发送自动停止通知
                yield {'type': 'auto_stopped', 'content': f'讨论自动停止: {auto_stop_reason}'}
                
                # 发送讨论完成状态
                yield {'type': 'discussion_complete', 'status': 'auto_stopped', 'reason': 'round_limit_reached'}
                
                cleanup_stop_event(discussion_id)
                return
            
            # 添加到讨论历史
            discussion_messages.append({
                'agent_id': agent_id,
                'content': agent_response_text,
                'question': current_question,
                'round': round_count
            })
            
            yield {
                'type': 'agent_complete',
                'agent_id': agent_id,
                'model': 'Zhipu' if 'Zhipu' in str(type(agent)) else 'Spark'
            }
            
            round_count += 1
            
            # 如果当前子问题每个agent都回答过一轮，进入下一个子问题
            if round_count % len(agents) == 0:
                current_question_idx += 1
        
        # 讨论结束，生成总结
        if discussion_messages:
            yield {'type': 'summary_start', 'content': '正在生成讨论总结...'}
            
            summary_prompt = f"""
【原始问题】{user_message}

【完整讨论记录】
{chr(10).join([f"Agent {msg['agent_id']}: {msg['content']}" for msg in discussion_messages])}

请基于以上讨论内容，生成一个全面的总结回答用户的原始问题。
总结应该：
1. 整合各个Agent的观点
2. 提供平衡和客观的分析
3. 给出明确的结论或建议
4. 指出讨论中的关键洞察

请提供结构化的总结回答。
"""
            
            try:
                summary = main_agent.chat(summary_prompt, agent_id=current_agent_idx, eval_targets=[], use_memory=False)
                
                summary_message = {
                    'agent_id': agent_id_map[current_agent_idx],
                    'name': f'Agent{agent_id_map[current_agent_idx]}',
                    'avatar': f'img/Zhipu.png',
                    'content': summary,
                    'type': 'summary'
                }
                
                yield {
                    'type': 'summary_complete',
                    'content': summary,
                    'message': summary_message
                }
                
                discussion_messages.append({
                    'agent_id': agent_id_map[current_agent_idx],
                    'content': summary,
                    'type': 'summary'
                })
                
            except Exception as e:
                app_logger.error(f"生成总结失败: {str(e)}")
                yield {'type': 'summary_complete', 'content': '总结生成失败', 'message': None}
        
        # 保存讨论记录到便笺
        discussion_data['messages'] = discussion_messages
        
        # 提取讨论结论
        conclusion = f"讨论主题: {user_message}\n参与Agent: {list(agent_id_map.values())}\n讨论轮次: {round_count}\n主要观点: " + "; ".join([msg['content'][:100] + "..." for msg in discussion_messages[:3]])
        
        # 将结论添加到每个agent的记忆中
        for agent_id in agent_id_map.values():
            if agent_id in context_engineering_instances:
                ce = context_engineering_instances[agent_id]
                ce.add_to_memory("conclusions", conclusion, {
                    "source": "discussion",
                    "discussion_id": discussion_id,
                    "timestamp": datetime.datetime.now().isoformat()
                })
        
        # 清理便笺
        scratchpad.clear()
        
        # 标记讨论完成
        discussion_data['status'] = 'completed'
        yield {'type': 'discussion_complete', 'status': 'completed', 'conclusion': conclusion}
        
    except Exception as e:
        app_logger.error(f"流式讨论处理异常: {str(e)}", exc_info=True)
        discussion_data['status'] = 'error'
        
        error_message = {
            'agent_id': agent_id_map[current_agent_idx] if 'current_agent_idx' in locals() else '1',
            'name': f'Agent{agent_id_map[current_agent_idx] if "current_agent_idx" in locals() else "1"}',
            'avatar': f'img/Zhipu.png',
            'content': f"讨论过程中发生错误: {str(e)}",
            'type': 'error'
        }
        
        discussion_data['messages'].append(error_message)
        yield {'type': 'error', 'content': f'讨论异常: {str(e)}', 'message': error_message}

def is_complex_question(message, agent_count=1, app_logger=None):
    """
    使用新的函数式模块判断问题是否复杂，需要多agent讨论
    """
    try:
        is_complex, analysis = is_complex_question_safe(message, agent_count)
        if app_logger:
            app_logger.info(f"复杂度判断结果: {is_complex}, 分析: {analysis}")
        return is_complex
    except Exception as e:
        if app_logger:
            app_logger.error(f"问题复杂度判断失败: {str(e)}")
        return False
