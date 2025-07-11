import re
import json
import requests
import datetime
import os
import pickle
from typing import List, Dict, Any, Optional, Union

SPARK_API = "BcSOEhzuOxuqbZVBjZWI:lLBQCKkaFGdmPdWNRICk"
ZHIPU_API = "3615e68f81697dc2db853f13cd8ae37d.gpGd71TLyTTkCepS"

# 内存上下文存储目录
MEMORY_DIR = "agent_memory"
os.makedirs(MEMORY_DIR, exist_ok=True)

class Agent:
    """
    Agent基类，提供基本的上下文管理和记忆功能
    """
    def set_agent_info(self, agent_id=None, eval_targets=None):
        """
        设置agent编号和评价对象编号
        """
        self.agent_id = agent_id
        self.eval_targets = eval_targets if eval_targets is not None else []
        # 初始化记忆文件路径
        if agent_id is not None:
            self.memory_file = os.path.join(MEMORY_DIR, f"agent_{agent_id}_memory.pkl")
            # 加载已有记忆
            self._load_memory()
    
    def __init__(self):
        # 基本系统提示
        self.system_prompt = "你是一个智能助手，请直接回答用户的问题。"
        
        # 公共上下文（当前会话）
        self.public_context = [{
            "role": "system",
            "content": self.system_prompt
        }]
        
        # 长期记忆（跨会话持久化）
        self.memory = {
            "facts": [],           # 事实性知识
            "user_preferences": [], # 用户偏好
            "conversation_history": [], # 重要对话历史
            "skills": [],          # 学习到的技能
            "relationships": []    # 与用户或其他agent的关系
        }
        
        # 记忆文件路径
        self.memory_file = None
        self.agent_id = None
        self.eval_targets = []
        
        # 记忆检索索引
        self.memory_index = {}
    
    def reset_public_context(self):
        """重置当前会话上下文"""
        self.public_context = [{
            "role": "system",
            "content": self.system_prompt
        }]
    
    def add_public_context(self, context):
        """
        添加公共上下文（当前会话）
        :param context: 上下文内容
        """
        self.public_context.append(context)
    
    def add_memory(self, memory_type: str, content: Dict[str, Any]):
        """
        添加长期记忆
        :param memory_type: 记忆类型（facts, user_preferences, conversation_history, skills, relationships）
        :param content: 记忆内容
        """
        if memory_type in self.memory:
            # 检查是否已存在类似内容，避免重复
            content_text = content.get("content", "")
            if not any(self._text_similarity(existing.get("content", ""), content_text) > 0.7 
                      for existing in self.memory[memory_type]):
                self.memory[memory_type].append(content)
                # 更新索引
                self._update_memory_index(memory_type, len(self.memory[memory_type])-1, content_text)
                # 保存记忆
                self._save_memory()
                return True
        return False
    
    def _text_similarity(self, text1: str, text2: str) -> float:
        """
        计算两段文本的简单相似度（0-1之间）
        """
        if not text1 or not text2:
            return 0
        
        # 简单实现，基于共同词汇比例
        words1 = set(text1.lower().split())
        words2 = set(text2.lower().split())
        
        if not words1 or not words2:
            return 0
            
        common_words = words1.intersection(words2)
        return len(common_words) / max(len(words1), len(words2))
    
    def _update_memory_index(self, memory_type: str, idx: int, content: str):
        """
        更新记忆检索索引
        """
        # 为内容中的每个词创建索引
        words = set(content.lower().split())
        for word in words:
            if word not in self.memory_index:
                self.memory_index[word] = []
            self.memory_index[word].append((memory_type, idx))
    
    def retrieve_memory(self, query: str, memory_type: Optional[str] = None, limit: int = 5) -> List[Dict[str, Any]]:
        """
        检索相关记忆
        :param query: 查询文本
        :param memory_type: 可选，限制检索的记忆类型
        :param limit: 返回结果数量限制
        :return: 相关记忆列表
        """
        query_words = set(query.lower().split())
        results = []
        
        # 收集所有匹配的记忆引用
        memory_refs = []
        for word in query_words:
            if word in self.memory_index:
                memory_refs.extend(self.memory_index[word])
        
        # 统计每个记忆项的匹配次数
        from collections import Counter
        memory_counts = Counter(memory_refs)
        
        # 按匹配次数排序
        sorted_refs = sorted(memory_counts.items(), key=lambda x: x[1], reverse=True)
        
        # 提取记忆内容
        for (mem_type, idx), _ in sorted_refs:
            if memory_type is not None and mem_type != memory_type:
                continue
                
            if idx < len(self.memory[mem_type]):
                results.append(self.memory[mem_type][idx])
                
            if len(results) >= limit:
                break
        
        return results
    
    def get_relevant_context(self, query: str, max_items: int = 3) -> str:
        """
        获取与查询相关的上下文信息，用于增强对话
        :param query: 用户查询
        :param max_items: 最大上下文项数
        :return: 格式化的上下文字符串
        """
        relevant_memories = self.retrieve_memory(query)
        
        if not relevant_memories:
            return ""
            
        context_parts = []
        
        # 添加事实
        facts = [m for m in relevant_memories if m in self.memory["facts"]]
        if facts and len(context_parts) < max_items:
            context_parts.append("相关事实：" + " ".join(f['content'] for f in facts[:2]))
        
        # 添加用户偏好
        preferences = [m for m in relevant_memories if m in self.memory["user_preferences"]]
        if preferences and len(context_parts) < max_items:
            context_parts.append("用户偏好：" + " ".join(p['content'] for p in preferences[:2]))
        
        # 添加对话历史
        history = [m for m in relevant_memories if m in self.memory["conversation_history"]]
        if history and len(context_parts) < max_items:
            history_items = []
            for h in history[:3]:
                prefix = "用户: " if h.get("type") == "user_message" else "助手: "
                history_items.append(f"{prefix}{h['content']}")
            context_parts.append("相关对话：\n" + "\n".join(history_items))
        
        return "\n\n".join(context_parts)
    
    def _save_memory(self):
        """保存记忆到文件"""
        if self.memory_file:
            try:
                with open(self.memory_file, 'wb') as f:
                    pickle.dump(self.memory, f)
            except Exception as e:
                print(f"保存记忆失败: {str(e)}")
    
    def _load_memory(self):
        """从文件加载记忆"""
        if self.memory_file and os.path.exists(self.memory_file):
            try:
                with open(self.memory_file, 'rb') as f:
                    self.memory = pickle.load(f)
                # 重建索引
                self._rebuild_memory_index()
            except Exception as e:
                print(f"加载记忆失败: {str(e)}")
    
    def _rebuild_memory_index(self):
        """重建记忆索引"""
        self.memory_index = {}
        for memory_type in self.memory:
            for idx, item in enumerate(self.memory[memory_type]):
                content = item.get("content", "")
                self._update_memory_index(memory_type, idx, content)
    
    def clear_memory(self, memory_type: Optional[str] = None):
        """
        清除记忆
        :param memory_type: 可选，指定要清除的记忆类型，None表示清除所有
        """
        if memory_type is None:
            # 清除所有记忆
            self.memory = {
                "facts": [],
                "user_preferences": [],
                "conversation_history": [],
                "skills": [],
                "relationships": []
            }
            self.memory_index = {}
        elif memory_type in self.memory:
            # 清除特定类型的记忆
            self.memory[memory_type] = []
            # 重建索引
            self._rebuild_memory_index()
        
        # 保存更改
        self._save_memory()
    
    def get_memory_summary(self) -> Dict[str, int]:
        """
        获取记忆摘要统计
        :return: 各类记忆的数量统计
        """
        return {memory_type: len(items) for memory_type, items in self.memory.items()}

class Spark(Agent):
    def __init__(self, api_key=SPARK_API):
        super().__init__()
        self.api_key = "Bearer " + api_key
        self.url = "https://spark-api-open.xf-yun.com/v1/chat/completions"
    
    def get_answer(self, message):
        headers = {
            'Authorization': self.api_key,
            'content-type': "application/json"
        }
        body = {
            "model": "lite",
            "messages": message,
            "stream": True,
            "temperature": 0.7,
            "max_tokens": 2048
        }
        full_response = ""
        try:
            print("[Spark.get_answer] 发送请求到Spark API...")
            response = requests.post(url=self.url, json=body, headers=headers, stream=True, timeout=30)
            
            # 检查HTTP状态码
            if response.status_code != 200:
                print(f"[Spark.get_answer] HTTP错误: {response.status_code}")
                return f"抱歉，服务暂时不可用，请稍后重试。"
                
            for line in response.iter_lines():
                if not line:
                    continue
                line = line.decode('utf-8').strip()
                if line == '[DONE]':
                    break
                if line.startswith('data: '):
                    line = line[6:]
                try:
                    chunk = json.loads(line)
                    if 'choices' in chunk and chunk['choices']:
                        delta = chunk['choices'][0].get('delta', {})
                        if 'content' in delta and delta['content']:
                            full_response += delta['content']
                except json.JSONDecodeError:
                    print(f"[Spark.get_answer] JSON解析错误: {line}")
                    continue
                except Exception as e:
                    print(f"[Spark.get_answer] 处理响应块时出错: {str(e)}")
                    continue
        except requests.exceptions.ConnectionError as e:
            print(f"[Spark.get_answer] 连接错误: {str(e)}")
            return "抱歉，网络连接异常，请检查网络后重试。"
        except requests.exceptions.Timeout as e:
            print(f"[Spark.get_answer] 请求超时: {str(e)}")
            return "抱歉，处理时间过长，请稍后重试。"
        except Exception as e:
            print(f"[Spark.get_answer] 未知错误: {type(e).__name__}: {str(e)}")
            return "抱歉，系统出现异常，请稍后重试。"
            
        # 检查响应是否为空
        if not full_response or not full_response.strip():
            print("[Spark.get_answer] 响应为空!")
            return "抱歉，暂时无法获取回答，请重新提问。"
            
        # 检查响应是否包含错误信息
        error_keywords = ['error', 'unauthorized', 'apikey', '异常', '失败', '错误']
        if any(keyword in full_response.lower() for keyword in error_keywords):
            print(f"[Spark.get_answer] 响应中检测到错误: {full_response}")
            return "抱歉，处理您的请求时遇到问题，请稍后重试。"
            
        print(f"[Spark.get_answer] 成功获取响应: {full_response[:100]}...")
        return full_response

    def getText(self, text, role, content):
        jsoncon = {}
        jsoncon["role"] = role
        jsoncon["content"] = content
        text.append(jsoncon)
        return text
    
    def getlength(self, text):
        length = 0
        for content in text:
            temp = content["content"]
            leng = len(temp)
            length += leng
        return length
    
    def checklen(self, text):
        while (self.getlength(text) > 11000):
            del text[0]
        return text

    def chat(self, Input, agent_id=None, eval_targets=None, use_memory=True, stream=False, context=None):
        # 设置agent信息（如果需要）
        if agent_id is not None and agent_id != getattr(self, 'agent_id', None):
            self.set_agent_info(agent_id, eval_targets)
        
        # 调试：打印当前上下文状态
        print(f"[Spark.chat] 开始处理，当前public_context长度: {len(self.public_context)}")
        for i, msg in enumerate(self.public_context):
            print(f"[Spark.chat] 上下文[{i}]: {msg.get('role')} - {msg.get('content')[:50]}...")
        
        # 简化系统信息，避免在回答中暴露内部处理逻辑
        sys_info = "你是一个智能助手，请直接回答用户的问题。"
        
        # 在多agent讨论模式下，不使用记忆上下文，避免混乱
        enhanced_input = Input
        
        # 如果提供了上下文，添加到输入中
        if context and isinstance(context, str) and context.strip():
            enhanced_input = context + "\n\n" + enhanced_input
        
        # 构建完整的对话上下文
        question = []
        
        # 首先添加系统消息
        question = self.getText(question, "system", sys_info)
        
        # 添加历史对话（跳过系统消息，避免重复）
        for msg in self.public_context:
            if msg.get("role") == "system":
                continue  # 跳过系统消息，因为我们已经添加了
            question = self.getText(question, msg.get("role"), msg.get("content"))
        
        # 添加当前用户输入
        question = self.getText(question, "user", enhanced_input)
        
        # 确保不超过长度限制
        question = self.checklen(question)
        
        print(f"[Spark.chat] 构建的question长度: {len(question)}")
        
        if stream:
            # 对于流式输出，我们需要特别处理上下文更新
            answer_generator = self.get_answer_stream(question)
            # 先将用户输入添加到上下文
            self.add_public_context({"role": "user", "content": Input})
            
            # 返回生成器，让调用者处理流式内容
            return answer_generator
        else:
            answer = self.get_answer(question)
            
            print(f"[Spark.chat] 获得回答: {answer[:100]}...")
            
            # 立即更新上下文
            self.add_public_context({"role": "user", "content": Input})
            self.add_public_context({"role": "assistant", "content": answer})
            
            print(f"[Spark.chat] 更新后public_context长度: {len(self.public_context)}")
            
            return answer

    def get_answer_stream(self, message):
        """真正的流式输出版本"""
        import time
        
        headers = {
            'Authorization': self.api_key,
            'content-type': "application/json"
        }
        body = {
            "model": "lite",
            "messages": message,
            "stream": True,
            "temperature": 0.7,
            "max_tokens": 2048
        }
        
        try:
            print("[Spark.get_answer_stream] 发送流式请求到Spark API...")
            response = requests.post(url=self.url, json=body, headers=headers, stream=True, timeout=30)
            
            # 检查HTTP状态码
            if response.status_code != 200:
                print(f"[Spark.get_answer_stream] HTTP错误: {response.status_code}")
                yield "抱歉，服务暂时不可用，请稍后重试。"
                return
                
            print("[Spark.get_answer_stream] 开始处理流式响应...")
            chunk_count = 0
            
            for line in response.iter_lines():
                if not line:
                    continue
                    
                line = line.decode('utf-8').strip()
                print(f"[Spark.get_answer_stream] 收到原始行: {repr(line)}")
                
                if line == '[DONE]':
                    print("[Spark.get_answer_stream] 收到结束标记")
                    break
                    
                if line.startswith('data: '):
                    line = line[6:]
                    
                try:
                    chunk = json.loads(line)
                    print(f"[Spark.get_answer_stream] 解析JSON成功: {chunk}")
                    
                    if 'choices' in chunk and chunk['choices']:
                        delta = chunk['choices'][0].get('delta', {})
                        if 'content' in delta and delta['content']:
                            content = delta['content']
                            chunk_count += 1
                            print(f"[Spark.get_answer_stream] 第{chunk_count}个内容块: {repr(content)}")
                            yield content
                            
                except json.JSONDecodeError as e:
                    print(f"[Spark.get_answer_stream] JSON解析错误: {line}, 错误: {str(e)}")
                    continue
                except Exception as e:
                    print(f"[Spark.get_answer_stream] 处理响应块时出错: {str(e)}")
                    continue
                    
            print(f"[Spark.get_answer_stream] 流式处理完成，总共处理了{chunk_count}个内容块")
            
        except requests.exceptions.ConnectionError as e:
            print(f"[Spark.get_answer_stream] 连接错误: {str(e)}")
            yield "抱歉，网络连接异常，请检查网络后重试。"
        except requests.exceptions.Timeout as e:
            print(f"[Spark.get_answer_stream] 请求超时: {str(e)}")
            yield "抱歉，处理时间过长，请稍后重试。"
        except Exception as e:
            print(f"[Spark.get_answer_stream] 未知错误: {type(e).__name__}: {str(e)}")
            yield "抱歉，系统出现异常，请稍后重试。"

from zhipuai import ZhipuAI
class Zhipu(Agent):
    def __init__(self, api_key=ZHIPU_API):
        super().__init__()
        self.client = ZhipuAI(api_key=api_key)

    def chat(self, Input, agent_id=None, eval_targets=None, use_memory=True, stream=False, context=None):
        # 设置agent信息（如果需要）
        if agent_id is not None and agent_id != getattr(self, 'agent_id', None):
            self.set_agent_info(agent_id, eval_targets)
        
        # 调试：打印当前上下文状态
        print(f"[Zhipu.chat] 开始处理，当前public_context长度: {len(self.public_context)}")
        for i, msg in enumerate(self.public_context):
            print(f"[Zhipu.chat] 上下文[{i}]: {msg.get('role')} - {msg.get('content')[:50]}...")
        
        # 简化系统信息，避免在回答中暴露内部处理逻辑
        sys_info = "你是一个智能助手，请直接回答用户的问题。"
        
        # 在多agent讨论模式下，不使用记忆上下文，避免混乱
        enhanced_input = Input
        
        # 如果提供了上下文，添加到输入中
        if context and isinstance(context, str) and context.strip():
            enhanced_input = context + "\n\n" + enhanced_input
        
        # 构建完整的对话上下文
        messages = [{"role": "system", "content": sys_info}]
        
        # 添加历史对话（跳过系统消息，避免重复）
        for msg in self.public_context:
            if msg.get("role") == "system":
                continue  # 跳过系统消息，因为我们已经添加了
            messages.append({"role": msg.get("role"), "content": msg.get("content")})
        
        # 添加当前用户输入
        messages.append({"role": "user", "content": enhanced_input})
        
        print(f"[Zhipu.chat] 构建的messages长度: {len(messages)}")
        
        tools = [
            {
                "type": "web_search",
                "web_search": {
                    "enable": "True",
                    "search_engine": "search_std",
                    "search_result": "True",
                    "count": "5",
                    "search_prompt": "你是一名智能助手，具备联网搜索能力。回答尽量简洁，信息如下。{{search_result}}",
                    "content_size": "medium"
                }
            }
        ]
        
        if stream:
            # 对于流式输出，我们需要特别处理上下文更新
            stream_generator = self.chat_stream(messages, tools, Input)
            # 先将用户输入添加到上下文
            self.add_public_context({"role": "user", "content": Input})
            
            # 返回生成器，让调用者处理流式内容
            return stream_generator
        else:
            try:
                response = self.client.chat.completions.create(
                    model="glm-4-flash-250414",
                    messages=messages,
                    tools=tools,
                    tool_choice="auto",
                    stream=False
                )
            except Exception as e:
                print(f"Zhipu API调用异常: {type(e).__name__}: {e}")
                return "抱歉，系统出现异常，请稍后重试。"
            
            msg = response.choices[0].message
            main_content = msg.content.strip()
            
            # 只返回模型总结的结果，不包含网页搜索内容
            final_reply = main_content
            
            print(f"[Zhipu.chat] 获得回答: {final_reply[:100]}...")
            
            # 立即更新上下文
            self.add_public_context({"role": "user", "content": Input})
            self.add_public_context({"role": "assistant", "content": final_reply})
            
            print(f"[Zhipu.chat] 更新后public_context长度: {len(self.public_context)}")
            
            return final_reply

    def chat_stream(self, messages, tools, original_input):
        """流式输出版本"""
        try:
            print("[Zhipu.chat_stream] 发送流式请求到Zhipu API...")
            response = self.client.chat.completions.create(
                model="glm-4-flash-250414",
                messages=messages,
                tools=tools,
                tool_choice="auto",
                stream=True
            )
            
            for chunk in response:
                if chunk.choices and len(chunk.choices) > 0:
                    delta = chunk.choices[0].delta
                    if hasattr(delta, 'content') and delta.content:
                        yield delta.content
                        
        except Exception as e:
            print(f"Zhipu API流式调用异常: {type(e).__name__}: {e}")
            yield "抱歉，系统出现异常，请稍后重试。"
