"""
上下文工程模块 - 实现高效的上下文管理、记忆系统和便笺功能
"""

import os
import json
import time
import pickle
import threading
import numpy as np
import re
from collections import defaultdict, Counter
from typing import Dict, List, Any, Optional, Union, Set, Tuple
from datetime import datetime
import logging
import atexit
import signal
import hashlib
from pathlib import Path

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("ContextEngineering")

# 常量定义
CONTEXT_DIR = "context_data"
SCRATCHPAD_DIR = os.path.join(CONTEXT_DIR, "scratchpads")
MEMORY_DIR = os.path.join(CONTEXT_DIR, "memories")
CHECKPOINT_DIR = os.path.join(CONTEXT_DIR, "checkpoints")

# 确保目录存在
for directory in [CONTEXT_DIR, SCRATCHPAD_DIR, MEMORY_DIR, CHECKPOINT_DIR]:
    os.makedirs(directory, exist_ok=True)

# 为了提升性能，禁用sentence-transformers重量级组件
# 使用轻量级的文本处理方法
SENTENCE_TRANSFORMER_AVAILABLE = False
logger.info("为优化性能，已禁用sentence-transformers，使用轻量级文本处理")

class ScratchpadManager:
    """
    便笺管理器 - 处理多agent讨论的临时推理内容
    
    特点:
    - 为每个agent维护单独的便笺空间
    - 支持并发安全的读写操作
    - 讨论结束后自动清理
    - 定期保存以防止数据丢失
    """
    
    def __init__(self, discussion_id: str = None):
        """
        初始化便笺管理器
        
        Args:
            discussion_id: 讨论ID，用于区分不同的讨论会话
        """
        self.discussion_id = discussion_id or f"discussion_{int(time.time())}"
        self.scratchpads = defaultdict(list)  # agent_id -> 便笺内容列表
        self.lock = threading.RLock()  # 递归锁，支持同一线程多次获取
        self.last_save_time = time.time()
        self.save_interval = 10  # 10秒保存一次
        self.modified = False
        
        # 尝试从磁盘恢复便笺数据
        self._recover_from_disk()
        
        # 注册退出时保存
        atexit.register(self.save_to_disk)
        
    def add_note(self, agent_id: str, content: str, metadata: Dict = None) -> bool:
        """
        添加便笺内容
        
        Args:
            agent_id: Agent ID
            content: 便笺内容
            metadata: 元数据，如时间戳、类型等
            
        Returns:
            bool: 是否成功添加
        """
        if not content.strip():
            return False
            
        with self.lock:
            note = {
                "content": content,
                "timestamp": datetime.now().isoformat(),
                "metadata": metadata or {}
            }
            self.scratchpads[agent_id].append(note)
            self.modified = True
            
            # 检查是否需要保存
            current_time = time.time()
            if current_time - self.last_save_time > self.save_interval:
                self.save_to_disk()
                self.last_save_time = current_time
                
            return True
    
    def get_notes(self, agent_id: Optional[str] = None, limit: int = None) -> List[Dict]:
        """
        获取便笺内容
        
        Args:
            agent_id: 可选，指定Agent的ID。如果为None，返回所有Agent的便笺
            limit: 可选，限制返回的便笺数量
            
        Returns:
            List[Dict]: 便笺内容列表
        """
        with self.lock:
            if agent_id is not None:
                notes = self.scratchpads.get(agent_id, [])
                return notes[-limit:] if limit else notes
            else:
                # 返回所有Agent的便笺，按时间排序
                all_notes = []
                for agent_notes in self.scratchpads.values():
                    all_notes.extend(agent_notes)
                
                # 按时间戳排序
                all_notes.sort(key=lambda x: x["timestamp"])
                return all_notes[-limit:] if limit else all_notes
    
    def get_latest_note(self, agent_id: str) -> Optional[Dict]:
        """获取指定Agent的最新便笺"""
        notes = self.get_notes(agent_id, limit=1)
        return notes[0] if notes else None
    
    def clear(self, agent_id: Optional[str] = None):
        """
        清除便笺内容
        
        Args:
            agent_id: 可选，指定要清除的Agent ID。如果为None，清除所有便笺
        """
        with self.lock:
            if agent_id is not None:
                if agent_id in self.scratchpads:
                    self.scratchpads[agent_id] = []
            else:
                self.scratchpads.clear()
            
            self.modified = True
            self.save_to_disk()
    
    def extract_conclusion(self) -> str:
        """
        从便笺中提取讨论结论
        
        Returns:
            str: 提取的结论
        """
        with self.lock:
            all_notes = self.get_notes()
            if not all_notes:
                return ""
            
            # 简单策略：使用最后几条便笺作为结论
            # 实际应用中可以使用更复杂的策略，如关键词提取、摘要等
            recent_notes = all_notes[-3:]  # 取最后3条
            conclusion_parts = [note["content"] for note in recent_notes]
            return "\n".join(conclusion_parts)
    
    def save_to_disk(self):
        """将便笺数据保存到磁盘"""
        if not self.modified:
            return
            
        with self.lock:
            try:
                # 使用临时文件+重命名方式确保原子写入
                file_path = os.path.join(SCRATCHPAD_DIR, f"{self.discussion_id}.json")
                temp_path = f"{file_path}.tmp"
                
                with open(temp_path, 'w', encoding='utf-8') as f:
                    json.dump({
                        "discussion_id": self.discussion_id,
                        "scratchpads": dict(self.scratchpads),
                        "last_updated": datetime.now().isoformat()
                    }, f, ensure_ascii=False, indent=2)
                
                # 原子重命名
                os.replace(temp_path, file_path)
                self.modified = False
                logger.debug(f"便笺数据已保存到 {file_path}")
            except Exception as e:
                logger.error(f"保存便笺数据失败: {str(e)}")
    
    def _recover_from_disk(self):
        """从磁盘恢复便笺数据"""
        file_path = os.path.join(SCRATCHPAD_DIR, f"{self.discussion_id}.json")
        if not os.path.exists(file_path):
            return
            
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
                
            self.scratchpads = defaultdict(list, data.get("scratchpads", {}))
            logger.info(f"从 {file_path} 恢复了便笺数据")
        except Exception as e:
            logger.error(f"恢复便笺数据失败: {str(e)}")


class MemoryManager:
    """
    记忆管理器 - 处理长期记忆的存储和检索
    
    特点:
    - 支持向量检索和关键词检索
    - 混合排序策略
    - 记忆分类和优先级
    """
    
    def __init__(self, agent_id: str = None):
        """
        初始化记忆管理器
        
        Args:
            agent_id: Agent ID
        """
        self.agent_id = agent_id or "default"
        self.memory_file = os.path.join(MEMORY_DIR, f"{self.agent_id}_memory.pkl")
        
        # 记忆存储
        self.memories = {
            "preferences": [],  # 用户偏好
            "conclusions": [],  # 重要结论
            "requirements": [], # 长期要求
            "facts": []         # 事实知识
        }
        
        # 关键词索引 {词: [(记忆类型, 索引), ...]}
        self.keyword_index = defaultdict(list)
        
        # 向量索引 {记忆类型: [(向量, 索引), ...]}
        self.vector_index = defaultdict(list)
        
        # 加载向量模型
        self.embedding_model = None
        if SENTENCE_TRANSFORMER_AVAILABLE:
            try:
                from sentence_transformers import SentenceTransformer
                # 使用轻量级模型
                self.embedding_model = SentenceTransformer('paraphrase-MiniLM-L6-v2')
                logger.info("成功加载Sentence Transformer模型")
            except Exception as e:
                logger.error(f"加载Sentence Transformer模型失败: {str(e)}")
        
        # 从磁盘加载记忆
        self._load_from_disk()
        
        # 注册退出时保存
        atexit.register(self.save_to_disk)
        
    def add_memory(self, memory_type: str, content: str, metadata: Dict = None) -> bool:
        """
        添加记忆
        
        Args:
            memory_type: 记忆类型 (preferences, conclusions, requirements, facts)
            content: 记忆内容
            metadata: 元数据
            
        Returns:
            bool: 是否成功添加
        """
        if memory_type not in self.memories:
            logger.warning(f"未知的记忆类型: {memory_type}")
            return False
            
        if not content.strip():
            return False
            
        # 检查是否已存在类似内容
        if self._is_similar_exists(memory_type, content):
            logger.debug(f"已存在类似内容，跳过添加: {content[:30]}...")
            return False
            
        # 创建记忆项
        memory_item = {
            "content": content,
            "timestamp": datetime.now().isoformat(),
            "metadata": metadata or {},
            "access_count": 0,  # 访问计数，用于优先级计算
            "last_accessed": None
        }
        
        # 添加到记忆存储
        self.memories[memory_type].append(memory_item)
        idx = len(self.memories[memory_type]) - 1
        
        # 更新关键词索引
        self._update_keyword_index(memory_type, idx, content)
        
        # 更新向量索引
        self._update_vector_index(memory_type, idx, content)
        
        # 保存到磁盘
        self.save_to_disk()
        
        return True
    
    def retrieve_memory(self, query: str, memory_types: List[str] = None, 
                       limit: int = 5, use_vector: bool = True) -> List[Dict]:
        """
        检索相关记忆
        
        Args:
            query: 查询文本
            memory_types: 要检索的记忆类型列表，None表示所有类型
            limit: 返回结果数量限制
            use_vector: 是否使用向量检索
            
        Returns:
            List[Dict]: 相关记忆列表
        """
        if not query.strip():
            return []
            
        memory_types = memory_types or list(self.memories.keys())
        
        # 关键词检索结果
        keyword_results = self._keyword_search(query, memory_types)
        
        # 向量检索结果
        vector_results = []
        if use_vector and self.embedding_model:
            vector_results = self._vector_search(query, memory_types)
        
        # 合并结果并排序
        combined_results = self._combine_search_results(keyword_results, vector_results)
        
        # 获取记忆内容
        results = []
        seen_items = set()  # 避免重复
        
        for score, memory_type, idx in combined_results[:limit]:
            if idx >= len(self.memories[memory_type]):
                continue
                
            memory_item = self.memories[memory_type][idx].copy()
            
            # 生成唯一标识，避免重复
            item_id = f"{memory_type}_{idx}"
            if item_id in seen_items:
                continue
                
            seen_items.add(item_id)
            
            # 更新访问信息
            self.memories[memory_type][idx]["access_count"] += 1
            self.memories[memory_type][idx]["last_accessed"] = datetime.now().isoformat()
            
            # 添加类型信息
            memory_item["type"] = memory_type
            memory_item["relevance_score"] = score
            
            results.append(memory_item)
        
        return results
    
    def clear_memory(self, memory_type: Optional[str] = None):
        """
        清除记忆
        
        Args:
            memory_type: 要清除的记忆类型，None表示清除所有
        """
        if memory_type is None:
            # 清除所有记忆
            for mem_type in self.memories:
                self.memories[mem_type] = []
            
            # 清除索引
            self.keyword_index.clear()
            self.vector_index.clear()
        elif memory_type in self.memories:
            # 清除特定类型的记忆
            self.memories[memory_type] = []
            
            # 更新关键词索引
            new_keyword_index = defaultdict(list)
            for word, refs in self.keyword_index.items():
                new_refs = [(t, i) for t, i in refs if t != memory_type]
                if new_refs:
                    new_keyword_index[word] = new_refs
            self.keyword_index = new_keyword_index
            
            # 清除向量索引
            if memory_type in self.vector_index:
                self.vector_index[memory_type] = []
        
        # 保存更改
        self.save_to_disk()
    
    def get_memory_summary(self) -> Dict[str, int]:
        """
        获取记忆摘要统计
        
        Returns:
            Dict[str, int]: 各类记忆的数量统计
        """
        return {memory_type: len(items) for memory_type, items in self.memories.items()}
    
    def save_to_disk(self):
        """将记忆数据保存到磁盘"""
        try:
            # 使用临时文件+重命名方式确保原子写入
            temp_path = f"{self.memory_file}.tmp"
            
            with open(temp_path, 'wb') as f:
                pickle.dump({
                    "agent_id": self.agent_id,
                    "memories": self.memories,
                    "keyword_index": dict(self.keyword_index),
                    "last_updated": datetime.now().isoformat()
                }, f)
            
            # 原子重命名
            os.replace(temp_path, self.memory_file)
            logger.debug(f"记忆数据已保存到 {self.memory_file}")
        except Exception as e:
            logger.error(f"保存记忆数据失败: {str(e)}")
    
    def _load_from_disk(self):
        """从磁盘加载记忆数据"""
        if not os.path.exists(self.memory_file):
            return
            
        try:
            with open(self.memory_file, 'rb') as f:
                data = pickle.load(f)
                
            self.memories = data.get("memories", self.memories)
            self.keyword_index = defaultdict(list, data.get("keyword_index", {}))
            
            # 重建向量索引
            self._rebuild_vector_index()
            
            logger.info(f"从 {self.memory_file} 加载了记忆数据")
        except Exception as e:
            logger.error(f"加载记忆数据失败: {str(e)}")
    
    def _is_similar_exists(self, memory_type: str, content: str) -> bool:
        """检查是否已存在类似内容"""
        # 先使用关键词检索快速筛选
        potential_matches = self._keyword_search(content, [memory_type], limit=5)
        
        for _, _, idx in potential_matches:
            if idx >= len(self.memories[memory_type]):
                continue
                
            existing_content = self.memories[memory_type][idx]["content"]
            similarity = self._calculate_text_similarity(content, existing_content)
            
            # 如果相似度超过阈值，认为已存在
            if similarity > 0.8:
                return True
                
        return False
    
    def _update_keyword_index(self, memory_type: str, idx: int, content: str):
        """更新关键词索引"""
        # 提取关键词
        keywords = self._extract_keywords(content)
        
        # 更新索引
        for word in keywords:
            self.keyword_index[word].append((memory_type, idx))
    
    def _update_vector_index(self, memory_type: str, idx: int, content: str):
        """更新向量索引"""
        if not self.embedding_model:
            return
            
        try:
            # 生成文本向量
            vector = self.embedding_model.encode(content)
            
            # 添加到索引
            self.vector_index[memory_type].append((vector, idx))
        except Exception as e:
            logger.error(f"生成文本向量失败: {str(e)}")
    
    def _rebuild_vector_index(self):
        """重建向量索引"""
        if not self.embedding_model:
            return
            
        self.vector_index.clear()
        
        for memory_type, items in self.memories.items():
            for idx, item in enumerate(items):
                self._update_vector_index(memory_type, idx, item["content"])
    
    def _extract_keywords(self, text: str) -> Set[str]:
        """提取文本中的关键词"""
        # 简单实现：分词并去除停用词
        words = re.findall(r'\w+', text.lower())
        
        # 简单的中文停用词
        stopwords = {'的', '了', '和', '是', '在', '我', '有', '这', '个', '你', '们', '就', '也', '都', '要'}
        
        # 简单的英文停用词
        stopwords.update({'the', 'a', 'an', 'and', 'or', 'but', 'is', 'are', 'was', 'were', 
                         'be', 'been', 'being', 'to', 'of', 'for', 'with', 'by', 'about', 
                         'against', 'between', 'into', 'through', 'during', 'before', 'after',
                         'above', 'below', 'from', 'up', 'down', 'in', 'out', 'on', 'off',
                         'over', 'under', 'again', 'further', 'then', 'once', 'here', 'there',
                         'when', 'where', 'why', 'how', 'all', 'any', 'both', 'each', 'few',
                         'more', 'most', 'other', 'some', 'such', 'no', 'nor', 'not', 'only',
                         'own', 'same', 'so', 'than', 'too', 'very', 's', 't', 'can', 'will',
                         'just', 'don', 'should', 'now'})
        
        # 过滤停用词和短词
        keywords = {word for word in words if word not in stopwords and len(word) > 1}
        return keywords
    
    def _keyword_search(self, query: str, memory_types: List[str], limit: int = 10) -> List[Tuple[float, str, int]]:
        """
        关键词搜索
        
        Returns:
            List[Tuple[float, str, int]]: [(得分, 记忆类型, 索引), ...]
        """
        query_keywords = self._extract_keywords(query)
        if not query_keywords:
            return []
            
        # 收集匹配的记忆引用
        matches = []
        for keyword in query_keywords:
            for memory_type, idx in self.keyword_index.get(keyword, []):
                if memory_type in memory_types:
                    matches.append((memory_type, idx))
        
        # 统计每个记忆项的匹配次数
        match_counts = Counter(matches)
        
        # 计算得分并排序
        results = []
        for (memory_type, idx), count in match_counts.items():
            # 得分 = 匹配关键词数 / 查询关键词数
            score = count / len(query_keywords)
            results.append((score, memory_type, idx))
        
        # 按得分降序排序
        results.sort(reverse=True)
        return results[:limit]
    
    def _vector_search(self, query: str, memory_types: List[str], limit: int = 10) -> List[Tuple[float, str, int]]:
        """
        向量搜索
        
        Returns:
            List[Tuple[float, str, int]]: [(得分, 记忆类型, 索引), ...]
        """
        if not self.embedding_model:
            return []
            
        try:
            # 生成查询向量
            query_vector = self.embedding_model.encode(query)
            
            # 计算相似度
            results = []
            for memory_type in memory_types:
                for vector, idx in self.vector_index.get(memory_type, []):
                    # 计算余弦相似度
                    similarity = self._cosine_similarity(query_vector, vector)
                    results.append((similarity, memory_type, idx))
            
            # 按相似度降序排序
            results.sort(reverse=True)
            return results[:limit]
        except Exception as e:
            logger.error(f"向量搜索失败: {str(e)}")
            return []
    
    def _combine_search_results(self, keyword_results: List[Tuple[float, str, int]], 
                              vector_results: List[Tuple[float, str, int]]) -> List[Tuple[float, str, int]]:
        """
        合并关键词搜索和向量搜索结果
        
        Returns:
            List[Tuple[float, str, int]]: [(得分, 记忆类型, 索引), ...]
        """
        # 如果其中一个为空，直接返回另一个
        if not keyword_results:
            return vector_results
        if not vector_results:
            return keyword_results
            
        # 合并结果
        combined = {}
        
        # 处理关键词结果
        for score, memory_type, idx in keyword_results:
            key = (memory_type, idx)
            combined[key] = score * 0.4  # 关键词得分权重
        
        # 处理向量结果
        for score, memory_type, idx in vector_results:
            key = (memory_type, idx)
            if key in combined:
                combined[key] += score * 0.6  # 向量得分权重
            else:
                combined[key] = score * 0.6
        
        # 转换回列表并排序
        results = [(score, memory_type, idx) for (memory_type, idx), score in combined.items()]
        results.sort(reverse=True)
        return results
    
    def _calculate_text_similarity(self, text1: str, text2: str) -> float:
        """计算两段文本的相似度"""
        if self.embedding_model:
            try:
                # 使用向量相似度
                vec1 = self.embedding_model.encode(text1)
                vec2 = self.embedding_model.encode(text2)
                return self._cosine_similarity(vec1, vec2)
            except Exception:
                pass
        
        # 备用方案：基于共同词汇的简单相似度
        words1 = set(re.findall(r'\w+', text1.lower()))
        words2 = set(re.findall(r'\w+', text2.lower()))
        
        if not words1 or not words2:
            return 0
            
        common_words = words1.intersection(words2)
        return len(common_words) / max(len(words1), len(words2))
    
    def _cosine_similarity(self, vec1: np.ndarray, vec2: np.ndarray) -> float:
        """计算两个向量的余弦相似度"""
        norm1 = np.linalg.norm(vec1)
        norm2 = np.linalg.norm(vec2)
        
        if norm1 == 0 or norm2 == 0:
            return 0
            
        return np.dot(vec1, vec2) / (norm1 * norm2)


class ContextCompressor:
    """
    上下文压缩器 - 使用PageRank算法压缩上下文
    
    特点:
    - 基于句子重要性的压缩
    - 考虑时间衰减因子
    - 自动触发压缩
    """
    
    def __init__(self, max_context_length: int = 8000):
        """
        初始化上下文压缩器
        
        Args:
            max_context_length: 上下文最大长度，超过此长度将触发压缩
        """
        self.max_context_length = max_context_length
        
        # 加载向量模型
        self.embedding_model = None
        if SENTENCE_TRANSFORMER_AVAILABLE:
            try:
                from sentence_transformers import SentenceTransformer
                self.embedding_model = SentenceTransformer('paraphrase-MiniLM-L6-v2')
            except Exception as e:
                logger.error(f"加载Sentence Transformer模型失败: {str(e)}")
    
    def compress_context(self, context: List[Dict], target_ratio: float = 0.5) -> List[Dict]:
        """
        压缩上下文
        
        Args:
            context: 上下文列表，每项包含role和content
            target_ratio: 目标压缩比例，0.5表示压缩到原来的一半
            
        Returns:
            List[Dict]: 压缩后的上下文
        """
        # 检查是否需要压缩
        context_length = sum(len(msg.get("content", "")) for msg in context)
        if context_length <= self.max_context_length:
            return context
            
        logger.info(f"上下文长度({context_length})超过最大限制({self.max_context_length})，开始压缩...")
        
        # 提取系统消息，这些通常不压缩
        system_messages = [msg for msg in context if msg.get("role") == "system"]
        other_messages = [msg for msg in context if msg.get("role") != "system"]
        
        # 如果没有需要压缩的消息，直接返回
        if not other_messages:
            return context
            
        # 计算需要保留的消息数量
        target_count = max(1, int(len(other_messages) * target_ratio))
        
        # 提取每条消息的句子
        all_sentences = []
        message_sentences = []
        
        for i, msg in enumerate(other_messages):
            content = msg.get("content", "")
            sentences = self._split_into_sentences(content)
            
            if not sentences:
                continue
                
            # 记录每个句子所属的消息索引
            for sentence in sentences:
                all_sentences.append(sentence)
                message_sentences.append((i, sentence))
        
        # 如果句子太少，不压缩
        if len(all_sentences) <= 3:
            return context
            
        # 计算句子重要性
        sentence_scores = self._calculate_sentence_importance(all_sentences)
        
        # 为每条消息选择最重要的句子
        message_importance = defaultdict(float)
        for (msg_idx, sentence), score in zip(message_sentences, sentence_scores):
            message_importance[msg_idx] += score
        
        # 按重要性排序消息
        sorted_messages = sorted(
            [(idx, score) for idx, score in message_importance.items()],
            key=lambda x: x[1],
            reverse=True
        )
        
        # 选择最重要的消息
        selected_indices = [idx for idx, _ in sorted_messages[:target_count]]
        selected_indices.sort()  # 保持原始顺序
        
        # 构建压缩后的上下文
        compressed_context = system_messages.copy()
        for idx in selected_indices:
            compressed_context.append(other_messages[idx])
        
        logger.info(f"上下文压缩完成: {len(context)} -> {len(compressed_context)} 条消息")
        return compressed_context
    
    def _split_into_sentences(self, text: str) -> List[str]:
        """将文本分割为句子"""
        # 简单的句子分割规则
        # 处理常见的中英文句子结束符
        text = re.sub(r'([.!?。！？])\s*', r'\1\n', text)
        sentences = [s.strip() for s in text.split('\n') if s.strip()]
        return sentences
    
    def _calculate_sentence_importance(self, sentences: List[str]) -> List[float]:
        """
        使用PageRank算法计算句子重要性
        
        Args:
            sentences: 句子列表
            
        Returns:
            List[float]: 每个句子的重要性得分
        """
        n = len(sentences)
        if n <= 1:
            return [1.0] * n
            
        # 构建相似度矩阵
        similarity_matrix = np.zeros((n, n))
        
        for i in range(n):
            for j in range(n):
                if i != j:
                    similarity_matrix[i][j] = self._sentence_similarity(sentences[i], sentences[j])
        
        # 行归一化
        for i in range(n):
            row_sum = similarity_matrix[i].sum()
            if row_sum > 0:
                similarity_matrix[i] = similarity_matrix[i] / row_sum
        
        # 应用PageRank算法
        scores = self._page_rank(similarity_matrix)
        
        # 应用时间衰减因子（越新的句子权重越高）
        for i in range(n):
            # 时间因子：位置越靠后（越新），权重越高
            time_factor = 1 + 0.1 * (i / n)
            scores[i] *= time_factor
            
        return scores
    
    def _sentence_similarity(self, s1: str, s2: str) -> float:
        """计算两个句子的相似度"""
        if self.embedding_model:
            try:
                # 使用向量相似度
                vec1 = self.embedding_model.encode(s1)
                vec2 = self.embedding_model.encode(s2)
                
                # 计算余弦相似度
                norm1 = np.linalg.norm(vec1)
                norm2 = np.linalg.norm(vec2)
                
                if norm1 == 0 or norm2 == 0:
                    return 0
                    
                return np.dot(vec1, vec2) / (norm1 * norm2)
            except Exception:
                pass
        
        # 备用方案：基于共同词汇的简单相似度
        words1 = set(re.findall(r'\w+', s1.lower()))
        words2 = set(re.findall(r'\w+', s2.lower()))
        
        if not words1 or not words2:
            return 0
            
        common_words = words1.intersection(words2)
        return len(common_words) / max(len(words1), len(words2))
    
    def _page_rank(self, matrix: np.ndarray, d: float = 0.85, max_iter: int = 100, tol: float = 1e-6) -> np.ndarray:
        """
        实现PageRank算法
        
        Args:
            matrix: 转移矩阵
            d: 阻尼系数
            max_iter: 最大迭代次数
            tol: 收敛容差
            
        Returns:
            np.ndarray: PageRank得分
        """
        n = matrix.shape[0]
        
        # 初始化得分
        scores = np.ones(n) / n
        
        # 迭代计算
        for _ in range(max_iter):
            new_scores = (1 - d) / n + d * matrix.T.dot(scores)
            
            # 检查收敛
            if np.abs(new_scores - scores).sum() < tol:
                break
                
            scores = new_scores
            
        return scores


class StorageManager:
    """
    存储管理器 - 处理高频写入和数据恢复
    
    特点:
    - 缓冲写入
    - 原子操作
    - 崩溃恢复
    """
    
    def __init__(self, storage_id: str = None, auto_flush_interval: int = 5):
        """
        初始化存储管理器
        
        Args:
            storage_id: 存储ID
            auto_flush_interval: 自动刷新间隔（秒）
        """
        self.storage_id = storage_id or f"storage_{int(time.time())}"
        self.checkpoint_file = os.path.join(CHECKPOINT_DIR, f"{self.storage_id}.pkl")
        self.log_file = os.path.join(CHECKPOINT_DIR, f"{self.storage_id}.log")
        
        # 内存缓冲区
        self.buffer = {}
        self.buffer_lock = threading.RLock()
        
        # 修改标记
        self.modified = False
        self.last_flush_time = time.time()
        self.auto_flush_interval = auto_flush_interval
        
        # 启动自动刷新线程
        self.stop_event = threading.Event()
        self.flush_thread = threading.Thread(target=self._auto_flush_thread, daemon=True)
        self.flush_thread.start()
        
        # 注册信号处理器和退出处理（只在主线程中）
        try:
            if threading.current_thread() is threading.main_thread():
                signal.signal(signal.SIGINT, self._signal_handler)
                signal.signal(signal.SIGTERM, self._signal_handler)
            else:
                logger.debug("非主线程，跳过信号处理器注册")
        except ValueError as e:
            logger.warning(f"无法设置信号处理器: {str(e)}")
        
        atexit.register(self.flush)
        
        # 尝试恢复数据
        self._recover()
        
    def set(self, key: str, value: Any) -> bool:
        """
        设置键值对
        
        Args:
            key: 键
            value: 值
            
        Returns:
            bool: 是否成功设置
        """
        with self.buffer_lock:
            # 检查是否有变化
            if key in self.buffer and self.buffer[key] == value:
                return True
                
            self.buffer[key] = value
            self.modified = True
            
            # 记录操作日志
            self._append_log("set", key, value)
            
            # 检查是否需要刷新
            current_time = time.time()
            if current_time - self.last_flush_time > self.auto_flush_interval:
                self.flush()
                
            return True
    
    def get(self, key: str, default: Any = None) -> Any:
        """
        获取键值
        
        Args:
            key: 键
            default: 默认值
            
        Returns:
            Any: 值
        """
        with self.buffer_lock:
            return self.buffer.get(key, default)
    
    def delete(self, key: str) -> bool:
        """
        删除键值对
        
        Args:
            key: 键
            
        Returns:
            bool: 是否成功删除
        """
        with self.buffer_lock:
            if key not in self.buffer:
                return False
                
            del self.buffer[key]
            self.modified = True
            
            # 记录操作日志
            self._append_log("delete", key)
            
            return True
    
    def flush(self) -> bool:
        """
        将缓冲区数据刷新到磁盘
        
        Returns:
            bool: 是否成功刷新
        """
        with self.buffer_lock:
            if not self.modified:
                return True
                
            try:
                # 创建检查点
                temp_checkpoint = f"{self.checkpoint_file}.tmp"
                
                with open(temp_checkpoint, 'wb') as f:
                    pickle.dump({
                        "storage_id": self.storage_id,
                        "data": self.buffer,
                        "timestamp": datetime.now().isoformat()
                    }, f)
                
                # 原子重命名
                os.replace(temp_checkpoint, self.checkpoint_file)
                
                # 清除日志
                if os.path.exists(self.log_file):
                    os.remove(self.log_file)
                
                self.modified = False
                self.last_flush_time = time.time()
                
                logger.debug(f"存储数据已刷新到 {self.checkpoint_file}")
                return True
            except Exception as e:
                logger.error(f"刷新存储数据失败: {str(e)}")
                return False
    
    def _append_log(self, operation: str, key: str, value: Any = None):
        """记录操作日志"""
        try:
            log_entry = {
                "timestamp": datetime.now().isoformat(),
                "operation": operation,
                "key": key
            }
            
            if operation == "set":
                log_entry["value"] = value
                
            with open(self.log_file, 'a', encoding='utf-8') as f:
                f.write(json.dumps(log_entry) + "\n")
        except Exception as e:
            logger.error(f"记录操作日志失败: {str(e)}")
    
    def _recover(self):
        """从检查点和日志恢复数据"""
        # 先从检查点恢复
        if os.path.exists(self.checkpoint_file):
            try:
                with open(self.checkpoint_file, 'rb') as f:
                    checkpoint_data = pickle.load(f)
                    self.buffer = checkpoint_data.get("data", {})
                    logger.info(f"从检查点 {self.checkpoint_file} 恢复了数据")
            except Exception as e:
                logger.error(f"从检查点恢复数据失败: {str(e)}")
        
        # 再应用日志中的操作
        if os.path.exists(self.log_file):
            try:
                with open(self.log_file, 'r', encoding='utf-8') as f:
                    for line in f:
                        line = line.strip()
                        if not line:
                            continue
                            
                        log_entry = json.loads(line)
                        operation = log_entry.get("operation")
                        key = log_entry.get("key")
                        
                        if operation == "set" and key:
                            self.buffer[key] = log_entry.get("value")
                        elif operation == "delete" and key:
                            if key in self.buffer:
                                del self.buffer[key]
                
                logger.info(f"从日志 {self.log_file} 应用了操作")
            except Exception as e:
                logger.error(f"应用日志操作失败: {str(e)}")
    
    def _auto_flush_thread(self):
        """自动刷新线程"""
        while not self.stop_event.is_set():
            time.sleep(self.auto_flush_interval)
            with self.buffer_lock:
                if self.modified:
                    self.flush()
    
    def _signal_handler(self, sig, frame):
        """信号处理器"""
        logger.info(f"接收到信号 {sig}，正在保存数据...")
        self.flush()
        
    def close(self):
        """关闭存储管理器"""
        self.stop_event.set()
        self.flush_thread.join(timeout=1)
        self.flush()


class ContextEngineering:
    """
    上下文工程 - 主类，整合所有功能
    """
    
    def __init__(self, agent_id: str = None):
        """
        初始化上下文工程
        
        Args:
            agent_id: Agent ID
        """
        self.agent_id = agent_id or "default"
        
        # 初始化组件
        self.scratchpad_manager = ScratchpadManager(f"scratchpad_{self.agent_id}")
        self.memory_manager = MemoryManager(self.agent_id)
        self.context_compressor = ContextCompressor()
        self.storage_manager = StorageManager(f"storage_{self.agent_id}")
        
        logger.info(f"上下文工程初始化完成，Agent ID: {self.agent_id}")
    
    def add_to_scratchpad(self, content: str, metadata: Dict = None) -> bool:
        """
        添加内容到便笺
        
        Args:
            content: 内容
            metadata: 元数据
            
        Returns:
            bool: 是否成功添加
        """
        return self.scratchpad_manager.add_note(self.agent_id, content, metadata)
    
    def get_scratchpad_content(self, agent_id: str = None, limit: int = None) -> List[Dict]:
        """
        获取便笺内容
        
        Args:
            agent_id: 可选，指定Agent ID
            limit: 可选，限制返回数量
            
        Returns:
            List[Dict]: 便笺内容
        """
        return self.scratchpad_manager.get_notes(agent_id, limit)
    
    def clear_scratchpad(self, agent_id: str = None):
        """
        清除便笺
        
        Args:
            agent_id: 可选，指定要清除的Agent ID
        """
        self.scratchpad_manager.clear(agent_id)
    
    def extract_discussion_conclusion(self) -> str:
        """
        提取讨论结论
        
        Returns:
            str: 讨论结论
        """
        return self.scratchpad_manager.extract_conclusion()
    
    def add_to_memory(self, memory_type: str, content: str, metadata: Dict = None) -> bool:
        """
        添加内容到记忆
        
        Args:
            memory_type: 记忆类型
            content: 内容
            metadata: 元数据
            
        Returns:
            bool: 是否成功添加
        """
        return self.memory_manager.add_memory(memory_type, content, metadata)
    
    def retrieve_from_memory(self, query: str, memory_types: List[str] = None, limit: int = 5) -> List[Dict]:
        """
        从记忆中检索内容
        
        Args:
            query: 查询文本
            memory_types: 要检索的记忆类型
            limit: 返回结果数量限制
            
        Returns:
            List[Dict]: 检索结果
        """
        return self.memory_manager.retrieve_memory(query, memory_types, limit)
    
    def compress_context(self, context: List[Dict], target_ratio: float = 0.5) -> List[Dict]:
        """
        压缩上下文
        
        Args:
            context: 上下文列表
            target_ratio: 目标压缩比例
            
        Returns:
            List[Dict]: 压缩后的上下文
        """
        return self.context_compressor.compress_context(context, target_ratio)
    
    def store_data(self, key: str, value: Any) -> bool:
        """
        存储数据
        
        Args:
            key: 键
            value: 值
            
        Returns:
            bool: 是否成功存储
        """
        return self.storage_manager.set(key, value)
    
    def retrieve_data(self, key: str, default: Any = None) -> Any:
        """
        检索数据
        
        Args:
            key: 键
            default: 默认值
            
        Returns:
            Any: 检索到的值
        """
        return self.storage_manager.get(key, default)
    
    def process_user_input(self, user_input: str, context: List[Dict]) -> Tuple[str, List[Dict]]:
        """
        处理用户输入，增强上下文
        
        Args:
            user_input: 用户输入
            context: 当前上下文
            
        Returns:
            Tuple[str, List[Dict]]: (增强后的输入, 更新后的上下文)
        """
        # 检索相关记忆
        relevant_memories = self.memory_manager.retrieve_memory(user_input)
        
        # 构建记忆提示
        memory_prompt = ""
        if relevant_memories:
            memory_parts = []
            
            # 用户偏好
            preferences = [m for m in relevant_memories if m.get("type") == "preferences"]
            if preferences:
                memory_parts.append("用户偏好：" + " ".join(p["content"] for p in preferences[:2]))
            
            # 重要结论
            conclusions = [m for m in relevant_memories if m.get("type") == "conclusions"]
            if conclusions:
                memory_parts.append("相关结论：" + " ".join(c["content"] for c in conclusions[:2]))
            
            # 长期要求
            requirements = [m for m in relevant_memories if m.get("type") == "requirements"]
            if requirements:
                memory_parts.append("用户要求：" + " ".join(r["content"] for r in requirements[:2]))
            
            memory_prompt = "\n\n参考信息：\n" + "\n".join(memory_parts)
        
        # 增强输入
        enhanced_input = user_input
        if memory_prompt:
            enhanced_input += memory_prompt
        
        # 检查是否需要压缩上下文
        compressed_context = self.context_compressor.compress_context(context)
        
        return enhanced_input, compressed_context
    
    def extract_user_preferences(self, user_input: str) -> List[str]:
        """
        从用户输入中提取偏好
        
        Args:
            user_input: 用户输入
            
        Returns:
            List[str]: 提取的偏好列表
        """
        preferences = []
        
        # 偏好关键词模式
        preference_patterns = [
            r"我喜欢(.*?)(?:。|，|,|\.|\n|$)",
            r"我不喜欢(.*?)(?:。|，|,|\.|\n|$)",
            r"我更喜欢(.*?)(?:。|，|,|\.|\n|$)",
            r"我讨厌(.*?)(?:。|，|,|\.|\n|$)",
            r"我想要(.*?)(?:。|，|,|\.|\n|$)",
            r"我不想要(.*?)(?:。|，|,|\.|\n|$)",
            r"我希望(.*?)(?:。|，|,|\.|\n|$)",
            r"我不希望(.*?)(?:。|，|,|\.|\n|$)"
        ]
        
        for pattern in preference_patterns:
            matches = re.findall(pattern, user_input)
            for match in matches:
                if match.strip():
                    preferences.append(match.strip())
        
        return preferences
    
    def extract_long_term_requirements(self, user_input: str) -> List[str]:
        """
        从用户输入中提取长期要求
        
        Args:
            user_input: 用户输入
            
        Returns:
            List[str]: 提取的长期要求列表
        """
        requirements = []
        
        # 长期要求关键词模式
        requirement_patterns = [
            r"总是(.*?)(?:。|，|,|\.|\n|$)",
            r"永远(.*?)(?:。|，|,|\.|\n|$)",
            r"每次(.*?)(?:。|，|,|\.|\n|$)",
            r"记住(.*?)(?:。|，|,|\.|\n|$)",
            r"请记住(.*?)(?:。|，|,|\.|\n|$)",
            r"不要忘记(.*?)(?:。|，|,|\.|\n|$)",
            r"以后(.*?)(?:。|，|,|\.|\n|$)"
        ]
        
        for pattern in requirement_patterns:
            matches = re.findall(pattern, user_input)
            for match in matches:
                if match.strip():
                    requirements.append(match.strip())
        
        return requirements
    
    def update_memory_from_input(self, user_input: str) -> int:
        """
        从用户输入更新记忆
        
        Args:
            user_input: 用户输入
            
        Returns:
            int: 添加的记忆数量
        """
        count = 0
        
        # 提取偏好
        preferences = self.extract_user_preferences(user_input)
        for pref in preferences:
            if self.add_to_memory("preferences", pref):
                count += 1
        
        # 提取长期要求
        requirements = self.extract_long_term_requirements(user_input)
        for req in requirements:
            if self.add_to_memory("requirements", req):
                count += 1
        
        return count
    
    def get_compressed_context(self, query: str) -> str:
        """
        获取与查询相关的压缩上下文
        
        Args:
            query: 用户查询
            
        Returns:
            str: 压缩后的上下文字符串
        """
        try:
            # 从记忆中检索相关内容
            relevant_memories = self.memory_manager.retrieve_memory(query, limit=5)
            
            # 构建上下文字符串
            context_parts = []
            
            # 添加用户偏好
            preferences = [m for m in relevant_memories if m.get("type") == "preferences"]
            if preferences:
                context_parts.append("用户偏好:")
                for pref in preferences[:3]:  # 最多取3个偏好
                    context_parts.append(f"- {pref['content']}")
            
            # 添加长期要求
            requirements = [m for m in relevant_memories if m.get("type") == "requirements"]
            if requirements:
                context_parts.append("\n用户长期要求:")
                for req in requirements[:3]:  # 最多取3个要求
                    context_parts.append(f"- {req['content']}")
            
            # 添加重要结论
            conclusions = [m for m in relevant_memories if m.get("type") == "conclusions"]
            if conclusions:
                context_parts.append("\n重要结论:")
                for concl in conclusions[:3]:  # 最多取3个结论
                    context_parts.append(f"- {concl['content']}")
            
            # 添加相关事实
            facts = [m for m in relevant_memories if m.get("type") == "facts"]
            if facts:
                context_parts.append("\n相关事实:")
                for fact in facts[:5]:  # 最多取5个事实
                    context_parts.append(f"- {fact['content']}")
            
            # 如果有上下文内容，添加提示
            if context_parts:
                context_parts.insert(0, "以下是与用户查询相关的上下文信息，请在回答时考虑这些信息：\n")
            
            # 合并上下文
            compressed_context = "\n".join(context_parts)
            
            logger.info(f"为查询 '{query}' 生成压缩上下文，长度: {len(compressed_context)}")
            return compressed_context
            
        except Exception as e:
            logger.error(f"生成压缩上下文时出错: {str(e)}", exc_info=True)
            return ""  # 出错时返回空字符串
    
    def close(self):
        """关闭上下文工程，保存所有数据"""
        self.scratchpad_manager.save_to_disk()
        self.memory_manager.save_to_disk()
        self.storage_manager.flush()
        self.storage_manager.close()
        logger.info("上下文工程已关闭，所有数据已保存")


# 辅助函数

def get_context_engineering(agent_id: str = None) -> ContextEngineering:
    """
    获取上下文工程实例
    
    Args:
        agent_id: Agent ID
        
    Returns:
        ContextEngineering: 上下文工程实例
    """
    return ContextEngineering(agent_id)


# 示例用法
if __name__ == "__main__":
    # 初始化上下文工程
    ce = get_context_engineering("agent_0")
    
    # 添加便笺
    ce.add_to_scratchpad("这是一个测试便笺")
    
    # 添加记忆
    ce.add_to_memory("preferences", "用户喜欢简洁的回答")
    
    # 检索记忆
    memories = ce.retrieve_from_memory("简洁回答")
    print(f"检索到 {len(memories)} 条相关记忆")
    
    # 压缩上下文
    context = [
        {"role": "system", "content": "你是一个助手"},
        {"role": "user", "content": "你好"},
        {"role": "assistant", "content": "你好，有什么可以帮助你的？"}
    ]
    compressed = ce.compress_context(context)
    print(f"压缩前 {len(context)} 条消息，压缩后 {len(compressed)} 条消息")
    
    # 关闭
    ce.close()
