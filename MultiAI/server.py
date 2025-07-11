"""
MultiAI项目主服务器
提供多智能体协作、记忆管理、流式聊天等功能的Flask Web服务

主要功能模块：
1. 智能体管理 - 创建、删除、配置不同模型的智能体
2. 记忆系统 - 智能体长期记忆的存储、检索、管理
3. 流式聊天 - 支持实时流式输出的对话功能  
4. 多智能体讨论 - 智能体间的协作讨论机制
5. 上下文工程 - 智能的上下文压缩和管理
6. 用户干预 - 讨论过程中的用户参与和引导

架构设计：
- 模块化设计，功能分离
- 配置化管理，支持环境变量
- 异步处理，支持并发
- 流式输出，提升用户体验
- 错误处理，保证系统稳定性
"""

from flask import Flask, request, jsonify, Response, send_from_directory
from flask_cors import CORS
import traceback
import uuid
import threading
import time
import json
import datetime
import os
from typing import Dict, List, Any, Optional, Tuple

# 导入配置管理模块
from config import get_config, AppConfig

# 导入上下文工程模块
from context_engineering.ContextEngineering import ContextEngineering, get_context_engineering, ScratchpadManager
# 导入Agent记忆管理器
from context_engineering.agent_memory_manager import AgentMemoryManager, get_agent_memory_manager

# 导入新的函数式模块
from context_engineering.stream_manager import (
    SafeDiscussionStreamer, StreamOutputManager, StreamEventType,
    format_stream_response, sanitize_for_frontend, validate_stream_chunk
)
from context_engineering.context_safe_manager import (
    SafeContextManager, ensure_agent_context_safe, update_agent_context_atomic,
    get_context_manager, validate_context_structure
)
from context_engineering.discussion_state_manager import (
    SafeDiscussionManager, DiscussionStatus, UserInterventionType,
    is_complex_question_safe, should_show_discussion_panel,
    analyze_user_intervention_safe, filter_discussion_data_for_frontend,
    create_discussion_manager, get_discussion_manager, remove_discussion_manager
)

# 初始化Flask应用和配置
app = Flask(__name__)
config = get_config()  # 获取全局配置实例

# 配置CORS - 支持跨域访问
CORS(app, origins=config.CORS_ORIGINS)

# 添加静态文件路由
@app.route('/')
def index():
    return send_from_directory('.', 'UserInterface.html')

@app.route('/<path:filename>')
def static_files(filename):
    return send_from_directory('.', filename)

from agent import Spark, Zhipu, Agent
# 导入重构后的函数模块
import item

agent_instances = {}  # id(str) -> Agent对象
context_engineering_instances = {}  # id(str) -> ContextEngineering对象

def is_complex_question(message, agent_count=1):
    """使用item.py中的函数判断问题复杂度"""
    return item.is_complex_question(message, agent_count, app.logger)

def select_best_agent_for_query(query, agents, agent_id_map):
    """使用item.py中的函数智能选择最适合的agent"""
    return item.select_best_agent_for_query(query, agents, agent_id_map, context_engineering_instances, app.logger)

# 自动创建默认Zhipu agent
agent_instances['1'] = Zhipu()
context_engineering_instances['1'] = get_context_engineering('1')

# 存储正在进行的讨论
discussions = {}  # discussion_id -> discussion_data

# 记忆管理API
@app.route('/agent/<id>/memory', methods=['GET'])
def get_agent_memory_summary(id):
    """获取指定agent的记忆摘要"""
    agent_id = str(id)
    if agent_id not in agent_instances:
        return jsonify({'error': 'Agent not found'}), 404
    
    # 使用上下文工程的记忆管理器
    if agent_id in context_engineering_instances:
        ce = context_engineering_instances[agent_id]
        memory_summary = ce.memory_manager.get_memory_summary()
    else:
        # 兼容旧版API
        agent = agent_instances[agent_id]
        memory_summary = agent.get_memory_summary()
    
    return jsonify({
        'agent_id': agent_id,
        'memory_summary': memory_summary
    })

@app.route('/agent/<id>/memory/<memory_type>', methods=['GET'])
def get_agent_memory_by_type(id, memory_type):
    """获取指定agent的特定类型记忆"""
    agent_id = str(id)
    if agent_id not in agent_instances:
        return jsonify({'error': 'Agent not found'}), 404
    
    # 使用上下文工程的记忆管理器
    if agent_id in context_engineering_instances:
        ce = context_engineering_instances[agent_id]
        
        # 使用配置类验证记忆类型是否有效
        if not config.validate_memory_type(memory_type):
            return jsonify({'error': f'Invalid memory type: {memory_type}. Valid types: {config.VALID_MEMORY_TYPES}'}), 400
        
        # 获取特定类型的记忆
        memories = ce.memory_manager.memories.get(memory_type, [])
    else:
        # 兼容旧版API
        agent = agent_instances[agent_id]
        
        # 验证记忆类型是否有效
        if memory_type not in agent.memory:
            return jsonify({'error': f'Invalid memory type: {memory_type}'}), 400
        
        memories = agent.memory.get(memory_type, [])
    
    return jsonify({
        'agent_id': agent_id,
        'memory_type': memory_type,
        'memories': memories
    })

@app.route('/agent/<id>/memory/search', methods=['POST'])
def search_agent_memory(id):
    """搜索指定agent的记忆"""
    agent_id = str(id)
    if agent_id not in agent_instances:
        return jsonify({'error': 'Agent not found'}), 404
    
    data = request.get_json()
    query = data.get('query', '')
    memory_type = data.get('memory_type')  # 可选
    limit = data.get('limit', 5)  # 默认返回5条结果
    
    if not query:
        return jsonify({'error': 'Query parameter is required'}), 400
    
    # 使用上下文工程的记忆管理器
    if agent_id in context_engineering_instances:
        ce = context_engineering_instances[agent_id]
        memory_types = [memory_type] if memory_type else None
        memories = ce.retrieve_from_memory(query, memory_types, limit)
    else:
        # 兼容旧版API
        agent = agent_instances[agent_id]
        memories = agent.retrieve_memory(query, memory_type, limit)
    
    return jsonify({
        'agent_id': agent_id,
        'query': query,
        'memory_type': memory_type,
        'results': memories
    })

@app.route('/agent/<id>/memory/<memory_type>', methods=['POST'])
def add_agent_memory(id, memory_type):
    """为指定agent添加记忆"""
    agent_id = str(id)
    if agent_id not in agent_instances:
        return jsonify({'error': 'Agent not found'}), 404
    
    data = request.get_json()
    content = data.get('content')
    
    if not content:
        return jsonify({'error': 'Content is required'}), 400
    
    # 构建记忆项元数据
    metadata = {
        'timestamp': data.get('timestamp', datetime.datetime.now().isoformat()),
        'source': data.get('source', 'manual_input')
    }
    
    # 添加额外字段（如果有）
    for key, value in data.items():
        if key not in ['content', 'timestamp', 'source']:
            metadata[key] = value
    
    # 使用上下文工程的记忆管理器
    if agent_id in context_engineering_instances:
        ce = context_engineering_instances[agent_id]
        # 使用配置类验证记忆类型是否有效
        if not config.validate_memory_type(memory_type):
            return jsonify({'error': f'Invalid memory type: {memory_type}. Valid types: {config.VALID_MEMORY_TYPES}'}), 400
            
        success = ce.add_to_memory(memory_type, content, metadata)
    else:
        # 兼容旧版API
        agent = agent_instances[agent_id]
        # 验证记忆类型是否有效
        if memory_type not in agent.memory:
            return jsonify({'error': f'Invalid memory type: {memory_type}'}), 400
            
        # 构建旧版记忆项
        memory_item = {
            'content': content,
            **metadata
        }
        success = agent.add_memory(memory_type, memory_item)
    
    if success:
        return jsonify({
            'success': True,
            'message': f'Memory added to {memory_type}',
            'content': content,
            'metadata': metadata
        })
    else:
        return jsonify({
            'success': False,
            'message': 'Memory not added (possibly duplicate)'
        })

@app.route('/agent/<id>/memory/<memory_type>', methods=['DELETE'])
def clear_agent_memory(id, memory_type):
    """清除指定agent的记忆"""
    agent_id = str(id)
    if agent_id not in agent_instances:
        return jsonify({'error': 'Agent not found'}), 404
    
    # 使用上下文工程的记忆管理器
    if agent_id in context_engineering_instances:
        ce = context_engineering_instances[agent_id]
        
        # 清除所有记忆
        if memory_type == 'all':
            ce.memory_manager.clear_memory()
            return jsonify({
                'success': True,
                'message': 'All memories cleared'
            })
        
        # 使用配置类验证记忆类型是否有效
        if not config.validate_memory_type(memory_type):
            return jsonify({'error': f'Invalid memory type: {memory_type}. Valid types: {config.VALID_MEMORY_TYPES}'}), 400
        
        # 清除特定类型的记忆
        ce.memory_manager.clear_memory(memory_type)
    else:
        # 兼容旧版API
        agent = agent_instances[agent_id]
        
        # 清除所有记忆
        if memory_type == 'all':
            agent.clear_memory()
            return jsonify({
                'success': True,
                'message': 'All memories cleared'
            })
        
        # 验证记忆类型是否有效
        if memory_type not in agent.memory:
            return jsonify({'error': f'Invalid memory type: {memory_type}'}), 400
        
        # 清除特定类型的记忆
        agent.clear_memory(memory_type)
    
    return jsonify({
        'success': True,
        'message': f'Memories of type {memory_type} cleared'
    })

@app.route('/agent/<id>/memory/context', methods=['POST'])
def get_memory_context(id):
    """获取与查询相关的记忆上下文"""
    agent_id = str(id)
    if agent_id not in agent_instances:
        return jsonify({'error': 'Agent not found'}), 404
    
    data = request.get_json()
    query = data.get('query', '')
    max_items = data.get('max_items', 3)
    
    if not query:
        return jsonify({'error': 'Query parameter is required'}), 400
    
    # 使用上下文工程的记忆管理器
    if agent_id in context_engineering_instances:
        ce = context_engineering_instances[agent_id]
        memories = ce.retrieve_from_memory(query, limit=max_items)
        
        # 构建上下文
        context = []
        for memory in memories:
            context.append({
                'type': memory.get('type', 'unknown'),
                'content': memory.get('content', ''),
                'relevance': memory.get('relevance_score', 0)
            })
    else:
        # 兼容旧版API
        agent = agent_instances[agent_id]
        context = agent.get_relevant_context(query, max_items)
    
    return jsonify({
        'agent_id': agent_id,
        'query': query,
        'context': context
    })

@app.route('/agent', methods=['POST'])
def create_agent():
    data = request.get_json()
    agent_id = str(data.get('id'))
    model = data.get('model')
    app.logger.info(f"收到Agent创建请求: ID={agent_id}, 模型={model}")
    if agent_id in agent_instances:
        return jsonify({'error': 'Agent already exists'}), 400
    try:
        if model == 'Spark':
            agent_instances[agent_id] = Spark()
        elif model == 'Zhipu':
            agent_instances[agent_id] = Zhipu()
        else:
            return jsonify({'error': 'Unknown model'}), 400
            
        # 为新创建的Agent初始化上下文工程实例
        context_engineering_instances[agent_id] = get_context_engineering(agent_id)
        agent_instances[agent_id].set_agent_info(agent_id=agent_id)
        
    except Exception as e:
        app.logger.error(f"创建Agent失败: {str(e)}", exc_info=True)
        return jsonify({'error': f'创建Agent失败: {str(e)}'}), 500
    app.logger.info(f"成功创建Agent: ID={agent_id}, 模型={model}")
    return jsonify({'msg': 'Agent created', 'id': agent_id, 'model': model})

@app.route('/agent/<id>', methods=['DELETE'])
def delete_agent(id):
    agent_id = str(id)
    if agent_id in agent_instances:
        # 关闭上下文工程实例，确保数据保存
        if agent_id in context_engineering_instances:
            try:
                context_engineering_instances[agent_id].close()
                del context_engineering_instances[agent_id]
            except Exception as e:
                app.logger.error(f"关闭上下文工程实例失败: {str(e)}", exc_info=True)
        
        del agent_instances[agent_id]
        app.logger.info(f"删除Agent成功: ID={agent_id}")
        return jsonify({'msg': 'Agent deleted', 'id': agent_id})
    else:
        return jsonify({'error': 'Agent not found'}), 404

@app.route('/chat_stream', methods=['POST', 'GET'])
def chat_stream():
    """流式聊天端点"""
    if request.method == 'GET':
        # 处理EventSource的GET请求
        agent_ids_str = request.args.get('agent_ids', '[]')
        try:
            agent_ids = [str(id) for id in json.loads(agent_ids_str) if str(id).isdigit()]
        except:
            agent_ids = []
        message = request.args.get('message', '')
        current_agent = request.args.get('current_agent', '1')
        enable_discussion = request.args.get('enable_discussion', 'false').lower() == 'true'
        use_context = request.args.get('use_context', 'true').lower() == 'true'
        
        data = {
            'agent_ids': agent_ids,
            'message': message,
            'current_agent': current_agent,
            'enable_discussion': enable_discussion,
            'use_context': use_context
        }
    else:
        # 处理POST请求
        data = request.get_json()
        agent_ids = [str(id) for id in data.get('agent_ids', []) if str(id).isdigit()]
    
    app.logger.info(f"Received stream chat request: {data}")
    
    if not agent_ids:
        app.logger.error("agent_ids参数为空")
        return jsonify({'error': 'agent_ids参数不能为空，请提供有效的Agent ID列表'}), 400
    
    message = data.get('message', '')
    if not message:
        app.logger.error("message参数为空")
        return jsonify({'error': 'message参数不能为空，请提供有效的消息内容'}), 400
    
    current_agent = str(data.get('current_agent')) if data.get('current_agent') is not None else '1'
    enable_discussion = data.get('enable_discussion', False)
    use_context = data.get('use_context', True)  # 默认使用上下文
    
    # 构建agent列表
    agents = []
    agent_id_map = {}
    invalid_agent_ids = []
    
    for idx, agent_id in enumerate(agent_ids):
        agent_id_str = str(agent_id)
        agent = agent_instances.get(agent_id_str)
        if agent is not None:
            agents.append(agent)
            agent_id_map[idx] = agent_id_str
        else:
            invalid_agent_ids.append(agent_id_str)

    if invalid_agent_ids:
        available_agents = list(agent_instances.keys())
        error_msg = f'提供的agent_ids中包含无效的Agent ID: {", ".join(invalid_agent_ids)}，请先创建Agent。当前可用Agent: {available_agents}'
        app.logger.error(error_msg)
        return jsonify({'error': error_msg}), 400

    # 单agent情况 - 流式输出
    if len(agents) == 1:
        agent = agents[0]
        agent_model = 'Spark' if isinstance(agent, Spark) else 'Zhipu'
        
        def generate():
            try:
                # 发送开始消息
                yield f"data: {json.dumps({'type': 'start', 'agent_id': agent_ids[0], 'model': agent_model})}\n\n"
                
                # 流式获取回复
                content_buffer = ""
                app.logger.info(f"开始流式聊天，agent_id={agent_ids[0]}, model={agent_model}")
                
                # 获取上下文（如果启用）
                context = ""
                if use_context and agent_ids[0] in context_engineering_instances:
                    ce = context_engineering_instances[agent_ids[0]]
                    # 获取压缩后的上下文
                    context = ce.get_compressed_context(message)
                    app.logger.info(f"获取到压缩上下文，长度: {len(context)}")
                
                # 直接调用agent的流式方法
                try:
                    stream_generator = agent.chat(message, agent_id=1, eval_targets=[], use_memory=False, stream=True, context=context)
                    app.logger.info(f"获取到流式生成器: {type(stream_generator)}")
                    
                    # 逐个处理流式块
                    chunk_count = 0
                    for chunk in stream_generator:
                        chunk_count += 1
                        app.logger.info(f"收到第{chunk_count}个chunk: '{chunk}'")
                        
                        if chunk:  # 只有非空chunk才发送
                            content_buffer += chunk
                            # 立即发送当前块
                            chunk_data = json.dumps({'type': 'content', 'content': chunk})
                            yield f"data: {chunk_data}\n\n"
                            app.logger.info(f"发送chunk: {chunk_data}")
                    
                    app.logger.info(f"流式接收完成，总共{chunk_count}个chunk，完整内容长度: {len(content_buffer)}")
                    
                except Exception as stream_error:
                    app.logger.error(f"流式处理异常: {str(stream_error)}", exc_info=True)
                    # 如果流式失败，尝试普通调用
                    app.logger.info("流式失败，尝试普通调用")
                    content_buffer = agent.chat(message, agent_id=1, eval_targets=[], use_memory=False, stream=False, context=context)
                    yield f"data: {json.dumps({'type': 'content', 'content': content_buffer})}\n\n"
                
                # 发送结束消息
                yield f"data: {json.dumps({'type': 'end', 'full_content': content_buffer})}\n\n"
                app.logger.info("流式聊天完成")
                
            except Exception as e:
                app.logger.error(f"流式聊天异常: {str(e)}", exc_info=True)
                error_msg = f'[模型调用异常: {str(e)}]'
                yield f"data: {json.dumps({'type': 'error', 'content': error_msg})}\n\n"
        
        return Response(generate(), mimetype='text/event-stream', headers={
            'Cache-Control': 'no-cache',
            'Connection': 'keep-alive',
            'Access-Control-Allow-Origin': '*',
            'Access-Control-Allow-Headers': 'Content-Type'
        })
    
    # 多agent情况 - 流式讨论
    def generate_discussion():
        try:
            # 先进行问题复杂度判断，决定处理模式
            need_discussion = False
            complexity_analysis = {'reason': 'single_agent', 'complexity_score': 0}
            
            if len(agents) > 1 and enable_discussion:
                # 使用规则判断问题复杂度，传入agent数量
                need_discussion, complexity_analysis = is_complex_question_safe(message, len(agents))
                app.logger.info(f"复杂度判断结果 - 需要讨论: {need_discussion}, 分析: {complexity_analysis}")
            
            # 根据复杂度判断决定处理模式
            if not need_discussion:
                # 简单问题：直接选择最适合的agent回答，不显示讨论悬浮框
                app.logger.info(f"简单问题处理模式，原因: {complexity_analysis.get('reason', 'unknown')}")
                
                # 发送简单模式的开始消息
                yield f"data: {json.dumps({'type': 'start', 'mode': 'simple', 'complexity_analysis': complexity_analysis})}\n\n"
                
                # 智能选择最适合的agent
                selected_agent_idx, selected_agent_id = select_best_agent_for_query(message, agents, agent_id_map)
                selected_agent = agents[selected_agent_idx]
                
                # 获取上下文（如果启用）
                context = ""
                if use_context and selected_agent_id in context_engineering_instances:
                    ce = context_engineering_instances[selected_agent_id]
                    context = ce.get_compressed_context(message)
                    app.logger.info(f"获取到压缩上下文，长度: {len(context)}")
                
                reply = selected_agent.chat(message, agent_id=selected_agent_idx, eval_targets=[], use_memory=False, context=context)
                
                agent_model = 'Spark' if isinstance(selected_agent, Spark) else 'Zhipu'
                yield f"data: {json.dumps({'type': 'agent_message', 'agent_id': selected_agent_id, 'model': agent_model, 'content': reply})}\n\n"
                yield f"data: {json.dumps({'type': 'end', 'status': 'completed'})}\n\n"
                return
            
            # 复杂问题：进入真正的讨论模式
            app.logger.info(f"复杂问题讨论模式，复杂度分数: {complexity_analysis.get('complexity_score', 0)}")
            
            # 启动异步讨论并流式返回结果
            discussion_id = str(uuid.uuid4())
            
            discussion_data = {
                'id': discussion_id,
                'user_message': message,
                'agents': agents,
                'agent_id_map': agent_id_map,
                'current_agent_idx': 0,
                'framework': None,
                'sub_questions': [],
                'messages': [],
                'status': 'starting',
                'stop_requested': False,
                'paused': False,
                'stream_mode': True,  # 标记为流式模式
                'complexity_analysis': complexity_analysis  # 保存复杂度分析结果
            }
            
            discussions[discussion_id] = discussion_data
            
            # 发送讨论模式的开始消息和discussion_id（这时才会触发悬浮框显示）
            yield f"data: {json.dumps({'type': 'start', 'mode': 'discussion', 'discussion_id': discussion_id, 'complexity_analysis': complexity_analysis})}\n\n"
            
            # 在同一线程中运行讨论，以便流式返回
            app.logger.info(f"开始流式讨论，discussion_id={discussion_id}")
            
            # 运行流式讨论
            for chunk in run_discussion_stream(discussion_id):
                yield f"data: {json.dumps(chunk)}\n\n"
                
        except Exception as e:
            app.logger.error(f"流式讨论异常: {str(e)}", exc_info=True)
            error_msg = f'[讨论异常: {str(e)}]'
            yield f"data: {json.dumps({'type': 'error', 'content': error_msg})}\n\n"
    
    return Response(generate_discussion(), mimetype='text/event-stream', headers={
        'Cache-Control': 'no-cache',
        'Connection': 'keep-alive',
        'Access-Control-Allow-Origin': '*',
        'Access-Control-Allow-Headers': 'Content-Type'
    })

# 上下文压缩API
@app.route('/agent/<id>/compress_context', methods=['POST'])
def compress_context(id):
    """获取压缩后的上下文"""
    agent_id = str(id)
    if agent_id not in agent_instances:
        return jsonify({'error': 'Agent not found'}), 404
    
    data = request.get_json()
    query = data.get('query', '')
    
    if not query:
        return jsonify({'error': 'Query parameter is required'}), 400
    
    # 使用上下文工程的上下文压缩
    if agent_id in context_engineering_instances:
        ce = context_engineering_instances[agent_id]
        compressed_context = ce.get_compressed_context(query)
    else:
        # 兼容旧版API
        compressed_context = ""
    
    return jsonify({
        'agent_id': agent_id,
        'query': query,
        'compressed_context': compressed_context
    })

@app.route('/chat', methods=['POST'])
def chat():
    data = request.get_json()
    app.logger.info(f"Received chat request: {data}")
    agent_ids = [str(id) for id in data.get('agent_ids', []) if str(id).isdigit()]
    
    if not agent_ids:
        app.logger.error("agent_ids参数为空")
        return jsonify({'error': 'agent_ids参数不能为空，请提供有效的Agent ID列表'}), 400
    
    message = data.get('message', '')
    if not message:
        app.logger.error("message参数为空")
        return jsonify({'error': 'message参数不能为空，请提供有效的消息内容'}), 400
    
    current_agent = str(data.get('current_agent')) if data.get('current_agent') is not None else '1'
    enable_discussion = data.get('enable_discussion', False)
    use_context = data.get('use_context', True)  # 默认使用上下文
    
    # 构建agent列表
    agents = []
    agent_id_map = {}
    invalid_agent_ids = []
    
    for idx, agent_id in enumerate(agent_ids):
        agent_id_str = str(agent_id)
        agent = agent_instances.get(agent_id_str)
        if agent is not None:
            agents.append(agent)
            agent_id_map[idx] = agent_id_str
        else:
            invalid_agent_ids.append(agent_id_str)

    if invalid_agent_ids:
        available_agents = list(agent_instances.keys())
        error_msg = f'提供的agent_ids中包含无效的Agent ID: {", ".join(invalid_agent_ids)}，请先创建Agent。当前可用Agent: {available_agents}'
        app.logger.error(error_msg)
        return jsonify({'error': error_msg}), 400

    # 单agent情况
    if len(agents) == 1:
        agent = agents[0]
        agent_model = 'Spark' if isinstance(agent, Spark) else 'Zhipu'
        
        try:
            # 获取上下文（如果启用）
            context = ""
            if use_context and agent_ids[0] in context_engineering_instances:
                ce = context_engineering_instances[agent_ids[0]]
                # 获取压缩后的上下文
                context = ce.get_compressed_context(message)
                app.logger.info(f"获取到压缩上下文，长度: {len(context)}")
            
            # 获取历史上下文 - 直接从agent对象获取，确保上下文连续性
            reply = agent.chat(message, agent_id=1, eval_targets=[], use_memory=True, context=context)
            
            # 记录上下文长度，用于调试
            app.logger.info(f"Agent回复后的上下文长度: {len(agent.public_context)}")
            
            if not reply or reply.strip().startswith('抱歉，'):
                reply = '抱歉，暂时无法获取回答，请重新提问。'
            
            discussion = [{
                'agent_id': agent_ids[0],
                'name': f'Agent{agent_ids[0]}',
                'avatar': f'img/{agent_model}.png',
                'content': reply,
                'type': 'agent_response'
            }]
            
            return jsonify({
                'discussion': discussion,
                'status': 'completed',
                'loading': False,
                'mode': 'single'
            })
            
        except Exception as e:
            app.logger.error(f"Agent聊天异常: {str(e)}", exc_info=True)
            error_reply = f'[模型调用异常: {str(e)}]'
            
            discussion = [{
                'agent_id': agent_ids[0],
                'name': f'Agent{agent_ids[0]}',
                'avatar': f'img/{agent_model}.png',
                'content': error_reply,
                'type': 'agent_response'
            }]
            
            return jsonify({
                'discussion': discussion,
                'status': 'completed',
                'loading': False,
                'mode': 'single'
            })

    # 多agent情况
    try:
        if not agents:
            return jsonify({'error': '提供的agent_ids中没有找到有效的Agent，请检查Agent是否已创建或ID是否正确'}), 400
        
        discussion = []
        
        # 确定主agent
        current_agent_idx = -1
        if current_agent is not None:
            for idx, agent_id in agent_id_map.items():
                if str(agent_id) == str(current_agent):
                    current_agent_idx = idx
                    break
        
        if current_agent_idx < 0 or current_agent_idx >= len(agents):
            current_agent_idx = 0
        
        main_agent = agents[current_agent_idx]
            
        # 判断问题复杂度
        need_discussion = False
        if len(agents) > 1 and enable_discussion:
            # 使用规则判断问题复杂度，传入agent数量
            need_discussion, complexity_analysis = is_complex_question_safe(message, len(agents))
            app.logger.info(f"规则判断问题复杂度 - 需要讨论: {need_discussion}, 分析: {complexity_analysis}")
        
        # 如果不需要讨论，直接回复
        if len(agents) <= 1 or not enable_discussion or not need_discussion:
            # 获取上下文（如果启用）
            context = ""
            if use_context and agent_id_map[current_agent_idx] in context_engineering_instances:
                ce = context_engineering_instances[agent_id_map[current_agent_idx]]
                # 获取压缩后的上下文
                context = ce.get_compressed_context(message)
                app.logger.info(f"获取到压缩上下文，长度: {len(context)}")
                
            reply = main_agent.chat(message, agent_id=current_agent_idx, eval_targets=[], use_memory=False, context=context)
            
            if not reply or reply.strip().startswith('抱歉，'):
                reply = '抱歉，暂时无法获取回答，请重新提问。'
            
            agent_model = 'Spark' if isinstance(main_agent, Spark) else 'Zhipu'
            discussion.append({
                'agent_id': agent_id_map[current_agent_idx],
                'name': f'Agent{agent_id_map[current_agent_idx]}',
                'avatar': f'img/{agent_model}.png',
                'content': reply,
                'type': 'agent_response'
            })
            
            return jsonify({
                'discussion': discussion,
                'status': 'completed',
                'loading': False,
                'mode': 'single'
            })
        
        # 启动异步讨论
        discussion_id = str(uuid.uuid4())
        
        discussion_data = {
            'id': discussion_id,
            'user_message': message,
            'agents': agents,
            'agent_id_map': agent_id_map,
            'current_agent_idx': current_agent_idx,
            'framework': None,
            'sub_questions': [],
            'messages': [],
            'status': 'starting',
            'stop_requested': False
        }
        
        discussions[discussion_id] = discussion_data
        
        # 启动异步线程处理讨论
        threading.Thread(target=run_discussion, args=(discussion_id,)).start()
        
        return jsonify({
            'discussion_id': discussion_id,
            'status': 'starting',
            'mode': 'discussion'
        })
        
    except Exception as e:
        app.logger.error(f"多Agent协作异常: {str(e)}", exc_info=True)
        return jsonify({
            'error': f'[多Agent协作异常] {type(e).__name__}: {e}',
            'status': 'error',
            'loading': False
        })

# 讨论状态端点
@app.route('/discussion_status/<discussion_id>', methods=['GET'])
def get_discussion_status(discussion_id):
    if discussion_id not in discussions:
        return jsonify({'error': 'Discussion not found'}), 404
    
    discussion_data = discussions[discussion_id]
    
    # 获取新消息
    new_messages = []
    if 'last_sent_index' in discussion_data:
        last_idx = discussion_data['last_sent_index']
        new_messages = discussion_data['messages'][last_idx:]
        discussion_data['last_sent_index'] = len(discussion_data['messages'])
    else:
        new_messages = discussion_data['messages']
        discussion_data['last_sent_index'] = len(discussion_data['messages'])
    
    # 获取下一个要发言的agent
    next_agent_id = None
    if discussion_data['status'] != 'completed' and discussion_data['status'] != 'error':
        if 'current_question_index' in discussion_data and discussion_data['sub_questions']:
            next_q_idx = discussion_data.get('current_question_index', 0)
            if next_q_idx < len(discussion_data['sub_questions']):
                agent_idx = next_q_idx % len(discussion_data['agents'])
                next_agent_id = discussion_data['agent_id_map'].get(agent_idx)
        
        if next_agent_id is None and 'current_agent_idx' in discussion_data:
            next_agent_id = discussion_data['agent_id_map'].get(discussion_data['current_agent_idx'])
        elif next_agent_id is None and len(discussion_data['agent_id_map']) > 0:
            next_agent_id = discussion_data['agent_id_map'].get(0)
    
    has_streaming = any(msg.get('streaming', False) for msg in discussion_data['messages'])
    
    status = discussion_data['status']
    if has_streaming and status == 'completed':
        status = 'processing'
    
    return jsonify({
        'status': status,
        'new_messages': new_messages,
        'total_messages': len(discussion_data['messages']),
        'next_agent_id': next_agent_id,
        'has_streaming': has_streaming,
        'stream': discussion_data.get('stream', False)
    })

# 停止讨论端点
@app.route('/stop_discussion', methods=['POST'])
def stop_discussion():
    data = request.get_json()
    discussion_id = data.get('discussion_id')
    
    if discussion_id not in discussions:
        return jsonify({'error': 'Discussion not found'}), 404
    
    # 同时设置传统标志和新的停止事件
    discussions[discussion_id]['stop_requested'] = True
    discussions[discussion_id]['status'] = 'stopping'
    
    # 触发停止事件，立即中断流式输出
    item.signal_stop(discussion_id)
    
    app.logger.info(f"用户请求停止讨论 {discussion_id}")
    
    return jsonify({'status': 'stopping', 'message': '正在停止讨论，请稍候...'})

# 暂停讨论端点
@app.route('/pause_discussion', methods=['POST'])
def pause_discussion():
    data = request.get_json()
    discussion_id = data.get('discussion_id')
    
    if discussion_id not in discussions:
        return jsonify({'error': 'Discussion not found'}), 404
    
    discussions[discussion_id]['paused'] = True
    discussions[discussion_id]['status'] = 'paused'
    
    return jsonify({'status': 'paused'})

# 继续讨论端点
@app.route('/resume_discussion', methods=['POST'])
def resume_discussion():
    data = request.get_json()
    discussion_id = data.get('discussion_id')
    
    if discussion_id not in discussions:
        return jsonify({'error': 'Discussion not found'}), 404
    
    discussions[discussion_id]['paused'] = False
    discussions[discussion_id]['status'] = 'running'
    
    return jsonify({'status': 'running'})

# 用户插入发言端点
@app.route('/user_intervention', methods=['POST'])
def user_intervention():
    data = request.get_json()
    discussion_id = data.get('discussion_id')
    message = data.get('message')
    
    if discussion_id not in discussions:
        return jsonify({'error': 'Discussion not found'}), 404
    
    if not message:
        return jsonify({'error': 'Message is required'}), 400
    
    discussion_data = discussions[discussion_id]
    
    # 添加用户消息到讨论历史
    user_message = {
        'agent_id': 'user',
        'name': '用户',
        'avatar': '',
        'content': message,
        'type': 'user_intervention',
        'timestamp': datetime.datetime.now().isoformat()
    }
    
    discussion_data['messages'].append(user_message)
    discussion_data['user_intervention'] = message
    discussion_data['paused'] = False
    discussion_data['status'] = 'running'
    
    return jsonify({'status': 'user_intervention_added'})

def identify_target_agent(user_message, discussion_messages, agent_id_map):
    """使用item.py中的函数识别用户发言针对的agent"""
    return item.identify_target_agent(user_message, discussion_messages, agent_id_map, app.logger)

def build_targeted_response_prompt(user_message, intervention_analysis, response_reason, original_question, discussion_history, agent_position=None):
    """使用item.py中的函数构建针对性回应提示"""
    return item.build_targeted_response_prompt(user_message, intervention_analysis, response_reason, original_question, discussion_history, agent_position)

def analyze_user_intervention(user_intervention, original_question, discussion_messages, main_agent):
    """使用item.py中的函数分析用户插入发言"""
    return item.analyze_user_intervention(user_intervention, original_question, discussion_messages, main_agent, app.logger)

def adjust_discussion_framework(original_question, user_intervention, original_framework, main_agent):
    """使用item.py中的函数调整讨论框架"""
    return item.adjust_discussion_framework(original_question, user_intervention, original_framework, main_agent, app.logger)

# 异步讨论处理函数
def run_discussion(discussion_id):
    """使用item.py中的函数处理异步讨论"""
    if discussion_id not in discussions:
        app.logger.error(f"讨论ID不存在: {discussion_id}")
        return
    
    discussion_data = discussions[discussion_id]
    # 调用item.py中的函数
    item.run_discussion_logic(discussion_data, agent_instances, context_engineering_instances, app.logger)

# 流式讨论处理函数
def run_discussion_stream(discussion_id):
    """使用item.py中的函数处理流式讨论"""
    if discussion_id not in discussions:
        yield {'type': 'error', 'content': f'讨论ID不存在: {discussion_id}'}
        return
    
    discussion_data = discussions[discussion_id]
    # 调用item.py中的函数
    for chunk in item.run_discussion_stream_logic(discussion_data, agent_instances, context_engineering_instances, app.logger):
        yield chunk

if __name__ == '__main__':
    # 使用配置系统启动服务器
    app.run(
        host=config.SERVER_HOST,
        port=config.SERVER_PORT,
        debug=config.DEBUG_MODE
    )
