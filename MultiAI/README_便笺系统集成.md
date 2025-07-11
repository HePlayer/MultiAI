# 便笺系统集成说明

本文档说明了在多Agent讨论过程中集成便笺系统的实现方法和测试方法。

## 功能概述

在多Agent讨论过程中，我们实现了以下功能：

1. Agent的推理内容不计入历史上下文中，而是存储在便笺系统中
2. 讨论结束后，提取讨论结论，并将结论添加到每个Agent的记忆中
3. 便笺系统的内容会被持久化存储，可以在需要时查看

## 实现方法

### 1. 创建Agent记忆管理器

我们创建了`agent_memory_manager.py`文件，实现了Agent记忆管理器，用于管理Agent的上下文历史。主要功能包括：

- 持久化存储上下文历史
- 线程安全的读写操作
- 自动保存机制

### 2. 修改server.py

在`server.py`中，我们修改了`run_discussion_stream`函数，集成了便笺系统：

- 在讨论开始时，创建便笺管理器
- 在Agent开始思考时，将思考状态添加到便笺中
- 在Agent生成内容块时，将内容块添加到便笺中
- 在Agent完成回答时，将完整回答添加到便笺中
- 在讨论结束时，提取讨论结论，并将结论添加到每个Agent的记忆中，然后清理便笺

### 3. 创建测试脚本

我们创建了两个测试脚本：

- `test_server_agent_with_output.py` - 用于测试服务器和Agent的集成，包括上下文工程功能
- `test_server_agent_multiple_instances.py` - 用于测试多个Agent实例和便笺系统的集成

## 测试方法

### 1. 启动服务器

```bash
python server.py
```

### 2. 运行测试脚本

```bash
python test_server_agent_with_output.py
```

或者

```bash
python test_server_agent_multiple_instances.py
```

### 3. 查看测试结果

测试脚本会输出详细的测试过程和结果，包括：

- Agent创建响应
- 讨论请求响应
- 讨论状态和消息
- 便笺文件检查
- 记忆文件检查
- 记忆检索测试

## 文件结构

```
.
├── server.py                             # 服务器主文件
├── agent.py                              # Agent实现
├── agent_memory_manager.py               # Agent记忆管理器
├── ContextEngineering.py                 # 上下文工程模块
├── test_server_agent_with_output.py      # 测试脚本1
├── test_server_agent_multiple_instances.py # 测试脚本2
├── UserInterface.html                    # 用户界面
└── context_data/                         # 数据存储目录
    ├── contexts/                         # 上下文历史存储
    ├── memories/                         # 记忆存储
    └── scratchpads/                      # 便笺存储
```

## 便笺系统数据格式

便笺系统的数据以JSON格式存储在`context_data/scratchpads/`目录下，文件名格式为`{discussion_id}.json`。数据格式如下：

```json
{
  "discussion_id": "讨论ID",
  "last_updated": "最后更新时间",
  "scratchpads": {
    "agent_id": [
      {
        "content": "便笺内容",
        "metadata": {
          "type": "便笺类型",
          "timestamp": "时间戳",
          "...": "其他元数据"
        }
      }
    ]
  }
}
```

## 记忆系统数据格式

记忆系统的数据以Pickle格式存储在`context_data/memories/`目录下，文件名格式为`{agent_id}_memory.pkl`。数据格式如下：

```python
{
  "agent_id": "Agent ID",
  "memories": {
    "preferences": [...],
    "conclusions": [...],
    "requirements": [...],
    "facts": [...]
  }
}
```

## 注意事项

1. 便笺系统只在多Agent讨论模式下使用，单Agent聊天模式下不使用
2. 便笺系统的内容不会影响Agent的历史上下文，但讨论结论会被添加到Agent的记忆中
3. 便笺系统的内容会被持久化存储，可以在需要时查看
4. 在讨论结束后，便笺系统的内容会被清理，但记忆中的结论会保留
