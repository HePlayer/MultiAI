#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import sys
import traceback

def test_import(module_name, description=""):
    try:
        print(f"测试导入 {module_name} {description}...", end="")
        if module_name == "server":
            import server
        elif module_name == "agent":
            import agent
        elif module_name == "ContextEngineering":
            import ContextEngineering
        elif module_name == "agent_memory_manager":
            import agent_memory_manager
        elif module_name == "stream_manager":
            import stream_manager
        elif module_name == "context_safe_manager":
            import context_safe_manager
        elif module_name == "discussion_state_manager":
            import discussion_state_manager
        else:
            exec(f"import {module_name}")
        print(" ✓ 成功")
        return True
    except Exception as e:
        print(f" ✗ 失败: {e}")
        print(f"详细错误信息:")
        traceback.print_exc()
        print("-" * 50)
        return False

print("开始诊断...")
print("=" * 60)

# 测试基础依赖
print("1. 测试基础Python模块:")
test_import("json")
test_import("datetime") 
test_import("os")
test_import("threading")
test_import("uuid")
test_import("time")

print("\n2. 测试Flask相关模块:")
test_import("flask")
test_import("flask_cors")

print("\n3. 测试项目核心模块:")
success_agent = test_import("agent")
success_ce = test_import("ContextEngineering")
success_amm = test_import("agent_memory_manager")

print("\n4. 测试新的函数式模块:")
success_stream = test_import("stream_manager")
success_context = test_import("context_safe_manager")
success_discussion = test_import("discussion_state_manager")

print("\n5. 测试server模块:")
success_server = test_import("server")

print("\n" + "=" * 60)
print("诊断总结:")
print(f"- agent模块: {'✓' if success_agent else '✗'}")
print(f"- ContextEngineering模块: {'✓' if success_ce else '✗'}")
print(f"- agent_memory_manager模块: {'✓' if success_amm else '✗'}")
print(f"- stream_manager模块: {'✓' if success_stream else '✗'}")
print(f"- context_safe_manager模块: {'✓' if success_context else '✗'}")
print(f"- discussion_state_manager模块: {'✓' if success_discussion else '✗'}")
print(f"- server模块: {'✓' if success_server else '✗'}")

if success_server:
    print("\n6. 检查server模块关键函数:")
    try:
        import server
        functions_to_check = [
            'is_complex_question',
            'analyze_user_intervention', 
            'chat_stream',
            'chat',
            'create_agent',
            'delete_agent'
        ]
        
        for func_name in functions_to_check:
            if hasattr(server, func_name):
                print(f"  ✓ {func_name}")
            else:
                print(f"  ✗ {func_name}")
                
    except Exception as e:
        print(f"  检查server函数时出错: {e}")

print("\n诊断完成!")
