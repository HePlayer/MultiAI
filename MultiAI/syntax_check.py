#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import ast
import sys

def check_syntax(filename):
    """检查Python文件的语法错误"""
    try:
        with open(filename, 'r', encoding='utf-8') as f:
            content = f.read()
        
        # 尝试解析AST
        ast.parse(content)
        print(f"✓ {filename} 语法正确")
        return True
        
    except SyntaxError as e:
        print(f"✗ {filename} 语法错误:")
        print(f"  行 {e.lineno}: {e.text.strip() if e.text else ''}")
        print(f"  错误: {e.msg}")
        return False
        
    except Exception as e:
        print(f"✗ {filename} 检查失败: {e}")
        return False

def check_imports(filename):
    """检查导入语句是否有明显问题"""
    try:
        with open(filename, 'r', encoding='utf-8') as f:
            lines = f.readlines()
        
        print(f"\n检查 {filename} 的导入语句:")
        for i, line in enumerate(lines, 1):
            line = line.strip()
            if line.startswith('from ') or line.startswith('import '):
                if any(char in line for char in ['#', '"""', "'''"]):
                    continue  # 跳过注释
                print(f"  行 {i}: {line}")
                
    except Exception as e:
        print(f"检查导入失败: {e}")

def main():
    files_to_check = [
        'server.py',
        'agent.py', 
        'ContextEngineering.py',
        'agent_memory_manager.py',
        'stream_manager.py',
        'context_safe_manager.py',
        'discussion_state_manager.py'
    ]
    
    print("Python语法检查工具")
    print("=" * 50)
    
    for filename in files_to_check:
        try:
            check_syntax(filename)
            if filename == 'server.py':
                check_imports(filename)
        except FileNotFoundError:
            print(f"✗ {filename} 文件不存在")
        print()

if __name__ == '__main__':
    main()
