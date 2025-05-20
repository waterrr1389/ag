#!/usr/bin/env python3
# -*- coding: utf-8 -*- # 确保正确处理中文字符

import argparse
import os
import sys
import subprocess
import json
from openai import AsyncOpenAI, APIError, APITimeoutError, APIConnectionError

# 引入 prompt_toolkit 的异步 Session 和样式化文本
from prompt_toolkit import PromptSession
from prompt_toolkit.history import FileHistory # 用于保存命令历史
from prompt_toolkit.auto_suggest import AutoSuggestFromHistory # 基于历史的自动建议
from prompt_toolkit.formatted_text import FormattedText # 用于样式化提示

# --- 配置 ---
try:
    # 初始化 AsyncOpenAI 客户端
    # 它会自动从环境变量 OPENAI_API_KEY 读取密钥
    client = AsyncOpenAI(
        base_url="https://api.siliconflow.cn/v1",
        timeout=120.0, # 设置请求超时时间
    )
except Exception as e:
    print(f"AsyncOpenAI 初始化错误: {e}")
    sys.exit(1)

# 全局对话历史记录
conversation_history = []

# --- 大语言模型交互 ---
async def get_llm_response_stream(prompt: str, history: list):
    """
    异步函数，使用 AsyncOpenAI 库向 OpenAI API 发送请求并流式接收响应。
    """
    if not os.environ.get("OPENAI_API_KEY"):
       yield "\n错误：OPENAI_API_KEY 环境变量未设置。"
       return

    current_messages = []
    # 添加历史记录
    for h_entry in history:
        current_messages.append({"role": h_entry["role"], "content": h_entry["content"]})
    # 添加当前用户输入
    current_messages.append({"role": "user", "content": prompt})

    try:
        # 创建流式聊天请求
        stream = await client.chat.completions.create(
            model="Pro/deepseek-ai/DeepSeek-V3",  # 您可以根据需要更改模型，例如 "gpt-3.5-turbo"
            messages=current_messages,
            stream=True,
        )
        # 异步迭代处理响应流
        async for chunk in stream:
            if chunk.choices and chunk.choices[0].delta and chunk.choices[0].delta.content is not None:
                yield chunk.choices[0].delta.content
    except APITimeoutError:
        yield "\n错误: 请求 OpenAI API 超时。"
    except APIConnectionError as e:
        yield f"\n错误: 无法连接到 OpenAI API。详情: {e}"
    except APIError as e: # 处理 OpenAI API 特定错误
        status_code = e.status_code if hasattr(e, 'status_code') else '未知状态码'
        error_message = e.message if hasattr(e, 'message') else str(e)
        yield f"\nOpenAI API 返回错误: {status_code} - {error_message}"
    except Exception as e: # 处理其他未知错误
        yield f"\n与 OpenAI API 互动时发生未知错误: {e} (类型: {type(e).__name__})"

# --- Shell 命令执行 ---
def execute_shell_command(command: str):
    """执行 shell 命令并打印其标准输出和标准错误。"""
    try:
        # 移除 '!' 前缀和两端空格
        command_to_run = command.strip()[1:].strip()
        if not command_to_run:
            print("请输入要执行的命令。例如: !ls -l")
            return

        print(f"\033[93m执行中: {command_to_run}\033[0m") # 黄色显示执行的命令
        # 使用 Popen 以非阻塞方式执行命令，并实时获取输出
        process = subprocess.Popen(
            command_to_run,
            shell=True, # 注意: shell=True 存在安全风险，确保命令来源可信
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True, # 以文本模式处理输出
            bufsize=1, # 行缓冲
            universal_newlines=True # 兼容不同系统的换行符
        )
        
        # 实时打印标准输出
        if process.stdout:
            for line in process.stdout:
                print(line, end='')
        
        # 等待命令结束
        process.wait()
        # 打印标准错误（如果存在）
        if process.stderr:
            stderr_output_str = process.stderr.read()
            if stderr_output_str:
                print(f"\033[91m标准错误:\n{stderr_output_str}\033[0m", file=sys.stderr) # 红色显示错误

    except FileNotFoundError:
        print(f"\033[91m错误: 命令 '{command_to_run.split()[0]}' 未找到。\033[0m")
    except Exception as e:
        print(f"\033[91m执行命令时发生错误: {e}\033[0m")

# --- 主要应用逻辑 ---
async def interactive_chat():
    """互动式对话的主循环 (使用 prompt_toolkit)。"""
    global conversation_history
    print("欢迎使用 ag (AI命令行助手)! 输入 'exit'、'quit' 或 'bye' 退出。以 '!' 开头执行shell命令。")
    if not os.environ.get("OPENAI_API_KEY"):
        print("\033[91m警告：OPENAI_API_KEY 环境变量未设置。程序可能无法正常工作。\033[0m")
        print("请设置 export OPENAI_API_KEY='your_openai_api_key'")

    # 将历史记录保存在用户主目录的 .ag_history 文件中
    history_file = os.path.expanduser("~/.ag_history")
    session = PromptSession(
        history=FileHistory(history_file),
        auto_suggest=AutoSuggestFromHistory(), # 输入时根据历史提供建议
    )

    while True:
        try:
            # 使用 prompt_toolkit 的样式化文本定义提示符 (简体中文)
            prompt_message = FormattedText([
                ('green', '你: ') # 使用 prompt_toolkit 的命名颜色 'green'
            ])
            
            # 异步获取用户输入
            prompt = await session.prompt_async(prompt_message)
            prompt = prompt.strip() # 去除两端空格

            if not prompt: # 用户直接按 Enter
                continue
            if prompt.lower() in ["exit", "quit", "bye"]: # 退出命令
                print("再见!")
                break
            elif prompt.startswith("!"): # Shell 命令
                execute_shell_command(prompt)
                continue # 执行完 Shell 命令后继续等待下一个输入
            
            # AI 回应的颜色由 print 直接处理
            print("\033[94mAI: \033[0m", end='', flush=True) # 蓝色 "AI: "
            assistant_response_parts = []
            # 获取并打印 AI 的流式响应
            async for chunk in get_llm_response_stream(prompt, conversation_history):
                print(chunk, end='', flush=True)
                assistant_response_parts.append(chunk)
            print() # AI 回答完毕后换行

            # 将用户输入和AI的完整回复都添加到历史记录
            if prompt: # 确保用户有输入
                 conversation_history.append({"role": "user", "content": prompt})
            if assistant_response_parts: # 确保AI有回复
                 full_response = "".join(assistant_response_parts)
                 # 避免将错误信息本身当做AI的正式回复加入历史
                 if not full_response.startswith("\n错误:"):
                    conversation_history.append({"role": "assistant", "content": full_response})
            
            # 限制历史记录长度，防止超出API限制或占用过多内存
            if len(conversation_history) > 20: # 保留最近20条消息 (用户 + AI)
                conversation_history = conversation_history[-20:]

        except KeyboardInterrupt: # 用户按下 Ctrl+C
            # prompt_toolkit 默认会清空当前输入行，这里直接继续循环即可
            continue 
        except EOFError: # 用户按下 Ctrl+D
            print("再见! (收到 EOF)")
            break # 退出循环
        except Exception as e: # 其他未捕获的异常
            print(f"\033[91m发生未处理的错误: {e} (类型: {type(e).__name__})\033[0m")


async def single_query(query: str):
    """处理单次查询。"""
    # 单次查询的 AI 回应颜色也由 print 直接处理
    print("\033[94mAI: \033[0m", end='', flush=True)
    assistant_response_parts = []
    # 对于单次查询，历史记录为空
    async for chunk in get_llm_response_stream(query, []):
        print(chunk, end='', flush=True)
        assistant_response_parts.append(chunk)
    print()


async def main():
    # 配置命令行参数解析器
    parser = argparse.ArgumentParser(description="ag - AI驱动的命令行助手 (OpenAI版)。")
    parser.add_argument("query", nargs="?", type=str, help="直接向AI提问（单次查询模式）。如果未提供，则进入互动模式。")
    
    args = parser.parse_args()

    # 检查 API 密钥是否已设置
    if not os.environ.get("OPENAI_API_KEY"):
        print("\033[91m错误：OPENAI_API_KEY 环境变量未设置。\033[0m")
        print("请运行 export OPENAI_API_KEY='your_openai_api_key' 进行设置。")
        sys.exit(1)

    # 根据是否有查询参数决定运行模式
    if args.query:
        await single_query(args.query)
    else:
        await interactive_chat()

if __name__ == "__main__":
    try:
        import asyncio
        asyncio.run(main()) # 运行主异步函数
    except KeyboardInterrupt: # 在 main 循环之外捕获 Ctrl+C (例如，在 asyncio.run 之前)
        print("\n程序已退出。")
    except ImportError as e: # 处理依赖库未安装的情况
        if "prompt_toolkit" in str(e).lower():
            print("错误：请确保您已安装 `prompt_toolkit` 函数库。")
            print("请执行 `pip install prompt_toolkit` 来安装依赖。")
        elif "openai" in str(e).lower():
             print("错误：请确保您已安装 `openai` 函数库。")
             print("请执行 `pip install openai` 来安装依赖。")
        else:
            print(f"导入错误: {e}")
    except Exception as e: # 捕获其他启动时可能发生的严重错误
        print(f"程序启动时发生严重错误: {e}")

