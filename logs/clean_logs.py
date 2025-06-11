#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
日志清理脚本
用于删除日志文件中的进度条和重复的评估信息
"""

import os
import re
import argparse
import glob
from pathlib import Path


def clean_log_content(content):
    """
    清理日志内容，删除不必要的进度条和重复信息
    
    Args:
        content (str): 原始日志内容
        
    Returns:
        str: 清理后的日志内容
    """
    lines = content.split('\n')
    cleaned_lines = []
    
    # 匹配需要删除的模式
    patterns_to_remove = [
        # tqdm进度条格式: XX%|▓▓▓▓▓░░░| XX/XXX [XX:XX<XX:XX, X.XXit/s]
        r'^\s*\d+%\|[▓▒░\s▏▎▍▌▋▊▉█]*\|\s*\d+/\d+\s*\[[\d:]+<[\d:?]+,\s*[\d.]+it/s\]',
        
        # 包含EVAL和Top-1/Top-5的进度条行
        r'.*\[31m\[EVAL\].*Top-[0-9]:[0-9]+\.[0-9]+.*\d+%\|.*\d+/\d+.*\[[\d:]+<[\d:?]+.*it/s\]',
        
        # 单独的进度条行（不含EVAL）
        r'^\s*\d+%\|[▓▒░\s▏▎▍▌▋▊▉█]*\|\s*\d+/\d+\s*\[[\d:]+<[\d:?]+.*it/s\]\s*$',
        
        # 开头为0%且包含?it/s的行
        r'^\s*0%\|\s*\|\s*0/\d+\s*\[[\d:]+<\?,\s*\?it/s\]',
        
        # 包含多个连续的EVAL Top信息的行（这些通常是被覆盖的进度条）
        r'.*(\[31m\[EVAL\].*Top-[0-9]:[0-9.]+.*){2,}',
        
        # 匹配像这样的行: [31m[EVAL] Top-1:62.500| Top-5:82.812[0m: 0%| | 0/157 [00:00<?, ?it/s]
        r'.*\[31m\[EVAL\].*Top-[0-9]:[0-9.]+.*\[0m:\s*\d*%\|.*\d*/\d+.*\?it/s\].*',
        
        # 匹配纯进度条，没有其他有用信息的行
        r'^\s*\[31m\[EVAL\].*Top-[0-9]:[0-9.]+.*\[0m:\s*\d+%\|[▓▒░\s▏▎▍▌▋▊▉█]*\|\s*\d+/\d+\s*\[[\d:]+<[\d:?]+.*it/s\]\s*$'
    ]
    
    # 编译正则表达式
    compiled_patterns = [re.compile(pattern) for pattern in patterns_to_remove]
    
    for line in lines:
        # 检查是否匹配任何需要删除的模式
        should_remove = False
        for pattern in compiled_patterns:
            if pattern.match(line):
                should_remove = True
                break
        
        # 如果没有匹配到删除模式，保留该行
        if not should_remove:
            # 但是还要处理一些特殊情况，比如行中间有多个EVAL信息
            if '[31m[EVAL]' in line and line.count('[31m[EVAL]') > 1:
                # 提取最后一个完整的EVAL信息
                eval_matches = re.findall(r'\[31m\[EVAL\][^[]*\[0m', line)
                if eval_matches:
                    # 只保留最后一个EVAL信息，去掉进度条部分
                    last_eval = eval_matches[-1]
                    # 检查是否包含进度条信息，如果包含则跳过
                    if not re.search(r'\d+%\|.*\d+/\d+.*it/s', line):
                        cleaned_lines.append(last_eval)
            elif '[31m[EVAL]' in line and re.search(r'\d+%\|.*\d+/\d+.*it/s', line):
                # 如果是包含进度条的EVAL行，提取纯EVAL信息
                eval_match = re.search(r'\[31m\[EVAL\][^[]*\[0m', line)
                if eval_match:
                    eval_info = eval_match.group()
                    # 只有当这个EVAL信息不是重复的才添加
                    if not cleaned_lines or eval_info != (cleaned_lines[-1] if cleaned_lines[-1].startswith('[31m[EVAL]') else ''):
                        cleaned_lines.append(eval_info)
            else:
                cleaned_lines.append(line)
    
    return '\n'.join(cleaned_lines)


def process_log_file(file_path, backup=True, dry_run=False):
    """
    处理单个日志文件
    
    Args:
        file_path (str): 日志文件路径
        backup (bool): 是否备份原文件
        dry_run (bool): 是否只是预览，不实际修改文件
        
    Returns:
        tuple: (原始行数, 清理后行数, 是否有变化)
    """
    try:
        with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
            original_content = f.read()
        
        cleaned_content = clean_log_content(original_content)
        
        original_lines = len(original_content.split('\n'))
        cleaned_lines = len(cleaned_content.split('\n'))
        has_changes = original_content != cleaned_content
        
        if not dry_run and has_changes:
            # 备份原文件
            if backup:
                backup_path = f"{file_path}.backup"
                with open(backup_path, 'w', encoding='utf-8') as f:
                    f.write(original_content)
                print(f"原文件已备份到: {backup_path}")
            
            # 写入清理后的内容
            with open(file_path, 'w', encoding='utf-8') as f:
                f.write(cleaned_content)
        
        return original_lines, cleaned_lines, has_changes
        
    except Exception as e:
        print(f"处理文件 {file_path} 时出错: {str(e)}")
        return 0, 0, False


def find_log_files(logs_dir):
    """
    查找所有日志文件
    
    Args:
        logs_dir (str): 日志目录路径
        
    Returns:
        list: 日志文件路径列表
    """
    log_files = []
    
    # 支持的日志文件扩展名
    extensions = ['*.log', '*.txt']
    
    for ext in extensions:
        pattern = os.path.join(logs_dir, '**', ext)
        log_files.extend(glob.glob(pattern, recursive=True))
    
    return log_files


def main():
    parser = argparse.ArgumentParser(description='清理日志文件中的进度条和重复信息')
    parser.add_argument('--logs-dir', default='.', help='日志文件目录 (默认: 当前目录)')
    parser.add_argument('--no-backup', action='store_true', help='不备份原文件')
    parser.add_argument('--dry-run', action='store_true', help='预览模式，不实际修改文件')
    parser.add_argument('--file', help='处理指定的单个文件')
    parser.add_argument('--pattern', help='文件名模式，例如: "*.log"')
    
    args = parser.parse_args()
    
    # 确定要处理的文件
    if args.file:
        if os.path.exists(args.file):
            log_files = [args.file]
        else:
            print(f"文件不存在: {args.file}")
            return
    else:
        if args.pattern:
            pattern = os.path.join(args.logs_dir, '**', args.pattern)
            log_files = glob.glob(pattern, recursive=True)
        else:
            log_files = find_log_files(args.logs_dir)
    
    if not log_files:
        print("未找到日志文件")
        return
    
    print(f"找到 {len(log_files)} 个日志文件")
    if args.dry_run:
        print("预览模式 - 不会实际修改文件")
    print("-" * 50)
    
    total_original_lines = 0
    total_cleaned_lines = 0
    files_with_changes = 0
    
    for file_path in log_files:
        print(f"处理: {file_path}")
        original_lines, cleaned_lines, has_changes = process_log_file(
            file_path, 
            backup=not args.no_backup, 
            dry_run=args.dry_run
        )
        
        total_original_lines += original_lines
        total_cleaned_lines += cleaned_lines
        
        if has_changes:
            files_with_changes += 1
            removed_lines = original_lines - cleaned_lines
            print(f"  原始行数: {original_lines}, 清理后: {cleaned_lines}, 删除: {removed_lines} 行")
        else:
            print(f"  无需清理 (行数: {original_lines})")
        print()
    
    print("-" * 50)
    print("清理完成!")
    print(f"处理文件数: {len(log_files)}")
    print(f"有变化的文件数: {files_with_changes}")
    print(f"总原始行数: {total_original_lines}")
    print(f"总清理后行数: {total_cleaned_lines}")
    print(f"总删除行数: {total_original_lines - total_cleaned_lines}")
    
    if args.dry_run:
        print("\n注意: 这是预览模式，文件未被实际修改")
        print("要实际执行清理，请去掉 --dry-run 参数")


if __name__ == '__main__':
    main()
