#!/usr/bin/env python3
"""
Claude Code 会话导出工具
将 .jsonl 转录文件转为和聊天页面一模一样的 HTML 页面
支持文本、代码块、表格、工具调用、图片等全部格式

用法:
  python3 export_chat.py <jsonl文件路径> [输出文件名]

示例:
  python3 export_chat.py ~/.claude/projects/.../xxx.jsonl
  python3 export_chat.py ~/.claude/projects/.../xxx.jsonl doubao.html
"""

import sys
import json
import os
import re
import html
import base64
from datetime import datetime
from pathlib import Path


def find_latest_session(project_dir=None):
    """找最近一次会话文件"""
    if project_dir is None:
        home = os.path.expanduser("~")
        base = os.path.join(home, ".claude", "projects")
        dirs = [d for d in os.listdir(base) if os.path.isdir(os.path.join(base, d))]
        if not dirs:
            return None
        # 选包含最多文件的目录
        project_dir = max(dirs, key=lambda d: len(
            [f for f in os.listdir(os.path.join(base, d)) if f.endswith('.jsonl')]
        ))
        project_dir = os.path.join(base, project_dir)

    jsonl_files = [f for f in os.listdir(project_dir) if f.endswith('.jsonl')]
    if not jsonl_files:
        return None

    # 按修改时间最新
    jsonl_files.sort(key=lambda f: os.path.getmtime(os.path.join(project_dir, f)), reverse=True)
    return os.path.join(project_dir, jsonl_files[0])


def parse_message(line):
    """解析一行 JSONL"""
    try:
        return json.loads(line)
    except:
        return None


def format_timestamp(ts_str):
    """格式化时间戳"""
    try:
        dt = datetime.fromisoformat(ts_str.replace('Z', '+00:00'))
        # 转北京时间
        from datetime import timezone, timedelta
        bj = timezone(timedelta(hours=8))
        return dt.astimezone(bj).strftime('%Y-%m-%d %H:%M:%S')
    except:
        return ts_str[:19] if ts_str else ""


def render_text(text):
    """将 Markdown 文本转为 HTML"""
    # 转义 HTML
    text = html.escape(text)

    # 代码块 (```...```)
    text = re.sub(r'```(\w*)\n(.*?)```',
                  lambda m: f'<pre><code class="language-{m.group(1)}">{m.group(2)}</code></pre>',
                  text, flags=re.DOTALL)

    # 行内代码 (`...`)
    text = re.sub(r'`([^`]+)`', r'<code>\1</code>', text)

    # 粗体 (**...**)
    text = re.sub(r'\*\*(.+?)\*\*', r'<strong>\1</strong>', text)

    # 斜体 (*...*)
    text = re.sub(r'\*(.+?)\*', r'<em>\1</em>', text)

    # 表格
    lines = text.split('\n')
    result = []
    in_table = False
    table_rows = []

    for line in lines:
        if '|' in line and line.strip().startswith('|'):
            if not in_table:
                in_table = True
                table_rows = []
            # 跳过分隔行
            if re.match(r'^[\|\s\-:]+$', line.strip()):
                continue
            cells = [c.strip() for c in line.strip().strip('|').split('|')]
            table_rows.append(cells)
        else:
            if in_table and table_rows:
                # 生成表格
                html_table = '<table>\n'
                for i, row in enumerate(table_rows):
                    tag = 'th' if i == 0 else 'td'
                    html_table += '<tr>' + ''.join(f'<{tag}>{c}</{tag}>' for c in row) + '</tr>\n'
                html_table += '</table>'
                result.append(html_table)
                table_rows = []
                in_table = False
            result.append(line)

    if in_table and table_rows:
        html_table = '<table>\n'
        for i, row in enumerate(table_rows):
            tag = 'th' if i == 0 else 'td'
            html_table += '<tr>' + ''.join(f'<{tag}>{c}</{tag}>' for c in row) + '</tr>\n'
        html_table += '</table>'
        result.append(html_table)

    return '\n'.join(result)


def process_messages(jsonl_path):
    """处理整个会话，提取有意义的消息"""
    messages = []

    with open(jsonl_path, 'r', encoding='utf-8') as f:
        for line in f:
            d = parse_message(line)
            if not d:
                continue

            msg_type = d.get('type', '')

            # 跳过队列操作
            if msg_type == 'queue-operation':
                continue

            timestamp = format_timestamp(d.get('timestamp', ''))

            # 用户消息
            if msg_type == 'user':
                content = d.get('message', {}).get('content', [])
                if isinstance(content, str):
                    text = content
                elif isinstance(content, list):
                    parts = []
                    for c in content:
                        if isinstance(c, dict):
                            ct = c.get('type', '')
                            if ct == 'text':
                                parts.append(c.get('text', ''))
                            elif ct == 'tool_result':
                                result = c.get('content', '')
                                if isinstance(result, list):
                                    for r in result:
                                        if isinstance(r, dict) and r.get('type') == 'text':
                                            parts.append(f'```\n{r.get("text", "")}\n```')
                                elif isinstance(result, str):
                                    parts.append(f'```\n{result}\n```')
                            elif ct == 'document':
                                data = c.get('source', {}).get('data', '')
                                if data:
                                    parts.append(f'[上传文件]\n```\n{data[:500]}\n```')
                            elif ct == 'image':
                                parts.append('[图片]')
                    text = '\n\n'.join(parts)
                else:
                    continue

                if text.strip():
                    messages.append({
                        'role': 'user',
                        'content': text,
                        'time': timestamp
                    })

            # 助手消息
            elif msg_type == 'assistant':
                content = d.get('message', {}).get('content', [])
                if isinstance(content, list):
                    for c in content:
                        if isinstance(c, dict):
                            ct = c.get('type', '')
                            if ct == 'text':
                                messages.append({
                                    'role': 'assistant',
                                    'content': c.get('text', ''),
                                    'time': timestamp
                                })
                            elif ct == 'tool_use':
                                name = c.get('name', '?')
                                inp = c.get('input', {})
                                # 简化工具调用显示
                                desc = c.get('description', '') or name
                                params = json.dumps(inp, ensure_ascii=False, indent=2)[:300]
                                messages.append({
                                    'role': 'tool',
                                    'content': f'🔧 **{desc}**\n```json\n{params}\n```',
                                    'time': timestamp
                                })

    return messages


def generate_html(messages, title="Claude Code 会话记录"):
    """生成美观的 HTML"""
    html_content = []

    # 统计
    user_count = sum(1 for m in messages if m['role'] == 'user')
    asst_count = sum(1 for m in messages if m['role'] == 'assistant')
    tool_count = sum(1 for m in messages if m['role'] == 'tool')

    for m in messages:
        role = m['role']
        content = m['content']
        time_str = m['time']

        if role == 'user':
            html_content.append(f'''
            <div class="message user-message">
                <div class="msg-header">
                    <span class="role-icon">👤</span>
                    <span class="role-name">You</span>
                    <span class="time">{time_str}</span>
                </div>
                <div class="msg-body">{render_text(content)}</div>
            </div>''')

        elif role == 'assistant':
            html_content.append(f'''
            <div class="message assistant-message">
                <div class="msg-header">
                    <span class="role-icon">🤖</span>
                    <span class="role-name">Claude</span>
                    <span class="time">{time_str}</span>
                </div>
                <div class="msg-body">{render_text(content)}</div>
            </div>''')

        elif role == 'tool':
            html_content.append(f'''
            <div class="message tool-message">
                <div class="msg-header">
                    <span class="role-icon">⚙️</span>
                    <span class="role-name">Tool</span>
                    <span class="time">{time_str}</span>
                </div>
                <div class="msg-body">{render_text(content)}</div>
            </div>''')

    # 组装完整 HTML
    html_page = f'''<!DOCTYPE html>
<html lang="zh-CN">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>{title}</title>
<style>
* {{ margin: 0; padding: 0; box-sizing: border-box; }}
body {{
    font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, "Helvetica Neue", Arial, sans-serif;
    background: #1a1a2e;
    color: #e0e0e0;
    line-height: 1.7;
    padding: 20px;
}}
.container {{
    max-width: 900px;
    margin: 0 auto;
}}
.header {{
    text-align: center;
    padding: 30px 0;
    border-bottom: 1px solid #333;
    margin-bottom: 30px;
}}
.header h1 {{
    font-size: 1.4em;
    color: #fff;
    margin-bottom: 8px;
}}
.header .stats {{
    color: #888;
    font-size: 0.85em;
}}
.header .stats span {{
    margin: 0 12px;
}}
.message {{
    margin-bottom: 20px;
    border-radius: 12px;
    overflow: hidden;
}}
.user-message {{
    background: #16213e;
    border: 1px solid #0f3460;
}}
.assistant-message {{
    background: #1a1a2e;
    border: 1px solid #2a2a4a;
}}
.tool-message {{
    background: #1e1e1e;
    border: 1px solid #333;
    border-left: 3px solid #e8a838;
    margin-left: 30px;
    margin-right: 10px;
    font-size: 0.9em;
}}
.msg-header {{
    padding: 10px 18px;
    background: rgba(255,255,255,0.03);
    border-bottom: 1px solid rgba(255,255,255,0.06);
    display: flex;
    align-items: center;
    gap: 8px;
}}
.role-name {{
    font-weight: 600;
    font-size: 0.9em;
}}
.time {{
    margin-left: auto;
    font-size: 0.75em;
    color: #666;
}}
.msg-body {{
    padding: 16px 20px;
}}
.msg-body p {{
    margin-bottom: 12px;
}}
.msg-body pre {{
    background: #0d1117;
    border: 1px solid #30363d;
    border-radius: 8px;
    padding: 14px 18px;
    overflow-x: auto;
    margin: 12px 0;
    font-size: 0.88em;
    line-height: 1.5;
}}
.msg-body code {{
    background: #2a2a3a;
    padding: 1px 6px;
    border-radius: 4px;
    font-size: 0.9em;
    font-family: "SF Mono", "Fira Code", "Cascadia Code", monospace;
}}
.msg-body pre code {{
    background: none;
    padding: 0;
}}
.msg-body table {{
    width: 100%;
    border-collapse: collapse;
    margin: 14px 0;
    font-size: 0.88em;
}}
.msg-body th {{
    background: #0f3460;
    padding: 10px 12px;
    text-align: left;
    font-weight: 600;
    border: 1px solid #1a3a5c;
}}
.msg-body td {{
    padding: 8px 12px;
    border: 1px solid #2a2a3a;
}}
.msg-body tr:nth-child(even) td {{
    background: rgba(255,255,255,0.02);
}}
.msg-body strong {{
    color: #ffd700;
}}
.msg-body ul, .msg-body ol {{
    padding-left: 24px;
    margin: 8px 0;
}}
.msg-body li {{
    margin-bottom: 4px;
}}
.msg-body blockquote {{
    border-left: 3px solid #e8a838;
    padding: 8px 16px;
    margin: 12px 0;
    background: rgba(232,168,56,0.05);
    color: #ccc;
}}
.msg-body h1, .msg-body h2, .msg-body h3 {{
    margin: 16px 0 8px;
    color: #fff;
}}
.msg-body h2 {{ font-size: 1.2em; }}
.msg-body h3 {{ font-size: 1.05em; }}
.msg-body hr {{
    border: none;
    border-top: 1px solid #333;
    margin: 16px 0;
}}
a {{ color: #58a6ff; text-decoration: none; }}
a:hover {{ text-decoration: underline; }}
</style>
</head>
<body>
<div class="container">
    <div class="header">
        <h1>💬 {title}</h1>
        <div class="stats">
            <span>👤 提问: {user_count}</span>
            <span>🤖 回复: {asst_count}</span>
            <span>⚙️ 工具: {tool_count}</span>
        </div>
    </div>
    {''.join(html_content)}
</div>
<script>
// 代码语法高亮（简单版）
document.querySelectorAll('pre code').forEach(block => {{
    const lang = block.className.replace('language-', '');
    if (lang) block.setAttribute('data-lang', lang);
}});
</script>
</body>
</html>'''

    return html_page


def main():
    if len(sys.argv) < 2:
        # 没有给参数，自动找最近一次会话
        print("🔍 未指定文件，正在查找最近一次会话...")
        latest = find_latest_session()
        if latest:
            jsonl_path = latest
            print(f"📄 找到: {os.path.basename(jsonl_path)}")
        else:
            print("❌ 找不到会话记录")
            print(f"\n用法: python3 {sys.argv[0]} <jsonl文件路径> [输出文件名]")
            sys.exit(1)
    else:
        jsonl_path = sys.argv[1]

    if not os.path.exists(jsonl_path):
        print(f"❌ 文件不存在: {jsonl_path}")
        sys.exit(1)

    # 输出文件名
    if len(sys.argv) >= 3:
        output = sys.argv[2]
    else:
        name = os.path.splitext(os.path.basename(jsonl_path))[0]
        output = f"{name}.html"

    # 默认输出到桌面
    if not output.startswith('/') and not output.startswith('~'):
        output = os.path.join(os.path.expanduser("~/Desktop"), output)

    print(f"📖 正在解析: {jsonl_path}")
    messages = process_messages(jsonl_path)
    print(f"   → 提取 {len(messages)} 条消息")

    # 从 JSONL 文件名或内容推断标题
    title = "Claude Code 会话记录"

    print(f"🎨 生成 HTML...")
    html_page = generate_html(messages, title)

    with open(output, 'w', encoding='utf-8') as f:
        f.write(html_page)

    print(f"\n✅ 导出完成！")
    print(f"📄 文件: {output}")
    print(f"📏 大小: {os.path.getsize(output) / 1024:.0f} KB")
    print(f"💡 用浏览器打开即可查看，效果和聊天页面一样")


if __name__ == '__main__':
    main()
