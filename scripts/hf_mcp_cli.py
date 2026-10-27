#!/usr/bin/env python3
import sys
import json
import subprocess

def run_mcp_command(tool_name: str, arguments: dict):
    proc = subprocess.Popen(
        ['npx', '-y', '@llmindset/hf-mcp-server'],
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True
    )
    init_msg = {
        'jsonrpc': '2.0',
        'id': 1,
        'method': 'initialize',
        'params': {
            'protocolVersion': '2024-11-05',
            'capabilities': {},
            'clientInfo': {'name': 'mark-cli', 'version': '1.0'}
        }
    }
    proc.stdin.write(json.dumps(init_msg) + '\n')
    proc.stdin.flush()
    _ = proc.stdout.readline()

    call_msg = {
        'jsonrpc': '2.0',
        'id': 2,
        'method': 'tools/call',
        'params': {
            'name': tool_name,
            'arguments': arguments
        }
    }
    proc.stdin.write(json.dumps(call_msg) + '\n')
    proc.stdin.flush()
    res_line = proc.stdout.readline()
    proc.terminate()
    try:
        return json.loads(res_line)
    except Exception as e:
        return {'error': str(e), 'raw': res_line}

if __name__ == '__main__':
    if len(sys.argv) < 2:
        print("Usage: hf_mcp_cli.py <whoami|search|details|fs> [args...]")
        sys.exit(1)
    
    cmd = sys.argv[1]
    if cmd == 'whoami':
        res = run_mcp_command('hf_whoami', {})
        print(json.dumps(res, indent=2))
    elif cmd == 'search':
        query = sys.argv[2] if len(sys.argv) > 2 else 'Qwen'
        res = run_mcp_command('hub_repo_search', {'query': query, 'limit': 5})
        print(json.dumps(res, indent=2))
    elif cmd == 'fs':
        path = sys.argv[2] if len(sys.argv) > 2 else 'hf://models/trending'
        res = run_mcp_command('hf_fs', {'operations': [{'cmd': 'ls', 'args': [path, '--limit', '5']}]})
        print(json.dumps(res, indent=2))
