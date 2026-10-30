"""Run a command on the authorized experiment host; never print credentials."""
import sys
from pathlib import Path
import paramiko

cfg = dict(line.split(':', 1) for line in Path('login.md').read_text().splitlines() if ':' in line)
client = paramiko.SSHClient()
client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
client.connect(cfg['ip'], port=int(cfg['port']), username=cfg['user'], password=cfg['passwd'], timeout=30)
try:
    if sys.argv[1] == 'put':
        with client.open_sftp() as sftp:
            sftp.put(sys.argv[2], sys.argv[3])
    elif sys.argv[1] == 'get':
        with client.open_sftp() as sftp:
            sftp.get(sys.argv[2], sys.argv[3])
    else:
        _, out, err = client.exec_command(sys.argv[1], timeout=180)
        print(out.read().decode('utf-8', errors='replace'))
        print(err.read().decode('utf-8', errors='replace'), file=sys.stderr)
        sys.exit(out.channel.recv_exit_status())
finally:
    client.close()
