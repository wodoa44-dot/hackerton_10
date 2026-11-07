"""Stream an archive over an existing SSH alias without an SFTP subsystem."""
import argparse
from pathlib import Path
import subprocess


def main():
    p=argparse.ArgumentParser()
    p.add_argument('archive',type=Path)
    p.add_argument('host',choices=['ff9ebb72-6ae84c15.hackathon','devopen-gpu-shell-ff9ebb72-6ae84c15'])
    p.add_argument('target')
    a=p.parse_args()
    if not a.target.startswith('/data/') or any(c not in '/_-.'+'0123456789abcdefghijklmnopqrstuvwxyz' for c in a.target):
        p.error('Target must be a simple absolute /data archive path')
    with a.archive.open('rb') as source:
        subprocess.run(['ssh','-o','BatchMode=yes','-o','ConnectTimeout=8',a.host,
                        'cat > '+a.target],stdin=source,check=True)
    print('TRANSFERRED',a.archive.stat().st_size,'bytes',a.host,a.target)


if __name__=='__main__':main()
