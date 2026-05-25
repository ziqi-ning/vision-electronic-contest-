"""Download and install pytest-asyncio 0.23.8 (compatible with Python 3.10)"""
import urllib.request
import json
import os
import sys
import subprocess
import hashlib

DEST = r'E:\Workspace\Ziqi-MultiProduct\AllProjectBackUp\VEC-Version2\FlightVersionOnRaspirryPi\wheels'
PYVER = '310'

def get_wheels(package, version):
    url = f'https://pypi.org/pypi/{package}/{version}/json'
    req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0'})
    data = json.loads(urllib.request.urlopen(req, timeout=30).read())
    return data['urls']

def download_and_install(url, filename):
    filepath = os.path.join(DEST, filename)
    if os.path.exists(filepath):
        print(f'  Already: {filename}')
    else:
        print(f'  Downloading {filename}...')
        req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0'})
        with urllib.request.urlopen(req, timeout=60) as resp:
            data = resp.read()
        with open(filepath, 'wb') as f:
            f.write(data)
        print(f'  Downloaded {len(data)//1024}KB')
    # Install
    result = subprocess.run(
        [sys.executable, '-m', 'pip', 'install', '--force-reinstall', '--no-deps', filepath],
        capture_output=True, text=True, timeout=60
    )
    if result.returncode == 0:
        print(f'  Installed OK')
    else:
        print(f'  FAILED: {result.stderr[-200:]}')

if __name__ == '__main__':
    os.makedirs(DEST, exist_ok=True)

    # Install a compatible version of pytest-asyncio
    version = '0.23.8'
    print(f'=== pytest-asyncio {version} ===')
    urls = get_wheels('pytest-asyncio', version)
    for u in urls:
        fn = u['filename']
        is_any = 'none-any' in fn
        is_cpython_win = f'cp{PYVER}-cp{PYVER}-win_amd64' in fn
        if is_any or is_cpython_win:
            download_and_install(u['url'], fn)
