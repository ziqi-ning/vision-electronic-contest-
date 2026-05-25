"""Download and install pytest wheels via PyPI JSON API (urllib, bypasses proxy)"""
import urllib.request
import json
import os
import sys
import subprocess
import hashlib

PYTHON_VERSION = '310'
ARCH = 'cp310-cp310-win_amd64'
WIN_PY3_ANY = f'py{PYTHON_VERSION}-none-any'
DEST = r'E:\Workspace\Ziqi-MultiProduct\AllProjectBackUp\VEC-Version2\FlightVersionOnRaspirryPi\wheels'


def get_pypi_json(package_name):
    url = f'https://pypi.org/pypi/{package_name}/json'
    req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0'})
    with urllib.request.urlopen(req, timeout=30) as resp:
        return json.loads(resp.read())


def find_compatible_wheels(releases):
    results = []
    for item in releases:
        if not item.get('url'):
            continue
        filename = item['filename']
        if filename.endswith('.whl'):
            if ARCH in filename or WIN_PY3_ANY in filename:
                results.append(item)
    return results


def download_and_install(package_name):
    print(f'\n=== {package_name} ===')
    try:
        data = get_pypi_json(package_name)
        version = data['info']['version']
        releases = data['releases'][version]
        wheels = find_compatible_wheels(releases)
        print(f'  Version: {version}, Wheels: {len(wheels)}')

        if not wheels:
            for r in releases:
                if r['filename'].endswith('.whl') and 'py3-none-any' in r['filename']:
                    wheels = [r]
                    break

        if wheels:
            wheel = wheels[-1]
            url = wheel['url']
            filename = wheel['filename']
            filepath = os.path.join(DEST, filename)
            digests = wheel['digests']

            if os.path.exists(filepath):
                print(f'  Already exists: {filename}')
            else:
                print(f'  Downloading {filename}...')
                req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0'})
                with urllib.request.urlopen(req, timeout=60) as resp:
                    data_bytes = resp.read()
                sha256_hash = hashlib.sha256(data_bytes).hexdigest()
                expected = digests['sha256']
                if sha256_hash != expected:
                    print(f'  SHA256 mismatch! Aborting.')
                    return
                with open(filepath, 'wb') as f:
                    f.write(data_bytes)
                print(f'  Downloaded ({len(data_bytes)//1024}KB), SHA256 OK')

            python = sys.executable
            print(f'  Installing...')
            result = subprocess.run(
                [python, '-m', 'pip', 'install', '--force-reinstall', '--no-deps', filepath],
                capture_output=True, text=True, timeout=60
            )
            if result.returncode == 0:
                print(f'  Installed OK')
            else:
                print(f'  FAILED: {result.stderr[-300:]}')
        else:
            print(f'  No compatible wheel found!')
    except Exception as e:
        print(f'  Error: {e}')


if __name__ == '__main__':
    os.makedirs(DEST, exist_ok=True)
    for pkg in ['pytest', 'pluggy', 'iniconfig', 'packaging',
                 'pytest-asyncio', 'exceptiongroup']:
        download_and_install(pkg)
