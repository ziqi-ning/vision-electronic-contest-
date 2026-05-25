import urllib.request
import json

# Check pytest-asyncio requirements
url = 'https://pypi.org/pypi/pytest-asyncio/json'
req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0'})
with urllib.request.urlopen(req, timeout=30) as resp:
    data = json.loads(resp.read())

print('pytest-asyncio versions:')
for v in list(data['releases'].keys())[-10:]:
    print(f'  {v}')
print()
info = data['info']
print(f'Requires: {info.get("requires", [])}')
print(f'Requires Python: {info.get("requires_python")}')
print(f'Latest: {info["version"]}')
