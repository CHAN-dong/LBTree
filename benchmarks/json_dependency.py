"""Stage the PyPI-distributed upstream JSON headers for an offline CMake build."""
from pathlib import Path
import shutil
import nlohmann_json
root = Path(nlohmann_json.__file__).parent
headers = next(root.rglob('json.hpp')).parent
target = Path('build-local/json-include/nlohmann')
shutil.copytree(headers, target, dirs_exist_ok=True)
print('JSON headers staged at', target)
