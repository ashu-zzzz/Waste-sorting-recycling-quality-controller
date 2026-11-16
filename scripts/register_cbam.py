
import importlib
import inspect
import shutil
from pathlib import Path
import ultralytics.nn.tasks as tasks

target = Path(tasks.__file__)
source = target.read_text(encoding="utf-8")

import_marker = "from ultralytics.nn.modules import (\n"
branch_marker = "        elif m is AIFI:\n            args = [ch[f], *args]\n"

changed = False

if "    CBAM,\n" not in source:
    if import_marker not in source:
        raise RuntimeError("Could not find the module import section.")

    source = source.replace(
        import_marker,
        import_marker + "    CBAM,\n",
        1
    )
    changed = True

if "        elif m is CBAM:\n" not in source:
    if branch_marker not in source:
        raise RuntimeError("Could not find the model parser section.")

    source = source.replace(
        branch_marker,
        "        elif m is CBAM:\n"
        "            c2 = ch[f]\n"
        "            args = [c2, *args]\n"
        + branch_marker,
        1
    )
    changed = True

if changed:
    backup = target.with_name("tasks.py.before_cbam")

    if backup.exists():
        raise FileExistsError(
            f"Backup already exists: {backup}. No changes were made."
        )

    shutil.copy2(target, backup)
    target.write_text(source, encoding="utf-8")
    importlib.invalidate_caches()
    importlib.reload(tasks)

    print(f"Updated: {target}")
    print(f"Backup: {backup}")
else:
    print("CBAM parser support is already registered.")

print("CBAM visible to parser:", hasattr(tasks, "CBAM"))
print(
    "CBAM handled by parser:",
    "elif m is CBAM:" in inspect.getsource(tasks.parse_model)
)
