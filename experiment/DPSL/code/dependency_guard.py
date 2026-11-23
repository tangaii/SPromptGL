"""Record project imports and file access, and reject adjacent project files."""
from __future__ import annotations

import hashlib
import os
import sys
from pathlib import Path

from standalone_paths import DPSL_ROOT


class ExternalProjectAccess(PermissionError):
    pass


class DependencyGuard:
    def __init__(self) -> None:
        self.root = DPSL_ROOT
        self.project_roots = (Path('/model/spromptgl'), Path('/data/adodas/SPromptGL-main'))
        self.files: set[str] = set()
        self.denied: set[str] = set()
        self.open_events = 0

    def check(self, filename: object) -> str | None:
        if not isinstance(filename, (str, bytes, os.PathLike)):
            return None
        name = os.fsdecode(filename)
        if not name or name.startswith('<'):
            return None
        absolute = Path(os.path.abspath(name))
        if not absolute.is_relative_to(self.root) and any(absolute.is_relative_to(parent) for parent in self.project_roots):
            self.denied.add(str(absolute))
            raise ExternalProjectAccess(f'External project file rejected: {absolute}')
        resolved = Path(os.path.realpath(absolute))
        if resolved.is_relative_to(self.root):
            return resolved.relative_to(self.root).as_posix()
        if any(resolved.is_relative_to(parent) for parent in self.project_roots):
            self.denied.add(str(resolved))
            raise ExternalProjectAccess(f'External project file rejected: {resolved}')
        return None

    def hook(self, event: str, args: tuple) -> None:
        if event in {'open', 'os.listdir', 'os.scandir'} and args:
            local = self.check(args[0])
            if event == 'open':
                self.open_events += 1
                flags = args[2] if len(args) > 2 and isinstance(args[2], int) else 0
                writing = flags & (os.O_WRONLY | os.O_RDWR)
                if local is not None and not writing:
                    self.files.add(local)
        elif event == 'import' and len(args) > 1:
            self.check(args[1])
        elif event in {'os.remove', 'os.mkdir', 'os.rmdir', 'os.chmod'} and args:
            self.check(args[0])
        elif event in {'os.rename', 'os.symlink', 'os.link'} and len(args) > 1:
            self.check(args[0])
            self.check(args[1])

    def report(self) -> dict:
        modules = {}
        for name, module in tuple(sys.modules.items()):
            filename = getattr(module, '__file__', None)
            try:
                local = self.check(filename)
            except ExternalProjectAccess:
                continue
            if local is not None:
                modules[name] = local
        files = sorted(self.files)
        return {
            'status': 'PASS' if not self.denied else 'FAIL',
            'working_directory': os.getcwd(), 'python_open_events': self.open_events,
            'local_file_count': len(files), 'local_files': files,
            'local_file_list_sha256': hashlib.sha256('\n'.join(files).encode()).hexdigest(),
            'local_imports': dict(sorted(modules.items())),
            'external_project_access': sorted(self.denied),
        }
