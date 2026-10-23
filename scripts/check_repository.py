"""Audit publication contents without importing project code or running experiments."""

import argparse
import ast
import json
import re
import subprocess
import sys
from pathlib import Path
from urllib.parse import unquote, urlsplit

ROOT = Path(__file__).resolve().parents[1]
ALLOWED_ARTIFACT_DOCS = {'data/README.md', 'data/raw/.gitkeep', 'data/processed/.gitkeep',
                         'artifacts/README.md', 'artifacts/.gitkeep'}
SECRET_PATTERNS = [r'ghp_[A-Za-z0-9]{30,}', r'github_pat_[A-Za-z0-9_]{50,}',
                   r'-----BEGIN (?:RSA |EC |OPENSSH )?PRIVATE KEY-----']


def publication_files():
    result = subprocess.run(['git', 'ls-files', '--cached', '--others', '--exclude-standard', '-z'],
                            cwd=ROOT, capture_output=True, check=True)
    return sorted(set(result.stdout.decode().split('\0')) - {''})


def audit():
    names = publication_files()
    public = set(names)
    errors = []
    counts = {'files': len(names), 'python_sources': 0, 'json_files': 0, 'notebooks': 0,
              'markdown_links': 0}
    for name in names:
        path = ROOT / name
        if not path.is_file():
            errors.append(f'Missing publication file: {name}')
            continue
        if path.is_symlink():
            errors.append(f'Symlink requires explicit publication review: {name}')
        if path.stat().st_size > 5_000_000:
            errors.append(f'Unexpected large publication file: {name}')
        if (name.startswith(('data/', 'artifacts/')) and name not in ALLOWED_ARTIFACT_DOCS
            or path.suffix in {'.safetensors', '.pt', '.pth', '.ckpt', '.zip', '.arrow', '.jsonl'}
            or path.name in {'private-author.tex', 'course-submission.tex', 'course-submission.pdf'}
            or path.name.startswith('.env')):
            errors.append(f'Private/data/model artifact in publication set: {name}')
        if path.suffix in {'.py', '.json', '.ipynb', '.md', '.tex', '.yml', '.yaml', '.toml', '.cff'}:
            text = path.read_text()
            if any(re.search(pattern, text) for pattern in SECRET_PATTERNS):
                errors.append(f'Credential-shaped content requires review: {name}')
        if path.suffix == '.py':
            ast.parse(path.read_text(), filename=name)
            counts['python_sources'] += 1
        if path.suffix in {'.json', '.ipynb'}:
            value = json.loads(path.read_text())
            counts['json_files'] += 1
            if path.suffix == '.ipynb':
                counts['notebooks'] += 1
                for number, cell in enumerate(value['cells'], 1):
                    if cell['cell_type'] != 'code':
                        continue
                    if cell.get('outputs') or cell.get('execution_count') is not None:
                        errors.append(f'Notebook contains executed output: {name} cell {number}')
                    source = ''.join(cell.get('source', []))
                    for match in re.finditer(r'^\s*(RUN_[A-Z0-9_]+)\s*=\s*(True|False)\b', source, re.M):
                        if match.group(2) != 'False':
                            errors.append(f'Enabled notebook switch: {name}: {match.group(1)}')
                    if not source.lstrip().startswith('%%'):
                        cleaned = '\n'.join('pass' if line.lstrip().startswith(('%', '!')) else line
                                            for line in source.splitlines())
                        ast.parse(cleaned, filename=f'{name}:cell{number}')
        if path.suffix == '.md':
            text = re.sub(r'```.*?```', '', path.read_text(), flags=re.S)
            for match in re.finditer(r'!?\[[^\]]*\]\(([^)]+)\)', text):
                target = match.group(1).strip().strip('<>')
                parts = urlsplit(target)
                if parts.scheme or not parts.path:
                    continue
                resolved = (path.parent / unquote(parts.path)).resolve()
                try:
                    relative = resolved.relative_to(ROOT).as_posix()
                except ValueError:
                    errors.append(f'Local link leaves repository: {name} -> {target}')
                    continue
                counts['markdown_links'] += 1
                if relative not in public and not any(x.startswith(relative.rstrip('/') + '/') for x in public):
                    errors.append(f'Link target absent from public Git files: {name} -> {target}')
    report = subprocess.run([sys.executable, str(ROOT / 'scripts/build_report_tables.py'), '--check'],
                            cwd=ROOT, capture_output=True, text=True)
    if report.returncode:
        errors.append('Generated paper tables/binding differ: ' + report.stderr.strip())
    return {'status': 'passed' if not errors else 'failed', 'counts': counts,
            'errors': errors, 'experiments_or_test_suite_run': False,
            'models_or_data_downloaded': False, 'report_binding_check': report.returncode == 0}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--json', type=Path, help='Optional local audit receipt path.')
    args = parser.parse_args()
    result = audit()
    if args.json:
        args.json.parent.mkdir(parents=True, exist_ok=True)
        args.json.write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps(result, indent=2))
    if result['errors']:
        raise SystemExit(1)


if __name__ == '__main__':
    main()
