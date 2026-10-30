"""Fail when Git-tracked text files contain common plaintext credential patterns."""
import re
import subprocess
from pathlib import Path


ROOT = Path(__file__).resolve().parent
PATTERNS = {
    "Google API key": re.compile(rb"AIza[0-9A-Za-z_-]{20,}"),
    "Google API key (AQ format)": re.compile(rb"AQ\.[0-9A-Za-z_-]{20,}"),
    "Assigned API key": re.compile(
        rb"(?i)(?:api[_-]?key|secret|token)\s*[=:]\s*['\"][^'\"\r\n]{12,}['\"]"
    ),
    "Private key": re.compile(rb"-----BEGIN (?:RSA |EC |OPENSSH )?PRIVATE KEY-----"),
}
HISTORY_PATTERNS = {
    "Google API key": r"AIza[0-9A-Za-z_-]{20,}",
    "Google API key (AQ format)": r"AQ\.[0-9A-Za-z_-]{20,}",
    "Private key": r"^-----BEGIN [A-Z ]*PRIVATE KEY-----$",
}
SKIP_SUFFIXES = {
    ".png", ".jpg", ".jpeg", ".gif", ".pdf", ".npz", ".npy", ".joblib",
    ".pkl", ".pt", ".pth", ".woff", ".woff2", ".ttf", ".ico",
}


def repository_files():
    output = subprocess.check_output(
        ["git", "ls-files", "--cached", "--others", "--exclude-standard", "-z"], cwd=ROOT
    )
    return [ROOT / item.decode("utf-8") for item in output.split(b"\0") if item]


findings = []
for path in repository_files():
    if path.suffix.lower() in SKIP_SUFFIXES or not path.is_file():
        continue
    data = path.read_bytes()
    for label, pattern in PATTERNS.items():
        if pattern.search(data):
            findings.append(f"{path.relative_to(ROOT)}: {label}")

for label, expression in HISTORY_PATTERNS.items():
    history = subprocess.run(
        ["git", "log", "--all", "-G", expression, "--format=%H"],
        cwd=ROOT, capture_output=True, text=True, check=True,
    ).stdout.splitlines()
    if history:
        findings.append(f"Git history: {label} in {len(set(history))} commit(s)")

if findings:
    print("SECRET_SCAN_FAILED")
    for finding in findings:
        print(f"- {finding}")
    raise SystemExit(1)

print("SECRET_SCAN_OK")
