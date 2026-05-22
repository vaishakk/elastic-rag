#!/usr/bin/env python3

from __future__ import annotations

import base64
import json
import os
import ssl
import urllib.error
import urllib.request
from pathlib import Path


DEFAULT_ES_URL = "https://192.168.1.53:9200"

# Edit these values directly before running the script.
METHOD = "DELETE"
BODY = None
PATH = "/rag-documents"

def load_env_file(path: Path) -> None:
    if not path.exists():
        return

    for raw_line in path.read_text().splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue

        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip()

        if len(value) >= 2 and value[0] == value[-1] and value[0] in {'"', "'"}:
            value = value[1:-1]

        os.environ.setdefault(key, value)


def resolve_ca_cert(script_dir: Path) -> str:
    ca_cert = os.environ.get("ES_CA_CERT")
    if ca_cert:
        ca_path = Path(ca_cert)
        if not ca_path.is_absolute():
            ca_path = script_dir / ca_path
        return str(ca_path)

    if (script_dir / "http_ca.crt").exists():
        return str(script_dir / "http_ca.crt")

    es_home = os.environ.get("ES_HOME")
    if es_home:
        return str(Path(es_home) / "config/certs/http_ca.crt")

    raise SystemExit("Set ES_CA_CERT or ES_HOME, or place http_ca.crt next to this script.")


def normalize_path(path: str) -> str:
    path = path.strip()
    if not path:
        return "/"
    if path.startswith("http://") or path.startswith("https://"):
        return path
    if not path.startswith("/"):
        return "/" + path
    return path


def build_ssl_context(ca_cert: str) -> ssl.SSLContext:
    return ssl.create_default_context(cafile=ca_cert)


def build_auth_header(password: str) -> str:
    token = f"elastic:{password}".encode("utf-8")
    return "Basic " + base64.b64encode(token).decode("ascii")


def print_response(status: int, reason: str, headers: object, body: bytes) -> None:
    print(f"Status: {status} {reason}")

    content_type = headers.get_content_type()
    text = body.decode("utf-8", errors="replace")

    if content_type == "application/json":
        try:
            payload = json.loads(text)
        except json.JSONDecodeError:
            print(text)
        else:
            print(json.dumps(payload, indent=2, sort_keys=True))
    else:
        print(text)


def main() -> int:
    script_dir = Path(__file__).resolve().parent
    load_env_file(script_dir / ".env")

    es_url = os.environ.get("ES_URL", DEFAULT_ES_URL)
    password = os.environ.get("ELASTIC_PASSWORD")
    if not password:
        raise SystemExit("ELASTIC_PASSWORD must be set in the environment or .env")

    ca_cert = resolve_ca_cert(script_dir)
    ssl_context = build_ssl_context(ca_cert)
    auth_header = build_auth_header(password)

    target = normalize_path(PATH)
    if target.startswith("http://") or target.startswith("https://"):
        url = target
    else:
        url = es_url.rstrip("/") + target

    body_bytes = None
    request_headers = {
        "Authorization": auth_header,
        "Accept": "application/json",
    }

    if BODY is not None:
        if isinstance(BODY, (dict, list)):
            body_text = json.dumps(BODY)
        else:
            body_text = str(BODY)
        body_bytes = body_text.encode("utf-8")
        request_headers.setdefault("Content-Type", "application/json")

    req = urllib.request.Request(
        url,
        data=body_bytes,
        method=METHOD.upper(),
        headers=request_headers,
    )

    try:
        with urllib.request.urlopen(req, context=ssl_context) as response:
            payload = response.read()
            print_response(response.status, response.reason, response.headers, payload)
            return 0
    except urllib.error.HTTPError as exc:
        payload = exc.read()
        print_response(exc.code, exc.reason, exc.headers, payload)
        return exc.code


if __name__ == "__main__":
    raise SystemExit(main())
