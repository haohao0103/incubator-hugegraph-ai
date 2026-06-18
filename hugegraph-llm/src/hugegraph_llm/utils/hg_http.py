# Licensed to the Apache Software Foundation (ASF) under one
# or more contributor license agreements.  See the NOTICE file
# distributed with this work for additional information
# regarding copyright ownership.  The ASF licenses this file
# to you under the Apache License, Version 2.0 (the
# "License"); you may not use this file except in compliance
# with the License.  You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing,
# software distributed under the License is distributed on an
# "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
# KIND, either express or implied.  See the License for the
# specific language governing permissions and limitations
# under the License.

"""Gzip-safe HTTP utility for HugeGraph REST API.

HugeGraph Server 1.5+ automatically gzip-compresses responses when the payload
is large enough. The Python `requests` library handles this transparently, but
`urllib.request` does NOT — it returns raw gzip bytes that `json.loads()` cannot
parse. This module provides a single entry point that works with both libraries
and always returns decoded JSON.

Key bug: HugeGraph Server IGNORES the `Accept-Encoding: identity` header and
still sends gzip. So we must always handle gzip decompression.

Usage:
    from hugegraph_llm.utils.hg_http import hg_get, hg_post

    data = hg_get("http://localhost:8080/graphspaces/DEFAULT/graphs/mygraph/graph/vertices?limit=100")
    result = hg_post("http://localhost:8080/graphspaces/DEFAULT/graphs/mygraph/graph/vertices", body=vertex_data)
"""

import json
import logging
import zlib
from typing import Any, Dict, Optional, Tuple

logger = logging.getLogger(__name__)

# HugeGraph Server default auth (None = no auth, useful for Bearer-token APIs)
_DEFAULT_AUTH: Optional[Tuple[str, str]] = ("admin", "admin")
_DEFAULT_TIMEOUT = 15


def _decompress_if_gzip(raw: bytes) -> bytes:
    """Decompress gzip bytes if the content is gzip-encoded.

    Checks for the gzip magic bytes (0x1f 0x8b) at the start of the data.
    If found, decompresses using zlib with gzip wrapper support.
    Otherwise returns the data unchanged.
    """
    if len(raw) >= 2 and raw[0] == 0x1F and raw[1] == 0x8B:
        return zlib.decompress(raw, 16 + zlib.MAX_WBITS)
    return raw


def hg_get(
    url: str,
    headers: Optional[Dict[str, str]] = None,
    auth: Optional[Tuple[str, str]] = _DEFAULT_AUTH,
    timeout: int = _DEFAULT_TIMEOUT,
) -> Dict[str, Any]:
    """Send a GET request to HugeGraph REST API with gzip-safe response handling.

    Uses the `requests` library if available (preferred, handles gzip transparently),
    otherwise falls back to `urllib.request` with explicit gzip decompression.

    Args:
        url: Full URL to the REST endpoint.
        headers: Optional dict of additional HTTP headers.
        auth: Basic auth tuple (username, password). Pass None to skip auth
              (e.g. for Bearer-token APIs like OpenAI-compatible endpoints).
        timeout: Request timeout in seconds.

    Returns:
        Parsed JSON response as a dict. Returns {"error": ...} on failure.
    """
    merged_headers = {"Content-Type": "application/json"}
    if headers:
        merged_headers.update(headers)

    # Try requests first (auto-decompresses gzip)
    try:
        import requests

        resp = requests.get(url, auth=auth, headers=merged_headers, timeout=timeout)
        resp.raise_for_status()
        return resp.json()
    except ImportError:
        pass  # Fall back to urllib
    except Exception as e:
        logger.warning("hg_get requests failed for %s: %s, falling back to urllib", url, e)

    # Fallback: urllib with explicit gzip decompression
    import urllib.request

    req = urllib.request.Request(url, headers=merged_headers)
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            raw = resp.read()
            decompressed = _decompress_if_gzip(raw)
            return json.loads(decompressed.decode("utf-8"))
    except Exception as e:
        logger.error("hg_get urllib failed for %s: %s", url, e)
        return {"error": str(e)}


def hg_post(
    url: str,
    body: Optional[Dict[str, Any]] = None,
    headers: Optional[Dict[str, str]] = None,
    auth: Optional[Tuple[str, str]] = _DEFAULT_AUTH,
    timeout: int = _DEFAULT_TIMEOUT,
) -> Dict[str, Any]:
    """Send a POST request to REST API with gzip-safe response handling.

    Args:
        url: Full URL to the REST endpoint.
        body: JSON-serializable dict to send as the request body.
        headers: Optional dict of additional HTTP headers.
        auth: Basic auth tuple (username, password). Pass None to skip auth.
        timeout: Request timeout in seconds.

    Returns:
        Parsed JSON response as a dict. Returns {"error": ...} on failure.
    """
    merged_headers = {"Content-Type": "application/json"}
    if headers:
        merged_headers.update(headers)

    data = json.dumps(body).encode("utf-8") if body else None

    # Try requests first
    try:
        import requests

        resp = requests.post(url, auth=auth, headers=merged_headers, data=data, timeout=timeout)
        resp.raise_for_status()
        if resp.status_code == 204:
            return {}
        return resp.json()
    except ImportError:
        pass
    except Exception as e:
        logger.warning("hg_post requests failed for %s: %s, falling back to urllib", url, e)

    # Fallback: urllib with explicit gzip decompression
    import urllib.request

    req = urllib.request.Request(url, data=data, headers=merged_headers, method="POST")
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            raw = resp.read()
            decompressed = _decompress_if_gzip(raw)
            text = decompressed.decode("utf-8")
            if not text:
                return {}
            return json.loads(text)
    except Exception as e:
        logger.error("hg_post urllib failed for %s: %s", url, e)
        return {"error": str(e)}


def hg_put(
    url: str,
    body: Optional[Dict[str, Any]] = None,
    headers: Optional[Dict[str, str]] = None,
    auth: Optional[Tuple[str, str]] = _DEFAULT_AUTH,
    timeout: int = _DEFAULT_TIMEOUT,
) -> Dict[str, Any]:
    """Send a PUT request to REST API with gzip-safe response handling.

    Args are the same as hg_post.
    """
    merged_headers = {"Content-Type": "application/json"}
    if headers:
        merged_headers.update(headers)

    data = json.dumps(body).encode("utf-8") if body else None

    # Try requests first
    try:
        import requests

        resp = requests.put(url, auth=auth, headers=merged_headers, data=data, timeout=timeout)
        resp.raise_for_status()
        if resp.status_code == 204:
            return {}
        return resp.json()
    except ImportError:
        pass
    except Exception as e:
        logger.warning("hg_put requests failed for %s: %s, falling back to urllib", url, e)

    # Fallback: urllib with explicit gzip decompression
    import urllib.request

    req = urllib.request.Request(url, data=data, headers=merged_headers, method="PUT")
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            raw = resp.read()
            decompressed = _decompress_if_gzip(raw)
            text = decompressed.decode("utf-8")
            if not text:
                return {}
            return json.loads(text)
    except Exception as e:
        logger.error("hg_put urllib failed for %s: %s", url, e)
        return {"error": str(e)}


def hg_delete(
    url: str,
    headers: Optional[Dict[str, str]] = None,
    auth: Optional[Tuple[str, str]] = _DEFAULT_AUTH,
    timeout: int = _DEFAULT_TIMEOUT,
) -> Dict[str, Any]:
    """Send a DELETE request to REST API with gzip-safe response handling.

    Args are the same as hg_get (no body).
    """
    merged_headers = {"Content-Type": "application/json"}
    if headers:
        merged_headers.update(headers)

    # Try requests first
    try:
        import requests

        resp = requests.delete(url, auth=auth, headers=merged_headers, timeout=timeout)
        resp.raise_for_status()
        if resp.status_code == 204:
            return {}
        return resp.json()
    except ImportError:
        pass
    except Exception as e:
        logger.warning("hg_delete requests failed for %s: %s, falling back to urllib", url, e)

    # Fallback: urllib with explicit gzip decompression
    import urllib.request

    req = urllib.request.Request(url, headers=merged_headers, method="DELETE")
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            raw = resp.read()
            decompressed = _decompress_if_gzip(raw)
            text = decompressed.decode("utf-8")
            if not text:
                return {}
            return json.loads(text)
    except Exception as e:
        logger.error("hg_delete urllib failed for %s: %s", url, e)
        return {"error": str(e)}
