# -*- coding: utf-8 -*-
"""在线 API 共用：带重试/指数退避的 JSON POST 与成本计数（评审第二梯队 #6）。"""
from __future__ import annotations

import json
import time
import urllib.error
import urllib.request


def post_json_with_retry(url, payload, headers, timeout, attempts=3, backoff=0.5,
                         stats=None):
    """POST JSON；5xx/429/网络错误按指数退避重试，其余（4xx 等）立即抛出。

    stats dict 计数（调用方持有）：requests / retries / errors。
    重试语义：attempts=3 → 最多 3 次请求、2 次重试；退避 0.5s→1s→2s。
    """
    n = max(1, int(attempts))
    last_err = None
    delay = float(backoff)
    for attempt in range(n):
        if stats is not None:
            stats["requests"] = stats.get("requests", 0) + 1
        try:
            req = urllib.request.Request(
                url, data=json.dumps(payload).encode("utf-8"),
                headers=headers, method="POST")
            with urllib.request.urlopen(req, timeout=timeout) as resp:
                return json.loads(resp.read().decode("utf-8"))
        except urllib.error.HTTPError as e:
            last_err = e
            retryable = e.code == 429 or 500 <= e.code < 600
            if not retryable or attempt == n - 1:
                if stats is not None:
                    stats["errors"] = stats.get("errors", 0) + 1
                raise
        except (urllib.error.URLError, TimeoutError, ConnectionError, OSError) as e:
            last_err = e
            if attempt == n - 1:
                if stats is not None:
                    stats["errors"] = stats.get("errors", 0) + 1
                raise
        if stats is not None:
            stats["retries"] = stats.get("retries", 0) + 1
        time.sleep(delay)
        delay *= 2
    raise last_err  # pragma: no cover（循环内必 raise 或 return）
