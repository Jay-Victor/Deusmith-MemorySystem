# -*- coding: utf-8 -*-
"""证据对象存储 —— 本地目录内容寻址（sha256），DB 只存指针与哈希（评审第三梯队 #8）。

设计（§七 治理）：对象存储保留原文证据，数据库只存 evidence 表的指针行；
换 MinIO/S3 时实现同接口（save/read/verify），pointer_uri 换 s3:// 即可。
"""
from __future__ import annotations

import hashlib
from pathlib import Path


class LocalEvidenceStore:
    """本地目录内容寻址存储：ev_id = "ev_" + sha256 全文哈希。

    同内容天然去重（同哈希同文件）；verify 重算哈希检测篡改。
    """

    def __init__(self, root):
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)

    def save(self, content: str, source: str = "user") -> dict:
        data = (content or "").encode("utf-8")
        h = hashlib.sha256(data).hexdigest()
        ev_id = "ev_" + h
        rel = self.root / h[:2] / (h + ".txt")
        if not rel.exists():
            rel.parent.mkdir(parents=True, exist_ok=True)
            rel.write_bytes(data)
        return {
            "evidence_id": ev_id,
            "content_hash": h,
            "pointer_uri": rel.resolve().as_uri(),
            "source": source,
        }

    def _path_of(self, evidence_id: str) -> Path:
        h = evidence_id[3:] if evidence_id.startswith("ev_") else evidence_id
        if len(h) != 64 or any(c not in "0123456789abcdef" for c in h):
            raise ValueError(f"非法 evidence_id: {evidence_id!r}")
        return self.root / h[:2] / (h + ".txt")

    def read(self, evidence_id: str) -> str:
        p = self._path_of(evidence_id)
        if not p.exists():
            raise FileNotFoundError(f"证据不存在: {evidence_id}")
        return p.read_text(encoding="utf-8")

    def verify(self, evidence_id: str) -> bool:
        """重算哈希与文件名比对 —— 检测落盘后被篡改。"""
        try:
            p = self._path_of(evidence_id)
            if not p.exists():
                return False
            h = hashlib.sha256(p.read_bytes()).hexdigest()
            return ("ev_" + h) == evidence_id
        except (ValueError, OSError):
            return False
