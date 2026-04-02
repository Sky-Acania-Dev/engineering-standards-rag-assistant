from __future__ import annotations

from dataclasses import asdict, dataclass
import json
from pathlib import Path


@dataclass(frozen=True)
class StoredChunk:
    chunk_uid: str
    text: str
    doc_id: str
    title: str
    section: str
    chunk_id: int
    page: int | None = None


class JsonlChunkStore:
    """Simple append-safe chunk store persisted as JSONL."""

    def __init__(self) -> None:
        self._records: dict[str, StoredChunk] = {}

    def upsert_many(self, records: list[StoredChunk]) -> None:
        for record in records:
            self._records[record.chunk_uid] = record

    def get(self, chunk_uid: str) -> StoredChunk | None:
        return self._records.get(chunk_uid)

    def get_many(self, chunk_uids: list[str]) -> list[StoredChunk]:
        return [record for chunk_uid in chunk_uids if (record := self._records.get(chunk_uid)) is not None]

    def __len__(self) -> int:
        return len(self._records)

    def save(self, output_path: str | Path) -> None:
        path = Path(output_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", encoding="utf-8") as file:
            for chunk_uid in sorted(self._records.keys()):
                file.write(json.dumps(asdict(self._records[chunk_uid]), ensure_ascii=False) + "\n")

    @classmethod
    def load(cls, input_path: str | Path) -> JsonlChunkStore:
        path = Path(input_path)
        store = cls()
        if not path.exists():
            return store

        with path.open("r", encoding="utf-8") as file:
            for line in file:
                if not line.strip():
                    continue
                payload = json.loads(line)
                store.upsert_many([StoredChunk(**payload)])

        return store
