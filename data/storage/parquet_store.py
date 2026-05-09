"""Partitioned parquet store for bars/ticks/books.

Layout:
    {root}/{kind}/{symbol}/{yyyy}/{mm}/{yyyy-mm-dd}.parquet
"""
from __future__ import annotations

from pathlib import Path
from typing import Iterable, Literal

import pandas as pd

Kind = Literal["bars", "ticks", "book", "features"]


class ParquetStore:
    def __init__(self, root: Path | str) -> None:
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)

    def _path(self, kind: Kind, symbol: str, day: pd.Timestamp) -> Path:
        return (
            self.root / kind / symbol / f"{day.year:04d}" / f"{day.month:02d}"
            / f"{day.strftime('%Y-%m-%d')}.parquet"
        )

    def write(self, kind: Kind, symbol: str, df: pd.DataFrame) -> list[Path]:
        if df.empty:
            return []
        if not isinstance(df.index, pd.DatetimeIndex):
            raise ValueError("DataFrame must have DatetimeIndex")
        df = df.tz_convert("UTC") if df.index.tz else df.tz_localize("UTC")
        written: list[Path] = []
        for day, chunk in df.groupby(df.index.normalize()):
            p = self._path(kind, symbol, pd.Timestamp(day))
            p.parent.mkdir(parents=True, exist_ok=True)
            chunk.to_parquet(p, compression="zstd")
            written.append(p)
        return written

    def read(
        self,
        kind: Kind,
        symbol: str,
        start: pd.Timestamp,
        end: pd.Timestamp,
        columns: Iterable[str] | None = None,
    ) -> pd.DataFrame:
        days = pd.date_range(start.normalize(), end.normalize(), freq="D")
        frames: list[pd.DataFrame] = []
        for d in days:
            p = self._path(kind, symbol, d)
            if p.exists():
                frames.append(pd.read_parquet(p, columns=list(columns) if columns else None))
        if not frames:
            return pd.DataFrame()
        out = pd.concat(frames).sort_index()
        return out[(out.index >= start) & (out.index <= end)]
