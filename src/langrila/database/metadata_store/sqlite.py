import contextlib
import copy
import sqlite3
from pathlib import Path
from typing import Any

import numpy as np

from ...core.metadata import BaseMetadataStore


@contextlib.contextmanager
def sqlite_open(path: Path):
    conn = sqlite3.connect(path)
    cursor = conn.cursor()
    try:
        yield cursor
        conn.commit()
    finally:
        conn.rollback()
        conn.close()


class SQLiteMetadataStore(BaseMetadataStore):
    def __init__(self, persistence_directory: str, collection_name: str):
        self.persistence_directory = persistence_directory
        self.collection_name = collection_name
        self._path = Path(persistence_directory) / collection_name / "metadata.sqlite3"
        self._path.parent.mkdir(parents=True, exist_ok=True)
        self.p_key = "__id"

    def store(self, ids: list[int], metadatas: list[dict[str, str]]) -> None:
        _metadatas = copy.deepcopy(metadatas)
        all_keys = list(set([k for metadata in _metadatas for k in metadata.keys()]))

        # check if all keys are the same
        for _id, metadata in zip(ids, _metadatas, strict=True):
            for key in all_keys:
                if key not in metadata:
                    metadata[key] = None

            metadata[self.p_key] = _id

        # insert primary key
        all_keys = [self.p_key] + all_keys

        # create table
        db_columns = ",".join([key for key in all_keys])
        db_columns_with_property = ",".join(
            [key + " PRIMARY KEY" if key == self.p_key else key for key in all_keys]
        )

        with sqlite_open(self._path) as cursor:
            cursor.execute(f"CREATE TABLE IF NOT EXISTS metadata({db_columns_with_property})")
            cursor.executemany(
                f"INSERT OR REPLACE INTO metadata({db_columns}) VALUES ({','.join(['?']*len(all_keys))})",
                [tuple(metadata[key] for key in all_keys) for metadata in _metadatas],
            )

    def fetch(self, ids: list[int]) -> list[dict[str, Any]]:
        with sqlite_open(self._path) as cursor:
            cursor.execute(
                f"SELECT * FROM metadata WHERE {self.p_key} IN ({','.join(['?']*len(ids))})",
                tuple(ids),
            )
            _metadatas = cursor.fetchall()
            descriptions = cursor.description
            columns = [description[0] for description in descriptions]

        p_key_index = columns.index(self.p_key)

        _metadatas = [
            {
                columns[i]: metadata[i]
                for i in range(len(metadata))
                if i != p_key_index and metadata[i] is not None
            }
            for metadata in _metadatas
        ]

        _metadatas = [_metadatas[i] for i in np.argsort(np.argsort(ids))]
        return _metadatas

    def delete_records(self, ids: list[int]) -> None:
        with sqlite_open(self._path) as cursor:
            cursor.execute(
                f"DELETE * FROM metadata WHERE {self.p_key} IN ({','.join(['?']*len(ids))})",
                tuple(ids),
            )

    def delete_table(self) -> None:
        with sqlite_open(self._path) as cursor:
            cursor.execute("DROP TABLE IF EXISTS metadata")
