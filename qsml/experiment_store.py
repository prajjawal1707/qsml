
"""SQLite-backed storage for QSML optimization experiments."""

import json
import sqlite3
import time
from pathlib import Path
from contextlib import contextmanager


class ExperimentStore:

    def __init__(self, database_path="qsml_experiments.sqlite3"):
        self.database_path = str(database_path)

        Path(self.database_path).parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        self._initialize()

    @contextmanager
    def _connect(self):
        """Open, transact and always close a database connection."""

        connection = sqlite3.connect(
            self.database_path,
            timeout=30,
        )

        connection.row_factory = sqlite3.Row

        try:
            yield connection
            connection.commit()

        except Exception:
            connection.rollback()
            raise

        finally:
            connection.close()

    def _initialize(self):

        with self._connect() as connection:

            connection.execute("""
                CREATE TABLE IF NOT EXISTS experiments (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    name TEXT NOT NULL,
                    created_at REAL NOT NULL,
                    config_json TEXT NOT NULL,
                    result_json TEXT NOT NULL,
                    tags_json TEXT NOT NULL
                )
            """)

            connection.execute("""
                CREATE INDEX IF NOT EXISTS idx_experiments_name
                ON experiments(name)
            """)

    @staticmethod
    def _json_default(value):

        if hasattr(value, "tolist"):
            return value.tolist()

        if hasattr(value, "item"):
            return value.item()

        raise TypeError(
            f"Object of type {type(value).__name__} "
            "is not JSON serializable"
        )

    def save(self, name, config, result, tags=None):

        if not isinstance(name, str) or not name.strip():
            raise ValueError(
                "Experiment name must be a non-empty string"
            )

        if not isinstance(config, dict):
            raise ValueError("config must be a dictionary")

        if not isinstance(result, dict):
            raise ValueError("result must be a dictionary")

        if tags is None:
            tags = []

        if not isinstance(tags, (list, tuple)) or not all(
            isinstance(tag, str) for tag in tags
        ):
            raise ValueError(
                "tags must be a list or tuple of strings"
            )

        config_json = json.dumps(
            config,
            default=self._json_default,
            allow_nan=False,
        )

        result_json = json.dumps(
            result,
            default=self._json_default,
            allow_nan=False,
        )

        tags_json = json.dumps(
            list(tags),
            allow_nan=False,
        )

        with self._connect() as connection:

            cursor = connection.execute(
                """
                INSERT INTO experiments
                (
                    name,
                    created_at,
                    config_json,
                    result_json,
                    tags_json
                )
                VALUES (?, ?, ?, ?, ?)
                """,
                (
                    name.strip(),
                    time.time(),
                    config_json,
                    result_json,
                    tags_json,
                ),
            )

            experiment_id = cursor.lastrowid

        return int(experiment_id)

    def get(self, experiment_id):

        with self._connect() as connection:

            row = connection.execute(
                "SELECT * FROM experiments WHERE id = ?",
                (int(experiment_id),),
            ).fetchone()

        if row is None:
            return None

        return self._decode(row)

    def list(self, name=None, limit=100):

        if (
            not isinstance(limit, int)
            or isinstance(limit, bool)
            or limit < 1
        ):
            raise ValueError(
                "limit must be a positive integer"
            )

        with self._connect() as connection:

            if name is None:

                rows = connection.execute(
                    """
                    SELECT * FROM experiments
                    ORDER BY id DESC
                    LIMIT ?
                    """,
                    (limit,),
                ).fetchall()

            else:

                rows = connection.execute(
                    """
                    SELECT * FROM experiments
                    WHERE name = ?
                    ORDER BY id DESC
                    LIMIT ?
                    """,
                    (name, limit),
                ).fetchall()

        return [
            self._decode(row)
            for row in rows
        ]

    @staticmethod
    def _decode(row):

        return {
            "id": row["id"],
            "name": row["name"],
            "created_at": row["created_at"],
            "config": json.loads(row["config_json"]),
            "result": json.loads(row["result_json"]),
            "tags": json.loads(row["tags_json"]),
        }

    def delete(self, experiment_id):

        with self._connect() as connection:

            cursor = connection.execute(
                "DELETE FROM experiments WHERE id = ?",
                (int(experiment_id),),
            )

            deleted = cursor.rowcount == 1

        return deleted

    def compare(self, experiment_ids, metric="energy"):

        if (
            not isinstance(experiment_ids, (list, tuple))
            or not experiment_ids
        ):
            raise ValueError(
                "experiment_ids must be a non-empty list"
            )

        records = []

        for experiment_id in experiment_ids:

            record = self.get(experiment_id)

            if record is None:
                raise ValueError(
                    f"Experiment {experiment_id} was not found"
                )

            if metric not in record["result"]:
                raise ValueError(
                    f"Experiment {experiment_id} "
                    f"has no metric {metric!r}"
                )

            value = record["result"][metric]

            if (
                isinstance(value, bool)
                or not isinstance(value, (int, float))
            ):
                raise ValueError(
                    f"Metric {metric!r} must be numeric"
                )

            records.append({
                "id": record["id"],
                "name": record["name"],
                "metric": metric,
                "value": float(value),
            })

        return records
