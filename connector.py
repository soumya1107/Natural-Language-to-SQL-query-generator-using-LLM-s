"""
db/connector.py
───────────────
MySQL connection manager and safe query executor.

Features:
- Connection pooling via a context manager
- Read-only guard: blocks mutating statements
- Row cap to prevent runaway result sets
- Returns results as (columns, rows) tuples
"""

import re
import mysql.connector
from mysql.connector import Error as MySQLError
from contextlib import contextmanager
from typing import Generator

from config.settings import get_db_config, get_app_config
from utils.logger import get_logger

logger = get_logger(__name__)

# Statements that are blocked in read-only mode
_MUTATING_KEYWORDS = re.compile(
    r"^\s*(INSERT|UPDATE|DELETE|DROP|CREATE|ALTER|TRUNCATE|REPLACE|MERGE|CALL|EXEC)\b",
    re.IGNORECASE,
)


class ReadOnlyViolationError(Exception):
    """Raised when a mutating query is attempted in read-only mode."""


class DBConnector:
    """
    Thin wrapper around mysql.connector.
    Instantiate once and reuse; the underlying connection is lazy.
    """

    def __init__(self, host=None, port=None, user=None, password=None, database=None):
        cfg = get_db_config()
        self.host = host or cfg.host
        self.port = port or cfg.port
        self.user = user or cfg.user
        self.password = password or cfg.password
        self.database = database or cfg.database
        self._conn = None

    # ── Connection lifecycle ───────────────────────────────────────────────

    def connect(self):
        cfg = get_db_config()
        self._conn = mysql.connector.connect(
            host=self.host,
            port=self.port,
            user=self.user,
            password=self.password,
            database=self.database,
            connect_timeout=cfg.connect_timeout,
        )
        logger.info(f"Connected to MySQL: {self.host}/{self.database}")

    def disconnect(self):
        if self._conn and self._conn.is_connected():
            self._conn.close()
            logger.info("MySQL connection closed.")

    def test_connection(self) -> tuple[bool, str]:
        """Returns (success, message)."""
        try:
            self.connect()
            self.disconnect()
            return True, "Connection successful."
        except MySQLError as e:
            return False, str(e)

    @contextmanager
    def get_cursor(self) -> Generator:
        """Context manager that yields a cursor and auto-commits / rolls back."""
        if self._conn is None or not self._conn.is_connected():
            self.connect()
        cursor = self._conn.cursor()
        try:
            yield cursor
            self._conn.commit()
        except Exception:
            self._conn.rollback()
            raise
        finally:
            cursor.close()

    # ── Query execution ────────────────────────────────────────────────────

    def execute_query(
        self,
        sql: str,
        params: tuple = (),
        read_only: bool | None = None,
    ) -> tuple[list[str], list[tuple]]:
        """
        Execute *sql* and return (column_names, rows).

        Args:
            sql:       The SQL string to execute.
            params:    Optional parameterised values.
            read_only: Override the global READ_ONLY_GUARD setting.

        Returns:
            (columns, rows) — columns is a list of strings,
            rows is a list of tuples.

        Raises:
            ReadOnlyViolationError: If a mutating statement is detected and
                                    read-only mode is active.
        """
        app_cfg = get_app_config()
        effective_read_only = read_only if read_only is not None else app_cfg.read_only_guard

        if effective_read_only and _MUTATING_KEYWORDS.match(sql.strip()):
            raise ReadOnlyViolationError(
                f"Mutating SQL blocked (read-only mode is ON).\n"
                f"Statement starts with: {sql.strip()[:80]}"
            )

        logger.info(f"Executing SQL:\n{sql.strip()}")

        with self.get_cursor() as cursor:
            cursor.execute(sql, params)

            # SELECT-like queries return rows
            if cursor.description:
                columns = [desc[0] for desc in cursor.description]
                rows = cursor.fetchmany(app_cfg.max_result_rows)
                logger.info(f"Query returned {len(rows)} row(s).")
                return columns, rows
            else:
                # DDL / DML — return affected rows count
                return ["affected_rows"], [(cursor.rowcount,)]
