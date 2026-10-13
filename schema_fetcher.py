"""
db/schema_fetcher.py
─────────────────────
Fetches the full schema of a MySQL database:
- Table names
- Column names + types + nullability + keys
- Foreign key relationships
- Row count estimates

The result is both a Python dict (for programmatic use) and a
compact text representation (for injection into LLM prompts).
"""

from __future__ import annotations
from dataclasses import dataclass, field
from functools import lru_cache

from db.connector import DBConnector
from utils.logger import get_logger

logger = get_logger(__name__)


@dataclass
class ColumnInfo:
    name: str
    data_type: str
    nullable: bool
    key: str          # PRI, UNI, MUL, or ""
    default: str | None
    extra: str        # e.g. "auto_increment"

    def to_text(self) -> str:
        parts = [f"{self.name} {self.data_type}"]
        if self.key == "PRI":
            parts.append("PRIMARY KEY")
        elif self.key == "UNI":
            parts.append("UNIQUE")
        if not self.nullable:
            parts.append("NOT NULL")
        if self.extra:
            parts.append(self.extra.upper())
        return " ".join(parts)


@dataclass
class TableSchema:
    name: str
    columns: list[ColumnInfo] = field(default_factory=list)
    foreign_keys: list[dict] = field(default_factory=list)   # {column, ref_table, ref_column}
    row_count: int = 0

    def to_text(self) -> str:
        lines = [f"TABLE {self.name} (~{self.row_count} rows):"]
        for col in self.columns:
            lines.append(f"  {col.to_text()}")
        for fk in self.foreign_keys:
            lines.append(
                f"  FK: {fk['column']} → {fk['ref_table']}.{fk['ref_column']}"
            )
        return "\n".join(lines)


@dataclass
class DatabaseSchema:
    database: str
    tables: dict[str, TableSchema] = field(default_factory=dict)

    def to_text(self) -> str:
        """Full schema as a single string, suitable for LLM prompts."""
        parts = [f"DATABASE: {self.database}\n"]
        for table in self.tables.values():
            parts.append(table.to_text())
        return "\n\n".join(parts)

    def table_names(self) -> list[str]:
        return list(self.tables.keys())


class SchemaFetcher:
    """
    Fetches and caches the schema for a connected MySQL database.
    """

    def __init__(self, connector: DBConnector):
        self.connector = connector
        self._schema: DatabaseSchema | None = None

    def fetch(self, force_refresh: bool = False) -> DatabaseSchema:
        """Fetch schema (cached unless force_refresh=True)."""
        if self._schema is not None and not force_refresh:
            return self._schema

        db_name = self.connector.database
        logger.info(f"Fetching schema for database: {db_name}")

        schema = DatabaseSchema(database=db_name)

        # 1. Get all tables
        cols, rows = self.connector.execute_query(
            "SELECT TABLE_NAME, TABLE_ROWS "
            "FROM information_schema.TABLES "
            "WHERE TABLE_SCHEMA = %s AND TABLE_TYPE = 'BASE TABLE' "
            "ORDER BY TABLE_NAME",
            params=(db_name,),
        )
        tables_rows = rows

        for (table_name, row_count) in tables_rows:
            table = TableSchema(name=table_name, row_count=row_count or 0)

            # 2. Get columns for this table
            _, col_rows = self.connector.execute_query(
                "SELECT COLUMN_NAME, DATA_TYPE, IS_NULLABLE, COLUMN_KEY, COLUMN_DEFAULT, EXTRA "
                "FROM information_schema.COLUMNS "
                "WHERE TABLE_SCHEMA = %s AND TABLE_NAME = %s "
                "ORDER BY ORDINAL_POSITION",
                params=(db_name, table_name),
            )
            for (col_name, data_type, is_nullable, col_key, default, extra) in col_rows:
                table.columns.append(
                    ColumnInfo(
                        name=col_name,
                        data_type=data_type,
                        nullable=(is_nullable == "YES"),
                        key=col_key or "",
                        default=default,
                        extra=extra or "",
                    )
                )

            # 3. Get foreign keys
            _, fk_rows = self.connector.execute_query(
                "SELECT kcu.COLUMN_NAME, kcu.REFERENCED_TABLE_NAME, kcu.REFERENCED_COLUMN_NAME "
                "FROM information_schema.KEY_COLUMN_USAGE kcu "
                "JOIN information_schema.REFERENTIAL_CONSTRAINTS rc "
                "  ON kcu.CONSTRAINT_NAME = rc.CONSTRAINT_NAME "
                "  AND kcu.TABLE_SCHEMA = rc.CONSTRAINT_SCHEMA "
                "WHERE kcu.TABLE_SCHEMA = %s AND kcu.TABLE_NAME = %s "
                "  AND kcu.REFERENCED_TABLE_NAME IS NOT NULL",
                params=(db_name, table_name),
            )
            for (col_name, ref_table, ref_col) in fk_rows:
                table.foreign_keys.append(
                    {"column": col_name, "ref_table": ref_table, "ref_column": ref_col}
                )

            schema.tables[table_name] = table

        self._schema = schema
        logger.info(f"Schema fetched: {len(schema.tables)} tables.")
        return schema

    def invalidate(self):
        """Clear cached schema (call after schema changes)."""
        self._schema = None
