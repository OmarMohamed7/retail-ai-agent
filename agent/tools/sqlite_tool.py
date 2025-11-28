import sqlite3
from typing import Dict, Any
from agent.config import AgentConfig
import re
def quote_identifier(name: str) -> str:
    # Escape internal quotes
    return f'"{name.replace("\"", "\"\"")}"'

class DatabaseInterface:
    def __init__(self, config: AgentConfig):
        self.config = config
        self.schema_cache = None
        self.tables = []
        self.columns = {}
        self._load_schema()

    def _load_schema(self):
        print(f"🗄️  Loading schema from {self.config.db_path}")

        if not self.config.db_path.exists():
            print("❌ Database not found")
            self.schema_cache = ""
            return

        try:
            conn = sqlite3.connect(self.config.db_path)
            cursor = conn.cursor()

            # Read all tables as exact names
            cursor.execute("SELECT name FROM sqlite_master WHERE type='table' ORDER BY name")
            tables = [row[0] for row in cursor.fetchall()]

            self.tables = tables

               

            schema_parts = []

            for table in tables:
                quoted = quote_identifier(table)
                cursor.execute(f"PRAGMA table_info({quoted})")
                columns = cursor.fetchall()

                self.columns[table] = [col[1] for col in columns]

                col_defs = [
                    f"{col[1]} {col[2]}"
                    for col in columns
                ]

                schema_parts.append(
                    f"Table: {table} (access as {quoted})\nColumns: {', '.join(col_defs)}\n"
                )

            self.schema_cache = "\n".join(schema_parts)
            conn.close()

            print(f"✅ Loaded schema for {len(tables)} tables")

        except Exception as e:
            print(f"❌ Schema loading error: {e}")
            self.schema_cache = ""  
