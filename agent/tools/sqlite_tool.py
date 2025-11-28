import sqlite3
from typing import Dict, Any
from agent.config import AgentConfig

class DatabaseInterface:
    """Handles database schema and query execution"""
    
    def __init__(self, config: AgentConfig):
        self.config = config
        self.schema_cache = None
        self._load_schema()
    
    def _load_schema(self):
        """Cache database schema"""
        print(f"🗄️  Loading database schema from {self.config.db_path}")
        
        if not self.config.db_path.exists():
            print(f"❌ Database not found: {self.config.db_path}")
            self.schema_cache = ""
            return
        
        try:
            conn = sqlite3.connect(self.config.db_path)
            cursor = conn.cursor()
            
            # Get all tables
            cursor.execute("SELECT name FROM sqlite_master WHERE type='table' ORDER BY name")
            tables = [row[0] for row in cursor.fetchall()]
            
            schema_parts = []
            for table in tables:
                cursor.execute(f"PRAGMA table_info({table})")
                columns = cursor.fetchall()
                
                col_defs = ', '.join([f"{col[1]} {col[2]}" for col in columns])
                schema_parts.append(f"Table: {table}\nColumns: {col_defs}\n")
            
            self.schema_cache = '\n'.join(schema_parts)
            conn.close()
            print(f"✅ Loaded schema for {len(tables)} tables")
            
        except Exception as e:
            print(f"❌ Schema loading error: {e}")
            self.schema_cache = ""
    
    def get_schema(self) -> str:
        """Get cached schema"""
        return self.schema_cache
    
    def execute(self, query: str) -> Dict[str, Any]:
        """Execute SQL query safely"""
        try:
            conn = sqlite3.connect(self.config.db_path)
            cursor = conn.cursor()
            
            # Execute query
            cursor.execute(query)
            results = cursor.fetchall()
            columns = [desc[0] for desc in cursor.description] if cursor.description else []
            
            conn.close()
            
            return {
                'success': True,
                'columns': columns,
                'rows': results,
                'error': None,
                'row_count': len(results)
            }
            
        except sqlite3.Error as e:
            return {
                'success': False,
                'columns': [],
                'rows': [],
                'error': str(e),
                'row_count': 0
            }
