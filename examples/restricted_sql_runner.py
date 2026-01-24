"""
Restricted PostgreSQL SQL Runner for Vanna AI.

This module provides a PostgreSQL runner that restricts queries to only
the public.v_bike_events view for security and simplicity.
"""

import re
import pandas as pd
from typing import Optional
from vanna.integrations.postgres import PostgresRunner
from vanna.capabilities.sql_runner import RunSqlToolArgs
from vanna.core.tool import ToolContext


class RestrictedPostgresRunner(PostgresRunner):
    """
    PostgreSQL runner that only allows queries on specific tables.

    This is useful when you want to limit Vanna AI to only query certain
    tables in your database for security or organizational reasons.
    """

    def __init__(
        self,
        allowed_tables: list[str] = None,
        connection_string: Optional[str] = None,
        host: Optional[str] = None,
        port: Optional[int] = 5432,
        database: Optional[str] = None,
        user: Optional[str] = None,
        password: Optional[str] = None,
        **kwargs,
    ):
        """
        Initialize the restricted PostgreSQL runner.

        Args:
            allowed_tables: List of allowed tables in format "schema.table" or just "table"
                          If just "table" is provided, assumes "public.table"
                          Example: ["public.v_bike_events", "public.users"]
            connection_string: PostgreSQL connection string
            host: Database host address
            port: Database port (default: 5432)
            database: Database name
            user: Database user
            password: Database password
            **kwargs: Additional psycopg2 connection parameters
        """
        super().__init__(
            connection_string=connection_string,
            host=host,
            port=port,
            database=database,
            user=user,
            password=password,
            **kwargs,
        )

        # Set default to public.v_bike_events if no tables specified
        if allowed_tables is None:
            allowed_tables = ["public.v_bike_events"]

        # Normalize table names (ensure schema.table format)
        self.allowed_tables = []
        for table in allowed_tables:
            if "." not in table:
                # If no schema specified, assume public
                self.allowed_tables.append(f"public.{table}")
            else:
                self.allowed_tables.append(table)

        print(f"✓ SQL queries restricted to tables: {', '.join(self.allowed_tables)}")

    def _validate_sql(self, sql: str) -> tuple[bool, str]:
        """
        Validate that SQL query only accesses allowed tables.

        Args:
            sql: SQL query to validate

        Returns:
            Tuple of (is_valid, error_message)
        """
        # Convert SQL to lowercase for case-insensitive matching
        sql_lower = sql.lower()

        # Remove comments and strings to avoid false positives
        # Simple approach: remove single-line comments
        sql_cleaned = re.sub(r"--[^\n]*", "", sql_lower)

        # Extract table references from FROM and JOIN clauses
        # This is a simplified pattern - production code might need more robust parsing
        from_pattern = r"\bfrom\s+([a-z0-9_\.]+)"
        join_pattern = r"\bjoin\s+([a-z0-9_\.]+)"

        referenced_tables = set()

        # Find all FROM clauses
        for match in re.finditer(from_pattern, sql_cleaned):
            table = match.group(1)
            # Normalize: if no schema, assume public
            if "." not in table:
                table = f"public.{table}"
            referenced_tables.add(table)

        # Find all JOIN clauses
        for match in re.finditer(join_pattern, sql_cleaned):
            table = match.group(1)
            # Normalize: if no schema, assume public
            if "." not in table:
                table = f"public.{table}"
            referenced_tables.add(table)

        # Check if all referenced tables are allowed
        for table in referenced_tables:
            if table not in self.allowed_tables:
                return False, (
                    f"Access denied: Table '{table}' is not in the allowed list. "
                    f"Only these tables are accessible: {', '.join(self.allowed_tables)}"
                )

        # Additional security: reject queries that don't reference any tables
        # (could be information_schema queries, etc.)
        if not referenced_tables and ("from" in sql_cleaned or "join" in sql_cleaned):
            # Check if it's a simple utility query (like SELECT version())
            if "information_schema" in sql_cleaned or "pg_" in sql_cleaned:
                return False, (
                    "Access denied: System table queries are not allowed. "
                    f"Only these tables are accessible: {', '.join(self.allowed_tables)}"
                )

        return True, ""

    async def run_sql(self, args: RunSqlToolArgs, context: ToolContext) -> pd.DataFrame:
        """
        Execute SQL query with table access validation.

        Args:
            args: SQL query arguments
            context: Tool execution context

        Returns:
            DataFrame with query results

        Raises:
            ValueError: If query attempts to access non-allowed tables
            psycopg2.Error: If query execution fails
        """
        # Validate SQL before execution
        is_valid, error_message = self._validate_sql(args.sql)

        if not is_valid:
            raise ValueError(error_message)

        # If validation passes, execute the query
        return await super().run_sql(args, context)

    def get_table_info(self) -> str:
        """
        Get information about allowed tables for the LLM context.

        Returns:
            String describing the allowed tables
        """
        return (
            f"You have access to the following tables: {', '.join(self.allowed_tables)}"
        )


async def get_bike_events_view_schema(runner: PostgresRunner) -> str:
    """
    Retrieve the schema of the public.v_bike_events view.

    Args:
        runner: PostgreSQL runner instance

    Returns:
        String describing the v_bike_events view schema
    """
    from vanna.capabilities.sql_runner import RunSqlToolArgs
    from vanna.core.tool import ToolContext
    from vanna import User
    from vanna.integrations.local.agent_memory import DemoAgentMemory
    import uuid

    schema_query = """
    SELECT 
        column_name,
        data_type,
        character_maximum_length,
        is_nullable,
        column_default
    FROM information_schema.columns
    WHERE table_schema = 'public' 
      AND table_name = 'v_bike_events'
    ORDER BY ordinal_position;
    """

    try:
        # Create temporary unrestricted runner to fetch schema
        temp_runner = PostgresRunner(
            connection_string=runner.connection_string
            if hasattr(runner, "connection_string") and runner.connection_string
            else None,
            host=runner.connection_params["host"]
            if hasattr(runner, "connection_params") and runner.connection_params
            else None,
            port=runner.connection_params["port"]
            if hasattr(runner, "connection_params") and runner.connection_params
            else 5432,
            database=runner.connection_params["database"]
            if hasattr(runner, "connection_params") and runner.connection_params
            else None,
            user=runner.connection_params["user"]
            if hasattr(runner, "connection_params") and runner.connection_params
            else None,
            password=runner.connection_params["password"]
            if hasattr(runner, "connection_params") and runner.connection_params
            else None,
        )

        test_user = User(id="schema-fetch", username="schema-fetch")
        context = ToolContext(
            user=test_user,
            conversation_id="schema-fetch",
            request_id=str(uuid.uuid4()),
            agent_memory=DemoAgentMemory(),
        )
        args = RunSqlToolArgs(sql=schema_query)

        result = await temp_runner.run_sql(args, context)

        if result.empty:
            return "View 'public.v_bike_events' not found or has no columns."

        schema_desc = "public.v_bike_events view schema:\n"
        for _, row in result.iterrows():
            schema_desc += f"  - {row['column_name']}: {row['data_type']}"
            if pd.notna(row["character_maximum_length"]):
                schema_desc += f"({row['character_maximum_length']})"
            if row["is_nullable"] == "NO":
                schema_desc += " NOT NULL"
            if pd.notna(row["column_default"]):
                schema_desc += f" DEFAULT {row['column_default']}"
            schema_desc += "\n"

        return schema_desc

    except Exception as e:
        return f"Could not fetch v_bike_events view schema: {e}"
