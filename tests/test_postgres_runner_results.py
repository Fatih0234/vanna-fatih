import pytest
from unittest.mock import MagicMock

from vanna.integrations.postgres import PostgresRunner
from vanna.capabilities.sql_runner import RunSqlToolArgs


@pytest.mark.asyncio
async def test_with_select_returns_rows():
    runner = PostgresRunner(connection_string="postgresql://user:pass@localhost/db")

    fake_cursor = MagicMock()
    fake_cursor.description = ("a",)
    fake_cursor.statusmessage = "SELECT 2"
    fake_cursor.fetchall.return_value = [{"a": 1}, {"a": 2}]

    fake_conn = MagicMock()
    fake_conn.cursor.return_value = fake_cursor

    runner.psycopg2.connect = MagicMock(return_value=fake_conn)

    df = await runner.run_sql(
        RunSqlToolArgs(sql="WITH x AS (SELECT 1) SELECT 1 AS a UNION ALL SELECT 2"),
        context=None,
    )

    assert df.to_dict(orient="records") == [{"a": 1}, {"a": 2}]
    fake_conn.commit.assert_not_called()


@pytest.mark.asyncio
async def test_insert_returning_commits_and_returns_rows():
    runner = PostgresRunner(connection_string="postgresql://user:pass@localhost/db")

    fake_cursor = MagicMock()
    fake_cursor.description = ("id",)
    fake_cursor.statusmessage = "INSERT 0 1"
    fake_cursor.fetchall.return_value = [{"id": 123}]

    fake_conn = MagicMock()
    fake_conn.cursor.return_value = fake_cursor

    runner.psycopg2.connect = MagicMock(return_value=fake_conn)

    df = await runner.run_sql(
        RunSqlToolArgs(sql="INSERT INTO t(id) VALUES (123) RETURNING id"),
        context=None,
    )

    assert df.to_dict(orient="records") == [{"id": 123}]
    fake_conn.commit.assert_called_once()


@pytest.mark.asyncio
async def test_non_returning_statement_returns_rows_affected():
    runner = PostgresRunner(connection_string="postgresql://user:pass@localhost/db")

    fake_cursor = MagicMock()
    fake_cursor.description = None
    fake_cursor.rowcount = 7
    fake_cursor.statusmessage = "UPDATE 7"

    fake_conn = MagicMock()
    fake_conn.cursor.return_value = fake_cursor

    runner.psycopg2.connect = MagicMock(return_value=fake_conn)

    df = await runner.run_sql(
        RunSqlToolArgs(sql="UPDATE t SET x = 1"),
        context=None,
    )

    assert df.to_dict(orient="records") == [{"rows_affected": 7}]
    fake_conn.commit.assert_called_once()

