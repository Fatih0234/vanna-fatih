import pandas as pd
import pytest

from vanna.capabilities.sql_runner import RunSqlToolArgs, SqlRunner
from vanna.core.tool import ToolContext
from vanna.core.user.models import User
from vanna.integrations.local.agent_memory import DemoAgentMemory
from vanna.integrations.local.file_system import LocalFileSystem
from vanna.tools import RunSqlTool


class FakeSqlRunner(SqlRunner):
    async def run_sql(self, args: RunSqlToolArgs, context: ToolContext) -> pd.DataFrame:
        return pd.DataFrame([{"district": "A", "cnt": 1}, {"district": "B", "cnt": 2}])


@pytest.mark.asyncio
async def test_run_sql_tool_with_cte_saves_csv_and_returns_table(tmp_path):
    tool = RunSqlTool(
        sql_runner=FakeSqlRunner(),
        file_system=LocalFileSystem(working_directory=str(tmp_path)),
    )

    context = ToolContext(
        user=User(id="u1"),
        conversation_id="c1",
        request_id="r1",
        agent_memory=DemoAgentMemory(),
    )

    result = await tool.execute(
        context,
        RunSqlToolArgs(sql="WITH x AS (SELECT 1) SELECT * FROM x"),
    )

    assert result.success is True
    assert result.metadata["row_count"] == 2
    assert "output_file" in result.metadata

    output_file = result.metadata["output_file"]
    assert await tool.file_system.exists(output_file, context) is True

