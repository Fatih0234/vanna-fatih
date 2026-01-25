"""Tool for visualizing DataFrame data from CSV files."""

from typing import Optional, Type
import logging
import json
import uuid
import pandas as pd
from pydantic import BaseModel, Field

from vanna.core.tool import Tool, ToolContext, ToolResult
from vanna.components import (
    UiComponent,
    ChartComponent,
    NotificationComponent,
    ComponentType,
    SimpleTextComponent,
)

from .file_system import FileSystem, LocalFileSystem
from vanna.integrations.plotly import PlotlyChartGenerator

logger = logging.getLogger(__name__)


class VisualizeDataArgs(BaseModel):
    """Arguments for visualize_data tool."""

    filename: str = Field(description="Name of the CSV file to visualize")
    title: Optional[str] = Field(
        default=None, description="Optional title for the chart"
    )


class VisualizeDataTool(Tool[VisualizeDataArgs]):
    """Tool that reads CSV files and generates visualizations using dependency injection."""

    def __init__(
        self,
        file_system: Optional[FileSystem] = None,
        plotly_generator: Optional[PlotlyChartGenerator] = None,
    ):
        """Initialize the tool with FileSystem and PlotlyChartGenerator.

        Args:
            file_system: FileSystem implementation for reading CSV files (defaults to LocalFileSystem)
            plotly_generator: PlotlyChartGenerator for creating Plotly charts (defaults to PlotlyChartGenerator())
        """
        self.file_system = file_system or LocalFileSystem()
        self.plotly_generator = plotly_generator or PlotlyChartGenerator()

    @property
    def name(self) -> str:
        return "visualize_data"

    @property
    def description(self) -> str:
        return "Create a visualization from a CSV file. The tool automatically selects an appropriate chart type based on the data."

    def get_args_schema(self) -> Type[VisualizeDataArgs]:
        return VisualizeDataArgs

    async def execute(
        self, context: ToolContext, args: VisualizeDataArgs
    ) -> ToolResult:
        """Read CSV file and generate visualization."""
        try:
            logger.info(f"Starting visualization for file: {args.filename}")

            # Read the CSV file using FileSystem
            csv_content = await self.file_system.read_file(args.filename, context)
            logger.info(f"Read {len(csv_content)} bytes from CSV file")

            # Parse CSV into DataFrame
            import io

            df = pd.read_csv(io.StringIO(csv_content))
            logger.info(
                f"Parsed DataFrame with shape {df.shape}, columns: {df.columns.tolist()}, dtypes: {df.dtypes.to_dict()}"
            )

            # Generate title
            title = args.title or f"Visualization of {args.filename}"

            # Generate chart using PlotlyChartGenerator
            logger.info("Generating chart...")
            chart_dict = self.plotly_generator.generate_chart(df, title)
            logger.info(
                f"Chart generated, type: {type(chart_dict)}, keys: {list(chart_dict.keys()) if isinstance(chart_dict, dict) else 'N/A'}"
            )

            # Create result message
            row_count = len(df)
            col_count = len(df.columns)
            result = f"Created visualization from '{args.filename}' ({row_count} rows, {col_count} columns)."

            # Create ChartComponent
            logger.info("Creating ChartComponent...")
            chart_component = ChartComponent(
                chart_type="plotly",
                data=chart_dict,
                title=title,
                config={
                    "data_shape": {"rows": row_count, "columns": col_count},
                    "source_file": args.filename,
                },
            )
            logger.info("ChartComponent created successfully")

            # Persist chart artifacts for downstream workflows (e.g., /report).
            chart_id = f"chart_{uuid.uuid4().hex[:8]}"
            chart_json_filename = f"charts/{chart_id}.json"
            chart_html_filename = f"charts/{chart_id}.html"

            # Save chart JSON (plotly figure dict)
            await self.file_system.write_file(
                chart_json_filename,
                json.dumps(chart_dict, ensure_ascii=False),
                context,
                overwrite=True,
            )

            # Save a self-contained embeddable HTML snippet (includes plotly.js inline)
            import plotly.io as pio

            fig = pio.from_json(json.dumps(chart_dict, ensure_ascii=False))
            chart_html = pio.to_html(
                fig,
                full_html=False,
                include_plotlyjs="inline",
                config={"responsive": True, "displayModeBar": False},
            )
            await self.file_system.write_file(
                chart_html_filename, chart_html, context, overwrite=True
            )

            chart_artifact_marker = (
                "VANNA_CHART_ARTIFACT "
                + json.dumps(
                    {
                        "id": chart_id,
                        "title": title,
                        "source_csv": args.filename,
                        "rows": row_count,
                        "columns": df.columns.tolist(),
                        "chart_json_file": chart_json_filename,
                        "chart_html_file": chart_html_filename,
                    },
                    ensure_ascii=False,
                )
            )

            logger.info("Creating ToolResult...")
            tool_result = ToolResult(
                success=True,
                # Store a parseable marker in the tool message so report workflows can
                # find charts later via chat history (conversation stores tool text).
                result_for_llm=f"{result}\n\n{chart_artifact_marker}",
                ui_component=UiComponent(
                    rich_component=chart_component,
                    simple_component=SimpleTextComponent(text=result),
                ),
                metadata={
                    "filename": args.filename,
                    "rows": row_count,
                    "columns": col_count,
                    "chart": chart_dict,
                    "chart_id": chart_id,
                    "chart_json_file": chart_json_filename,
                    "chart_html_file": chart_html_filename,
                },
            )
            logger.info("ToolResult created successfully")
            return tool_result

        except FileNotFoundError as e:
            logger.error(f"File not found: {args.filename}", exc_info=True)
            error_message = f"File not found: {args.filename}"
            return ToolResult(
                success=False,
                result_for_llm=error_message,
                ui_component=UiComponent(
                    rich_component=NotificationComponent(
                        type=ComponentType.NOTIFICATION,
                        level="error",
                        message=error_message,
                    ),
                    simple_component=SimpleTextComponent(text=error_message),
                ),
                error=str(e),
                metadata={"error_type": "file_not_found"},
            )
        except pd.errors.ParserError as e:
            logger.error(f"CSV parse error for {args.filename}", exc_info=True)
            error_message = f"Failed to parse CSV file '{args.filename}': {str(e)}"
            return ToolResult(
                success=False,
                result_for_llm=error_message,
                ui_component=UiComponent(
                    rich_component=NotificationComponent(
                        type=ComponentType.NOTIFICATION,
                        level="error",
                        message=error_message,
                    ),
                    simple_component=SimpleTextComponent(text=error_message),
                ),
                error=str(e),
                metadata={"error_type": "csv_parse_error"},
            )
        except ValueError as e:
            logger.error(f"Visualization error for {args.filename}", exc_info=True)
            error_message = f"Cannot visualize data: {str(e)}"
            return ToolResult(
                success=False,
                result_for_llm=error_message,
                ui_component=UiComponent(
                    rich_component=NotificationComponent(
                        type=ComponentType.NOTIFICATION,
                        level="error",
                        message=error_message,
                    ),
                    simple_component=SimpleTextComponent(text=error_message),
                ),
                error=str(e),
                metadata={"error_type": "visualization_error"},
            )
        except Exception as e:
            logger.error(
                f"Unexpected error creating visualization for {args.filename}",
                exc_info=True,
            )
            error_message = f"Error creating visualization: {str(e)}"
            return ToolResult(
                success=False,
                result_for_llm=error_message,
                ui_component=UiComponent(
                    rich_component=NotificationComponent(
                        type=ComponentType.NOTIFICATION,
                        level="error",
                        message=error_message,
                    ),
                    simple_component=SimpleTextComponent(text=error_message),
                ),
                error=str(e),
                metadata={"error_type": "general_error"},
            )
