import pandas as pd

from vanna.integrations.plotly.chart_generator import PlotlyChartGenerator


def test_grouped_bar_with_numeric_metric_does_not_count_rows() -> None:
    # Pre-aggregated data: one row per (week, category) with a numeric metric.
    df = pd.DataFrame(
        {
            "week": ["2026-01-05", "2026-01-12", "2026-01-05", "2026-01-12"],
            "bike_issue_category": ["A", "A", "B", "B"],
            "growth_rate": [0.10, 0.25, 0.05, 0.15],
        }
    )

    gen = PlotlyChartGenerator()
    fig = gen._create_grouped_bar_chart(  # type: ignore[attr-defined]
        df,
        ["week", "bike_issue_category"],
        "Growth Rate",
        value_col="growth_rate",
    )

    assert fig.data, "Expected plotly traces"
    assert all(t.type == "bar" for t in fig.data)

    ys = []
    for t in fig.data:
        ys.extend(list(t.y or []))

    assert ys, "Expected y values in traces"
    assert any(y != 1 for y in ys), "Should plot metric values, not row counts"
