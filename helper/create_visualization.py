import plotly.graph_objects as go
from plotly.subplots import make_subplots


def create_visualization(time, data):

    # Create visualization
    fig = make_subplots(rows=3, cols=1,
                        shared_xaxes=True,
                        vertical_spacing=0.02)

    row_idx = 1
    fig.add_trace(
        go.Scatter(
            x=time, y=data["throttle"], name="throttle"),
        row=row_idx, col=1
    )
    fig.add_trace(
        go.Scatter(
            x=time, y=data["gimbal"], name="gimbal"),
        row=row_idx, col=1
    )
    row_idx = row_idx + 1

    fig.add_trace(
        go.Scatter(
            x=time, y=data["velocity_x"], name="velocity_x"),
        row=row_idx, col=1
    )
    fig.add_trace(
        go.Scatter(
            x=time, y=data["velocity_y"], name="velocity_y"),
        row=row_idx, col=1
    )
    # Add y-axis label for each subplot (corresponding to each column)
    fig.update_yaxes(title_text="velocity", row=row_idx, col=1)
    row_idx = row_idx + 1

    fig.add_trace(
        go.Scatter(
            x=time, y=data["position_x"], name="position_x"),
        row=row_idx, col=1
    )
    fig.add_trace(
        go.Scatter(
            x=time, y=data["position_y"], name="position_y"),
        row=row_idx, col=1
    )
    # Add y-axis label for each subplot (corresponding to each column)
    fig.update_yaxes(title_text="position", row=row_idx, col=1)

    # Update the x-axis label for all subplots
    fig.update_xaxes(title_text="Time", row=row_idx, col=1)
    fig.show()

    return fig
