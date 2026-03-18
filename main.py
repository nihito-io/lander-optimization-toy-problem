import numpy as np
import pandas as pd

from helper.create_visualization import create_visualization
from helper.export_video import export_video
from ocp.lander_ocp import N_TIMESTEPS, solve_lander_ocp


def lander_main(
    init_pos_x=100.0,
    init_pos_y=100.0,
    init_vel_x=-0.0,
    init_vel_y=10.0,
    init_zenith_angle=0,
    init_angular_vel=0,
):

    results_dict = solve_lander_ocp(
        init_pos_x,
        init_pos_y,
        init_vel_x,
        init_vel_y,
        init_zenith_angle,
        init_angular_vel,
    )

    # Flatten the dictionary for states and inputs
    data = {}

    # Generate time vector
    delta_t = results_dict["delta_t"][0, 0]  # Uniform time difference
    time = np.arange(0, N_TIMESTEPS * delta_t, delta_t)  # Time vector
    data["time"] = time

    for state, values in results_dict["states"].items():
        data[state] = values
    for inp, values in results_dict["inputs"].items():
        data[inp] = np.append(values, np.nan)  # Add NaN to make same length

    # Convert to a DataFrame
    df = pd.DataFrame(data)

    csv_filename = "results.csv"
    # Save to a CSV file
    df.to_csv(csv_filename, index=False)

    video_filename = "out.mp4"
    export_video(results_dict, video_filename, 1920, 1080)

    fig = create_visualization(time, data)

    return tuple([csv_filename, video_filename, fig])


if __name__ == "__main__":
    init_pos_x = 100.0
    init_pos_y = 100.0
    init_vel_x = 5.0
    init_vel_y = 0.0
    init_zenith_angle = 0
    init_angular_vel = 0

    lander_main(
        init_pos_x,
        init_pos_y,
        init_vel_x,
        init_vel_y,
        init_zenith_angle,
        init_angular_vel,
    )
