from ocp.lander_ocp import N_TIMESTEPS


import cv2
import numpy as np


from math import cos, sin


def export_video(results_dict, video_filename, image_width, image_height):

    flight_path = np.array([
        results_dict['states']['position_x'],
        results_dict['states']['position_y']
    ])

    video = cv2.VideoWriter(video_filename, cv2.VideoWriter_fourcc(
        'm', 'p', '4', 'v'), 60, (image_width, image_height))
    for i in range(N_TIMESTEPS-1):
        for interp in np.arange(0.1, 1.0, 0.2):
            image = 255 * np.ones((image_height, image_width, 3), np.uint8)

            this_zenith_angle = results_dict["states"]["zenith_angle"][i] + interp * (
                results_dict["states"]["zenith_angle"][i+1] -
                results_dict["states"]["zenith_angle"][i]
            )
            this_pos = flight_path[:, i] + interp*(
                flight_path[:, i+1] - flight_path[:, i]
            )

            cz = cos(this_zenith_angle)
            sz = sin(this_zenith_angle)

            def transform_lander(x): return np.array(
                [[cz, sz], [-sz, cz]]) @ x + np.tile(this_pos, (x.shape[1], 1)).T

            def transform(x): return [np.array((np.array([[7.0, 0], [
                0, -7.0]]) @ x + np.array([[image_width/2.0], [image_height-30]])).T, np.int32)]

        # Draw flight path
            image = cv2.polylines(image, transform(
                flight_path), False, (255, 100, 0), 1, cv2.LINE_AA)

        # Draw lander
            lander_polygon = transform_lander(np.array(
                [[-1, -3, -2, -1, -1, 1, 1, 2, 3, 1], [-1, -2, 1, 1, 3, 3, 1, 1, -2, -1]]))
            image = cv2.polylines(image, transform(
                lander_polygon), True, (0, 0, 0), 2, cv2.LINE_AA)

        # Draw ground
            image = cv2.polylines(image, transform(
                np.array([[-50, 50], [-2.5, -2.5]])), False, (0, 0, 0), 2, cv2.LINE_AA)

        # Draw exhaust
            throttle_value = results_dict["inputs"]["throttle"][i]
            gimbal_value = results_dict["inputs"]["gimbal"][i]
            exhaust_polygon = np.array(
                [[-0.5, -5*float(throttle_value)*float(gimbal_value), 0.5], [-1, -1-(5*float(throttle_value)), -1]])
            image = cv2.polylines(image, transform(transform_lander(
                exhaust_polygon)), False, (0, 0, 255), 2, cv2.LINE_AA)

            video.write(image)

    video.release()
