import os

import cv2
import matplotlib.pyplot as plt
from ultralytics import YOLO


def get_subfolders(folder):
    return [f.path + "/" for f in os.scandir(folder) if f.is_dir()]


def get_data_from_mask(r):
    current_shape = r.masks.data[0].shape
    original_shape = r.orig_shape
    scale_factor_x = original_shape[0] / current_shape[0]
    scale_factor_y = original_shape[1] / current_shape[1]
    if "L" in r.path.split("/")[-1]:
        left = True
    elif "R" in r.path.split("/")[-1]:
        left = False
    else:
        raise ValueError("Could not determine left or right claw")

    filename = r.path.split("/")[-1]
    data = dict(
        animal_id=filename.split("V")[0],
        left=left,
        date=r.path.split("/")[-2],
    )
    claw_mask1 = r.masks.data[0].numpy()
    claw_mask2 = r.masks.data[1].numpy()

    highest_point1 = claw_mask1.max(axis=1).argmax()
    highest_point2 = claw_mask2.max(axis=1).argmax()

    claw1_is_inner = highest_point1 < highest_point2 and data["left"]
    inner_claw = claw_mask1 if claw1_is_inner else claw_mask2
    outer_claw = claw_mask2 if claw1_is_inner else claw_mask1

    data["outer_claw_area"] = outer_claw.sum()
    data["inner_claw_area"] = inner_claw.sum()
    data["relative_claw_area"] = round(
        data["outer_claw_area"] / data["inner_claw_area"], 4
    )

    # compute center of claw as an x, y tuple
    data["inner_claw_center"] = (
        inner_claw.max(axis=0).argmax() * scale_factor_x,
        inner_claw.mean(axis=1).argmax() * scale_factor_y,
    )

    data["outer_claw_center"] = (
        outer_claw.max(axis=0).argmax() * scale_factor_x,
        outer_claw.mean(axis=1).argmax() * scale_factor_y,
    )
    return data


model = YOLO("./claw_segmentation/models/yolov8_claw_segmentation_v0.pt")

folder = "./claw_segmentation/data/"
subfolders = get_subfolders(folder)


for subfolder in subfolders:
    try:
        res = model.predict(subfolder, conf=0.1, iou=0.45, max_det=2)
    except FileNotFoundError:
        print("FileNotFoundError for folder: ", subfolder)
        continue

    date = subfolder.split("/")[-2]
    for r in res:
        if "V" not in r.path.split("/")[-1]:
            continue
        result_data = get_data_from_mask(r)

        im_array = r.plot(
            boxes=False, labels=False, probs=False
        )  # plot a BGR numpy array of predictions
        fig = plt.imshow(cv2.cvtColor(im_array, cv2.COLOR_BGR2RGB))
        # Add annotations in the center of the inner claw, with a large white font

        plt.annotate(
            "Innere Klaue",
            xy=result_data["inner_claw_center"],
            color="white",
            size=14,
            backgroundcolor="black",
        )
        plt.annotate(
            "Äußere Klaue\n"
            + "Relative Fläche: "
            + str(result_data["relative_claw_area"]),
            xy=result_data["outer_claw_center"],
            color="white",
            size=14,
            backgroundcolor="black",
        )

        left_right = "links" if result_data["left"] else "rechts"
        plt.title(f"{result_data['animal_id']} Fuß {left_right} - {date}")
        plt.show()  # show image

print(res)
