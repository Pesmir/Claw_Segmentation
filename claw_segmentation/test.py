

import os

import openpyxl
from openpyxl import load_workbook
import cv2
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from tqdm import tqdm
from ultralytics import YOLO


# create all subfolders
def create_folders(path: str):
    if not os.path.exists(path):
        os.makedirs(path)


def create_plot(r, result_data):
    filename = r.path.split("/")[-1]
    date = result_data["date"]
    folder = "./claw_segmentation/data/ergebnisse/grafiken/" + date
    create_folders(folder)
    if os.path.exists(folder + filename):
        return
    im_array = r.plot(
        boxes=False, labels=False, probs=False
    )  # plot a BGR numpy array of predictions
    plt.imshow(cv2.cvtColor(im_array, cv2.COLOR_BGR2RGB), resample=True)
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
    plt.savefig(folder + filename)
    plt.close()


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


def process_result(r):
    if "V" not in r.path.split("/")[-1]:
        return None
    result_data = get_data_from_mask(r)
    create_plot(r, result_data)
    # Reverse the date to get the date in the format YYYY-MM-DD
    date_yyyy_mm_dd = "_".join(result_data["date"].split("_")[::-1])
    row = {
        "Ohrmarkennummer": result_data["animal_id"],
        "Datum": date_yyyy_mm_dd,
        "Relative Klauenfläche (=Aussen/Innen)": result_data[
            "relative_claw_area"
        ],
        "Klaue": "links" if result_data["left"] else "rechts",
    }
    return row


def create_result_plots(df, result_folder, category):
    # Make a boxplot over time for the relative claw area in seaborn
    sns.boxplot(
        hue=category,
        x="Mess Nr.",
        y="Relative Klauenfläche (=Aussen/Innen)",
        data=df,
    )
    plt.savefig(result_folder + f"{category}_boxplot.png")
    plt.close()

    sns.lineplot(
        hue=category,
        x="Mess Nr.",
        y="Relative Klauenfläche (=Aussen/Innen)",
        data=df,
        err_style="bars",
    )
    plt.savefig(result_folder + f"{category}_lineplot.png")
    plt.close()

    sns.histplot(
        hue=category,
        x="Relative Klauenfläche (=Aussen/Innen)",
        data=df,
        kde=True,
    )
    plt.savefig(result_folder + f"{category}_histplot.png")
    plt.close()

def add_categories(df, folder):
    # Read the "Klauenerziehung_Urliste.xlsx" file and also get the background color data
    excel_file = folder + "Klauenerziehung_Urliste.xlsx"
    wb = load_workbook(excel_file, data_only=True)
    start_row = 7
    end_row = 82
    sh = wb["Liste"]
    categories = []
    ear_tags = df["Ohrmarkennummer"].unique()
    for row_num in range(start_row, end_row):
        cell = sh[f"E{row_num}"]
        if cell.value in ear_tags:
            color_in_hex = cell.fill.start_color.index  # this gives you Hexadecimal value of the color
            if color_in_hex == "00000000":
                group = "Weiß"
            else:
                group = "Gelb"
            categories.append({"Ohrmarkennummer": cell.value, "Gruppe": group})
    categories_df = pd.DataFrame(categories)
    # Add group data to the dataframe
    df = df.join(categories_df.set_index("Ohrmarkennummer", drop=True), on="Ohrmarkennummer", how="left")
    return df


model = YOLO("./claw_segmentation/models/yolov8_claw_segmentation_v0.pt")

folder = "./claw_segmentation/data/"
subfolders = get_subfolders(folder)

result_folder = "./claw_segmentation/data/ergebnisse/"
result_file = result_folder + "ergebnisse.csv"
todo_dates = set(map(lambda x: x.split("/")[-2], get_subfolders(folder)))

# Get dates from resutlts file if exists
if os.path.exists(result_file):
    res_df = pd.read_csv(result_folder + "ergebnisse.csv", dtype=str)
    res_df["Relative Klauenfläche (=Aussen/Innen)"] = res_df[
        "Relative Klauenfläche (=Aussen/Innen)"
    ].astype(float)
    result_dates = set(res_df["Datum"].unique())
    dates = set()
    for date in result_dates:
        date = "_".join(date.split("_")[::-1])
        dates.add(date)
    todo_dates = todo_dates - dates
else:
    res_df = pd.DataFrame()

plot_folder = result_folder + "grafiken/"
create_folders(plot_folder)

rows = []

all_res = []
for subfolder in todo_dates:
    if "ergebnisse" in subfolder:
        continue
    try:
        res = model.predict(folder + subfolder, conf=0.1, iou=0.45, max_det=2)
    except FileNotFoundError:
        print("FileNotFoundError for folder: ", folder + subfolder)
        continue
    all_res.extend(res)


for r in tqdm(all_res):
    row = process_result(r)
    if row is not None:
        rows.append(row)

df = pd.DataFrame(
    rows,
    columns=[
        "Ohrmarkennummer",
        "Datum",
        "Klaue",
        "Relative Klauenfläche (=Aussen/Innen)",
    ],
    dtype=object,
)
df["Mess Nr."] = df.groupby(["Ohrmarkennummer", "Klaue"]).cumcount() + 1
# Combine results with existing results
df = pd.concat([res_df, df], ignore_index=True).drop_duplicates()
df = df.sort_values(by=["Ohrmarkennummer", "Klaue", "Datum"]).reset_index(
    drop=True
)
df["Mess Nr."] = df.groupby(["Ohrmarkennummer", "Klaue"]).cumcount() + 1

df = add_categories(df, folder)
df.to_csv(result_file, mode="w", index=False)

create_result_plots(df, result_folder, "Klaue")
create_result_plots(df, result_folder, "Gruppe")

# Join left an right claw into one row
df_joined = df.pivot_table(
    index=["Ohrmarkennummer", "Datum", "Mess Nr.", "Gruppe"],
    values=["Relative Klauenfläche (=Aussen/Innen)"],
    columns=["Klaue"],
)

df_joined.columns = df_joined.columns.droplevel(0)
df_joined = df_joined.reset_index()
sns.scatterplot(
    x="links",
    y="rechts",
    hue="Gruppe",
    data=df_joined,
)

plt.show()
