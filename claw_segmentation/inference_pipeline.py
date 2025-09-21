"""Inference pipeline utilities for the claw segmentation project."""

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Optional

import cv2
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from tqdm import tqdm
from ultralytics import YOLO


DEFAULT_MODEL_PATH = Path("./claw_segmentation/models/yolov8_claw_segmentation_v0.pt")
DEFAULT_DATA_DIR = Path("./claw_segmentation/data")
DEFAULT_RESULT_DIR = DEFAULT_DATA_DIR / "ergebnisse"
DEFAULT_EXCEL_FILENAME = "Klauenerziehung_Urliste.xlsx"


@dataclass
class InferenceConfig:
    """Lightweight container for inference configuration."""

    model_path: Path = DEFAULT_MODEL_PATH
    data_dir: Path = DEFAULT_DATA_DIR
    result_dir: Path = DEFAULT_RESULT_DIR
    conf: float = 0.1
    iou: float = 0.45
    max_det: int = 2
    excel_filename: str = DEFAULT_EXCEL_FILENAME

    @property
    def excel_path(self) -> Path:
        return self.data_dir / self.excel_filename


class ClawSegmentationInference:
    """Encapsulates model inference, result aggregation and plotting."""

    def __init__(self, config: Optional[InferenceConfig] = None) -> None:
        self.config = config or InferenceConfig()
        self.model = YOLO(str(self.config.model_path))

    def run(self) -> pd.DataFrame:
        """Execute inference on unseen folders and update artifacts."""

        data_dir = self.config.data_dir
        result_dir = self.config.result_dir
        ensure_directory(result_dir)

        previous_results, processed_dates = self._load_previous_results(
            result_dir / "ergebnisse.csv"
        )

        candidate_folders = self._collect_candidate_folders(data_dir, processed_dates)
        if not candidate_folders:
            return previous_results

        inference_results = self._predict_folders(candidate_folders)
        if not inference_results:
            return previous_results

        rows = []
        for inference_result in tqdm(inference_results, desc="Processing predictions"):
            row = self._process_result(inference_result)
            if row is not None:
                rows.append(row)

        if not rows:
            return previous_results

        new_results = pd.DataFrame(
            rows,
            columns=[
                "Ohrmarkennummer",
                "Datum",
                "Klaue",
                "Relative Klauenfläche (=Aussen/Innen)",
            ],
            dtype=object,
        )
        new_results["Mess Nr."] = (
            new_results.groupby(["Ohrmarkennummer", "Klaue"]).cumcount() + 1
        )

        combined = pd.concat([previous_results, new_results], ignore_index=True)
        combined = combined.drop_duplicates()
        combined = combined.sort_values(
            by=["Ohrmarkennummer", "Klaue", "Datum"]
        ).reset_index(drop=True)
        combined["Mess Nr."] = (
            combined.groupby(["Ohrmarkennummer", "Klaue"]).cumcount() + 1
        )

        combined = self._append_category_information(combined)
        combined.to_csv(result_dir / "ergebnisse.csv", mode="w", index=False)

        self._create_result_plots(combined, result_dir, "Klaue")
        self._create_result_plots(combined, result_dir, "Gruppe")
        self._create_joint_plot(combined, result_dir)

        return combined

    def _collect_candidate_folders(
        self, data_dir: Path, processed_dates: Iterable[str]
    ) -> List[Path]:
        processed = set(processed_dates)
        candidates: List[Path] = []
        for child in data_dir.iterdir():
            if not child.is_dir():
                continue
            if child.name == self.config.result_dir.name:
                continue
            if child.name in processed:
                continue
            candidates.append(child)
        return sorted(candidates)

    def _predict_folders(self, folders: Iterable[Path]):
        all_results = []
        for folder in folders:
            try:
                results = self.model.predict(
                    source=str(folder),
                    conf=self.config.conf,
                    iou=self.config.iou,
                    max_det=self.config.max_det,
                )
            except FileNotFoundError:
                continue
            all_results.extend(results)
        return all_results

    def _load_previous_results(
        self, result_file: Path
    ) -> tuple[pd.DataFrame, List[str]]:
        if result_file.exists():
            previous = pd.read_csv(result_file, dtype=str)
            if not previous.empty:
                previous[
                    "Relative Klauenfläche (=Aussen/Innen)"
                ] = previous["Relative Klauenfläche (=Aussen/Innen)"].astype(float)
            processed_dates = [
                "_".join(date.split("_")[::-1]) for date in previous["Datum"].unique()
            ]
            return previous, processed_dates
        return pd.DataFrame(), []

    def _process_result(self, result):
        filename = Path(result.path)
        if "V" not in filename.name:
            return None
        result_data = self._extract_mask_data(result)
        self._create_prediction_visual(result, result_data)
        formatted_date = "_".join(result_data["date"].split("_")[::-1])
        return {
            "Ohrmarkennummer": result_data["animal_id"],
            "Datum": formatted_date,
            "Relative Klauenfläche (=Aussen/Innen)": result_data[
                "relative_claw_area"
            ],
            "Klaue": "links" if result_data["left"] else "rechts",
        }

    def _extract_mask_data(self, result):
        current_shape = result.masks.data[0].shape
        original_shape = result.orig_shape
        scale_factor_x = original_shape[0] / current_shape[0]
        scale_factor_y = original_shape[1] / current_shape[1]

        filename = Path(result.path)
        left = "L" in filename.name
        if "L" not in filename.name and "R" not in filename.name:
            raise ValueError("Could not determine left or right claw from filename")

        data = {
            "animal_id": filename.name.split("V")[0],
            "left": left,
            "date": filename.parent.name,
        }

        claw_mask1 = result.masks.data[0].numpy()
        claw_mask2 = result.masks.data[1].numpy()

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

        data["inner_claw_center"] = (
            inner_claw.max(axis=0).argmax() * scale_factor_x,
            inner_claw.mean(axis=1).argmax() * scale_factor_y,
        )
        data["outer_claw_center"] = (
            outer_claw.max(axis=0).argmax() * scale_factor_x,
            outer_claw.mean(axis=1).argmax() * scale_factor_y,
        )
        return data

    def _create_prediction_visual(self, result, result_data) -> None:
        filename = Path(result.path)
        date_folder = "_".join(result_data["date"].split("_")[::-1])
        plot_dir = self.config.result_dir / "grafiken" / date_folder
        ensure_directory(plot_dir)
        plot_path = plot_dir / filename.name
        if plot_path.exists():
            return

        image = result.plot(boxes=False, labels=False, probs=False)
        plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB), resample=True)
        plt.annotate(
            "Innere Klaue",
            xy=result_data["inner_claw_center"],
            color="white",
            size=14,
            backgroundcolor="black",
        )
        plt.annotate(
            "Äußere Klaue\nRelative Fläche: "
            + str(result_data["relative_claw_area"]),
            xy=result_data["outer_claw_center"],
            color="white",
            size=14,
            backgroundcolor="black",
        )
        foot_side = "links" if result_data["left"] else "rechts"
        plt.title(
            f"{result_data['animal_id']} Fuß {foot_side} - {date_folder.replace('_', '-')}"
        )
        plt.savefig(plot_path)
        plt.close()

    def _create_result_plots(
        self, df: pd.DataFrame, result_folder: Path, category: str
    ) -> None:
        ensure_directory(result_folder)
        sns.boxplot(
            hue=category,
            x="Mess Nr.",
            y="Relative Klauenfläche (=Aussen/Innen)",
            data=df,
        )
        plt.grid()
        plt.savefig(result_folder / f"{category}_boxplot.png")
        plt.close()

        sns.lineplot(
            hue=category,
            x="Mess Nr.",
            y="Relative Klauenfläche (=Aussen/Innen)",
            data=df,
            err_style="bars",
        )
        plt.grid()
        plt.savefig(result_folder / f"{category}_lineplot.png")
        plt.close()

        sns.histplot(
            hue=category,
            x="Relative Klauenfläche (=Aussen/Innen)",
            data=df,
            kde=True,
        )
        plt.grid()
        plt.savefig(result_folder / f"{category}_histplot.png")
        plt.close()

    def _create_joint_plot(self, df: pd.DataFrame, result_folder: Path) -> None:
        pivot = df.pivot_table(
            index=["Ohrmarkennummer", "Datum", "Mess Nr.", "Gruppe"],
            values=["Relative Klauenfläche (=Aussen/Innen)"],
            columns=["Klaue"],
        )
        pivot.columns = pivot.columns.droplevel(0)
        pivot = pivot.reset_index()
        joint = sns.jointplot(
            x="links",
            y="rechts",
            hue="Gruppe",
            data=pivot,
        )
        joint.figure.suptitle("")
        plt.grid()
        plt.savefig(result_folder / "links_rechts_jointplot.png")
        plt.close(joint.figure)

    def _append_category_information(self, df: pd.DataFrame) -> pd.DataFrame:
        excel_path = self.config.excel_path
        if not excel_path.exists():
            return df
        from openpyxl import load_workbook

        wb = load_workbook(excel_path, data_only=True)
        sh = wb["Liste"]
        start_row = 7
        end_row = 82
        ear_tags = df["Ohrmarkennummer"].unique()
        categories = []
        for row_num in range(start_row, end_row):
            cell = sh[f"E{row_num}"]
            if cell.value in ear_tags:
                color_in_hex = cell.fill.start_color.index
                group = "Weiß" if color_in_hex == "00000000" else "Gelb"
                categories.append({"Ohrmarkennummer": cell.value, "Gruppe": group})
        categories_df = pd.DataFrame(categories)
        df = df.drop(columns=["Gruppe"], errors="ignore")
        return df.join(
            categories_df.set_index("Ohrmarkennummer", drop=True),
            on="Ohrmarkennummer",
            how="left",
        )


def ensure_directory(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def run_inference(config: Optional[InferenceConfig] = None) -> pd.DataFrame:
    """Convenience wrapper to execute the inference pipeline."""

    pipeline = ClawSegmentationInference(config)
    return pipeline.run()
