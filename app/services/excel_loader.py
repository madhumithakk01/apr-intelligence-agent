import pandas as pd


class ExcelLoader:

    def __init__(self, file_path: str):
        self.file_path = file_path

    def load(self):
        """Load all sheets and merge into one dataframe."""
        workbook = pd.read_excel(self.file_path, sheet_name=None)
        frames = []

        for sheet_name, frame in workbook.items():
            frame.columns = frame.columns.str.strip()
            frame = frame.copy()
            frame["source_sheet"] = sheet_name
            frames.append(frame)

        if not frames:
            return pd.DataFrame()

        merged = pd.concat(frames, ignore_index=True)
        if "Application ID" in merged.columns:
            merged = merged.drop_duplicates(subset=["Application ID"], keep="last")

        return merged