from typing import Dict, List

import pandas as pd


class ExcelLoader:

    def __init__(self, file_path: str):
        self.file_path = file_path

    def load(self):
        """Load all sheets and merge into one dataframe.

        Never deduplicates by Application ID -- a collision is a data
        problem to surface, not a decision this loader gets to make
        silently (see find_duplicate_application_ids). Every row from
        every sheet is returned.
        """
        workbook = pd.read_excel(self.file_path, sheet_name=None)
        frames = []

        for sheet_name, frame in workbook.items():
            frame.columns = frame.columns.str.strip()
            frame = frame.copy()
            frame["source_sheet"] = sheet_name
            # +2: pandas' 0-based row index -> 1-based, plus the header row.
            frame["source_row"] = frame.index + 2
            frames.append(frame)

        if not frames:
            return pd.DataFrame()

        return pd.concat(frames, ignore_index=True)

    @staticmethod
    def find_duplicate_application_ids(df: pd.DataFrame) -> Dict[str, List[dict]]:
        """Application IDs that occur more than once across the loaded rows.

        Returns {application_id: [{"source_sheet", "source_row",
        "application_name"}, ...]} for every colliding ID; {} if none.
        """
        if "Application ID" not in df.columns:
            return {}

        collisions: Dict[str, List[dict]] = {}
        ids = df["Application ID"].astype(str).str.strip()
        for application_id, group in df.groupby(ids):
            if len(group) <= 1:
                continue
            collisions[application_id] = [
                {
                    "source_sheet": row.get("source_sheet"),
                    "source_row": row.get("source_row"),
                    "application_name": row.get("Application Name"),
                }
                for _, row in group.iterrows()
            ]
        return collisions
