from typing import List

import numpy as np
import pandas as pd
import os

from src.utils.basic.io import get_file_list


class DataFileGenerator(object):
    def __init__(
        self,
        root_dir,
        nmco_file_loc,
        output_dir,
        marker_file_loc,
        image_dir_loc,
        drop_nmco_feats: List = [],
        drop_nmco_feats_searchstr: List = [],
    ):
        super().__init__()
        self.root_dir = root_dir
        self.nmco_file_loc = nmco_file_loc
        self.output_dir = output_dir
        self.drop_nmco_feats = drop_nmco_feats
        self.drop_nmco_feats_searchstr = drop_nmco_feats_searchstr
        self.marker_file_loc = marker_file_loc
        self.image_dir_loc = image_dir_loc

        self.nmco_label_df = None
        self.image_loc_label_df = None

    def create_data_label_dfs(self):
        nmco_label_dfs = []
        image_loc_label_dfs = []
        timepoint_dirs = [
            os.path.join(self.root_dir, timepoint_dir)
            for timepoint_dir in os.listdir(self.root_dir)
        ]
        for timepoint_dir in timepoint_dirs:
            timepoint = os.path.split(timepoint_dir)[1].split("_")[1]
            patient_tp_dirs = [
                os.path.join(timepoint_dir, patient_dir)
                for patient_dir in os.listdir(timepoint_dir)
            ]
            for patient_tp_dir in patient_tp_dirs:
                patient = os.path.split(patient_tp_dir)[1].split("_")[0]

                nmco_label_df = self.get_nmco_label_df(patient_tp_dir)
                nmco_label_df["patient"] = patient
                nmco_label_df["timepoint"] = timepoint
                nmco_label_dfs.append(nmco_label_df)

                image_loc_label_df = self.get_image_loc_label_df(patient_tp_dir)
                image_loc_label_df["patient"] = patient
                image_loc_label_df["timepoint"] = timepoint
                image_loc_label_dfs.append(image_loc_label_df)

        self.nmco_label_df = pd.concat(nmco_label_dfs)
        self.nmco_label_df = self.nmco_label_df.loc[
            :, (self.nmco_label_df != self.nmco_label_df.iloc[0]).any()
        ]
        self.nmco_label_df = self.nmco_label_df.dropna(axis=1, how="any")
        self.nmco_label_df = self.nmco_label_df.dropna(axis=0, how="any")

        self.image_loc_label_df = pd.concat(image_loc_label_dfs)
        self.image_loc_label_df = self.image_loc_label_df.dropna(axis=0, how="any")

    def get_nmco_label_df(self, patient_tp_dir):
        marker_label_df = pd.read_csv(
            os.path.join(patient_tp_dir, self.marker_file_loc), index_col=0
        )
        for col in marker_label_df.columns:
            marker_label_df.loc[:, col] = marker_label_df.loc[:, col].map(
                {0: "{}-".format(col), 1: "{}+".format(col)}
            )
        nmco_df = pd.read_csv(
            os.path.join(patient_tp_dir, self.nmco_file_loc), index_col=0
        )
        nmco_label_df = nmco_df.join(marker_label_df)
        nmco_label_df = nmco_label_df.drop(columns=self.drop_nmco_feats)

        to_drop = []
        for s in self.drop_nmco_feats_searchstr:
            to_drop += [col for col in list(nmco_label_df.columns) if s in col]
        nmco_label_df = nmco_label_df.drop(columns=to_drop)

        return nmco_label_df

    def get_image_loc_label_df(self, patient_tp_dir):
        marker_label_df = pd.read_csv(
            os.path.join(patient_tp_dir, self.marker_file_loc), index_col=0
        )
        for col in marker_label_df.columns:
            marker_label_df.loc[:, col] = marker_label_df.loc[:, col].map(
                {0: "{}-".format(col), 1: "{}+".format(col)}
            )
        file_locs = get_file_list(os.path.join(patient_tp_dir, self.image_dir_loc), absolute_path=True)
        file_ids = get_file_list(
            os.path.join(patient_tp_dir, self.image_dir_loc),
            absolute_path=False,
            file_ending=False,
        )
        image_file_df = pd.DataFrame(
            np.array(file_locs), columns=["image_loc"], index=file_ids
        )
        image_loc_label_df = image_file_df.join(marker_label_df)
        return image_loc_label_df

    def save_data_label_dfs(self):
        if self.nmco_label_df is not None:
            self.nmco_label_df.to_csv(
                os.path.join(self.output_dir, "nmco_feats_and_labels.csv")
            )
        if self.image_loc_label_df is not None:
            self.image_loc_label_df.to_csv(
                os.path.join(self.output_dir, "image_locs_and_labels.csv")
            )
