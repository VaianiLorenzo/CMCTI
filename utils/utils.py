import argparse

import numpy as np
import pandas as pd

from utils.config import get_cfg_defaults


def parse_cmd_line_params():
    parser = argparse.ArgumentParser(description="SemEval 2022 MAMI Solution")
    parser.add_argument(
        "--cfg",
        help="Allows to specify values for the configuration entries to override default settings",
        nargs=argparse.REMAINDER, required=False)

    args = parser.parse_args()
    return args


def read_config():
    cfg = get_cfg_defaults()

    cmd_args = parse_cmd_line_params()

    cmd_cfg = cmd_args.cfg
    if cmd_cfg is not None:
        try:
            cfg.merge_from_list(cmd_cfg)
        except:
            print("Command-line parameter parsing error")
            exit()

    cfg.freeze()

    return cfg


def read_csv_data(path_dataset, use_redundant_labels: bool, random_state=None):
    df = pd.read_csv(path_dataset, sep=";")
    if random_state is not None:
        df = df.sample(frac=1, random_state=random_state)
    names = list(df["file_name"])
    misogynous = list(df['misogynous'])
    '''
    shaming = list(df['shaming'])
    stereotype = list(df['stereotype'])
    objectification = list(df['objectification'])
    violence = list(df['violence'])
    '''
    text = list(df["Text Transcription"])

    # Construct type labels
    if use_redundant_labels:
        type_label = df[["misogynous", "shaming", "stereotype", "objectification", "violence"]]
    else:
        type_label = df[["shaming", "stereotype", "objectification", "violence"]]
    type_label = type_label.to_numpy()

    # Construct source identification labels
    source = np.zeros((len(df), 5))
    for i in range(len(df)):
        row = df.iloc[i]
        label = int(row["text"] * 1 + row["image"] * 2 + row["either"] * 3 + row["both"] * 4)
        source[i, label] = 1

    return names, text, misogynous, type_label, source
