"""How the patient-ID matching holds up: the overlap under other confidence rules, under a
random split, and the NIH images that more than one ChestMNIST image claims."""

from __future__ import annotations

import numpy as np
import pandas as pd


def matching_checks(table: pd.DataFrame, confident: pd.Series) -> list[tuple[str, int, str]]:
    out: list[tuple[str, int, str]] = []
    shared_by_rule = {}
    for floor, margin in ((0.95, 0.0), (0.95, 0.005), (0.99, 0.0), (0.90, 0.0), (0.90, 0.02)):
        rule = (table["correlation"] >= floor) & (
            table["runner_up"].isna() | (table["correlation"] - table["runner_up"] >= margin)
        )
        train = set(table.loc[(table["split"] == "train") & rule, "patient_id"])
        held = set(table.loc[(table["split"] == "test") & rule, "patient_id"])
        shared_by_rule[(floor, margin)] = len(train & held)
    out.append(("sharedRuleNoMargin", shared_by_rule[(0.95, 0.0)], "{}"))
    out.append(("sharedRuleSmallMargin", shared_by_rule[(0.95, 0.005)], "{}"))
    out.append(("sharedRuleHighFloor", shared_by_rule[(0.99, 0.0)], "{}"))
    out.append(("sharedRuleLowFloor", shared_by_rule[(0.90, 0.0)], "{}"))
    out.append(("sharedRuleLowFloorMargin", shared_by_rule[(0.90, 0.02)], "{}"))
    sure_rows = table[confident]
    rng = np.random.default_rng(0)
    null = []
    for _ in range(5):
        shuffled = rng.permutation(sure_rows["split"].to_numpy())
        ids = sure_rows["patient_id"].to_numpy()
        null.append(len(set(ids[shuffled == "train"]) & set(ids[shuffled == "test"])))
    out.append(("nullSharedMin", min(null), "{:,}"))
    out.append(("nullSharedMax", max(null), "{:,}"))
    claimed = table["nih_image"].replace("", np.nan).dropna().value_counts()
    out.append(("nihClaimedTwice", int((claimed > 1).sum()), "{}"))
    sure_claimed = sure_rows["nih_image"].value_counts()
    out.append(("nihClaimedTwiceSure", int((sure_claimed > 1).sum()), "{}"))
    out.append(("noCandidate", int(table["patient_id"].isna().sum()), "{}"))
    out.append(("sureNoRunnerUp", int((confident & table["runner_up"].isna()).sum()), "{}"))
    return out
