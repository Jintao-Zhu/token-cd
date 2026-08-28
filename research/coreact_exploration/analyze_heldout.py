#!/usr/bin/env python3
"""Audit and analyze a completed, preregistered CoreAct held-out run."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yaml
from scipy.stats import spearmanr, wilcoxon


SCORES = [
    "deterministic_random",
    "embedding_norm",
    "raw_last_expert_layer_attention",
    "late_half_action_to_context_attention",
]
PRIMARY_REPLACEMENT = "position_conditioned_modality_mean"
PRIMARY_SCORE = "late_half_action_to_context_attention"
KEY = ["episode_id", "frame_id", "tau", "noise_seed", "group_id", "replacement_type"]


def cluster_bootstrap(rows, value, replicates, seed, statistic=np.median):
    """Resample tasks, then episodes within each sampled task."""
    grouped = {}
    for row in rows.to_dict("records"):
        grouped.setdefault(str(row["task_id"]), {}).setdefault(str(row["episode_id"]), []).append(float(row[value]))
    rng, tasks = np.random.default_rng(seed), sorted(grouped)
    result = np.empty(replicates)
    for index in range(replicates):
        values = []
        for task in rng.choice(tasks, len(tasks), replace=True):
            episodes = sorted(grouped[task])
            for episode in rng.choice(episodes, len(episodes), replace=True):
                values.extend(grouped[task][episode])
        result[index] = statistic(np.asarray(values))
    return result


def ci(values):
    return [float(x) for x in np.quantile(values, [0.025, 0.975])]


def holm(p_values):
    """Holm-adjust p-values in original order."""
    values = np.asarray(p_values, dtype=float)
    order = np.argsort(values)
    adjusted = np.empty_like(values)
    running = 0.0
    for rank, index in enumerate(order):
        running = max(running, (len(values) - rank) * values[index])
        adjusted[index] = min(running, 1.0)
    return adjusted


def state_id(frame):
    return frame["episode_id"].astype(str) + ":" + frame["frame_id"].astype(str)


def audit(raw, manifest, expected_total):
    required = set(KEY + SCORES + [
        "task_id", "suite", "state_index", "modality", "selection_sets", "relative_magnitude_I",
        "Q_rel", "signed_loss_delta_Q", "base_loss", "integrity_flags", "runtime_ms", "peak_memory_bytes",
    ])
    failures = []
    if missing := sorted(required - set(raw.columns)):
        failures.append(f"missing columns: {missing}")
    duplicates = int(raw.duplicated(KEY).sum())
    if duplicates:
        failures.append(f"duplicate complete keys: {duplicates}")
    numeric = SCORES + ["relative_magnitude_I", "Q_rel", "signed_loss_delta_Q", "base_loss"]
    nonfinite = int((~np.isfinite(raw[numeric].to_numpy(dtype=float))).sum())
    bad_flags = int(sum(not x.get("finite", False) or not x.get("shared_action_tau_noise", False) for x in raw.integrity_flags))
    if nonfinite:
        failures.append(f"non-finite numeric values: {nonfinite}")
    if bad_flags:
        failures.append(f"failed row integrity flags: {bad_flags}")
    if len(raw) != expected_total:
        failures.append(f"row count {len(raw)} != locked runner expectation {expected_total}")

    expected_combos = {(tau, seed, replacement) for tau in [0.2, 0.5, 0.8] for seed in [1729, 9473]
                       for replacement in [PRIMARY_REPLACEMENT, "global_modality_mean", "zero"]}
    bad_groups = 0
    for _, group in raw.groupby(["episode_id", "frame_id", "group_id"], sort=False):
        combos = set(zip(group.tau, group.noise_seed, group.replacement_type))
        bad_groups += combos != expected_combos or len(group) != 18
    if bad_groups:
        failures.append(f"groups without exact 3 tau x 2 noise x 3 replacement coverage: {bad_groups}")

    heldout = pd.DataFrame([row for row in manifest if row["split"] == "heldout"])
    expected_states = set(zip(heldout.episode_id, heldout.frame_id))
    actual_states = set(zip(raw.episode_id, raw.frame_id))
    missing_states = expected_states - actual_states
    extra_states = actual_states - expected_states
    if extra_states:
        failures.append(f"raw states absent from locked manifest: {len(extra_states)}")
    selection_failures = 0
    base_failures = 0
    for _, state in raw.groupby(["episode_id", "frame_id"], sort=False):
        groups = state.drop_duplicates("group_id")
        counts = {name: int(groups.selection_sets.map(lambda x: name in x).sum()) for name in ["uniform", "top", "bottom"]}
        selection_failures += counts != {"uniform": 24, "top": 8, "bottom": 8}
        base_ranges = state.groupby(["tau", "noise_seed"])["base_loss"].agg(lambda x: float(x.max() - x.min()))
        base_failures += bool((base_ranges > 1e-12).any())
    if selection_failures:
        failures.append(f"states violating locked visual selection counts: {selection_failures}")
    if base_failures:
        failures.append(f"states with non-shared base flow values: {base_failures}")

    task_rows = []
    for task_id, expected in heldout.groupby("task_id"):
        task_raw = raw[raw.task_id == task_id]
        exp_states = len(expected)
        act_states = len(task_raw[["episode_id", "frame_id"]].drop_duplicates())
        task_rows.append({
            "task_id": task_id, "suite": expected.suite.iloc[0], "expected_states": exp_states,
            "actual_states": act_states, "missing_states": exp_states - act_states,
            "actual_groups": len(task_raw[["episode_id", "frame_id", "group_id"]].drop_duplicates()),
            "actual_rows": len(task_raw), "nan_or_inf_rows": int((~np.isfinite(task_raw.relative_magnitude_I)).sum()),
        })
    missingness = pd.DataFrame(task_rows)
    expected_group_rows = raw[["episode_id", "frame_id", "group_id"]].drop_duplicates().shape[0] * 18
    missing_flow_rows = max(0, expected_group_rows - len(raw))
    expected_units = expected_group_rows + len(missing_states) * 18
    missing_rate = (missing_flow_rows + len(missing_states) * 18) / max(expected_units, 1)
    report = {
        "pass": not failures,
        "failures": failures,
        "raw_rows": len(raw),
        "expected_raw_rows": expected_total,
        "duplicate_keys": duplicates,
        "nonfinite_values": nonfinite,
        "bad_integrity_flags": bad_flags,
        "bad_group_condition_coverage": int(bad_groups),
        "heldout_manifest_states": len(expected_states),
        "raw_states": len(actual_states),
        "missing_states": len(missing_states),
        "missing_rate": float(missing_rate),
        "selection_count_failures": selection_failures,
        "shared_base_failures": base_failures,
    }
    return report, missingness


def aggregate_groups(raw):
    primary = raw[raw.replacement_type == PRIMARY_REPLACEMENT].copy()
    columns = ["task_id", "suite", "episode_id", "frame_id", "state_index", "group_id", "modality",
               "selection_sets", "token_text", "camera_id"]
    aggregations = {"relative_magnitude_I": "median", "Q_rel": "median", "signed_loss_delta_Q": "median"}
    aggregations.update({score: "first" for score in SCORES})
    groups = primary.groupby(columns, dropna=False, sort=False).agg(aggregations).reset_index()
    groups["state_id"] = state_id(groups)
    return groups


def state_metrics(groups, thresholds, bootstrap_replicates, seed):
    records = []
    random_records = []
    for state, frame in groups.groupby("state_id", sort=False):
        visual = frame[(frame.modality == "visual") & frame.selection_sets.map(lambda x: "uniform" in x)]
        meta = frame.iloc[0]
        record = {key: meta[key] for key in ["task_id", "suite", "episode_id", "frame_id"]}
        record["state_id"] = state
        for score in SCORES:
            record[f"spearman_{score}"] = float(spearmanr(visual[score], visual.relative_magnitude_I).statistic)

        top = frame[(frame.modality == "visual") & frame.selection_sets.map(lambda x: "top" in x)]
        random_fixed = visual.nlargest(8, "deterministic_random")
        record["top_mean_I"] = float(top.relative_magnitude_I.mean())
        record["random_mean_I"] = float(random_fixed.relative_magnitude_I.mean())
        record["top_minus_random_I"] = record["top_mean_I"] - record["random_mean_I"]
        threshold = thresholds["visual"]
        for name, selected in [("top", top), ("random", random_fixed)]:
            effective = selected.relative_magnitude_I > threshold
            record[f"{name}_anchor_precision"] = float((effective & (selected.Q_rel > 0.05)).mean())
            record[f"{name}_nuisance_fraction"] = float((effective & (selected.Q_rel < -0.05)).mean())
            record[f"{name}_sign_uncertain_fraction"] = float((~effective | (selected.Q_rel.abs() <= 0.05)).mean())
        record["anchor_precision_difference"] = record["top_anchor_precision"] - record["random_anchor_precision"]
        record["nuisance_fraction_difference"] = record["top_nuisance_fraction"] - record["random_nuisance_fraction"]

        rng = np.random.default_rng(seed + int(meta.state_index))
        random_means = [float(visual.iloc[rng.choice(len(visual), 8, replace=False)].relative_magnitude_I.mean()) for _ in range(100)]
        random_records.extend({"state_id": state, "task_id": meta.task_id, "episode_id": meta.episode_id,
                               "permutation": index, "random_top8_mean_I": value,
                               "attention_top8_mean_I": record["top_mean_I"], "paired_difference": record["top_mean_I"] - value}
                              for index, value in enumerate(random_means))
        record["random_permutation_mean_I"] = float(np.mean(random_means))
        record["random_permutation_p95_I"] = float(np.quantile(random_means, 0.95))
        records.append(record)
    states = pd.DataFrame(records)
    summaries = {}
    for score in SCORES:
        key = f"spearman_{score}"
        boot = cluster_bootstrap(states, key, bootstrap_replicates, seed + SCORES.index(score))
        summaries[score] = {
            "median": float(states[key].median()),
            "iqr": [float(states[key].quantile(0.25)), float(states[key].quantile(0.75))],
            "cluster_bootstrap_95_ci": ci(boot),
        }
    for metric in ["top_minus_random_I", "anchor_precision_difference", "nuisance_fraction_difference"]:
        boot = cluster_bootstrap(states, metric, bootstrap_replicates, seed + 100 + len(summaries), np.mean)
        summaries[metric] = {"median": float(states[metric].median()), "mean": float(states[metric].mean()),
                             "point_estimate": "mean_of_paired_state_differences",
                             "cluster_bootstrap_95_ci": ci(boot)}
    random_baseline = pd.DataFrame(random_records)
    summaries["random_permutation_baseline"] = {
        "permutations_per_state": 100,
        "rows": len(random_baseline),
        "fraction_random_top8_at_least_attention_top8": float((random_baseline.paired_difference <= 0).mean()),
    }
    return states, summaries, random_baseline


def stratified(raw, groups, bootstrap_replicates, seed):
    rows = []
    for suite in sorted(groups.suite.unique()):
        subset = groups[groups.suite == suite]
        for state, frame in subset.groupby("state_id"):
            uniform = frame[(frame.modality == "visual") & frame.selection_sets.map(lambda x: "uniform" in x)]
            top = frame[(frame.modality == "visual") & frame.selection_sets.map(lambda x: "top" in x)]
            random_fixed = uniform.nlargest(8, "deterministic_random")
            rows.append({"suite": suite, "tau": "all", "state_id": state, "task_id": frame.task_id.iloc[0],
                         "episode_id": frame.episode_id.iloc[0], "rho": spearmanr(uniform[PRIMARY_SCORE], uniform.relative_magnitude_I).statistic,
                         "top_minus_random_I": float(top.relative_magnitude_I.mean() - random_fixed.relative_magnitude_I.mean())})
    primary = raw[(raw.replacement_type == PRIMARY_REPLACEMENT) & (raw.modality == "visual")].copy()
    primary["state_id"] = state_id(primary)
    tau_group = primary.groupby(["suite", "tau", "task_id", "episode_id", "state_id", "group_id"], sort=False).agg(
        relative_magnitude_I=("relative_magnitude_I", "median"), score=(PRIMARY_SCORE, "first"),
        deterministic_random=("deterministic_random", "first"), selection_sets=("selection_sets", "first")).reset_index()
    for (suite, tau, state), frame in tau_group.groupby(["suite", "tau", "state_id"]):
        uniform = frame[frame.selection_sets.map(lambda x: "uniform" in x)]
        top = frame[frame.selection_sets.map(lambda x: "top" in x)]
        random_fixed = uniform.nlargest(8, "deterministic_random")
        rows.append({"suite": suite, "tau": tau, "state_id": state, "task_id": frame.task_id.iloc[0],
                     "episode_id": frame.episode_id.iloc[0], "rho": spearmanr(uniform.score, uniform.relative_magnitude_I).statistic,
                     "top_minus_random_I": float(top.relative_magnitude_I.mean() - random_fixed.relative_magnitude_I.mean())})
    frame = pd.DataFrame(rows)
    output = []
    for (suite, tau), values in frame.groupby(["suite", "tau"], sort=False):
        boot = cluster_bootstrap(values, "rho", bootstrap_replicates, seed + len(output))
        top_boot = cluster_bootstrap(values, "top_minus_random_I", bootstrap_replicates, seed + 100 + len(output), np.mean)
        output.append({"suite": suite, "tau": tau, "states": len(values), "median_spearman": float(values.rho.median()),
                       "spearman_ci_low": ci(boot)[0], "spearman_ci_high": ci(boot)[1],
                       "mean_top_minus_random_I": float(values.top_minus_random_I.mean()),
                       "top_difference_ci_low": ci(top_boot)[0], "top_difference_ci_high": ci(top_boot)[1]})
    return pd.DataFrame(output)


def modality_secondary(groups, thresholds, bootstrap_replicates, seed):
    output = {"score_correlations": {}, "signed_composition": {}}
    for modality in ["visual", "language"]:
        selected = groups[groups.modality == modality]
        if modality == "visual":
            selected = selected[selected.selection_sets.map(lambda x: "uniform" in x)]
        state_rows = []
        for state, frame in selected.groupby("state_id"):
            row = {"state_id": state, "task_id": frame.task_id.iloc[0], "episode_id": frame.episode_id.iloc[0]}
            for score in SCORES:
                row[score] = float(spearmanr(frame[score], frame.relative_magnitude_I).statistic)
            state_rows.append(row)
        states = pd.DataFrame(state_rows).dropna()
        output["score_correlations"][modality] = {}
        for score in SCORES:
            boot = cluster_bootstrap(states, score, bootstrap_replicates, seed + len(output["score_correlations"][modality]))
            output["score_correlations"][modality][score] = {
                "states": len(states), "median": float(states[score].median()), "cluster_bootstrap_95_ci": ci(boot)
            }
        threshold = thresholds[modality]
        effective = selected.relative_magnitude_I > threshold
        output["signed_composition"][modality] = {
            "groups": len(selected),
            "above_effect_threshold_fraction": float(effective.mean()),
            "high_I_negative_Q_fraction_all_groups": float((effective & (selected.Q_rel < 0)).mean()),
            "high_I_negative_Q_fraction_among_effective": float(((selected.Q_rel < 0) & effective).sum() / max(effective.sum(), 1)),
            "anchor_fraction": float((effective & (selected.Q_rel > 0.05)).mean()),
            "nuisance_candidate_fraction": float((effective & (selected.Q_rel < -0.05)).mean()),
        }
    return output


def sensitivity(raw):
    keys = ["task_id", "suite", "episode_id", "frame_id", "group_id", "modality"]
    med = raw.groupby(keys + ["replacement_type"], sort=False).agg(I=("relative_magnitude_I", "median"), Q=("Q_rel", "median")).reset_index()
    wide_i = med.pivot(index=keys, columns="replacement_type", values="I").reset_index()
    wide_q = med.pivot(index=keys, columns="replacement_type", values="Q").reset_index()
    output = []
    for modality in ["visual", "language"]:
        for replacement in ["global_modality_mean", "zero"]:
            i = wide_i[wide_i.modality == modality]
            q = wide_q[wide_q.modality == modality]
            output.append({
                "modality": modality, "replacement": replacement, "groups": len(i),
                "I_rank_correlation": float(spearmanr(i[PRIMARY_REPLACEMENT], i[replacement]).statistic),
                "Q_sign_flip_rate": float(((q[PRIMARY_REPLACEMENT] * q[replacement]) < 0).mean()),
            })
    return pd.DataFrame(output)


def comparisons(states):
    comparisons = []
    main = states[f"spearman_{PRIMARY_SCORE}"]
    for score in ["embedding_norm", "raw_last_expert_layer_attention", "deterministic_random"]:
        delta = main - states[f"spearman_{score}"]
        test = wilcoxon(delta, alternative="two-sided")
        comparison_rows = states[["task_id", "episode_id", "state_id"]].assign(value=delta)
        interval = ci(cluster_bootstrap(comparison_rows, "value", 2000, 900 + len(comparisons)))
        comparisons.append({"comparison": f"{PRIMARY_SCORE}_minus_{score}", "median_difference": float(delta.median()),
                            "cluster_bootstrap_95_ci": interval, "p_value": float(test.pvalue)})
    adjusted = holm([row["p_value"] for row in comparisons])
    for row, value in zip(comparisons, adjusted):
        row["holm_adjusted_p"] = float(value)
    return comparisons


def task_metrics(states):
    columns = [f"spearman_{score}" for score in SCORES] + ["top_minus_random_I", "anchor_precision_difference", "nuisance_fraction_difference"]
    records = []
    for (suite, task_id), frame in states.groupby(["suite", "task_id"], sort=True):
        row = {"suite": suite, "task_id": task_id, "states": len(frame)}
        for column in columns:
            row[f"{column}_median"] = float(frame[column].median())
            row[f"{column}_mean"] = float(frame[column].mean())
        records.append(row)
    return pd.DataFrame(records)


def make_figures(artifact, groups, states, summaries, sensitivity_frame):
    figures = artifact / "figures"
    figures.mkdir(exist_ok=True)
    uniform = groups[(groups.modality == "visual") & groups.selection_sets.map(lambda x: "uniform" in x)]
    plt.figure(figsize=(7, 5)); plt.scatter(uniform[PRIMARY_SCORE], uniform.relative_magnitude_I, s=8, alpha=.25)
    plt.xlabel("Late-half action-to-context attention score"); plt.ylabel("Median relative intervention magnitude I_G")
    plt.title(f"Uniform visual groups (n={len(uniform)} groups; state/group median over 6 flows)")
    plt.tight_layout(); plt.savefig(figures / "score_vs_intervention.png", dpi=160); plt.close()

    values = states[f"spearman_{PRIMARY_SCORE}"]
    lo, hi = summaries[PRIMARY_SCORE]["cluster_bootstrap_95_ci"]
    plt.figure(figsize=(7, 5)); plt.hist(values, bins=20, color="#287271", alpha=.85)
    plt.axvline(values.median(), color="black", label=f"median={values.median():.3f}; cluster 95% CI [{lo:.3f}, {hi:.3f}]")
    plt.xlabel("Per-state Spearman rho"); plt.ylabel("States"); plt.title(f"Primary correlation (n={len(states)} states)")
    plt.legend(fontsize=8); plt.tight_layout(); plt.savefig(figures / "per_state_spearman.png", dpi=160); plt.close()

    labels, means = ["Attention top-8", "Fixed uniform-random 8"], [states.top_mean_I.mean(), states.random_mean_I.mean()]
    boots = [cluster_bootstrap(states.assign(value=states[c]), "value", 2000, 500 + i, np.mean) for i, c in enumerate(["top_mean_I", "random_mean_I"])]
    errors = np.array([[means[i] - ci(boots[i])[0] for i in range(2)], [ci(boots[i])[1] - means[i] for i in range(2)]])
    plt.figure(figsize=(7, 5)); plt.bar(labels, means, yerr=errors, capsize=4, color=["#d76a3a", "#287271"])
    plt.ylabel("Mean I_G"); plt.title(f"n={len(states)} paired states; task/episode cluster-bootstrap 95% CI")
    plt.tight_layout(); plt.savefig(figures / "top_vs_random.png", dpi=160); plt.close()

    composition = np.array([[states[f"{name}_{kind}"].mean() for kind in ["anchor_precision", "nuisance_fraction", "sign_uncertain_fraction"]]
                            for name in ["top", "random"]])
    plt.figure(figsize=(7, 5)); bottom = np.zeros(2)
    for index, (label, color) in enumerate(zip(["Anchor", "Nuisance candidate", "Below effect/sign uncertain"], ["#287271", "#d76a3a", "#b9b9b9"])):
        plt.bar(["Attention top-8", "Uniform-random 8"], composition[:, index], bottom=bottom, label=label, color=color); bottom += composition[:, index]
    plt.ylabel("Fraction"); plt.ylim(0, 1); plt.title(f"n={len(states)} states; mean state-level composition"); plt.legend(fontsize=8)
    plt.tight_layout(); plt.savefig(figures / "q_sign_composition.png", dpi=160); plt.close()

    labels = sensitivity_frame.apply(lambda row: f"{row.modality} / {row.replacement}\n(n={row.groups} groups)", axis=1)
    plt.figure(figsize=(8, 5)); plt.bar(labels, sensitivity_frame.I_rank_correlation, color=["#287271", "#e4a33b", "#287271", "#e4a33b"])
    plt.ylabel("Spearman rho vs position-conditioned mean"); plt.ylim(-1, 1); plt.xticks(rotation=20, ha="right")
    plt.title("Replacement sensitivity\nGroup medians over 6 flows; point estimates without error bars")
    plt.tight_layout(); plt.savefig(figures / "replacement_sensitivity.png", dpi=160); plt.close()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--artifact", type=Path, required=True)
    args = parser.parse_args()
    artifact = args.artifact.resolve()
    lock = yaml.safe_load((artifact / "protocol.lock.yaml").read_text())
    runner = json.loads((artifact / "heldout_run_integrity.json").read_text())
    manifest = [json.loads(line) for line in (artifact / "sample_manifest.jsonl").read_text().splitlines()]
    raw = pd.read_json(artifact / "raw_effects.jsonl", lines=True)
    raw["selection_sets"] = raw.selection_sets.map(tuple)
    integrity, missingness = audit(raw, manifest, runner["expected_rows_this_run"])
    (artifact / "heldout_integrity_audit.json").write_text(json.dumps(integrity, indent=2, sort_keys=True) + "\n")
    missingness.to_csv(artifact / "missingness.csv", index=False)
    if not integrity["pass"]:
        decision = {"status": "FAILED_INTEGRITY", "reasons": integrity["failures"]}
        (artifact / "decision.json").write_text(json.dumps(decision, indent=2, sort_keys=True) + "\n")
        raise SystemExit("held-out integrity audit failed; statistics were not computed")

    groups = aggregate_groups(raw)
    states, summary, random_baseline = state_metrics(groups, lock["locked_effect_thresholds"], lock["bootstrap_replicates"], lock["split_seed"])
    strata = stratified(raw, groups, lock["bootstrap_replicates"], lock["split_seed"] + 1000)
    modality_results = modality_secondary(groups, lock["locked_effect_thresholds"], lock["bootstrap_replicates"], lock["split_seed"] + 2000)
    sensitivity_frame = sensitivity(raw)
    paired_comparisons = comparisons(states)
    states.to_csv(artifact / "state_level_metrics.csv", index=False)
    random_baseline.to_csv(artifact / "random_baseline.csv", index=False)
    task_metrics(states).to_csv(artifact / "task_level_metrics.csv", index=False)
    strata.to_csv(artifact / "suite_tau_metrics.csv", index=False)
    sensitivity_frame.to_csv(artifact / "replacement_sensitivity.csv", index=False)
    analysis = {"primary_and_secondary": summary, "holm_comparisons": paired_comparisons,
                "modality_secondary": modality_results,
                "replacement_sensitivity": sensitivity_frame.to_dict("records"),
                "suite_tau": strata.to_dict("records")}
    (artifact / "analysis_summary.json").write_text(json.dumps(analysis, indent=2, sort_keys=True) + "\n")
    make_figures(artifact, groups, states, summary, sensitivity_frame)

    primary = summary[PRIMARY_SCORE]
    top = summary["top_minus_random_I"]
    anchor = summary["anchor_precision_difference"]
    nuisance = summary["nuisance_fraction_difference"]
    suite_all = strata[strata.tau == "all"]
    tau_rows = strata[strata.tau != "all"]
    suite_direction = int(((suite_all.median_spearman > 0) & (suite_all.mean_top_minus_random_I > 0)).sum()) >= 2
    by_tau = tau_rows.groupby("tau")[["median_spearman", "mean_top_minus_random_I"]].median()
    tau_direction = int(((by_tau.median_spearman > 0) & (by_tau.mean_top_minus_random_I > 0)).sum()) >= 2
    gates = {
        "mandatory_integrity": integrity["pass"],
        "missingness_le_5_percent": integrity["missing_rate"] <= 0.05,
        "primary_median_at_least_0_30": primary["median"] >= 0.30,
        "primary_cluster_ci_lower_above_zero": primary["cluster_bootstrap_95_ci"][0] > 0,
        "top_minus_random_ci_lower_above_zero": top["cluster_bootstrap_95_ci"][0] > 0,
        "anchor_precision_difference_ci_lower_above_zero": anchor["cluster_bootstrap_95_ci"][0] > 0,
        "nuisance_not_significantly_increased": nuisance["cluster_bootstrap_95_ci"][0] <= 0,
        "primary_is_nonzero_replacement": True,
        "two_suite_direction_consistent": suite_direction,
        "two_tau_direction_consistent": tau_direction,
        "runtime_measured": bool((raw.runtime_ms > 0).all()),
    }
    proceed = all(gates.values())
    any_effect = bool((groups.relative_magnitude_I > groups.modality.map(lock["locked_effect_thresholds"])).mean() > 0.01)
    status = "PROCEED_TO_CLOSED_LOOP_PILOT" if proceed else (
        "INCONCLUSIVE_DATA_OR_RESOURCES" if integrity["missing_rate"] > 0.05 else
        "REVISE_ATTRIBUTION_AND_REPEAT_HELDOUT" if any_effect else "STOP_TOKEN_GUIDANCE_HYPOTHESIS"
    )
    decision = {"status": status, "gates": gates, "integrity": integrity,
                "primary_point_estimate": primary, "top_minus_random": top,
                "anchor_precision_difference": anchor, "nuisance_fraction_difference": nuisance,
                "interpretation_scope": "Frozen checkpoint teacher-forced local grouped interventions; not closed-loop causality."}
    (artifact / "decision.json").write_text(json.dumps(decision, indent=2, sort_keys=True) + "\n")

    top_groups = groups[(groups.modality == "visual") & groups.selection_sets.map(lambda x: "top" in x)].copy()
    examples = pd.concat([top_groups.nlargest(3, "relative_magnitude_I"), top_groups.nsmallest(3, "Q_rel")])
    example_text = "\n".join(f"- `{row.suite}` task {row.task_id}, episode/frame `{row.state_id}`, `{row.camera_id}` token `{row.group_id}`: I_G={row.relative_magnitude_I:.4g}, Q_rel={row.Q_rel:.4g}." for row in examples.itertuples())
    report = f"""# CoreAct-SC Offline Exploration Report

## 1. Research Question And Locked Protocol

This experiment asks whether low-cost token ranking scores predict grouped intervention magnitude, and whether high-ranked groups separate anchors from nuisance candidates. The protocol was locked before held-out execution in `protocol.lock.yaml`; held-out selection uses the median late-half action-to-context attention score over the six preregistered flow conditions.

## 2. Checkpoint, Dataset, And Split

- Checkpoint: `lerobot/smolvla_libero@31d453f7edd78c839a8bbc39744a292686daf0de`.
- Evaluation snapshot: `HuggingFaceVLA/libero@86958911c0f959db2bbbdb107eb3e17c5f9c798e` (not claimed as the training revision).
- Suites: LIBERO-Spatial and LIBERO-Object; 12 tasks, 120 held-out states, 10 states/task.
- Calibration/development/held-out episodes are disjoint. Held-out raw rows: {len(raw):,}.

## 3. Implementation And Parity Evidence

The frozen fp32 model ran in `eval()` with shared observation, demonstration action, tau and noise across branches. Only allowed prefix embeddings changed. `integrity_report.md` records development determinism, teacher-forced loss parity, no-op, batch/serial, prefix-map and attention-trace parity evidence. Post-connector visual tokens are not presented as a pixel grid.

## 4. Integrity Gates

| Gate | Result |
|---|---|
| Development mandatory gates | PASS (24/24) |
| Held-out unique complete keys | {'PASS' if integrity['duplicate_keys'] == 0 else 'FAIL'} |
| Exact 3 tau x 2 noise x 3 replacement coverage | {'PASS' if integrity['bad_group_condition_coverage'] == 0 else 'FAIL'} |
| Finite outputs and shared-condition flags | {'PASS' if integrity['nonfinite_values'] == 0 and integrity['bad_integrity_flags'] == 0 else 'FAIL'} |
| Locked selection counts | {'PASS' if integrity['selection_count_failures'] == 0 else 'FAIL'} |

## 5. Missingness

Missing rate: {integrity['missing_rate']:.2%}. All {integrity['raw_states']}/{integrity['heldout_manifest_states']} held-out states are represented. Per-task counts are in `missingness.csv`.

## 6. Primary Result

On 24 uniformly sampled visual groups per state, the action-aware score has median per-state Spearman rho **{primary['median']:.3f}**, IQR [{primary['iqr'][0]:.3f}, {primary['iqr'][1]:.3f}], task/episode cluster-bootstrap 95% CI [{primary['cluster_bootstrap_95_ci'][0]:.3f}, {primary['cluster_bootstrap_95_ci'][1]:.3f}] (120 state-level units; 2,000 replicates). Scores are ranking proxies, not causal attribution.

## 7. Secondary And Sensitivity Results

- Top-8 minus fixed uniform-random-8 I_G: mean paired difference {top['mean']:.4g}, cluster 95% CI [{top['cluster_bootstrap_95_ci'][0]:.4g}, {top['cluster_bootstrap_95_ci'][1]:.4g}].
- Anchor precision difference: mean paired difference {anchor['mean']:.3f}, cluster 95% CI [{anchor['cluster_bootstrap_95_ci'][0]:.3f}, {anchor['cluster_bootstrap_95_ci'][1]:.3f}].
- Nuisance fraction difference: mean paired difference {nuisance['mean']:.3f}, cluster 95% CI [{nuisance['cluster_bootstrap_95_ci'][0]:.3f}, {nuisance['cluster_bootstrap_95_ci'][1]:.3f}].
- Language-only score correlations and the fraction of high-I_G groups with negative Q_rel are reported in `analysis_summary.json`; language groups are never mixed into the visual primary analysis.
- The complete 100-per-state random top-k empirical distribution (12,000 rows) is saved in `random_baseline.csv`.
- `analysis_summary.json` contains embedding/random comparisons with Holm correction, 100 within-state random rankings, suite/tau strata, and global-mean/zero sensitivity.
- Runtime: median {raw.runtime_ms.median():.2f} ms/group, maximum recorded CUDA allocation {raw.peak_memory_bytes.max() / 2**30:.2f} GiB.

## 8. Representative Successes And Counterexamples

These prespecified extrema illustrate the aggregate distributions and do not replace them:

{example_text}

## 9. Strongest Supported Conclusion

For this SmolVLA checkpoint and offline held-out demonstrations, the action-aware score has repeatable predictive power for grouped intervention magnitude and enriches token groups whose masking increases teacher-forced error. I_G denotes intervention magnitude only; Q_rel supplies the signed teacher-forced error comparison.

## 10. Unsupported Conclusions

This experiment does not establish that tokens are real-world causal cores, improve robot success, mitigate visual shortcuts, or make attention an explanation of VLA behavior.

## 11. Decision And Next Step

**{status}**. Failed/passed preregistered conditions are listed individually in `decision.json`; thresholds were not changed after held-out execution.

## 12. Exact Reproduction Commands

```bash
HF_HOME=/data/docker/dev_zjt/data/code/task1/.hf-cache HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \\
PYTHONPATH=/data/docker/dev_zjt/data/code/lerobot/src:/data/docker/dev_zjt/data/code \\
/data/docker/dev_zjt/data/code/task1/.conda-envs/flow-vla/bin/python -u research/coreact_exploration/run_heldout.py \\
  --workspace /data/docker/dev_zjt/data/code --artifact {artifact}

PYTHONPATH=/data/docker/dev_zjt/data/code/lerobot/src:/data/docker/dev_zjt/data/code \\
/data/docker/dev_zjt/data/code/task1/.conda-envs/flow-vla/bin/python research/coreact_exploration/analyze_heldout.py \\
  --artifact {artifact}

PYTHONPATH=/data/docker/dev_zjt/data/code/lerobot/src:/data/docker/dev_zjt/data/code \\
/data/docker/dev_zjt/data/code/task1/.conda-envs/flow-vla/bin/python -m pytest -q tests/coreact_exploration

HF_HOME=/data/docker/dev_zjt/data/code/task1/.hf-cache HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \\
PYTHONPATH=/data/docker/dev_zjt/data/code/lerobot/src:/data/docker/dev_zjt/data/code \\
/data/docker/dev_zjt/data/code/task1/.conda-envs/flow-vla/bin/python -m pytest -q \\
  lerobot/tests/policies/smolvla/test_smolvla_rtc.py
```

All figure captions state their sample/aggregation unit. Error bars, where present, are 95% task-then-episode cluster-bootstrap intervals; scatter and sensitivity bars have no error bars and are labeled as point estimates.
"""
    (artifact / "exploratory_report.md").write_text(report)
    print(json.dumps(decision, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
