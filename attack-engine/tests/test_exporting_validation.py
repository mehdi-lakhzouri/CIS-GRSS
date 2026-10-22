from __future__ import annotations

from attack_engine.exporting.builder import (
    assign_row_partitions,
    build_attacked_target,
    build_audit_dataset,
    build_detector_input,
    build_ground_truth,
)
from attack_engine.exporting.contracts import default_detector_context_columns
from attack_engine.exporting.manifest import build_attack_manifest
from attack_engine.exporting.validation import require_valid, run_consistency_checks
from attack_engine.scheduling.config import FamilyParameters, SchedulingConfig
from attack_engine.scheduling.models import AttackFamily
from attack_engine.scheduling.scheduler import AttackWindowScheduler


def _scheduling_config() -> SchedulingConfig:
    return SchedulingConfig(
        global_seed=14,
        attack_budget=0.3,
        budget_scope=("validation",),
        enabled_families=(AttackFamily.BIAS,),
        family_parameters={AttackFamily.BIAS: FamilyParameters(weight=1.0)},
        min_window_length=2,
        max_window_length=6,
        max_attempts=500,
    )


def test_consistency_checks_pass_on_a_valid_export(schema, partition_plan, make_observations):
    frame = make_observations(hours=72)
    scheduling_config = _scheduling_config()
    plan = AttackWindowScheduler(schema, partition_plan, scheduling_config).plan(frame)

    from attack_engine.attacks.config import AttackExecutionConfig, BiasParameters
    from attack_engine.attacks.executor import AttackExecutor

    attack_config = AttackExecutionConfig(
        global_seed=14, quantum=0.001, bias=BiasParameters(magnitude=0.05)
    )
    summary = AttackExecutor(schema, partition_plan, attack_config).execute(frame, plan)

    attacked = build_attacked_target(frame, schema, summary)
    partitions = assign_row_partitions(frame, schema, partition_plan)
    audit = build_audit_dataset(frame, schema, partition_plan, attacked, partitions)
    ground_truth = build_ground_truth(frame, schema, attacked, partitions)
    context_columns = default_detector_context_columns(schema)
    detector_input = build_detector_input(frame, schema, attacked, partitions, context_columns)
    manifest = build_attack_manifest(plan, summary)

    checks = run_consistency_checks(
        frame, schema, summary, audit, ground_truth, detector_input, manifest
    )

    require_valid(checks)
    assert all(check.passed for check in checks)


def test_manifest_reconciliation_fails_when_label_dropped(
    schema, partition_plan, make_observations
):
    frame = make_observations(hours=72)
    scheduling_config = _scheduling_config()
    plan = AttackWindowScheduler(schema, partition_plan, scheduling_config).plan(frame)

    from attack_engine.attacks.config import AttackExecutionConfig, BiasParameters
    from attack_engine.attacks.executor import AttackExecutor

    attack_config = AttackExecutionConfig(
        global_seed=14, quantum=0.001, bias=BiasParameters(magnitude=0.05)
    )
    summary = AttackExecutor(schema, partition_plan, attack_config).execute(frame, plan)
    assert summary.committed_total > 0, "fixture must actually modify something"

    attacked = build_attacked_target(frame, schema, summary)
    partitions = assign_row_partitions(frame, schema, partition_plan)
    audit = build_audit_dataset(frame, schema, partition_plan, attacked, partitions)
    ground_truth = build_ground_truth(frame, schema, attacked, partitions)
    context_columns = default_detector_context_columns(schema)
    detector_input = build_detector_input(frame, schema, attacked, partitions, context_columns)
    manifest = build_attack_manifest(plan, summary)

    # Tamper with one positive label so the manifest no longer reconciles.
    first_attacked_index = ground_truth.index[ground_truth["is_attacked"]][0]
    ground_truth.loc[first_attacked_index, "is_attacked"] = False

    checks = run_consistency_checks(
        frame, schema, summary, audit, ground_truth, detector_input, manifest
    )
    failing_codes = {check.code for check in checks if not check.passed}
    assert "manifest_reconciles_with_labels" in failing_codes or "no_overlapping_modifications" in (
        failing_codes
    )
