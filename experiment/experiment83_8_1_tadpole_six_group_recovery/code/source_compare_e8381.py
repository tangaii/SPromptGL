"""Compare the recovery implementation with the sealed E83.8 source."""
from __future__ import annotations

import ast
import hashlib
import json
from pathlib import Path

import run_e8381 as runner


PARENT = Path("/model/spromptgl/experiment/experiment83_8_tadpole_aloss_pairmean_optimization")
CRITICAL_FUNCTIONS = (
    "paper_aloss", "pair_mean_graph_supervision", "historical_loss", "run_warmup",
    "active_supervision_kind", "train_fold", "dpsl_optimizer_step",
    "build_model", "make_optimizers", "apply_intermediate_weight_decay",
)
EXPECTED_AUDIT_ONLY_FUNCTION_DIFFERENCES = {
    "initial_warmup_identity", "make_warmup_payload", "identity", "execute_fold",
}


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def function_ast(path: Path, name: str) -> str | None:
    tree = ast.parse(path.read_text())
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name == name:
            return ast.dump(node, include_attributes=False)
    return None


def main() -> None:
    parent_runner = PARENT / "code/run_e838.py"
    active_runner = runner.CODE_ROOT / "run_e8381.py"
    copied_files = []
    for source in sorted((PARENT / "code").rglob("*")):
        if not source.is_file() or "__pycache__" in source.parts:
            continue
        relative = source.relative_to(PARENT / "code")
        destination = runner.CODE_ROOT / relative
        if not destination.is_file():
            copied_files.append({"path": str(relative), "status": "MISSING"})
            continue
        copied_files.append({"path": str(relative), "status":
                             "IDENTICAL" if sha(source) == sha(destination) else "DIFFERS",
                             "e838_sha256": sha(source), "e8381_sha256": sha(destination)})

    config_files = []
    for group in (f"X{index}" for index in range(1, 7)):
        parent_config = json.loads((PARENT / "configs" / f"{group}.json").read_text())
        current_config = json.loads((runner.ROOT / "configs" / f"{group}.json").read_text())
        parent_identity = parent_config.pop("experiment", None)
        current_identity = current_config.pop("experiment", None)
        config_files.append({
            "group": group,
            "status": "IDENTICAL_EXCEPT_EXPERIMENT_IDENTITY"
            if parent_config == current_config and current_identity == "Experiment 83.8.1"
            else "METHOD_OR_CONFIG_DIFFERENCE",
            "e838_experiment_identity": parent_identity,
            "e8381_experiment_identity": current_identity,
            "e838_sha256": sha(PARENT / "configs" / f"{group}.json"),
            "e8381_sha256": sha(runner.ROOT / "configs" / f"{group}.json"),
            "method_fields_equal": parent_config == current_config,
        })

    functions = {}
    for name in CRITICAL_FUNCTIONS + tuple(sorted(EXPECTED_AUDIT_ONLY_FUNCTION_DIFFERENCES)):
        left, right = function_ast(parent_runner, name), function_ast(active_runner, name)
        functions[name] = {
            "status": "AST_IDENTICAL" if left == right else
            "GROUP_GATE_CHANGE" if name == "execute_fold" else
            "AUDIT_IDENTITY_EXTENSION" if name in EXPECTED_AUDIT_ONLY_FUNCTION_DIFFERENCES
            else "DIFFERS",
            "e838_function_present": left is not None,
            "e8381_function_present": right is not None,
        }

    parent_model = PARENT / "code/common/fidelity_model.py"
    current_model = runner.CODE_ROOT / "common/fidelity_model.py"
    parent_runner_sha = sha(parent_runner)
    result = {
        "experiment": "Experiment 83.8.1",
        "parent_experiment": "Experiment 83.8",
        "parent_runner_sha256": parent_runner_sha,
        "recovery_runner_sha256": sha(active_runner),
        "historical_model_fidelity_sha256": sha(parent_model),
        "recovery_model_fidelity_sha256": sha(current_model),
        "fidelity_model_identical": sha(parent_model) == sha(current_model),
        "inherited_code_files": copied_files,
        "inherited_code_files_all_identical": all(item["status"] == "IDENTICAL"
                                                   for item in copied_files),
        "group_configurations": config_files,
        "group_methods_equal_parent": all(
            item["status"] == "IDENTICAL_EXCEPT_EXPERIMENT_IDENTITY" for item in config_files),
        "training_function_ast_comparison": functions,
        "training_critical_functions_identical": all(
            functions[name]["status"] == "AST_IDENTICAL" for name in CRITICAL_FUNCTIONS),
        "permitted_non_method_differences": [
            "per-group preflight policy and audit scripts",
            "execute_fold per-group gate call",
            "resource probe audit fields and experiment identity",
            "warmup state hash fields used only for paired audit",
            "E83.8.1 identity strings",
        ],
    }
    runner.write_json(runner.ROOT / "audits/source_comparison.json", result)
    print(json.dumps({
        "fidelity_model_identical": result["fidelity_model_identical"],
        "inherited_code_files_all_identical": result["inherited_code_files_all_identical"],
        "group_methods_equal_parent": result["group_methods_equal_parent"],
        "training_critical_functions_identical": result["training_critical_functions_identical"],
        "function_differences": {name: value["status"] for name, value in functions.items()
                                 if value["status"] != "AST_IDENTICAL"},
    }, indent=2))
    if not (result["fidelity_model_identical"] and result["group_methods_equal_parent"]
            and result["training_critical_functions_identical"]
            and result["inherited_code_files_all_identical"]):
        raise SystemExit("E83.8.1 contains an unregistered training-method/source difference")


if __name__ == "__main__":
    main()
