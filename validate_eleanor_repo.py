import os
from pathlib import Path

# ------------------------------------------------------------
# CONFIGURATION — update this if your repo root path changes
# ------------------------------------------------------------
REPO_ROOT = Path(os.getcwd())  # or hardcode your repo folder


REQUIRED_TREE = {
    "engine": {
        "critics": {
            "rights.py": None,
            "risk.py": None,
            "fairness.py": None,
            "pragmatics.py": None,
            "truth.py": None,
            "base.py": None,
            "schemas": {
                "rights.yaml": None,
                "risk.yaml": None,
                "fairness.yaml": None,
                "pragmatics.yaml": None,
                "truth.yaml": None,
            },
            "prompts.yaml": None,
        },
        "detectors": {
            "base.py": None,
            "signals.py": None,
            "engine.py": None,
        },
        "recorder": {
            "__init__.py": None,
            "evidence_recorder.py": None,
            "db_sink.py": None,
        },
        "orchestrator": "*",  # allow anything
        "router": "*",  # allow anything
        "engine.py": None,
    }
}

# Detector domains (should exist as folders under engine/detectors)
DETECTOR_DOMAINS = [
    "discrimination",
    "autonomy",
    "privacy",
    "coercion",
    "dehumanization",
    "irreversible_harm",
    "physical_safety",
    "psychological_harm",
    "operational_risk",
    "cascading_failure",
    "disparate_treatment",
    "disparate_impact",
    "procedural_fairness",
    "embedding_bias",
    "structural_disadvantage",
    "factual_accuracy",
    "omission",
    "contradiction",
    "hallucination",
    "evidence_grounding",
    "feasibility",
    "resource_burden",
    "time_constraints",
    "environmental_impact",
    "cascading_pragmatic_failure",
]


def validate_tree(base: Path, required: dict, errors: list, prefix=""):
    """Recursively validate folder structure."""
    for name, subtree in required.items():
        target = base / name

        if subtree is None:
            # required file
            if not target.exists():
                errors.append(f"❌ Missing file: {prefix}{name}")
            else:
                print(f"✓ Found file: {prefix}{name}")
        elif subtree == "*":
            # wildcards — don't validate content
            if not target.exists():
                errors.append(f"❌ Missing directory: {prefix}{name}/")
            else:
                print(f"✓ Found directory: {prefix}{name}/ (contents not validated)")
        else:
            # required directory
            if not target.exists():
                errors.append(f"❌ Missing directory: {prefix}{name}/")
            else:
                print(f"✓ Found directory: {prefix}{name}/")
                validate_tree(target, subtree, errors, prefix + name + "/")


def validate_detectors(detectors_root: Path, errors: list):
    print("\n🔍 Validating detector domains...\n")

    for d in DETECTOR_DOMAINS:
        folder = detectors_root / d
        if not folder.exists():
            errors.append(f"❌ Missing detector folder: detectors/{d}/")
            continue

        detector_file = folder / "detector.py"
        if not detector_file.exists():
            errors.append(f"❌ detectors/{d}/ is missing detector.py")
        else:
            print(f"✓ detectors/{d}/detector.py present")


def main():
    print("\n=== ELEANOR V8 REPO STRUCTURE VALIDATOR ===\n")
    errors = []

    # Validate required tree
    validate_tree(REPO_ROOT, REQUIRED_TREE, errors)

    # Validate detector domain folders
    detectors_root = REPO_ROOT / "engine" / "detectors"
    if detectors_root.exists():
        validate_detectors(detectors_root, errors)
    else:
        errors.append("❌ Missing detectors root directory: engine/detectors/")

    # Summary
    print("\n===========================================")
    if errors:
        print("\n❌ VALIDATION FAILED — Issues found:\n")
        for e in errors:
            print("  - " + e)
        print("\nFix the issues above and re-run the validator.\n")
    else:
        print("\n✅ ALL GOOD — ELEANOR V8 structure is valid!\n")


if __name__ == "__main__":
    main()
