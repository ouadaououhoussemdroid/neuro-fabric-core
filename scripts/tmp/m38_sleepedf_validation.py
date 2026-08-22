#!/usr/bin/env python3
"""
M38 — Sleep-EDF Dataset Loader Validation Script

Validates the Tier-2 Sleep-EDF dataset loader for Joint-2312 compatibility:
  1. Dataset loader implemented (source, license, metadata)
  2. Channel expansion 7→62 (nearest-neighbour spatial interpolation)
  3. Sleep stage annotation parsing (5-stage: W, N1, N2, N3, REM)
  4. Sample rate resampling (100/128 Hz → 250 Hz)
  5. Window segmentation (4-second windows for Joint-2312)
  6. Manifest registration (KNOWN_DATASETS)
  7. Barrel export (datasets/index.ts)
  8. Unit test coverage

Usage:
    python scripts/tmp/m38_sleepedf_validation.py
"""
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

CHECKS_PASSED = 0
CHECKS_FAILED = 0
CHECKS_TOTAL = 0


def check(condition: bool, label: str) -> None:
    global CHECKS_PASSED, CHECKS_FAILED, CHECKS_TOTAL
    CHECKS_TOTAL += 1
    if condition:
        CHECKS_PASSED += 1
        print(f"  ✓ {label}")
    else:
        CHECKS_FAILED += 1
        print(f"  ✗ {label}")


def section(title: str) -> None:
    print(f"\n{'='*60}")
    print(f"  {title}")
    print(f"{'='*60}")


def file_exists(path: str) -> bool:
    return os.path.isfile(os.path.join(ROOT, path))


def file_contains(path: str, needle: str) -> bool:
    full = os.path.join(ROOT, path)
    if not os.path.isfile(full):
        return False
    with open(full, encoding="utf-8") as f:
        return needle in f.read()


section("CHECK 1: Dataset Loader Implementation")

check(file_exists("src/lib/datasets/sleep-edf.ts"), "Sleep-EDF loader file exists")
check(file_contains("src/lib/datasets/sleep-edf.ts", "preprocessSleepEDF"), "preprocessSleepEDF export")
check(file_contains("src/lib/datasets/sleep-edf.ts", "expandSleepToEEGPT"), "expandSleepToEEGPT export")
check(file_contains("src/lib/datasets/sleep-edf.ts", "parseSleepAnnotations"), "parseSleepAnnotations export")
check(file_contains("src/lib/datasets/sleep-edf.ts", "parseEDF"), "uses existing parseEDF")
check(file_contains("src/lib/datasets/sleep-edf.ts", "resampleSignal"), "uses resampleSignal")
check(file_contains("src/lib/datasets/sleep-edf.ts", "selectCbraModChannels"), "selects CBraMod channels")
check(file_contains("src/lib/datasets/sleep-edf.ts", "selectProdChannels"), "selects V2/Prod channels")
check(file_contains("src/lib/datasets/sleep-edf.ts", "selectEEGPTChannels"), "selects EEGPT channels")

section("CHECK 2: Dataset Metadata")

check(file_contains("src/lib/datasets/sleep-edf.ts", "name: \"Sleep-EDF\""), "name = Sleep-EDF")
check(file_contains("src/lib/datasets/sleep-edf.ts", "license: \"BSD-3-Clause\""), "license = BSD-3-Clause")
check(file_contains("src/lib/datasets/sleep-edf.ts", "nSubjects: 99"), "nSubjects = 99")
check(file_contains("src/lib/datasets/sleep-edf.ts", "nChannels: 7"), "nChannels = 7")
check(file_contains("src/lib/datasets/sleep-edf.ts", "sampleRate: 100"), "sampleRate = 100 Hz")
check(file_contains("src/lib/datasets/sleep-edf.ts", "nClasses: 5"), "nClasses = 5")
check(file_contains("src/lib/datasets/sleep-edf.ts", "BSD-3-Clause"), "references BSD-3-Clause license")

section("CHECK 3: Sleep Stage Labels (5-stage)")

sleep_stages = ["W", "N1", "N2", "N3", "REM"]
for stage in sleep_stages:
    check(file_contains("src/lib/datasets/sleep-edf.ts", f'"{stage}"'), f"stage label '{stage}' present")

check(file_contains("src/lib/datasets/sleep-edf.ts", "SleepStage"), "SleepStage type defined")
check(file_contains("src/lib/datasets/sleep-edf.ts", "SLEEP_STAGES"), "SLEEP_STAGES constant")
check(file_contains("src/lib/datasets/sleep-edf.ts", "SLEEP_STAGE_ID_TO_LABEL"), "stage ID→label map")
check(file_contains("src/lib/datasets/sleep-edf.ts", "SLEEP_STAGE_LABEL_TO_ID"), "stage label→ID map")

section("CHECK 4: Channel Expansion (7→62)")

check(file_contains("src/lib/datasets/sleep-edf.ts", "SLEEP_CHANNEL_TO_EEGPT_NEIGHBOUR"), "neighbour map defined")
check(file_contains("src/lib/datasets/sleep-edf.ts", "EEGPT_CHANNELS_62"), "references EEGPT_CHANNELS_62")
check(file_contains("src/lib/datasets/sleep-edf.ts", "7→62"), "documents 7→62 expansion")
check(file_contains("src/lib/datasets/sleep-edf.ts", "nearest-neighbour"), "uses nearest-neighbour interpolation")

# Verify all 62 EEGPT channels have a nearest-neighbour mapping
eegpt_channels = [
    "FP1", "FPZ", "FP2", "AF7", "AF3", "AF4", "AF8", "F7", "F5", "F3", "F1",
    "FZ", "F2", "F4", "F6", "F8", "FT7", "FC5", "FC3", "FC1", "FCZ", "FC2",
    "FC4", "FC6", "FT8", "T7", "C5", "C3", "C1", "CZ", "C2", "C4", "C6", "T8",
    "TP7", "CP5", "CP3", "CP1", "CPZ", "CP2", "CP4", "CP6", "TP8",
    "P7", "P5", "P3", "P1", "PZ", "P2", "P4", "P6", "P8",
    "PO7", "PO5", "PO3", "POZ", "PO4", "PO6", "PO8",
    "O1", "OZ", "O2",
]

sleep_loader = os.path.join(ROOT, "src/lib/datasets/sleep-edf.ts")
with open(sleep_loader, encoding="utf-8") as f:
    loader_content = f.read()

for ch in eegpt_channels:
    check(f"{ch}:" in loader_content, f"neighbour mapping for {ch}")

section("CHECK 5: Resampling & Preprocessing")

check(file_contains("src/lib/datasets/sleep-edf.ts", "JOINT_2312_SAMPLE_RATE_HZ = 250"), "target rate = 250 Hz")
check(file_contains("src/lib/datasets/sleep-edf.ts", "JOINT_WINDOW_SEC = 4"), "window = 4 seconds")
check(file_contains("src/lib/datasets/sleep-edf.ts", "SLEEP_EPOCH_SEC = 30"), "epoch = 30 seconds")
check(file_contains("src/lib/datasets/sleep-edf.ts", "preprocessSleepEDF"), "full pipeline function")
check(file_contains("src/lib/datasets/sleep-edf.ts", "window count mismatch"), "alignment verification (error guard)")

section("CHECK 6: Manifest Registration")

check(file_contains("src/lib/datasets/manifest.ts", "Sleep-EDF"), "Sleep-EDF in KNOWN_DATASETS")
check(file_contains("src/lib/datasets/manifest.ts", "M38"), "M38 comment marker in manifest")

section("CHECK 7: Barrel Export")

check(file_exists("src/lib/datasets/index.ts"), "datasets barrel exists")
check(file_contains("src/lib/datasets/index.ts", "sleep-edf"), "barrel re-exports sleep-edf")
check(file_contains("src/lib/datasets/index.ts", "sleepEDFDataset"), "exports sleepEDFDataset")
check(file_contains("src/lib/datasets/index.ts", "preprocessSleepEDF"), "exports preprocessSleepEDF")

section("CHECK 8: Unit Tests")

check(file_exists("src/lib/datasets/__tests__/sleep-edf.test.ts"), "test file exists")
check(file_contains("src/lib/datasets/__tests__/sleep-edf.test.ts", "describe"), "has test suite")
tests_count = loader_test_content.count("it(") if (loader_test_content := open(
    os.path.join(ROOT, "src/lib/datasets/__tests__/sleep-edf.test.ts"), encoding="utf-8").read()
) else 0
check(tests_count >= 15, f"15+ test cases ({tests_count} found)")

section("CHECK 9: Integration with Joint-2312 Pipeline")

check(file_contains("src/lib/datasets/sleep-edf.ts", "SleepEEGPreprocessResult"), "returns aligned result type")
check(file_contains("src/lib/datasets/sleep-edf.ts", "cbramodWindows"), "returns CBraMod windows")
check(file_contains("src/lib/datasets/sleep-edf.ts", "v2Windows"), "returns V2 windows")
check(file_contains("src/lib/datasets/sleep-edf.ts", "eegptWindows"), "returns EEGPT windows")

# ─── Final Summary ──────────────────────────────────────────────────────────────

print(f"\n{'='*60}")
print(f"  M38 Validation: {CHECKS_PASSED}/{CHECKS_TOTAL} checks passed, {CHECKS_FAILED} failed")
print(f"{'='*60}")

if CHECKS_FAILED > 0:
    print(f"\n  ❌ {CHECKS_FAILED} checks FAILED — review before proceeding.")
    sys.exit(1)
else:
    print(f"\n  ✅ All {CHECKS_TOTAL} checks PASSED — M38 ready for archive.")
    sys.exit(0)
