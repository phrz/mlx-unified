# Copyright © 2026 Apple Inc.
#
# mlx-unified: coverage tripwire against lmstudio-ai/mlx-engine (LM Studio).
#
# mlx-engine's support surface for architectures that need more than stock mlx-lm is
# small and enumerable from its source tree:
#
#   1. mlx_engine/model_kit/patches/<arch>.py — per-arch patches (vision text-side
#      semantics, tokenizer fixes) keyed by checkpoint model_type.
#   2. mlx_engine/external/models/<arch>/ — vendored model implementations mlx-lm
#      lacks upstream.
#
# This test discovers both lists from a clone of mlx-engine and asserts that every
# architecture LM Studio special-cases resolves in THIS fork:
#
#   - the checkpoint model_type(s) it serves import as an mlx_lm.models module via
#     the same MODEL_REMAPPING/_get_classes path mlx_lm.utils.load uses, and
#   - vision archs have the expected mlx_lm.multimodal TEXT_SIDE capability (or are
#     deliberately on the plain-injection default).
#
# The point: when mlx-engine grows a patch or vendored model for a new architecture,
# test_every_engine_arch_is_mapped starts failing and its message says exactly what
# appeared and what to add here. Update ENGINE_ARCHES (and the fork) to make it green.
#
# Discovery uses a local clone at /tmp/mlx-engine when present, else a shallow
# network clone; if the network fetch fails the whole class SKIPS (never hard-fail
# CI on network).

import importlib
import shutil
import subprocess
import tempfile
import unittest
from pathlib import Path

from mlx_lm.multimodal import TEXT_SIDE
from mlx_lm.utils import MODEL_REMAPPING

LOCAL_CLONE = Path("/tmp/mlx-engine")
ENGINE_REPO_URL = "https://github.com/lmstudio-ai/mlx-engine"
PATCHES_SUBDIR = Path("mlx_engine/model_kit/patches")
EXTERNAL_SUBDIR = Path("mlx_engine/external/models")

# mlx-engine arch name -> checkpoint model_type(s) it serves (what appears as
# "model_type" in config.json, which is also what MODEL_REMAPPING keys on).
# Keys are patch module stems and external/models directory names; they usually
# match the model_type but not always (e.g. the ernie_4_5 patch covers both the
# text and the vision-MoE checkpoints). A newly discovered arch must be added
# here AND supported in the fork before this suite goes green again.
ENGINE_ARCHES = {
    # --- per-arch patches: mlx_engine/model_kit/patches/<stem>.py ---
    "ernie_4_5": ("ernie4_5", "ernie4_5_moe_vl"),
    "gemma3n": ("gemma3n",),
    "gemma4": ("gemma4", "gemma4_unified"),
    "lfm2_vl": ("lfm2-vl",),
    "qwen3_5": ("qwen3_5", "qwen3_5_moe"),
    # --- vendored archs: mlx_engine/external/models/<dir>/ ---
    "ernie4_5": ("ernie4_5",),
    "ernie4_5_moe": ("ernie4_5_moe", "ernie4_5_moe_vl"),
}

# checkpoint model_type -> the mlx_lm.multimodal TEXT_SIDE capability this fork is
# expected to register for it. "plain" = the plain-injection default is the intended
# path (no TEXT_SIDE entry needed). None = text-only arch, no vision assertion
# (mlx-engine's ernie_4_5 patch is a tokenizer/grammar fix, not a vision one, so the
# plain text checkpoints carry no vision expectation).
EXPECTED_TEXT_SIDE = {
    "ernie4_5": None,
    "ernie4_5_moe": None,
    "ernie4_5_moe_vl": "ernie-visual",
    "gemma3n": "gemma3n-visual",
    "gemma4": "gemma-visual",
    "gemma4_unified": "gemma-visual",
    "lfm2-vl": "plain",
    "qwen3_5": "mrope",
    "qwen3_5_moe": "mrope",
}


class TestMlxEngineCoverage(unittest.TestCase):
    engine_root: Path
    _tmpdir: str = ""

    @classmethod
    def setUpClass(cls):
        if (LOCAL_CLONE / PATCHES_SUBDIR).is_dir():
            cls.engine_root = LOCAL_CLONE
            return
        cls._tmpdir = tempfile.mkdtemp(prefix="mlx-engine-")
        dst = str(Path(cls._tmpdir) / "mlx-engine")
        try:
            proc = subprocess.run(
                ["git", "clone", "--depth", "1", ENGINE_REPO_URL, dst],
                capture_output=True,
                text=True,
                timeout=180,
            )
        except (OSError, subprocess.TimeoutExpired) as e:
            raise unittest.SkipTest(f"could not clone mlx-engine: {e}")
        if proc.returncode != 0:
            raise unittest.SkipTest(
                f"could not clone mlx-engine (network?): {proc.stderr.strip()[:500]}"
            )
        cls.engine_root = Path(dst)

    @classmethod
    def tearDownClass(cls):
        if cls._tmpdir:
            shutil.rmtree(cls._tmpdir, ignore_errors=True)

    # --- discovery -------------------------------------------------------------

    def _engine_dir(self, subdir: Path) -> Path:
        path = self.engine_root / subdir
        # A missing directory in a successful clone means mlx-engine restructured;
        # fail loudly — the discovery paths above need updating.
        self.assertTrue(
            path.is_dir(),
            f"mlx-engine layout changed: {subdir} not found under {self.engine_root}",
        )
        return path

    def discovered_arches(self) -> set:
        patches = {
            p.stem
            for p in self._engine_dir(PATCHES_SUBDIR).glob("*.py")
            if not p.stem.startswith("_")
        }
        external = {
            d.name
            for d in self._engine_dir(EXTERNAL_SUBDIR).iterdir()
            if d.is_dir() and not d.name.startswith("_")
        }
        return patches | external

    def mapped_model_types(self) -> set:
        discovered = self.discovered_arches()
        return {
            mt for arch in discovered & set(ENGINE_ARCHES) for mt in ENGINE_ARCHES[arch]
        }

    # --- assertions ------------------------------------------------------------

    def test_every_engine_arch_is_mapped(self):
        unmapped = sorted(self.discovered_arches() - set(ENGINE_ARCHES))
        self.assertEqual(
            unmapped,
            [],
            "mlx-engine special-cases architectures this fork has no mapping for: "
            f"{unmapped}. For each: read the patch/vendored model to find the "
            "checkpoint model_type(s), add support in this fork (mlx_lm/models "
            "module, MODEL_REMAPPING and/or multimodal TEXT_SIDE as needed), then "
            "record it in ENGINE_ARCHES and EXPECTED_TEXT_SIDE in this test.",
        )

    def test_mapped_model_types_resolve_in_fork(self):
        # Mirrors mlx_lm.utils._get_classes: remap the checkpoint model_type, then
        # import mlx_lm.models.<name> and require the Model/ModelArgs contract.
        for model_type in sorted(self.mapped_model_types()):
            with self.subTest(model_type=model_type):
                module_name = MODEL_REMAPPING.get(model_type, model_type)
                try:
                    module = importlib.import_module(f"mlx_lm.models.{module_name}")
                except ImportError as e:
                    self.fail(
                        f"model_type {model_type!r} (module {module_name!r}) does "
                        f"not resolve in this fork: {e}"
                    )
                self.assertTrue(hasattr(module, "Model"), module_name)
                self.assertTrue(hasattr(module, "ModelArgs"), module_name)

    def test_vision_arches_have_expected_text_side(self):
        for model_type in sorted(self.mapped_model_types()):
            with self.subTest(model_type=model_type):
                self.assertIn(
                    model_type,
                    EXPECTED_TEXT_SIDE,
                    f"model_type {model_type!r} is mapped in ENGINE_ARCHES but has "
                    "no EXPECTED_TEXT_SIDE entry — declare its vision expectation "
                    '(a TEXT_SIDE capability, "plain", or None for text-only).',
                )
                expected = EXPECTED_TEXT_SIDE[model_type]
                if expected is None:
                    continue
                self.assertEqual(
                    TEXT_SIDE.get(model_type, "plain"),
                    expected,
                    f"vision arch {model_type!r}: this fork's TEXT_SIDE disagrees "
                    "with the expectation recorded here.",
                )


if __name__ == "__main__":
    unittest.main()
