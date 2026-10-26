"""Tests for the motif-discovery subpackage (deepISA.discover).

These tests exercise only the Python side -- attribution, NPZ/H5 building,
hits parsing, motif extraction -- and never invoke the external ``modisco`` /
``finemo`` CLIs. They therefore run regardless of whether those binaries are
installed.
"""

import numpy as np
import pandas as pd
import pytest
import torch

from deepISA.model.cnn import Conv
from deepISA.score.mapper import map_motif_finemo
from deepISA.quickstart import QuickStart
import os
from pathlib import Path

from deepISA.discover import (
    build_finemo_db,
    build_finemo_input,
    compute_attribution,
    extract_motifs_from_group,
    load_hits_with_annotation,
    load_motifs,
    prepare_modisco_input,
    run_modisco,
    run_finemo_scan,
    run_motif_report,
    cwm_to_meme,
    read_attribution_h5,
    select_top_regions,
    drop_non_acgt_regions,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------
@pytest.fixture(scope="session")
def device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


@pytest.fixture(scope="session")
def tiny_model(device):
    cfg = {"seq_len": 60, "ks": [7, 5], "cs": [8, 8], "ds": [1, 2], "dropout": 0.0}
    m = Conv(mode="regression", model_config=cfg).to(device).eval()
    return m


def _rand_onehot(n, L, seed=0):
    """Random valid one-hot (N, 4, L) with exactly one base per position."""
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, 4, size=(n, L))
    ohe = np.eye(4, dtype=np.float32)[idx]            # (N, L, 4)
    return np.ascontiguousarray(np.transpose(ohe, (0, 2, 1)))  # (N, 4, L)


# ---------------------------------------------------------------------------
# attribution (tangermeme DeepLIFT-SHAP)
# ---------------------------------------------------------------------------
class TestAttribution:
    def test_shape_single_track(self, tiny_model, device):
        seqs = _rand_onehot(2, 60, seed=10)
        hyp, act = compute_attribution(
            tiny_model, seqs, tracks=[0], device=device,
            n_refs=4, batch_size=2, show_progress=False,
        )
        assert hyp.shape == (1, 2, 4, 60)
        assert act.shape == (1, 2, 4, 60)
        assert np.isfinite(hyp).all()

    def test_hyp_has_values_at_all_bases(self, tiny_model, device):
        """hypothetical scores must have non-trivial values at ALL 4 bases per
        position (not just the observed base). This is the defining property
        that makes them the correct ``-a`` input for tf-modisco-lite."""
        seqs = _rand_onehot(3, 60, seed=13)
        hyp, _ = compute_attribution(
            tiny_model, seqs, tracks=[0], device=device,
            n_refs=4, batch_size=8, show_progress=False,
        )
        n_nonzero = np.sum(np.abs(hyp[0]) > 1e-8, axis=1)   # (N, L) bases nonzero per pos
        assert n_nonzero.mean() >= 3.0, f"hyp not at all bases: mean={n_nonzero.mean()}"

    def test_act_is_zero_at_non_observed_bases(self, tiny_model, device):
        """actual scores = hyp * one-hot, so non-observed bases must be ~0.
        This confirms the act/hyp distinction is preserved (regression for an
        earlier bug where act was set to phi directly)."""
        seqs = _rand_onehot(3, 60, seed=14)
        _, act = compute_attribution(
            tiny_model, seqs, tracks=[0], device=device,
            n_refs=4, batch_size=8, show_progress=False,
        )
        observed = seqs > 0
        not_observed = ~observed
        # Non-observed positions should be near zero.
        assert np.max(np.abs(act[0][not_observed])) < 1e-5, "act nonzero at absent bases"

    def test_different_tracks_give_different_attributions(self, device):
        """Dual-mode models have distinct regression/classification heads; their
        attributions must differ. (Regression for the silent track-averaging bug.)
        Builds its own dual-output model so the test always runs, never skips."""
        cfg = {"seq_len": 60, "ks": [7, 5], "cs": [8, 8], "ds": [1, 2], "dropout": 0.0}
        model = Conv(mode="dual", model_config=cfg).to(device).eval()
        seqs = _rand_onehot(2, 60, seed=15)
        hyp, _ = compute_attribution(
            model, seqs, tracks=[0, 1], device=device,
            n_refs=4, batch_size=8, show_progress=False,
        )
        assert hyp.shape[0] == 2
        diff = np.max(np.abs(hyp[0] - hyp[1]))
        assert diff > 1e-4, f"tracks 0 and 1 identical (diff={diff})"

    def test_h5_sink_writes_channels_first_schema(self, tiny_model, device, tmp_path):
        import h5py
        h5_path = str(tmp_path / "attr.h5")
        seqs = _rand_onehot(3, 60, seed=12)
        compute_attribution(
            tiny_model, seqs, tracks=[0, 1] if _model_n_outputs(tiny_model) >= 2 else [0],
            device=device, n_refs=3, batch_size=8, save_h5_path=h5_path,
            ids=["a", "b", "c"], show_progress=False,
        )
        with h5py.File(h5_path, "r") as f:
            assert f.attrs["layout"] == b"channels_first" or f.attrs["layout"] == "channels_first"
            assert f["sequences"].shape == (3, 4, 60)
            assert f["hyp_scores"].ndim == 4
            assert "id" in f


def _model_n_outputs(model):
    with torch.no_grad():
        x = torch.zeros(1, 4, 60)
        y = model(x)
    return int(y.shape[-1]) if y.ndim > 1 else 1


# ---------------------------------------------------------------------------
# unknown bases (N) -- regression for the crash on real genomes:
# one_hot_encode maps N to all-zero columns and tangermeme then raises
# "ValueError: X must be one-hot encoded ... cannot have unknown characters".
# ---------------------------------------------------------------------------
class TestUnknownBaseImputation:
    def test_impute_is_valid_onehot_deterministic_and_nondestructive(self):
        from deepISA.discover.attribution import _impute_unknown_bases
        from deepISA.utils import one_hot_encode

        seqs = one_hot_encode(["ACGTN", "NNNNN", "ACGTA"])
        out1 = _impute_unknown_bases(seqs, seed=7)
        out2 = _impute_unknown_bases(seqs, seed=7)

        # every position is now exactly one base
        assert np.array_equal(out1.sum(axis=1), np.ones((3, 5), dtype=np.float32))
        # known bases are untouched (seqs "ACGTN" prefix and "ACGTA" entirely)
        assert np.array_equal(out1[0, :, :4], seqs[0, :, :4])
        assert np.array_equal(out1[2], seqs[2])
        # seeded -> deterministic
        assert np.array_equal(out1, out2)
        # caller's array is not modified
        assert (seqs.sum(axis=1) == 0).any()

    def test_compute_attribution_tolerates_unknown_bases(self, tiny_model, device, tmp_path):
        """End-to-end repro of the collaborator crash: sequences containing N
        must pass through attribution, and the saved H5 must hold strictly
        one-hot sequences (what modisco consumes as -s)."""
        import h5py
        from deepISA.utils import one_hot_encode

        rng = np.random.default_rng(0)
        seq_list = []
        for _ in range(3):
            s = ["ACGT"[i] for i in rng.integers(0, 4, size=60)]
            s[7] = "N"
            s[33] = "N"
            seq_list.append("".join(s))
        seqs = one_hot_encode(seq_list)
        assert (seqs.sum(axis=1) == 0).any(), "fixture must contain N columns"

        h5_path = str(tmp_path / "attr.h5")
        hyp, act = compute_attribution(
            tiny_model, seqs, tracks=[0], device=device,
            n_refs=3, batch_size=8, save_h5_path=h5_path, show_progress=False,
        )
        assert hyp.shape == (1, 3, 4, 60)
        assert np.isfinite(hyp).all()
        with h5py.File(h5_path, "r") as f:
            saved = f["sequences"][:]
        assert np.array_equal(saved.sum(axis=1), np.ones((3, 60), dtype=np.float32))
        # the caller's original array keeps its zero columns (no in-place edit)
        assert (seqs.sum(axis=1) == 0).any()


# ---------------------------------------------------------------------------
# modisco input prep + orchestrator error paths
# ---------------------------------------------------------------------------
class TestModisco:
    def test_prepare_input_preserves_full_length(self, tiny_model, device, tmp_path):
        # Sequences are NOT trimmed: the full attribution length is written to
        # the NPZ. window is a separate -w flag handled by run_modisco().
        import h5py
        h5_path = str(tmp_path / "attr.h5")
        seqs = _rand_onehot(2, 60, seed=20)
        compute_attribution(
            tiny_model, seqs, tracks=[0], device=device,
            n_refs=3, batch_size=2, save_h5_path=h5_path, show_progress=False,
        )
        ohe_npz, hyp_npz = prepare_modisco_input(h5_path, str(tmp_path / "in"))
        with np.load(ohe_npz) as dz:
            arr = dz[dz.files[0]]
            assert arr.shape == (2, 4, 60)   # full length preserved
            assert arr.dtype == np.int8
        with np.load(hyp_npz) as dz:
            arr = dz[dz.files[0]]
            assert arr.shape == (2, 4, 60)
            assert arr.dtype == np.float32

    def test_prepare_input_trims_odd_length_by_one(self, tmp_path):
        # Odd-length attributions lose exactly one trailing position to satisfy
        # tf-modisco-lite's even-length requirement -- this is a parity fix,
        # not a content window.
        import h5py
        rng = np.random.default_rng(0)
        seqs_nlf = np.eye(4, dtype=np.float32)[rng.integers(0, 4, size=(2, 7))]  # (2,7,4)
        hyp_nlf = rng.standard_normal((2, 7, 4)).astype(np.float32)
        h5_path = str(tmp_path / "odd.h5")
        with h5py.File(h5_path, "w") as f:
            f.create_dataset("sequences", data=seqs_nlf)
            f.create_dataset("hyp_scores", data=hyp_nlf)
        ohe_npz, _ = prepare_modisco_input(h5_path, str(tmp_path / "odd_in"))
        with np.load(ohe_npz) as dz:
            assert dz[dz.files[0]].shape == (2, 4, 6)   # 7 -> 6

    def test_read_attribution_h5_legacy_layout(self, tmp_path):
        import h5py
        # mc000-style (N, L, 4) legacy schema, no layout attr.
        h5_path = str(tmp_path / "legacy.h5")
        with h5py.File(h5_path, "w") as f:
            f.create_dataset("sequences", data=np.eye(4, dtype=np.float32)[None].repeat(3, 0))
            f.create_dataset("hyp_scores", data=np.eye(4, dtype=np.float32)[None].repeat(3, 0))
        seqs, hyp, T = read_attribution_h5(h5_path)
        assert seqs.shape == (3, 4, 4)   # normalized to channels-first
        assert hyp.shape == (3, 4, 4)
        assert T == 1

    def test_run_modisco_missing_binary_raises(self, tmp_path, monkeypatch):
        monkeypatch.setattr("deepISA.discover.modisco.resolve_cli", lambda name: None)
        with pytest.raises(RuntimeError, match="tf-modisco-lite"):
            run_modisco("o.npz", "h.npz", str(tmp_path / "out.h5"))


# ---------------------------------------------------------------------------
# finemo input + hits parsing + db build
# ---------------------------------------------------------------------------
class TestFinemo:
    def test_build_input_writes_npz_and_bed(self, tmp_path):
        seqs = _rand_onehot(3, 41, seed=30)   # odd length to test trimming
        hyp = np.random.default_rng(31).standard_normal((3, 4, 41)).astype(np.float32)
        npz = build_finemo_input(seqs, hyp, str(tmp_path / "fin"), ids=np.array(["x", "y", "z"]))
        with np.load(npz) as dz:
            assert dz["sequences"].shape == (3, 4, 40)   # trimmed to even
            assert dz["contributions"].shape == (3, 4, 40)
        bed_lines = (tmp_path / "fin" / "regions.bed").read_text().strip().split("\n")
        assert len(bed_lines) == 3

    def test_build_db_and_load_hits_annotation(self, tmp_path):
        import h5py
        motifs = {
            "p0_main": {"cwm": np.eye(4)[:10], "seq": np.eye(4)[:10]},
            "p1_main": {"cwm": np.eye(4)[:10], "seq": np.eye(4)[:10]},
        }
        db = build_finemo_db(motifs, str(tmp_path / "db.h5"),
                             annotations={"p0_main": "GATA1", "p1_main": "TAL1"})

        # Write a fake hits.tsv referencing those motifs.
        hits_path = tmp_path / "hits.tsv"
        pd.DataFrame({
            "motif_name": ["pos_patterns.pattern_0", "pos_patterns.pattern_1"],
            "score": [0.9, 0.7],
        }).to_csv(hits_path, sep="\t", index=False)

        df = load_hits_with_annotation(str(hits_path), db)
        assert {"MC_ID", "TF_Name"} <= set(df.columns)
        assert list(df["TF_Name"]) == ["GATA1", "TAL1"]
        assert list(df["MC_ID"]) == ["p0_main", "p1_main"]

    def test_run_finemo_missing_binary_raises(self, tmp_path, monkeypatch):
        monkeypatch.setattr("deepISA.discover.modisco.resolve_cli", lambda name: None)
        with pytest.raises(RuntimeError, match="Fi-NeMo"):
            run_finemo_scan("in.npz", str(tmp_path / "out"), "db.h5")


# ---------------------------------------------------------------------------
# h5_io motif extraction
# ---------------------------------------------------------------------------
class TestH5IO:
    def test_extract_and_load_motifs(self, tmp_path):
        import h5py
        h5_path = str(tmp_path / "modisco.h5")
        # Realistic structure: pattern_N contains subpattern_M nodes that hold
        # the actual CWM/sequence, mirroring tf-modisco-lite output.
        with h5py.File(h5_path, "w") as f:
            pos = f.create_group("pos_patterns")
            p0 = pos.create_group("pattern_0")
            sub0 = p0.create_group("subpattern_0")
            sub0.create_dataset("contrib_scores", data=np.random.rand(50, 4).astype(np.float32))
            sub0.create_dataset("sequence", data=np.eye(4, dtype=np.float32)[:50])
            sub0.create_group("seqlets").create_dataset("n_seqlets", data=np.array([42]))

        raw = extract_motifs_from_group(h5py.File(h5_path, "r"), "task")
        assert any("p0" in k for k in raw), f"no p0 motif in {list(raw)}"

        loaded = load_motifs(h5_path, task_name="task", target_len=20)
        assert len(loaded) >= 1, f"load_motifs returned empty; raw={list(raw)}"
        for m in loaded.values():
            assert m["cwm"].shape == (20, 4)
            assert m["seq"].shape == (20, 4)

    def test_load_motifs_missing_file_raises(self):
        with pytest.raises(FileNotFoundError):
            load_motifs("/nonexistent.h5", "task")


# ---------------------------------------------------------------------------
# track_index selection (regression of the silent track-averaging bug)
# ---------------------------------------------------------------------------
class TestTrackSelection:
    def test_read_attribution_h5_selects_track(self, tiny_model, device, tmp_path):
        # Build a 2-track attribution file. read_attribution_h5 must return the
        # SELECTED track, not the average across tracks.
        import h5py
        h5_path = str(tmp_path / "attr.h5")
        seqs = _rand_onehot(2, 60, seed=40)
        # Manually write two tracks with distinct values so selection is detectable.
        hyp_t0 = np.zeros((2, 4, 60), dtype=np.float32)
        hyp_t1 = np.ones((2, 4, 60), dtype=np.float32) * 9.0
        with h5py.File(h5_path, "w") as f:
            f.attrs["layout"] = "channels_first"
            f.create_dataset("sequences", data=seqs)
            f.create_dataset("hyp_scores", data=np.stack([hyp_t0, hyp_t1], axis=0))

        _, hyp_a, T = read_attribution_h5(h5_path, track_index=0)
        _, hyp_b, _ = read_attribution_h5(h5_path, track_index=1)
        assert T == 2
        assert np.allclose(hyp_a, 0.0)      # track 0 selected, not averaged
        assert np.allclose(hyp_b, 9.0)      # track 1 selected

    def test_read_attribution_h5_rejects_bad_track(self, tmp_path):
        import h5py
        h5_path = str(tmp_path / "attr.h5")
        with h5py.File(h5_path, "w") as f:
            f.attrs["layout"] = "channels_first"
            f.create_dataset("sequences", data=np.zeros((1, 4, 10), dtype=np.float32))
            f.create_dataset("hyp_scores", data=np.zeros((1, 1, 4, 10), dtype=np.float32))
        with pytest.raises(IndexError):
            read_attribution_h5(h5_path, track_index=5)


# ---------------------------------------------------------------------------
# finemo merge prefix compatibility (regression of the silent "Unknown" bug)
# ---------------------------------------------------------------------------
class TestFinemoMergePrefix:
    def test_merge_matches_with_or_without_prefix(self, tmp_path):
        import h5py
        # DB has 2 patterns; build a hits.tsv where one row has the prefix and
        # the other does not. Both must be annotated.
        db = str(tmp_path / "db.h5")
        with h5py.File(db, "w") as f:
            pos = f.create_group("pos_patterns")
            g0 = pos.create_group("pattern_0")
            g0.attrs["mc_id"] = "M0"; g0.attrs["tf_label"] = "GATA1"
            g1 = pos.create_group("pattern_1")
            g1.attrs["mc_id"] = "M1"; g1.attrs["tf_label"] = "TAL1"

        hits = tmp_path / "hits.tsv"
        pd.DataFrame({
            # pattern_0 WITH prefix, pattern_1 WITHOUT prefix
            "motif_name": ["pos_patterns.pattern_0", "pattern_1"],
            "score": [0.9, 0.8],
        }).to_csv(hits, sep="\t", index=False)

        df = load_hits_with_annotation(str(hits), db)
        assert list(df["TF_Name"]) == ["GATA1", "TAL1"], "prefix mismatch must not drop annotation"
        assert list(df["MC_ID"]) == ["M0", "M1"]


# ---------------------------------------------------------------------------
# report.py
# ---------------------------------------------------------------------------
class TestReport:
    def test_run_report_missing_binary_raises(self, tmp_path, monkeypatch):
        monkeypatch.setattr("deepISA.discover.modisco.resolve_cli", lambda name: None)
        # Need a real file for the modisco_h5 existence check to pass.
        h5 = tmp_path / "r.h5"
        h5.write_bytes(b"")
        with pytest.raises(RuntimeError, match="tf-modisco-lite"):
            run_motif_report(str(h5), str(tmp_path / "rep"))

    def test_run_report_missing_h5_raises(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            run_motif_report(str(tmp_path / "nope.h5"), str(tmp_path / "rep"))

    def test_run_report_missing_meme_db_raises(self, tmp_path):
        h5 = tmp_path / "r.h5"
        h5.write_bytes(b"")
        with pytest.raises(FileNotFoundError, match="MEME"):
            run_motif_report(str(h5), str(tmp_path / "rep"), meme_db=str(tmp_path / "no.meme"))

    def test_cwm_to_meme_writes_valid_format(self, tmp_path):
        motifs = {
            "m0": {"cwm": np.array([[0.5, 0.2, 0.2, 0.1], [-0.3, 0.8, 0.1, 0.1]],
                                   dtype=np.float32)},
            "m1": {"cwm": np.array([[1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0]],
                                   dtype=np.float32)},
        }
        out = str(tmp_path / "motifs.meme")
        cwm_to_meme(motifs, out)
        text = open(out, encoding="utf-8").read()
        assert "MEME version 5" in text
        assert "ALPHABET= ACGT" in text
        assert "MOTIF m0" in text and "MOTIF m1" in text
        # Negative CWM values must be clipped before PPM normalization: m0's
        # -0.3 may not appear anywhere in the matrices (all values are %.4f).
        assert "-0.3000" not in text


# ---------------------------------------------------------------------------
# h5_io reproducibility + dedup regression
# ---------------------------------------------------------------------------
class TestH5IOReproducibility:
    def test_parse_motif_name_fallback_is_deterministic(self):
        from deepISA.discover.h5_io import parse_motif_name
        # A path with no pattern_/subpattern_ tokens hits the fallback hash,
        # which must be deterministic across calls (regression for hash()).
        a = parse_motif_name("/weird/node", "task")
        b = parse_motif_name("/weird/node", "task")
        assert a == b
        assert a.startswith("task_x")

    def test_load_motifs_skip_main_uses_anchored_suffix(self, tmp_path):
        # skip_main must drop only TRAILING "_v{n}" dedup suffixes, not any
        # name containing "_v" (regression for the substring match). Two nodes
        # that parse to the same id collide: the 2nd gets "_v2" and is dropped,
        # while the task name "evi1_v" itself must survive.
        import h5py
        h5_path = str(tmp_path / "m.h5")
        with h5py.File(h5_path, "w") as f:
            for parent in ("pos_patterns", "other_grp"):
                sub = f.create_group(f"{parent}/pattern_0/subpattern_0")
                sub.create_dataset("contrib_scores",
                                   data=np.random.rand(20, 4).astype(np.float32))
                sub.create_dataset("sequence",
                                   data=np.eye(4, dtype=np.float32)[:20])
        loaded = load_motifs(h5_path, task_name="evi1_v", target_len=10)
        assert "evi1_v_p0_sub0" in loaded, "legit '_v' in task name wrongly dropped"
        assert len(loaded) == 1, f"dedup-suffixed duplicate not dropped: {list(loaded)}"


# ---------------------------------------------------------------------------
# input-set curation: select_top_regions + drop_non_acgt_regions
# (+ the QuickStart-level composition _select_modisco_regions)
# ---------------------------------------------------------------------------
def _write_mini_fasta(path, seq):
    with open(path, "w") as fh:
        fh.write(">chrSyn\n" + seq + "\n")


class TestRegionSelection:
    def test_select_top_regions_basic_and_ceil(self):
        df = pd.DataFrame({"region": [f"r{i}" for i in range(10)]})
        out = select_top_regions(df, list(range(10)), top_frac=0.1)
        assert len(out) == 1 and out["region"].iloc[0] == "r9"
        assert out["activity_score"].iloc[0] == 9.0
        # 10 rows x 0.25 = 2.5 -> ceil -> 3 kept
        out = select_top_regions(df, list(range(10)), top_frac=0.25)
        assert len(out) == 3
        # descending order preserved
        assert list(out["region"]) == ["r9", "r8", "r7"]

    def test_select_top_regions_column_name_and_ties(self):
        df = pd.DataFrame({"region": list("abcde"), "sig": [1.0, 5.0, 5.0, 2.0, 0.0]})
        out = select_top_regions(df, "sig", top_frac=0.4)   # ceil(2.0) = 2
        # ties 'b' and 'c' keep original relative order (stable sort)
        assert list(out["region"]) == ["b", "c"]

    def test_select_top_regions_validates(self):
        df = pd.DataFrame({"region": list("abc")})
        with pytest.raises(ValueError):
            select_top_regions(df, [1, 2, 3], top_frac=0.0)
        with pytest.raises(ValueError):
            select_top_regions(df, [1, 2], top_frac=0.5)   # length mismatch

    def test_drop_non_acgt_regions(self, tmp_path):
        # 60bp layout: two clean 20bp regions, one N-containing in the middle
        seq = "ACGT" * 5 + "NNACGTACGTACGTACGTN" + "CGTA" * 5
        fa = str(tmp_path / "syn.fa")
        _write_mini_fasta(fa, seq)
        df = pd.DataFrame({
            "chrom": ["chrSyn"] * 3,
            "start": [0, 20, 40],
            "end": [20, 40, 60],
        })
        kept, n_dropped = drop_non_acgt_regions(df, fa)
        assert n_dropped == 1 and len(kept) == 2
        assert list(kept["start"]) == [0, 40]   # middle (N) region gone


class TestQuickStartSelection:
    """QuickStart-level composition: predict -> rank -> top-frac -> drop N."""

    def test_select_modisco_regions_end_to_end(self, tmp_path):
        from deepISA.quickstart import QuickStart

        seq = ("ACGT" * 5) + ("NNNN" + "ACGT" * 4) + ("TTTTACGG" * 5)
        fa = str(tmp_path / "syn.fa")
        _write_mini_fasta(fa, seq)
        df = pd.DataFrame({
            "chrom": ["chrSyn"] * 3,
            "start": [0, 20, 60],
            "end": [20, 40, 80],
        })
        pipe = QuickStart(
            results_dir=str(tmp_path / "res"),
            fasta_path=fa,
            df_input=df,
            device="cpu",
        )
        cfg = {"seq_len": 20, "ks": [7, 5], "cs": [8, 8], "ds": [1, 2], "dropout": 0.0}
        pipe.define_model(cfg, mode="regression")

        sel = pipe._select_modisco_regions(
            df, tracks=[0], top_frac=0.5, rank_by=None, drop_N=True, max_drop_frac=1.0
        )
        # N region dropped first (3 -> 2), then top 50% -> ceil(1.0) = 1 region
        assert len(sel) == 1
        assert "activity_score" in sel.columns
        assert sel["activity_score"].notna().all()

        # rank_by path: explicit column is used verbatim
        df_sig = df.copy()
        df_sig["signal"] = [1.0, 3.0, 2.0]
        sel2 = pipe._select_modisco_regions(
            df_sig, tracks=[0], top_frac=0.34, rank_by="signal", drop_N=False
        )
        # ceil(3 * 0.34) = 2 -> highest signals: 3.0 (row1) then 2.0 (row2)
        assert list(sel2["start"]) == [20, 60]

        # disabling both filters is a structural no-op (order/content kept)
        sel3 = pipe._select_modisco_regions(df, tracks=[0], top_frac=None, drop_N=False)
        assert len(sel3) == 3 and list(sel3["start"]) == [0, 20, 60]

        # drop_N=True over a fully-ACGT fasta drops nothing (0% N -> no-op)
        fa_clean = str(tmp_path / "clean.fa")
        _write_mini_fasta(fa_clean, "ACGT" * 15)
        df_clean = pd.DataFrame({"chrom": ["chrSyn"] * 3,
                                 "start": [0, 20, 40], "end": [20, 40, 60]})
        pipe_clean = QuickStart(results_dir=str(tmp_path / "res_clean"),
                                fasta_path=fa_clean, df_input=df_clean, device="cpu")
        sel4 = pipe_clean._select_modisco_regions(df_clean, tracks=[0],
                                                  top_frac=None, drop_N=True)
        assert list(sel4["start"]) == [0, 20, 40]

    def test_predict_activity_deterministic_and_dual_indexed(self, tmp_path):
        """Regression: _predict_activity must eval() the model first -- with
        dropout active (dropout=0.5 below) two calls on the same input
        disagree, making the top_frac ranking irreproducible. Also checks the
        pred[:, track] branch on a dual-output model."""
        from deepISA.quickstart import QuickStart

        fa = str(tmp_path / "syn.fa")
        _write_mini_fasta(fa, "ACGT" * 10)
        df = pd.DataFrame({"chrom": ["chrSyn"] * 2,
                           "start": [0, 20], "end": [20, 40]})
        pipe = QuickStart(results_dir=str(tmp_path / "res"), fasta_path=fa,
                          df_input=df, device="cpu")
        pipe.define_model({"seq_len": 20, "ks": [7, 5], "cs": [8, 8],
                           "ds": [1, 2], "dropout": 0.5}, mode="dual")

        s0a = pipe._predict_activity(df, track=0)
        s0b = pipe._predict_activity(df, track=0)
        s1 = pipe._predict_activity(df, track=1)
        assert s0a.shape == (2,)
        assert np.array_equal(s0a, s0b), "dropout still active: ranking not reproducible"
        assert not np.array_equal(s0a, s1), "dual heads must give distinct tracks"

    def test_n_fallback_when_drop_exceeds_cap(self, tmp_path):
        """Safety valve: when N regions exceed max_drop_frac (default 20%),
        nothing is dropped -- the regions are kept and their N bases will be
        imputed with random ACGT during attribution instead."""
        from deepISA.quickstart import QuickStart

        seq = ("NACG" + "ACGT" * 4) + ("ACGT" * 5) + ("NNNN" + "ACGT" * 4)
        fa = str(tmp_path / "syn.fa")
        _write_mini_fasta(fa, seq)                     # 2 of 3 regions contain N (67%)
        df = pd.DataFrame({
            "chrom": ["chrSyn"] * 3,
            "start": [0, 20, 40],
            "end": [20, 40, 60],
        })
        pipe = QuickStart(results_dir=str(tmp_path / "res"), fasta_path=fa,
                          df_input=df, device="cpu")
        pipe.define_model({"seq_len": 20, "ks": [7, 5], "cs": [8, 8],
                           "ds": [1, 2], "dropout": 0.0}, mode="regression")

        sel = pipe._select_modisco_regions(df, tracks=[0], top_frac=None, drop_N=True)
        assert len(sel) == 3, "67% N > 20% cap -> keep all, impute downstream"

        # boundary: exactly 20% (1 of 5) still counts as droppable
        seq5 = ("NACG" + "ACGT" * 4) + "ACGT" * 5 + "ACGT" * 5 + "CGTA" * 5 + "TGCA" * 5
        fa5 = str(tmp_path / "syn5.fa")
        _write_mini_fasta(fa5, seq5)
        df5 = pd.DataFrame({
            "chrom": ["chrSyn"] * 5,
            "start": [0, 20, 40, 60, 80],
            "end": [20, 40, 60, 80, 100],
        })
        pipe5 = QuickStart(results_dir=str(tmp_path / "res5"), fasta_path=fa5,
                           df_input=df5, device="cpu")
        pipe5.define_model({"seq_len": 20, "ks": [7, 5], "cs": [8, 8],
                            "ds": [1, 2], "dropout": 0.0}, mode="regression")
        sel5 = pipe5._select_modisco_regions(df5, tracks=[0], top_frac=None, drop_N=True)
        assert len(sel5) == 4 and list(sel5["start"]) == [20, 40, 60, 80]

        # all regions contain N and the cap allows dropping them all -> the
        # top-frac step must fail loudly instead of feeding modisco an empty set
        fa_all = str(tmp_path / "all_n.fa")
        _write_mini_fasta(fa_all, "N" * 40)
        df_all = pd.DataFrame({"chrom": ["chrSyn"] * 2,
                               "start": [0, 20], "end": [20, 40]})
        pipe_all = QuickStart(results_dir=str(tmp_path / "res_all"), fasta_path=fa_all,
                              df_input=df_all, device="cpu")
        with pytest.raises(ValueError, match="No regions left"):
            pipe_all._select_modisco_regions(df_all, tracks=[0], top_frac=0.5,
                                             drop_N=True, max_drop_frac=1.0)


# ---------------------------------------------------------------------------
# QuickStart orchestration: CLI-facing methods with the external CLI stubbed out
# ---------------------------------------------------------------------------
class TestQuickStartOrchestration:
    """run_isa(isa_config={"method": ["modisco-finemo"]}) end-to-end.

    The external CLIs are stubbed with realistic artefacts (a modisco results
    H5 in the true layout; a hits.tsv whose peak_name holds region strings);
    everything else — attribution, NPZ prep, hits annotation, map_motif_finemo
    and all downstream ISA stages — runs for real on a tiny model.
    """

    @staticmethod
    def _write_fake_modisco_h5(out_h5, motif_len=10):
        import h5py
        with h5py.File(out_h5, "w") as f:
            pos = f.create_group("pos_patterns")
            p0 = pos.create_group("pattern_0")
            sub = p0.create_group("subpattern_0")
            sub.create_dataset("contrib_scores",
                               data=np.random.rand(motif_len, 4).astype(np.float32))
            sub.create_dataset("sequence", data=np.eye(4, dtype=np.float32)[:motif_len])
            sub.create_group("seqlets").create_dataset("n_seqlets", data=np.array([7]))

    def test_run_isa_modisco_finemo_end_to_end(self, tmp_path, monkeypatch):
        import deepISA.quickstart as qs

        fa = str(tmp_path / "syn.fa")
        with open(fa, "w") as fh:
            fh.write(">chrA\n" + "ACGT" * 150 + "\n>chrB\n" + "ACGT" * 150 + "\n")
        df = pd.DataFrame({
            "chrom": ["chrA", "chrB"], "start": [0, 0], "end": [600, 600],
        })
        pipe = QuickStart(results_dir=str(tmp_path / "res"), fasta_path=fa,
                          df_input=df, device="cpu")
        pipe.define_model({"seq_len": 600, "ks": [7], "cs": [8], "ds": [1],
                           "dropout": 0.0}, mode="dual")

        # stub `modisco motifs`: write a valid results H5
        def fake_modisco_cli(ohe_npz, hyp_npz, out_h5, n_seqlets=50000, window=None):
            self._write_fake_modisco_h5(out_h5)
            return out_h5
        monkeypatch.setattr(qs, "run_modisco", fake_modisco_cli)

        # stub `finemo call-hits`: hits.tsv with region-string peak_names
        def fake_finemo_cli(npz_path, out_dir, motif_db_h5, lam=0.7, max_steps=10000):
            hits = pd.DataFrame({
                "peak_name": ["chrA:0-600"] * 3 + ["chrB:0-600"] * 2,
                "start": [50, 150, 250, 80, 180],
                "end":   [60, 160, 260, 90, 190],
                "strand": ["+", "-", "+", "+", "-"],
                "motif_name": ["pos_patterns.pattern_0"] * 5,
                "hit_coefficient_global": [1.5, 0.7, 0.9, 1.1, 0.6],
            })
            out = Path(out_dir) / "hits.tsv"
            out.parent.mkdir(parents=True, exist_ok=True)
            hits.to_csv(out, sep="\t", index=False)
            return str(out)
        monkeypatch.setattr(qs, "run_finemo_scan", fake_finemo_cli)

        # tracks=[1]: regression guard — discovery must use H5 slot 0 (the
        # first requested track), never the model-output index.
        pipe.run_isa(
            jaspar_path="unused (method has no JASPAR)",
            isa_config={
                "method": ["modisco-finemo"],
                "tracks": [1],
                "discovery_top_frac": None,      # keep both regions
                "n_samples": 12, "n_bins": 2,
                "num_regions_per_batch": 10, "pred_batch_size": 4,
                "receptive_field": 255,
            },
        )

        # motif_locs: map_motifs-identical contract; region start is 0 so
        # absolute == relative here — coordinate reversal verified separately
        motif_df = pd.read_csv(pipe.files["motif_locs"])
        assert list(motif_df.columns) == ["chrom", "start", "end", "start_rel",
                                          "end_rel", "tf", "score", "strand", "region"]
        assert len(motif_df) == 5
        row = motif_df.iloc[0]
        assert row["chrom"] == "chrA" and row["start"] == 50 and row["end"] == 60
        assert row["start_rel"] == 50 and row["strand"] == "+"

        # downstream stages ran on the finemo-derived motif_locs
        for key in ("isa_single", "isa_combi", "non_motif_isa", "imp_tf"):
            assert os.path.exists(pipe.files[key]), key
        assert len(pd.read_csv(pipe.files["isa_single"])) == 5

        # discovery products live under Data/Motif_discovery/
        assert "Motif_discovery" in pipe.files["attr_h5"].replace("\\", "/")
        assert os.path.exists(pipe.files["modisco_h5"])
        assert os.path.exists(pipe.files["finemo_hits"])
        assert os.path.exists(pipe.files["discovered_motifs"])

    def test_run_isa_unknown_method_rejected(self, tmp_path):
        pipe = QuickStart(results_dir=str(tmp_path / "res"), fasta_path="f.fa",
                          df_input=pd.DataFrame({"chrom": ["chr1"], "start": [0],
                                                 "end": [10]}), device="cpu")
        pipe.model = object()  # pass the define_model guard
        with pytest.raises(ValueError, match="Unknown isa_config method"):
            pipe.run_isa(jaspar_path="x",
                         isa_config={"method": ["gpt4"]},
                         start_from="map_motifs")


class TestMapMotifFinemo:
    """Contract of map_motif_finemo: identical columns to map_motifs output."""

    def _hits(self):
        return pd.DataFrame({
            "peak_name": ["chr1:1000-1600", "chr1:1000-1600", "chr2:50-650"],
            "start": [10, 300, 5], "end": [24, 314, 19],
            "strand": ["+", "-", "+"],
            "motif_name": ["pos_patterns.pattern_0",
                           "pos_patterns.pattern_0",
                           "pos_patterns.pattern_9"],
            "TF_Name": ["GATA1", "GATA1", "Unknown"],
            "hit_coefficient_global": [1.5, 0.7, 0.9],
        })

    def test_columns_match_map_motifs_contract(self, tmp_path):
        regions = pd.DataFrame({"chrom": ["chr1", "chr2"],
                                "start": [1000, 50], "end": [1600, 650]})
        mo, no = tmp_path / "motif.csv", tmp_path / "nonmotif.csv"
        out = map_motif_finemo(self._hits(), regions, str(mo), str(no))
        assert list(out.columns) == ["chrom", "start", "end", "start_rel",
                                     "end_rel", "tf", "score", "strand", "region"]

    def test_coordinates_reverse_the_fake_bed(self, tmp_path):
        regions = pd.DataFrame({"chrom": ["chr1", "chr2"],
                                "start": [1000, 50], "end": [1600, 650]})
        mo, no = tmp_path / "motif.csv", tmp_path / "nonmotif.csv"
        out = map_motif_finemo(self._hits(), regions, str(mo), str(no))
        # chr1 hits: absolute = 1000 + relative
        assert list(out.loc[out.chrom == "chr1", "start"]) == [1010, 1300]
        assert list(out.loc[out.chrom == "chr1", "end"]) == [1024, 1314]
        assert list(out.loc[out.chrom == "chr1", "start_rel"]) == [10, 300]
        # chr2 (region start 50): 50 + 5 = 55
        assert out.loc[out.chrom == "chr2", "start"].iloc[0] == 55

    def test_unknown_tf_falls_back_to_stripped_motif_id(self, tmp_path):
        regions = pd.DataFrame({"chrom": ["chr2"], "start": [50], "end": [650]})
        mo, no = tmp_path / "motif.csv", tmp_path / "nonmotif.csv"
        out = map_motif_finemo(self._hits(), regions, str(mo), str(no))
        tf_chr2 = out.loc[out.chrom == "chr2", "tf"].iloc[0]
        assert tf_chr2 == "PATTERN_9"          # pos_patterns. prefix stripped
        assert (out.loc[out.chrom == "chr1", "tf"] == "GATA1").all()

    def test_non_motifs_written_via_mapper_helper(self, tmp_path):
        regions = pd.DataFrame({"chrom": ["chr1"], "start": [1000], "end": [1600]})
        mo, no = tmp_path / "motif.csv", tmp_path / "nonmotif.csv"
        map_motif_finemo(self._hits(), regions, str(mo), str(no))
        non = pd.read_csv(no)
        # region is split around the two chr1 motifs -> non-empty complement
        assert len(non) >= 1

    def test_unparseable_peak_name_raises(self, tmp_path):
        regions = pd.DataFrame({"chrom": ["chr1"], "start": [0], "end": [600]})
        bad = self._hits()
        bad["peak_name"] = "seq_0"
        with pytest.raises(ValueError, match="peak_name"):
            map_motif_finemo(bad, regions, str(tmp_path / "m.csv"),
                             str(tmp_path / "n.csv"))

    def test_run_isa_union_of_jaspar_and_finemo(self, tmp_path, monkeypatch):
        """method=["JASPAR", "modisco-finemo"]: both sources run, their hits
        are unioned (deduplicated) into motif_locs, downstream ISA proceeds."""
        import deepISA.quickstart as qs

        fa = str(tmp_path / "syn.fa")
        with open(fa, "w") as fh:
            fh.write(">chrA\n" + "ACGT" * 150 + "\n")
        df = pd.DataFrame({"chrom": ["chrA"], "start": [0], "end": [600]})
        pipe = QuickStart(results_dir=str(tmp_path / "res"), fasta_path=fa,
                          df_input=df, device="cpu")
        pipe.define_model({"seq_len": 600, "ks": [7], "cs": [8], "ds": [1],
                           "dropout": 0.0}, mode="regression")

        # stub the JASPAR mapper: one known-TF hit at 400-410
        def fake_map_motifs(regions_df, jaspar_path, motif_outpath,
                            non_motif_outpath, expressed_tfs=None,
                            motif_score_thresh=500, remap_path=None):
            pd.DataFrame([{"chrom": "chrA", "start": 400, "end": 410,
                           "start_rel": 400, "end_rel": 410, "tf": "CTCF",
                           "score": 700, "strand": "+", "region": "chrA:0-600"}]
                         ).to_csv(motif_outpath, index=False)
        monkeypatch.setattr(qs, "map_motifs", fake_map_motifs)

        def fake_modisco_cli(ohe_npz, hyp_npz, out_h5, n_seqlets=50000, window=None):
            TestQuickStartOrchestration._write_fake_modisco_h5(out_h5)
            return out_h5
        monkeypatch.setattr(qs, "run_modisco", fake_modisco_cli)

        def fake_finemo_cli(npz_path, out_dir, motif_db_h5, lam=0.7, max_steps=10000):
            hits = pd.DataFrame({
                "peak_name": ["chrA:0-600"] * 2,
                "start": [50, 100],            # 40bp apart -> one combi pair
                "end": [60, 110],
                "strand": ["+", "+"],
                "motif_name": ["pos_patterns.pattern_0"] * 2,
                "hit_coefficient_global": [1.5, 0.9],
            })
            out = Path(out_dir) / "hits.tsv"
            out.parent.mkdir(parents=True, exist_ok=True)
            hits.to_csv(out, sep="\t", index=False)
            return str(out)
        monkeypatch.setattr(qs, "run_finemo_scan", fake_finemo_cli)

        pipe.run_isa(
            jaspar_path="stubbed",
            isa_config={
                "method": ["JASPAR", "modisco-finemo"],
                "discovery_top_frac": None,
                "n_samples": 12, "n_bins": 2,
                "num_regions_per_batch": 10, "pred_batch_size": 4,
                "receptive_field": 255,
            },
        )

        motif_df = pd.read_csv(pipe.files["motif_locs"])
        # JASPAR hit (400-410, CTCF) + finemo hits (50-60, 400-410) = 3 rows
        # (same interval from two sources is kept once per tf/score tuple)
        assert len(motif_df) == 3, motif_df.to_dict("records")
        tfs = set(motif_df["tf"])
        assert "CTCF" in tfs                                   # JASPAR side
        assert any(t.startswith("PATTERN") for t in tfs)       # finemo side
        # downstream stages ran on the union
        assert os.path.exists(pipe.files["isa_single"])
        assert len(pd.read_csv(pipe.files["isa_single"])) == 3
