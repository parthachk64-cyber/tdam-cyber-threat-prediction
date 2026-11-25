"""FORMAT tests on a tiny file in LANL's column layout. It is not LANL data."""

import pandas as pd
import pytest

from tdam.data import lanl
from tdam.data.lanl import episodes_from_redteam, preprocess_auth, write_sample_in_lanl_format


def _rows(out):
    parts = sorted(out.glob("part-*.csv.gz"))
    frame = pd.concat([pd.read_csv(p) for p in parts], ignore_index=True)
    return frame.groupby(["entity", "b", "avail"]).sum().reset_index()


def test_chunk_size_does_not_change_result(tmp_path):
    f = tmp_path / "auth.txt.gz"
    write_sample_in_lanl_format(f, 40)
    a = preprocess_auth(f, tmp_path / "a", ["C1", "C2"], 0, 10**6, chunksize=1000, resume=False)
    b = preprocess_auth(f, tmp_path / "b", ["C1", "C2"], 0, 10**6, chunksize=7, resume=False)
    pd.testing.assert_frame_equal(_rows(a), _rows(b))
    r = _rows(a)
    assert r["n"].sum() == 40 + 8  # every line counted exactly once (bins split across chunks are carried)
    c1 = r[r["entity"] == "C1"]
    assert (c1["nd"] == 1).all() and c1["nf"].sum() == 0
    assert r[r["entity"] == "C2"]["nf"].sum() == 8


def test_interrupted_run_resumes_to_same_result(tmp_path, monkeypatch):
    f = tmp_path / "auth.txt.gz"
    write_sample_in_lanl_format(f, 40)
    clean = preprocess_auth(f, tmp_path / "clean", ["C1", "C2"], 0, 10**6, chunksize=9, resume=False)

    real = lanl._read_auth
    calls = {"n": 0}

    def flaky(path, skip, chunksize):
        for chunk in real(path, skip, chunksize):
            calls["n"] += 1
            if calls["n"] == 3:
                raise RuntimeError("simulated crash")
            yield chunk

    monkeypatch.setattr(lanl, "_read_auth", flaky)
    with pytest.raises(RuntimeError):
        preprocess_auth(f, tmp_path / "resumed", ["C1", "C2"], 0, 10**6, chunksize=9, resume=False)
    monkeypatch.setattr(lanl, "_read_auth", real)
    resumed = preprocess_auth(f, tmp_path / "resumed", ["C1", "C2"], 0, 10**6, chunksize=9, resume=True)
    pd.testing.assert_frame_equal(_rows(clean), _rows(resumed))


def test_simulated_delay_is_separate_from_replay_assumption(tmp_path):
    f = tmp_path / "auth.txt.gz"
    write_sample_in_lanl_format(f, 20)
    base = _rows(preprocess_auth(f, tmp_path / "x", ["C1"], 0, 10**6, resume=False))
    assert (base["avail"] == base["b"]).all()  # replay assumption: arrival == event time
    sim = _rows(preprocess_auth(f, tmp_path / "y", ["C1"], 0, 10**6, simulated_delay_max_s=300, seed=1, resume=False))
    assert (sim["avail"] >= sim["b"]).all() and (sim["avail"] > sim["b"]).any()


def test_redteam_episodes_use_deterministic_gap_and_label_onset_as_proxy():
    red = pd.DataFrame(
        {"time": [100, 200, 5000, 100, 90000], "user": ["u"] * 5, "src": ["C1", "C1", "C1", "C2", "C2"], "dst": ["d"] * 5}
    ).sort_values("time")
    eps = episodes_from_redteam(red, gap_s=3600, label_delay_s=0, end_s=10**6)
    assert sorted(e["incident_id"] for e in eps) == ["C1-E001", "C1-E002", "C2-E001", "C2-E002"]
    assert all(e["kind"] == "lanl_redteam_episode_proxy_onset" for e in eps)


def test_missing_files_reported_not_replaced(tmp_path):
    from tdam.cli import main

    assert main(["data", "verify", "--lanl", str(tmp_path)]) == 2
    assert main(["run", "--lanl-dir", str(tmp_path / "nothing")]) == 2


def test_prepare_and_load_slice_end_to_end(tmp_path):
    import gzip

    from tdam.config import Config
    from tdam.data.lanl import load_lanl_dir, prepare_lanl
    from tdam.temporal.windows import build_windowset

    auth = tmp_path / "auth.txt.gz"
    write_sample_in_lanl_format(auth, 140)
    red = tmp_path / "redteam.txt.gz"
    with gzip.open(red, "wt") as fh:
        fh.write("3000,U1@D,C1,C2\n3100,U1@D,C1,C3\n")
    card = prepare_lanl(auth, red, tmp_path / "slice", start_day=0, max_days=0.1, top_k=5, log=lambda *_: None)
    assert card["episodes"] == 1 and card["arrival_model"]["assumption"].startswith("no ingestion")
    ds = load_lanl_dir(tmp_path / "slice")
    assert ds.source == "independent-lanl" and ds.incidents and ds.incidents[0].kind.endswith("proxy_onset")
    ws = build_windowset(ds, Config())
    assert len(ws) > 0 and ws.y.sum() > 0
