"""Command line interface: tdam <command>."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from tdam.config import load_config
from tdam.data.dataset import build_dataset, load_dir
from tdam.data.fixture import FixtureSpec, write_fixture
from tdam.data.io import read_events, write_json
from tdam.data.validate import lanl_missing, validate_dir
from tdam.evaluation.alerting import issue_alerts
from tdam.evaluation.experiment import run_experiment
from tdam.models.bundle import Bundle
from tdam.report import write_report
from tdam.timeutil import iso


def _cfg(args):
    cfg = load_config(getattr(args, "config", None))
    if getattr(args, "seed", None) is not None:
        cfg.seed = args.seed
    return cfg


def cmd_generate(args) -> int:
    spec = FixtureSpec(n_hosts=args.hosts, days=args.days, seed=args.seed or 7)
    card = write_fixture(spec, args.out)
    print(f"wrote {card['n_events']} events, {card['n_incidents']} incidents to {args.out}")
    return 0


def cmd_verify(args) -> int:
    if args.lanl:
        missing = lanl_missing(args.lanl)
        if missing:
            print("LANL benchmark files are missing; the run will NOT fall back to synthetic data.")
            print("\n".join("  - " + m for m in missing))
            print("Then run: tdam data prepare-lanl --auth <auth.txt.gz> --redteam <redteam.txt.gz> --out <dir>")
            return 2
        print("LANL files present.")
        return 0
    result = validate_dir(args.path, _cfg(args))
    print(json.dumps(result, indent=2, default=str))
    return 0 if result["ok"] else 1


def cmd_prepare_lanl(args) -> int:
    from tdam.data.lanl import prepare_lanl

    missing = lanl_missing(Path(args.auth).parent)
    if not Path(args.auth).exists() or not Path(args.redteam).exists():
        print("missing input files:\n  - " + "\n  - ".join(missing or [args.auth, args.redteam]))
        return 2
    card = prepare_lanl(args.auth, args.redteam, args.out, args.start_day, args.max_days, args.top_k, args.gap_minutes, args.label_delay_minutes, args.arrival_delay_s, args.simulated_delay_max_s, args.seed or 0)
    print(json.dumps(card, indent=2))
    return 0


def _run(ds, cfg, out: Path, repo_root: Path, report_to: Path | None, claims: Path | None) -> int:
    run_experiment(ds, cfg, out, repo_root)
    if report_to is not None:
        write_report(out, report_to, claims)
        print(f"report: {report_to}")
    print(f"artifacts: {out}")
    return 0


def cmd_run(args) -> int:
    cfg = _cfg(args)
    if args.lanl_dir:
        from tdam.data.lanl import load_lanl_dir

        if not (Path(args.lanl_dir) / "rows.csv.gz").exists():
            print("prepared LANL slice not found; run `tdam data prepare-lanl` first (no silent synthetic fallback)")
            return 2
        ds = load_lanl_dir(args.lanl_dir, cfg.protocol.bin_seconds)
    else:
        check = validate_dir(args.data, cfg)
        if not check["ok"]:
            print(json.dumps(check, indent=2))
            return 1
        ds = load_dir(args.data, "synthetic", cfg.protocol.bin_seconds)
    return _run(ds, cfg, Path(args.out), Path.cwd(), Path(args.report) if args.report else None, Path(args.claims) if args.claims else None)


def cmd_demo(args) -> int:
    cfg = _cfg(args)
    data = Path(args.data_dir)
    spec = FixtureSpec(n_hosts=args.hosts, days=args.days, seed=cfg.seed)
    write_fixture(spec, data)
    check = validate_dir(data, cfg)
    print("validator:", "ok" if check["ok"] else "FAILED", check["warnings"])
    if not check["ok"]:
        print(json.dumps(check, indent=2))
        return 1
    ds = load_dir(data, "synthetic", cfg.protocol.bin_seconds)
    claims = Path("references/reported/paper_claims.yaml")
    return _run(ds, cfg, Path(args.out), Path.cwd(), Path(args.report), claims if claims.exists() else None)


def cmd_train(args) -> int:
    from tdam.temporal.windows import build_windowset
    from tdam.training.pipeline import fit_bundle

    cfg = _cfg(args)
    ds = load_dir(args.data, "synthetic", cfg.protocol.bin_seconds)
    ws = build_windowset(ds, cfg)
    bundle, info, _, _ = fit_bundle(ds, ws, cfg)
    bundle.save(args.out)
    write_json(Path(args.out) / "train_info.json", {k: v for k, v in info.items()})
    print(f"bundle saved to {args.out}; blend weight {bundle.blend_weight}")
    return 0


def _predict_events(events, bundle, cfg, model: str, op: str, stride: int):
    from tdam.data.replay import replay

    entities = sorted({e.entity for e in events})
    cov = {"entities": {}}
    for ent in entities:
        ts = [e.event_time for e in events if e.entity == ent]
        cov["entities"][ent] = [[iso(min(ts)), iso(max(ts))]]
    ds = build_dataset("local", "user-supplied", events, [], cov, cfg.protocol.bin_seconds)
    first = bundle.lookback - 1
    ticks = np.arange(first, ds.timeline.n_bins, stride)
    preds, stats = replay(events, ds.entities, ds.timeline, ds.cover, bundle, ticks)
    frame = pd.DataFrame(preds)
    thr = bundle.thresholds.get(model, {}).get(op)
    if thr is None:
        raise ValueError(f"bundle has no {op} threshold for model {model}")
    frame["time_utc"] = [iso(int(t)) for t in frame["time_s"]]
    ent_idx = frame["entity"].astype("category").cat.codes.to_numpy()
    sel = issue_alerts(ent_idx, frame["time_s"].to_numpy(), frame[model].to_numpy(), thr, cfg.protocol.cooldown_minutes * 60)
    frame["alert"] = 0
    frame.loc[frame.index[sel], "alert"] = 1
    frame["threshold"] = thr
    return frame, stats


def cmd_predict(args) -> int:
    cfg = _cfg(args)
    bundle = Bundle.load(args.bundle)
    events = read_events(args.events)
    print("assumption: each entity is monitored from its first to its last event in the file")
    frame, stats = _predict_events(events, bundle, cfg, args.model, args.operating_point, args.stride)
    frame.drop(columns=["time_s"]).to_csv(args.out, index=False)
    print(f"{len(frame)} predictions, {int(frame['alert'].sum())} alerts -> {args.out}; replay stats {stats}")
    return 0


def cmd_replay(args) -> int:
    cfg = _cfg(args)
    bundle = Bundle.load(args.bundle)
    events = read_events(Path(args.data) / "events.jsonl")
    frame, stats = _predict_events(events, bundle, cfg, args.model, args.operating_point, args.stride)
    frame.drop(columns=["time_s"]).to_csv(args.out, index=False)
    print(f"replayed {len(events)} records in arrival order; {len(frame)} predictions; stats {stats}")
    return 0


def cmd_report(args) -> int:
    write_report(args.run, args.out, args.claims)
    print(f"wrote {args.out}")
    return 0


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="tdam", description=__doc__)
    sub = p.add_subparsers(dest="cmd", required=True)

    def common(sp, config=True):
        if config:
            sp.add_argument("--config", default=None)
        sp.add_argument("--seed", type=int, default=None)

    d = sub.add_parser("demo", help="generate synthetic data, train, evaluate, report (CPU)")
    common(d)
    d.add_argument("--data-dir", default="data/demo")
    d.add_argument("--out", default="results/demo")
    d.add_argument("--report", default="docs/results.md")
    d.add_argument("--hosts", type=int, default=16)
    d.add_argument("--days", type=float, default=8.0)
    d.set_defaults(fn=cmd_demo)

    data = sub.add_parser("data", help="data tools")
    ds = data.add_subparsers(dest="dcmd", required=True)
    g = ds.add_parser("generate")
    common(g, config=False)
    g.add_argument("--out", required=True)
    g.add_argument("--hosts", type=int, default=16)
    g.add_argument("--days", type=float, default=8.0)
    g.set_defaults(fn=cmd_generate)
    v = ds.add_parser("verify")
    common(v)
    v.add_argument("path", nargs="?", default="data/demo")
    v.add_argument("--lanl", default=None, help="directory that should contain auth.txt.gz and redteam.txt.gz")
    v.set_defaults(fn=cmd_verify)
    pl = ds.add_parser("prepare-lanl")
    pl.add_argument("--auth", required=True)
    pl.add_argument("--redteam", required=True)
    pl.add_argument("--out", required=True)
    pl.add_argument("--start-day", type=float, default=0.0)
    pl.add_argument("--max-days", type=float, default=7.0)
    pl.add_argument("--top-k", type=int, default=100)
    pl.add_argument("--gap-minutes", type=int, default=60)
    pl.add_argument("--label-delay-minutes", type=int, default=0)
    pl.add_argument("--arrival-delay-s", type=int, default=0)
    pl.add_argument("--simulated-delay-max-s", type=int, default=None)
    pl.add_argument("--seed", type=int, default=0)
    pl.set_defaults(fn=cmd_prepare_lanl)

    r = sub.add_parser("run", help="train + evaluate on a prepared dataset")
    common(r)
    r.add_argument("--data", default="data/demo")
    r.add_argument("--lanl-dir", default=None)
    r.add_argument("--out", default="runs/run")
    r.add_argument("--report", default=None)
    r.add_argument("--claims", default=None)
    r.set_defaults(fn=cmd_run)

    t = sub.add_parser("train", help="fit and save a model bundle")
    common(t)
    t.add_argument("--data", default="data/demo")
    t.add_argument("--out", required=True)
    t.set_defaults(fn=cmd_train)

    for name, fn, help_ in (("predict", cmd_predict, "score a local CSV/JSONL of events"), ("replay", cmd_replay, "replay a saved dataset in arrival order")):
        q = sub.add_parser(name, help=help_)
        common(q)
        q.add_argument("--bundle", required=True)
        q.add_argument("--out", required=True)
        q.add_argument("--model", default="ensemble")
        q.add_argument("--operating-point", default="budget", choices=["budget", "f1"])
        q.add_argument("--stride", type=int, default=1)
        if name == "predict":
            q.add_argument("--events", required=True)
        else:
            q.add_argument("--data", required=True)
        q.set_defaults(fn=fn)

    rp = sub.add_parser("report", help="regenerate results markdown from run artifacts")
    rp.add_argument("--run", required=True)
    rp.add_argument("--out", default="docs/results.md")
    rp.add_argument("--claims", default=None)
    rp.set_defaults(fn=cmd_report)
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        return int(args.fn(args))
    except (ValueError, FileNotFoundError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
