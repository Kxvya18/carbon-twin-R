import argparse
import json


def main():
    p = argparse.ArgumentParser(prog="carbontwin", description="CarbonTwin-R research CLI")
    sub = p.add_subparsers(dest="command", required=True)

    d = sub.add_parser("data-download", help="Download and prepare the real UCI Appliances Energy dataset")
    d.add_argument("--config", default="configs/real_uci.yaml")
    d.add_argument("--force", action="store_true")

    prep = sub.add_parser("prepare", help="Validate raw data and build the silver Parquet layer")
    prep.add_argument("--config", default="configs/real_uci.yaml")

    tr = sub.add_parser("train", help="Time-series CV hyperparameter search and final clean test evaluation")
    tr.add_argument("--config", default="configs/real_uci.yaml")
    tr.add_argument("--budget", choices=["quick", "research"], default=None)

    sim = sub.add_parser("simulate", help="Run an interactive-style fault scenario on real measured test data")
    sim.add_argument("--config", default="configs/real_uci.yaml")
    sim.add_argument("--fault", choices=["hvac_efficiency_drift", "standby_load", "lighting_schedule", "sensor_bias"], default=None)
    sim.add_argument("--severity", type=float, default=None)
    sim.add_argument("--start", type=float, default=None, help="Fraction of test period before fault begins")
    sim.add_argument("--zone", type=int, choices=range(1,10), default=None)
    sim.add_argument("--name", default="latest")

    rr = sub.add_parser("run-research", help="Prepare/train/tune and then run the configured scenario")
    rr.add_argument("--config", default="configs/real_uci.yaml")
    rr.add_argument("--budget", choices=["quick", "research"], default=None)

    ft = sub.add_parser("fine-tune", help="Optional PyTorch temporal pretrain + fine-tune experiment")
    ft.add_argument("--config", default="configs/real_uci.yaml")

    legacy = sub.add_parser("run-all", help="Legacy synthetic demo retained for unit/debug work")
    legacy.add_argument("--config", default="configs/demo.yaml")
    legacy.add_argument("--output", default="outputs/synthetic")

    args = p.parse_args()
    if args.command == "data-download":
        from .data.uci_appliances import download_uci_appliances
        from pathlib import Path
        import yaml
        cfg = yaml.safe_load(Path(args.config).read_text())
        path = download_uci_appliances(cfg["project"]["raw_dir"], force=args.force)
        print(path)
    elif args.command == "prepare":
        from .data.prepare import prepare_real_uci
        print(json.dumps(prepare_real_uci(args.config, download=True), indent=2))
    elif args.command == "train":
        from .research_pipeline import train_real_models
        print(json.dumps(train_real_models(args.config, budget=args.budget), indent=2, default=str))
    elif args.command == "simulate":
        from .research_pipeline import simulate_real_scenario
        summary, _, _ = simulate_real_scenario(args.config, fault_type=args.fault, severity=args.severity,
                                                start_fraction=args.start, zone=args.zone, output_name=args.name)
        print(json.dumps(summary, indent=2, default=str))
    elif args.command == "run-research":
        from .research_pipeline import run_research
        print(json.dumps(run_research(args.config, budget=args.budget), indent=2, default=str))
    elif args.command == "fine-tune":
        from .research_pipeline import run_temporal_finetune_experiment
        print(json.dumps(run_temporal_finetune_experiment(args.config), indent=2))
    elif args.command == "run-all":
        from .pipeline import run_all
        print(json.dumps(run_all(args.config, args.output), indent=2))


if __name__ == "__main__":
    main()
