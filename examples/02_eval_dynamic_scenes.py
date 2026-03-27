import argparse
import sys
import omnigibson as og
from realm.eval import evaluate


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="dynamic sim evals")
    parser.add_argument('--perturbation_id', type=int, required=False, default=0)
    parser.add_argument('--task_id', type=int, required=False, default=0)
    parser.add_argument('--repeats', type=int, required=False, default=5)
    parser.add_argument('--max_steps', type=int, required=False, default=500)
    parser.add_argument('--model', type=str, required=True, default=None)
    parser.add_argument('--port', type=int, required=True)
    parser.add_argument('--experiment_name', type=str, required=True)
    parser.add_argument('--run_id', type=str, required=False, default=None)
    parser.add_argument('--log_dir', type=str, required=False, default=None)
    parser.add_argument('--record_video', type=lambda x: x.lower() != 'false', default=True, help='Record videos (default: true). Pass --record_video false to disable.')
    args = parser.parse_args()
    assert args.model is not None
    assert args.experiment_name is not None
    log_dir = args.log_dir if args.log_dir is not None else "/app/logs"

    evaluate(
        task_id=args.task_id,
        perturbation_id=args.perturbation_id,
        repeats=args.repeats,
        max_steps=args.max_steps,
        model=args.model,
        port=args.port,
        log_dir=log_dir,
        record_video=args.record_video,
    )
    og.shutdown()
    sys.exit(0)
