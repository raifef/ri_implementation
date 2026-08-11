from __future__ import annotations
import argparse, json
from pathlib import Path
from .contracts import StepProtocol, build_control_inventory, build_run_plan

def main(argv=None):
    parser=argparse.ArgumentParser(description="Plan the source-budgeted 924-control step-response analogue")
    parser.add_argument("--config",type=Path,required=True); parser.add_argument("--output",type=Path,required=True)
    args=parser.parse_args(argv); values=json.loads(args.config.read_text(encoding="utf-8"))
    values.pop("launch_automatically",None); protocol=StepProtocol(**values)
    args.output.mkdir(parents=True,exist_ok=True)
    (args.output/"run_plan.json").write_text(json.dumps(build_run_plan(protocol),indent=2,sort_keys=True)+"\n",encoding="utf-8")
    (args.output/"control_inventory.json").write_text(json.dumps(build_control_inventory(protocol.controls,protocol.direction_coordinate),indent=2,sort_keys=True)+"\n",encoding="utf-8")
    return 0
if __name__=="__main__": raise SystemExit(main())
