from __future__ import annotations
import argparse, json
from pathlib import Path
from .contracts import DecoderIdentity, PriorSteeringProtocol, public_benchmark_contract

def main(argv=None):
    parser=argparse.ArgumentParser(description="Fail-closed offline decoder-prior preflight")
    parser.add_argument("--config",type=Path,required=True); parser.add_argument("--decoder-identity",type=Path,required=True)
    parser.add_argument("--output",type=Path,required=True); args=parser.parse_args(argv)
    config=json.loads(args.config.read_text(encoding="utf-8")); identity=DecoderIdentity(**json.loads(args.decoder_identity.read_text(encoding="utf-8")))
    ignored={"primary_method_doi","decoder_backend","launch_automatically","required_first_gate"}
    protocol=PriorSteeringProtocol(decoder=identity,**{k:v for k,v in config.items() if k not in ignored})
    payload=public_benchmark_contract(protocol); args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps(payload,indent=2,sort_keys=True)+"\n",encoding="utf-8"); return 0
if __name__=="__main__": raise SystemExit(main())
