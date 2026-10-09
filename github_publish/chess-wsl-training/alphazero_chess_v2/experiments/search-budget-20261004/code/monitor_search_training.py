import argparse
import subprocess
import sys
import json
from pathlib import Path

def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--experiment',type=Path,required=True)
    parser.add_argument('--cycle',type=int,required=True)
    args=parser.parse_args()
    root=Path(__file__).resolve().parent
    run=subprocess.run([sys.executable,str(root/'monitor_experiment.py'),'--experiment',str(args.experiment),'--cycle',str(args.cycle)],capture_output=True,text=True)
    if run.returncode not in (0,2):
        print(run.stdout+run.stderr,flush=True)
        return run.returncode
    import chess
    from azchess.checkpoint import load_variables
    from azchess.mcts import MCTS
    model,variables=load_variables(args.experiment/'checkpoints/latest.msgpack')
    agent=MCTS(model,variables)
    records=[]
    for position in json.loads((args.experiment/'repetition_manifest.json').read_text()):
        board=chess.Board()
        for uci in position['moves']:board.push_uci(uci)
        priors,value=agent.evaluate(board)
        move=chess.Move.from_uci(position['reference_move'])
        records.append(dict(game=position['game'],ply=position['ply'],reference_move=move.uci(),
            reference_prior=priors[move],value=value,
            priors={m.uci():p for m,p in priors.items()}))
    metrics=args.experiment/'metrics'/f'cycle-{args.cycle:06d}.json'
    result=json.loads(metrics.read_text())
    result['repetition_policy_probe']=records
    metrics.write_text(json.dumps(result,indent=2))
    print('MONITOR '+json.dumps(result),flush=True)
    return run.returncode

if __name__=='__main__':raise SystemExit(main())
