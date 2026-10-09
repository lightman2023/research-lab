from pathlib import Path
import shutil
root=Path('//wsl.localhost/Ubuntu/home/user/chess/alphazero_chess_v2')
win=Path('C:/Users/User/Desktop/chess_v2')
work=Path(__file__).resolve().parent
path=root/'check_search_cycle.py'
shutil.copy2(path,path.with_name(path.name+'.pre-partial-resume-20261005'))
shutil.copy2(work/'check_search_cycle.py',path)
path=root/'parallel_selfplay.py';text=path.read_text(encoding='utf-8')
shutil.copy2(path,path.with_name(path.name+'.pre-partial-resume-20261005'))
old='    args = parser.parse_args()\n    pending = deque((number, 0) for number in range(1, args.games + 1))'
new='''    parser.add_argument("--game-numbers", help="Comma-separated game numbers to generate; omitted means all games")
    args = parser.parse_args()
    numbers = list(range(1, args.games + 1)) if args.game_numbers is None else [int(x) for x in args.game_numbers.split(",")]
    if len(set(numbers)) != len(numbers) or any(n < 1 or n > args.games for n in numbers):
        parser.error("game-numbers must be unique integers within 1..games")
    pending = deque((number, 0) for number in numbers)'''
assert text.count(old)==1;text=text.replace(old,new)
text=text.replace('print(f"All {args.games} parallel self-play games finished", flush=True)','print(f"All requested {len(numbers)} parallel self-play games finished", flush=True)')
path.write_text(text,encoding='utf-8')
path=win/'training_controller_search_B200.py';text=path.read_text(encoding='utf-8')
shutil.copy2(path,path.with_name(path.name+'.pre-partial-resume-20261005'))
old='        if exit_code:\n            raise RuntimeError(last_line or f"終了コード {exit_code}")'
new=old+'\n        return last_line'
assert text.count(old)==1;text=text.replace(old,new)
old='''        self.run_stage(
            f"自己対局 {games}局（{workers}並列・一手{simulations}回探索）",
            [PYTHON, "-u", "parallel_selfplay.py", "--games", str(games),
             "--workers", str(workers), "--simulations", str(simulations),
             "--max-plies", "512", "--retries", "2",
             "--checkpoint", CHECKPOINT, "--output-dir", data_dir,
             "--late-temperature", "1", "--seed", str(20401004 + (self.cycles + 1) * 1000),
             "--experiment-cycle", str(self.cycles + 1)],
        )'''
new='''        missing = list(range(1, games + 1)) if verify else self.cycle_plan["missing_game_numbers"]
        completed = 0 if verify else len(self.cycle_plan["completed_game_numbers"])
        self.root.after(0, lambda: self.append_log(f"保存済み{completed}局を保持。残り{len(missing)}局だけを生成します。"))
        if missing:
            self.run_stage(
                f"自己対局 残り{len(missing)}局（{workers}並列・一手{simulations}回探索）",
                [PYTHON, "-u", "parallel_selfplay.py", "--games", str(games),
                 "--game-numbers", ",".join(str(number) for number in missing),
                 "--workers", str(workers), "--simulations", str(simulations),
                 "--max-plies", "512", "--retries", "2",
                 "--checkpoint", CHECKPOINT, "--output-dir", data_dir,
                 "--late-temperature", "1", "--seed", str(20401004 + (self.cycles + 1) * 1000),
                 "--experiment-cycle", str(self.cycles + 1)],
            )'''
assert text.count(old)==1;text=text.replace(old,new)
text=text.replace('            self.games_total += games','            self.games_total = (self.cycles + 1) * games')
old='''                    self.run_stage("サイクル開始前のデータ・学習状態を照合中",
                        [PYTHON, "-u", "check_search_cycle.py", "--experiment", EXPERIMENT, "--cycle", str(cycle), "--stage", "before"])'''
new='''                    planning = self.run_stage("サイクル開始前のデータ・学習状態を照合中",
                        [PYTHON, "-u", "check_search_cycle.py", "--experiment", EXPERIMENT, "--cycle", str(cycle), "--stage", "before", "--resume-partial"])
                    self.cycle_plan = json.loads(planning)'''
assert text.count(old)==1;text=text.replace(old,new)
path.write_text(text,encoding='utf-8')
shutil.copy2(path,work/'training_controller_search_B200_continuation.py')
print('Resume saved games by cycle/seed; generate missing game numbers only.')
