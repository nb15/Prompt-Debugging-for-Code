#from human_eval.data import read_problems, write_jsonl, stream_jsonl
from evalplus.data import get_human_eval_plus, get_mbpp_plus, write_jsonl, get_human_eval_plus
import glob 
from tqdm import tqdm
import argparse

from typing import Iterable, Dict
import gzip
import json
import os



def stream_jsonl(filename: str) -> Iterable[Dict]:
    """
    Parses each jsonl line and yields it as a dictionary
    """
    if filename.endswith(".gz"):
        with open(filename, "rb") as gzfp:
            with gzip.open(gzfp, 'rt') as fp:
                for line in fp:
                    if any(not x.isspace() for x in line):
                        yield json.loads(line)
    else:
        with open(filename, "r") as fp:
            for line in fp:
                if any(not x.isspace() for x in line):
                    yield json.loads(line)


def remove_unindented_lines(code, ok_starts):
    new_code = ""
    for line in code.splitlines():
        if any([line.startswith(t) for t in ok_starts]) or line.strip() == "":
            new_code += line + "\n"
            continue

        lspace = len(line) - len(line.lstrip())
        if lspace == 0:
            continue

        new_code += line + "\n"

    return new_code


parser = argparse.ArgumentParser()


#parser.add_argument('--file', default='./generated_code/mbpp_wizardcoder_7B_1.jsonl', type=str, help="")
#parser.add_argument('--file', required=True, type=str, help="")
#parser.add_argument('--out-file', default='./generated_code/deltas/mbpp_wizardcoder_7B_1', type=str, help="")
#parser.add_argument('--dataset', default='mbpp', type=str, help="")
parser.add_argument('--base-dir', default='generated_code', type=str, help="")
parser.add_argument('--out-dir', default='generated_code/deltas', type=str, help="")
parser.add_argument('--dataset', required=True, type=str, help="")
#parser.add_argument('--model', required=True, type=str, help="")

parser.add_argument('--num-deltas', type=int, default=7,help="")
#parser.add_argument('--run-id', type=int, required=True, help="")

args = parser.parse_args()

if args.dataset == 'humaneval':
    problems = get_human_eval_plus()
else:
    problems = get_mbpp_plus()


file = f'{args.base_dir}/{args.dataset}_{args.model}_run1.jsonl'
out_file = f'{args.out_dir}/{args.dataset}_{args.model}_run1'



codes = [c for c in stream_jsonl(file)]
for i,code in enumerate(codes): 
    all_code = code['all_code']

    task_idx = eval(code['task_id'].split('/')[1])
    task_id = list(problems.keys())[task_idx]

    if i%7 != 3:
        entry_point = problems[task_id]['entry_point']
    else:
        entry_point = 'func'
        if "def " + entry_point + "(" not in all_code:
            entry_point = problems[task_id]['entry_point']



    fndef = "def " + entry_point + "("
    if args.dataset == 'humaneval':
        if fndef in all_code:
            code_chunks = all_code.split(fndef)
            new_code = code_chunks[1]
            NON_CODE_EOFS = ["<|endoftext|>", "\n```", "\n</s>", "\n#", "### Testing", "# Testing", '\end', 'import']
            
            for eof in NON_CODE_EOFS:
                new_code = new_code.split(eof)[0]
            tmp = ""
            for line in new_code.splitlines():
                tmp += line + "\n"
                if line.startswith('def '):
                    break
            new_code = tmp

            new_code = fndef + new_code
            new_code = remove_unindented_lines(new_code, ok_starts=[fndef])
            new_code = code_chunks[0] + new_code
        else:
            new_code = ''
    
    code['completion'] = new_code
    if args.dataset == 'mbpp':
        code['task_id'] = task_id
    
    
#deltas_per_task = args.num_runs * args.num_deltas
#tasked_codes = [codes[i:i+deltas_per_task] for i in range(0,len(codes),deltas_per_task)]

all_tasks = {}
for delta in range(args.num_deltas):
    all_tasks[f'{out_file}_Delta_{delta}.jsonl'] = []

for idx, delta in enumerate(codes):
    #task = idx%args.num_runs
    delta_id = idx%args.num_deltas
    #delta_id = task&args.num_deltas
    all_tasks[f'{out_file}_Delta_{delta_id}.jsonl'].append(delta)

memorization = 0
if args.num_deltas == 7:
    for code in all_tasks[f'{out_file}_Delta_3.jsonl']:
        entry_point = problems[code['task_id']]['entry_point']
        if 'def func(' in code['completion']:
            code['completion'] = code['completion'].replace('def func(', f'def {entry_point}(')
        elif entry_point in code['completion']:
            #print("**Memorization**")
            memorization +=1
print(f'{memorization}')

for file, code in all_tasks.items():        
    #print("save to {}".format(file))
    write_jsonl(file, code)