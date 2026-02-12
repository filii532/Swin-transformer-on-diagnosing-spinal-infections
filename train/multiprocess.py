import concurrent.futures
import multiprocessing
import subprocess
import time
import os
os.system('clear')

def get_free_gpu():
    result = subprocess.check_output(["nvidia-smi", "--query-gpu=memory.free", "--format=csv,nounits,noheader"]).decode("utf-8")
    memory_free = [int(x) for x in result.strip().split('\n')]
    return [i for i, mem in enumerate(memory_free) if mem > 24064]

devices = get_free_gpu()

def run_command(model, seed, device):
    command = '/home/hzt/Application/miniconda3/envs/torch/bin/python /home/hzt/code/bone/pre_classification/pl_train.py'
    command = f'{command} --model {model} --seed {seed} --device {device} --epochs 200 --is_train True'
    # command = f'nohup {command} >pl{device}.log 2>&1 &'
    print(command)
    subprocess.run(command, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    devices.append(device)

if __name__ == "__main__":
    with concurrent.futures.ThreadPoolExecutor(max_workers=len(devices)) as executor:
        for i in range(1, 10):
            for _ in range(3):
                for model in ['swin_small', 'swin_large', 'deit_base', 'deit_small', ]:
                    executor.submit(run_command, model, i, devices[0])
                    time.sleep(10)