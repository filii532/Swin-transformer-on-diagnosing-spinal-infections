import os
import shutil
import subprocess
from utils import create_folder

dir = os.path.dirname(os.path.abspath(__file__))

print("\nCreating CSV file")
command = f'python {os.path.join(dir, "process_csv.py")}'
subprocess.run(command, shell=True)

print("\nNPZ File Creating")
command = f'python {os.path.join(dir, "process_npz.py")}'
subprocess.run(command, shell=True)

print("\nDeleting Ineffecitve Files & Layers")
command = f'python {os.path.join(dir, "process_del.py")}'
subprocess.run(command, shell=True)

print("\nFixing the Too-dark File")
command = f'python {os.path.join(dir, "process_clahe.py")}'
subprocess.run(command, shell=True)

print("\nSlicing CT")
command = f'python {os.path.join(dir, "slice_ct.py")}'
subprocess.run(command, shell=True)

print("\nSlicing MRI")
command = f'python {os.path.join(dir, "slice_mri.py")}'
subprocess.run(command, shell=True)

print("\nGenerating the JPG and Contoured JPG")
command = f'python {os.path.join(dir, "process_contour.py")}'
subprocess.run(command, shell=True)

print("\nCopy 3d image")
if os.path.exists(r'/data/npz_data/bone/inf/3d'):
    shutil.rmtree(r'/data/npz_data/bone/inf/3d')
shutil.copytree(r'/data/npz_data/bone/inf/trans/data', r'/data/npz_data/bone/inf/3d')