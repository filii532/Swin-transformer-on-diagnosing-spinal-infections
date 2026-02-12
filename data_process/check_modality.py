import os

data_path = r'F:\yqy\bone\data_npz\3d'
modalities = {}

for path in os.listdir(data_path):
    try:
        modalities[path[:10]].append(path[11])
    except:
        modalities[path[:10]] = [path[11]]

comp = 0
for patient in modalities.keys():
    miss = True
    for i in range(4):
        if str(i) not in modalities[patient]:
            if miss:
                print(f'\n{patient}:')
                miss = False
            print(i)
    if miss:
        comp += 1
print("\nCompleted Patient Amount: ", comp)