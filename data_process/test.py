import os
import SimpleITK as sitk

data_dir = r"E:\project\bone\data\raw"
modf_dir = r"E:\project\bone\grad_cam"
for i in os.listdir(data_dir):
    img_path = os.path.join(data_dir, i)
    msk_path = os.path.join(data_dir, i, "label.tar")
    name = str(sitk.ReadImage(rf"{img_path}\10000.dcm").GetMetaData('0010|0020')).zfill(10)
    os.rename(rf"{modf_dir}\{name}", rf"{modf_dir}\{i}")