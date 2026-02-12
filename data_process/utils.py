import numpy as np
import pydicom
import os
import scipy.ndimage
import shutil
import SimpleITK as sitk
import nrrd
import tarfile
import nibabel as nib
import warnings
warnings.filterwarnings("error")

def read_dicom(path, if_resample=False, wcct: int = 350, wwct: int = 1000, if_pydicom = False, if_del = False):
    '''
    输入dicom文件，进行排序，转换成以HU为单位的图像，便于返回三维数组
    :param path: dicom文件所在的目录地址
    :param if_resample: 是否需要resample
    :return: 三维 np数组图像
    '''
    paths = sitk.ImageSeriesReader.GetGDCMSeriesFileNames(path)
    if if_del:
        try:
            slice_locations = [float(sitk.ReadImage(s).GetMetaData('0020|1041')) for s in paths]
        except:
            slice_locations = [float(pydicom.dcmread(s)[0x00201041].value) for s in paths]
        indexes_to_remove = set()
        for i in range(len(slice_locations)):
            for j in range(i + 1, len(slice_locations)):
                if slice_locations[i] == slice_locations[j]:
                    indexes_to_remove.add(j)
        for idx, s in enumerate(paths):
            try:
                pydicom.dcmread(s)
            except Warning as e:
                # print(e)
                indexes_to_remove.add(idx)
            except:
                pass
        paths = [paths[i] for i in range(len(paths)) if i not in indexes_to_remove]

    if if_pydicom:
        slices = [pydicom.dcmread(s) for s in paths]
        image = np.array([s.pixel_array for s in slices], dtype=np.float64)
        if_sitk = False
    else:
        try:
            slices = [sitk.ReadImage(s) for s in paths]
            image = np.array([sitk.GetArrayFromImage(slice) for slice in slices], dtype=np.float64)
            if_sitk = True
        except:
            slices = [pydicom.dcmread(s) for s in paths]
            image = np.array([s.pixel_array for s in slices], dtype=np.float64)
            if_sitk = False

    # try:
    #     slice_thickness = np.abs(slices[0].ImagePositionPatient[2] - slices[1].ImagePositionPatient[2])
    # except:
    #     slice_thickness = np.abs(slices[0].SliceLocation - slices[1].SliceLocation)

    # for s in slices:
    #     s.SliceThickness = slice_thickness

    out = []
    for i, slice in enumerate(slices):
        img = image[i]
        '''
        Modality: 00080060
        Window Center: 00281050
        Window Width: 00281051
        Rescale Intercept: 00281052
        Rescale Slope: 00281053
        '''
        modality = slice.GetMetaData("0008|0060") if if_sitk else slice[0x00080060].value

        if modality == 'CT':
            if not if_sitk:
                try:
                    interception = int(slice[0x00281052].value[0])
                except:
                    interception = int(slice[0x00281052].value)
                try:
                    slope = int(slice[0x00281053].value[0])
                except:
                    slope = int(slice[0x00281053].value)
                img = img * slope + interception
            wc = wcct
            ww = wwct
        else:
            try:
                ww = int(slice.GetMetaData("0028|1051")[0]) if if_sitk else int(slice[0x00281051].value[0])
            except:
                ww = int(slice.GetMetaData("0028|1051")) if if_sitk else int(slice[0x00281051].value)
            try:
                wc = int(slice.GetMetaData("0028|1050")[0]) if if_sitk else int(slice[0x00281050].value[0])
            except:
                wc = int(slice.GetMetaData("0028|1050")) if if_sitk else int(slice[0x00281050].value)

        img_min = wc - ww // 2
        img_max = wc + ww // 2
        img[img < img_min] = img_min
        img[img > img_max] = img_max
        if (np.max(img) - np.min(img)) == 0:
            return read_dicom(path, if_pydicom=True, if_del=if_del)
        img = (img - np.min(img))  * 255 // (np.max(img) - np.min(img))
        out.append(img)
        
    image = np.array(out, dtype=np.int16)
    if if_resample:
        image, _ = resample(image, slices, [1, 1, 1])
    image = (image - np.min(image))  * 255 // (np.max(image) - np.min(image))
    return np.squeeze(np.array(image, dtype=np.uint8))

def resample(image, scan, new_spacing=[1,1,1]):
    '''

    :param image:
    :param scan:
    :param new_spacing:
    :return:
    '''
    # 确定当前像素间距:切片厚度+切片间距
    spacing = np.array([scan[0].SliceThickness] + list(scan[0].PixelSpacing), dtype=np.float32)

    resize_factor = spacing / new_spacing
    new_real_shape = image.shape * resize_factor
    new_shape = np.round(new_real_shape)  # 进行四舍五入，默认小数位0
    real_resize_factor = new_shape / image.shape
    new_spacing = spacing / real_resize_factor

    image = scipy.ndimage.zoom(image, real_resize_factor, mode='nearest')

    return image, new_spacing

def dicom_attr_de(path, attr: str):
    '''
    :param path:
    :param attr:
    :return: 
    Modality: 00080060
    Name: 00100010
    ID: 00100020
    scan_date: 00080020
    series: 0008103E 00081030 00400254
    birth: 00100030
    '''
    paths = sitk.ImageSeriesReader.GetGDCMSeriesFileNames(path)
    attr = attr.replace(' ', '').lower()
    try:
        slices = [pydicom.dcmread(s) for s in paths]
        if_sitk = False
    except:
        slices = [sitk.ReadImage(s) for s in paths]
        if_sitk = True
    try:
        attr = slices[0].GetMetaData(f"{attr[:4]}|{attr[4:]}") if if_sitk else slices[0][eval(f"0x{attr}")].value
    except:
        attr = slices[-1].GetMetaData(f"{attr[:4]}|{attr[4:]}") if if_sitk else slices[-1][eval(f"0x{attr}")].value

    return attr

def create_folder(path:str, empty:bool = True):
    if empty and os.path.exists(path):
        shutil.rmtree(path)
    if not os.path.exists(path):
        os.makedirs(path)

def read_nrrd(mask_path: str):
    mask, _ = nrrd.read(mask_path)
    mask = np.rot90(mask, k=3)
    mask = np.fliplr(mask)
    mask = np.transpose(mask, [2, 0, 1])
    mask[mask > 0] = 1
    return np.squeeze(np.array(mask, dtype=np.uint8))

def dice(a, b):
    return (2*np.sum(a*b)+1e-4) / (np.sum(a+b) + 1e-4)

def check_dim(a, b):
    if len(a) != len(b):
        return False
    for i in range(len(a)):
        if a[i] != b[i]:
            return False
    return True

def del_layer(img, layers):
    if isinstance(layers[0], list):
        for i in layers[::-1]:
            img = del_layer(img, i)
        return img
    else:
        if layers[0] == 0:
            return img[layers[1]+1:]
        else:
            return img[:layers[0]]

def read_nii(mask_path: str):
    temp_dir = r'temp'
    create_folder(temp_dir)
    with tarfile.open(mask_path, 'r') as tar:
        tar.extractall(temp_dir)
    for mask_path in os.listdir(temp_dir):
        if mask_path[-2:] == 'gz':
            break
    mask = nib.load(fr"{temp_dir}\{mask_path}").get_fdata()
    mask = np.moveaxis(mask, np.argmin(mask.shape), -1)
    mask = np.rot90(mask, k=3)
    mask = np.fliplr(mask)
    mask = np.transpose(mask, [2, 0, 1])
    mask[mask > 0] = 1
    shutil.rmtree(temp_dir)
    return np.squeeze(np.array(mask, dtype=np.uint8))