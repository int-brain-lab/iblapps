
from pathlib import Path
import requests
import re
from ibllib.io import params
from oneibl.webclient import http_download_file
import SimpleITK as sitk

def download_histology_data(subject, lab):

    if lab == 'hoferlab':
        lab_temp = 'mrsicflogellab'
    else:
        lab_temp = lab

    par = params.read('one_params')

    try:
        FLAT_IRON_HIST_REL_PATH = Path('histology', lab_temp, subject, 'downsampledStacks_25', 'sample2ARA')
        baseurl = (par.HTTP_DATA_SERVER + '/' + '/'.join(FLAT_IRON_HIST_REL_PATH.parts))
        r = requests.get(baseurl, auth=(par.HTTP_DATA_SERVER_LOGIN, par.HTTP_DATA_SERVER_PWD))
        r.raise_for_status()
    except Exception as err:
        print(err)
        try:
            subject_rem = subject.replace("_", "")
            FLAT_IRON_HIST_REL_PATH = Path('histology', lab_temp, subject_rem, 'downsampledStacks_25', 'sample2ARA')
            baseurl = (par.HTTP_DATA_SERVER + '/' + '/'.join(FLAT_IRON_HIST_REL_PATH.parts))
            r = requests.get(baseurl, auth=(par.HTTP_DATA_SERVER_LOGIN, par.HTTP_DATA_SERVER_PWD))
            r.raise_for_status()
        except Exception as err:
            print(err)
            path_to_nrrd = None
            return path_to_nrrd

    tif_files = []
    for line in r.text.splitlines():
        result = re.findall('href="(.*)RD.tif"', line)
        if result:
            tif_files.append(result[0] + 'RD.tif')

    CACHE_DIR = Path(Path.home(), 'Downloads', 'FlatIron', lab, 'Subjects', subject, 'histology')
    CACHE_DIR.mkdir(exist_ok=True, parents=True)
    path_to_image = Path(CACHE_DIR, tif_files[0])
    if not path_to_image.exists():
        url = (baseurl + '/' + tif_files[0])
        http_download_file(url, cache_dir=CACHE_DIR,
                           username=par.HTTP_DATA_SERVER_LOGIN,
                           password=par.HTTP_DATA_SERVER_PWD)

    path_to_nrrd = tif2nrrd(path_to_image)

    return path_to_nrrd


def tif2nrrd(path_to_image):
    path_to_nrrd = Path(path_to_image.parent, path_to_image.parts[-1][:-3] + 'nrrd')
    if not path_to_nrrd.exists():
        reader = sitk.ImageFileReader()
        reader.SetImageIO("TIFFImageIO")
        reader.SetFileName(str(path_to_image))
        img = reader.Execute()

        #img = sitk.ReadImage(str(path_to_image))
        new_img = sitk.PermuteAxes(img, [2, 1, 0])
        new_img = sitk.Flip(new_img, [True, False, False])
        writer = sitk.ImageFileWriter()
        writer.SetImageIO("NrrdImageIO")
        writer.SetFileName(str(path_to_nrrd))
        writer.Execute(new_img)
        #sitk.WriteImage(new_img, str(path_to_nrrd))

    return path_to_nrrd






