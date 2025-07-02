from abc import ABC, abstractmethod
import logging
import numpy as np
from pathlib import Path
import requests
import re
from typing import Dict, Union
import SimpleITK as sitk

from iblatlas.atlas import AllenAtlas
from one import params
from one.webclient import http_download_file

logger = logging.getLogger(__name__)

# TODO docstrings and typing and logging


class SliceLoader(ABC):
    def __init__(self, file_path: Path, brain_atlas: AllenAtlas):
        """
        Abstract base class for loading histology slices.

        Parameters
        ----------
        file_path : Path
            Directory containing histology files.
        brain_atlas : AllenAtlas
            Reference brain atlas.
        """
        self.file_path = file_path
        self.brain_atlas = brain_atlas
        self.hist_paths: Dict[str, Path] = {}
        self.get_paths()

    @abstractmethod
    def get_paths(self) -> None:
        """
        Locate and store relevant histology file paths in self.hist_paths.
        """

    @abstractmethod
    def load_volume(self, vol_path: Path) -> np.ndarray:
        """
        Load a 3D volume from a file.

        Parameters
        ----------
        vol_path : Path
            Path to the volume file.

        Returns
        -------
        np.ndarray
            Loaded 3D image volume.
        """

    def get_slices(self, xyz: np.ndarray) -> Dict[str, Dict[str, Union[np.ndarray, np.ndarray]]]:
        """
        Generate slice images for CCF, annotation, and loaded histology.

        Parameters
        ----------
        xyz : np.ndarray
            Nx3 array of XYZ coordinates.

        Returns
        -------
        Dict[str, Dict]
            Dictionary of image slices and metadata for each image type.
        """
        slices = {
            'CCF': self.get_slice(xyz, self.brain_atlas.image),
            'Annotation': self.get_slice(xyz, self.brain_atlas.label, annotation=True)
        }

        for key, vol_path in self.hist_paths.items():
            try:
                vol = self.load_volume(vol_path)
                slices[key] = self.get_slice(xyz, vol)
            except Exception as e:
                logger.error(f"Failed to load {key} volume at {vol_path}: {e}")

        return slices

    def get_slice(self, xyz: np.ndarray, vol: np.ndarray, annotation=False) -> Dict[str, np.ndarray]:
        """
        Extract a slice from a 3D volume using given coordinates.

        Parameters
        ----------
        xyz : np.ndarray
            Nx3 array of XYZ coordinates.
        vol : np.ndarray
            3D volume from which to extract a slice.

        Returns
        -------
        Dict[str, np.ndarray]
            Dictionary containing the 2D slice, scale, and offset.
        """
        index = self.brain_atlas.bc.xyz2i(xyz)[:, self.brain_atlas.xyz2dims]
        width = [self.brain_atlas.bc.i2x(0), self.brain_atlas.bc.i2x(456)]
        height = [self.brain_atlas.bc.i2z(index[0, 2]), self.brain_atlas.bc.i2z(index[-1, 2])]
        hist_slice = vol[index[:, 0], :, index[:, 2]]
        if annotation:
            hist_slice = self.brain_atlas._label2rgb(hist_slice)
        hist_slice = np.swapaxes(hist_slice, 0, 1)
        return {
            'slice': hist_slice,
            'scale': np.array([(width[-1] - width[0]) / hist_slice.shape[0],
                               (height[-1] - height[0]) / hist_slice.shape[1]]),
            'offset': np.array([width[0], height[0]])
        }


class NrrdSliceLoader(SliceLoader):
    def __init__(self, file_path: Path, brain_atlas: AllenAtlas):
        """
        SliceLoader for IBL histology.

        Parameters
        ----------
        file_path : Path
            Directory containing .nrrd files.
        brain_atlas : AllenAtlas
            Brain atlas for alignment.
        """
        super().__init__(file_path, brain_atlas)

    def get_paths(self) -> None:
        """
        Load histology file paths with predefined color channel suffixes.
        """
        col_map = {'red': 'RD', 'green': 'GR'}
        files = list(self.file_path.glob("*.nrrd"))

        for color, abbrev in col_map.items():
            match = next((f for f in files if abbrev in f.name), None)
            if match:
                self.hist_paths[f'Histology {color}'] = match

    def load_volume(self, vol_path: Path) -> np.ndarray:
        """
        Load a volume using AllenAtlas.

        Parameters
        ----------
        vol_path : Path
            Path to histology volume.

        Returns
        -------
        np.ndarray
            Loaded image volume.
        """
        return AllenAtlas._read_volume(vol_path)


def download_histology_data(subject: str, laboratory: str):

    # If we detect >= 2 nrrd file we assume the histology data already exists
    cache_dir = params.get_cache_dir().joinpath(laboratory, 'Subjects', subject, 'histology')
    expected_files = list(cache_dir.glob("*.nrrd"))

    if len(expected_files) >= 2:
        return expected_files, cache_dir

    # Otherwise we attempt to download files
    lab_hist = 'mrsicflogellab' if laboratory == 'hoferlab' else laboratory

    par = params.get()

    def _find_histology_folder(subj: str, lab: str):
        flatiron_path = Path('histology', lab, subj, 'downsampledStacks_25', 'sample2ARA')
        url = f"{par.HTTP_DATA_SERVER}/{'/'.join(flatiron_path.parts)}"
        try:
            response = requests.get(url, auth=(par.HTTP_DATA_SERVER_LOGIN, par.HTTP_DATA_SERVER_PWD))
            response.raise_for_status()
            return flatiron_path, response.text
        except Exception as e:
            logger.warning(f"Failed to find path for lab={lab}, subject={subj}: {e}")
            return None

    attempts = [
        (subject, lab_hist),
        (subject.replace("_", ""), lab_hist)
    ]

    if lab_hist == 'churchlandlab_ucla':
        attempts.append((subject, 'churchlandlab'))

    histology_folder = None
    for subj, lab in attempts:
        histology_folder = _find_histology_folder(subj, lab)
        if histology_folder:
            break

    if not histology_folder:
        logger.error(f"Could not find histology folder for subject={subject}, lab={laboratory}")
        return None, cache_dir

    rel_path, html_text = histology_folder
    base_url = f"{par.HTTP_DATA_SERVER}/{'/'.join(rel_path.parts)}"

    tif_files = [match + ".tif" for match in re.findall(r'href="(.*).tif"', html_text)]

    cache_dir.mkdir(exist_ok=True, parents=True)
    path_to_files = []
    for file in tif_files:
        img_path = Path(cache_dir, file)
        if not img_path.exists():
            file_url = f"{base_url}/{file}"
            http_download_file(file_url, target_dir=cache_dir,
                               username=par.HTTP_DATA_SERVER_LOGIN,
                               password=par.HTTP_DATA_SERVER_PWD)
        path_to_files.append(tif2nrrd(img_path))

    if len(path_to_files) > 3:
        path_to_files = path_to_files[1:3]

    return path_to_files, cache_dir


def tif2nrrd(path_to_image):

    path_to_nrrd = Path(path_to_image).with_suffix('.nrrd')
    if not path_to_nrrd.exists():
        reader = sitk.ImageFileReader()
        reader.SetImageIO("TIFFImageIO")
        reader.SetFileName(str(path_to_image))
        img = reader.Execute()

        new_img = sitk.PermuteAxes(img, [2, 1, 0])
        new_img = sitk.Flip(new_img, [True, False, False])
        new_img.SetSpacing([1, 1, 1])
        writer = sitk.ImageFileWriter()
        writer.SetImageIO("NrrdImageIO")
        writer.SetFileName(str(path_to_nrrd))
        writer.Execute(new_img)

    return path_to_nrrd
