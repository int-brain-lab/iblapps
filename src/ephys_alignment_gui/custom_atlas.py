
import logging

import nibabel as nib
import numpy as np
import SimpleITK as sitk
from iblatlas.atlas import ALLEN_CCF_LANDMARKS_MLAPDV_UM, BrainAtlas
from iblatlas.regions import BrainRegions

_logger = logging.getLogger(__name__)

# This is a custom atlas class that inherits from BrainAtlas
class CustomAtlas(BrainAtlas):
    image = None
    label = None

    def __init__(self,
                 atlas_image_file =None,
                 atlas_labels_file = None,
                 bregma = None,
                 force_um = None,
                 scaling = np.array([1,1,1])):
        self.atlas_image_file = atlas_image_file
        self.atlas_labels_file = atlas_labels_file
        if force_um is None:
            #dxyz = np.array(self.read_atlas_image())*np.array([1, -1, -1])*1e-6
            dxyz = np.array(self.read_atlas_image()) * 1000
            print('Atlas scaling', dxyz)
            self.res_um = dxyz[0] 
            dxyz = self.res_um * 1e-6 * np.array([1, 1, 1]) * scaling
            print('Resolution', self.res_um)
        else:
            _  = self.read_atlas_image()
            self.res_um = force_um
            dxyz = self.res_um * 1e-6 * np.array([1, -1, -1]) * scaling        
        self.read_atlas_labels()
        regions = BrainRegions()
        #_, im = ismember(self.label, regions.id)
        #label = np.reshape(im.astype(np.uint16), self.label.shape)
        # Make sure we're in uint16 already (avoid extra copies)
        if self.label.dtype != np.uint16:
            self.label = self.label.astype(np.uint16, copy=False)

        # Sort region IDs (only once)
        valid = np.sort(np.asarray(regions.id, dtype=self.label.dtype))

        # Binary search lookup for all labels
        idx = np.searchsorted(valid, self.label)
        mask = (idx < len(valid)) & (valid[idx] == self.label)

        # Apply replacement in-place
        self.label[~mask] = np.uint16(997)

        
        xyz2dims = np.array([0, 1, 2])  # this is the c-contiguous ordering
        dims2xyz = np.array([0, 1, 2])
        if bregma is None:
            bregma = [0,0,0]
        elif isinstance(bregma,str) and (bregma.lower() == 'allen'):
            bregma = (ALLEN_CCF_LANDMARKS_MLAPDV_UM['bregma'] / self.res_um)
        super().__init__(self.image, self.label, dxyz, regions, iorigin=list(self.offset), dims2xyz=dims2xyz, xyz2dims=xyz2dims)
        self.label[~np.isin(self.label,regions.id)]=997

    
    def read_atlas_image(self):
        if self.atlas_image_file.suffix == ".nrrd":
            print(f"Loading nrrd file: {self.atlas_image_file.as_posix()}")
            # Reads the 
            IMG = sitk.ReadImage(self.atlas_image_file)
            # Convert sitk to the (ap, ml, dv) np array needed by BrainAtlas
            self.original_image = IMG
            self.image = np.flip(sitk.GetArrayFromImage(IMG).T, axis=(0, 2))
            print('Shape', self.image.shape)
            self.offset = IMG.GetOrigin()
            self.spacing = IMG.GetSpacing()[0] * 1000
            return IMG.GetSpacing()
        else: # nii.gz file
            print(f"Loading nii.gz file: {self.atlas_image_file.as_posix()}")
            image_lazy_loaded = nib.load(self.atlas_image_file)
            self.image = image_lazy_loaded.dataobj
            print('Shape', self.image.shape)
            self.spacing = image_lazy_loaded.header.get_zooms()[0] * 1000
            self.offset = tuple(image_lazy_loaded.affine[:3, 3])
            return tuple(image_lazy_loaded.header.get_zooms())
        
    def read_atlas_labels(self):
        IMG = sitk.ReadImage(self.atlas_labels_file)
        # Convert sitk to the (ap, ml, dv) np array needed by BrainAtlas
        self.label = np.flip(sitk.GetArrayFromImage(IMG).astype(np.int32).T, axis=(0, 2))
