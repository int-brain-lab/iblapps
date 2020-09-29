from atlasview import atlasview  # mouais il va falloir changer ça - naming is terrible !!
av = atlasview.viewatlas()  # need to have an output argument here or the garbage collector will clean it all up

""" Roadmap
Mapping voxel to axes:
    - use Qtransforms to map voxels to real world coordinates.
    - use BrainAtlas as a model
    This will allow to support effortlessly
        -   Allen coordinates
        -   Voxel coordinates
        -   IBL coordinates
        -   different resolutions per axes (Needles)
        -   tilted slices
        -   slices for all 3 planes for all of the above

Create a new window with a view from above (use the surface in the AllenAtlas object).
Add 2 lines and a callback that will update the slice when the line is moved.
Should we have a coronal slice window and a sagittal slice window ?

Use the layer combobox to switch between image / brain regions

Label the brain region lookup at the bottom right with the hover function
"""
