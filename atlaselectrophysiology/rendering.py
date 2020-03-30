from pathlib import Path

import numpy as np
import cv2
import vtk

import mayavi.mlab as mlab


def figure():
    """
    Creates a mayavi figure with the brain atlas mesh
    :return: mayavi figure
    """
    fig = mlab.figure(bgcolor=(1, 1, 1))
    # engine = mlab.get_engine() # Returns the running mayavi engine.
    obj_file = "/home/olivier/.brainrender/Data/Meshes/Mouse/root.obj"
    reader = vtk.vtkOBJReader()
    reader.SetFileName(obj_file)
    reader.Update()
    mapper = vtk.vtkPolyDataMapper()
    mapper.SetInputConnection(reader.GetOutputPort())
    actor = vtk.vtkActor()
    actor.SetMapper(mapper)
    actor.GetProperty().SetOpacity(0.6)
    fig.scene.add_actor(actor)

    mlab.view(azimuth=180, elevation=0)
    mlab.view(azimuth=210, elevation=210, reset_roll=False)

    return fig


def rotating_video(output_file, mfig, fps=12, secs=6):

    file_video = Path(output_file)
    if file_video.suffix == '.avi':
        fourcc = cv2.VideoWriter_fourcc(*'XVID')
    elif file_video.suffix == '.webm':
        fourcc = cv2.VideoWriter_fourcc(*'VP80')
    else:
        NotImplementedError(f"Extension {file_video.suffix} not supported")

    mlab.view(azimuth=180, elevation=0)
    mfig.scene.render()
    frame = mlab.screenshot(figure=mfig, mode='rgb', antialiased=False)
    w, h, _ = frame.shape
    video = cv2.VideoWriter(str(file_video), fourcc, float(fps), (h, w))

    # import time
    for e in np.linspace(-180, 180, secs * fps):
        # frame = np.random.randint(0, 256, (w, h, 3), dtype=np.uint8)
        print(e)
        mlab.view(azimuth=0, elevation=e, reset_roll=False)
        mfig.scene.render()
        frame = mlab.screenshot(figure=mfig, mode='rgb', antialiased=False)
        video.write(frame)
        # time.sleep(0.05)

    video.release()
    cv2.destroyAllWindows()
