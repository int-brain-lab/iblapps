from pathlib import Path

import numpy as np
import cv2
import vtk
from matplotlib import pyplot as plt  # noqa
import mayavi.mlab as mlab


def add_mesh(fig, obj_file, color=(1., 1., 1.), opacity=0.4):
    """
    Adds a mesh object from an *.obj file to the mayavi figure
    :param fig: mayavi figure
    :param obj_file: full path to a local *.obj file
    :param color: rgb tuple of floats between 0 and 1
    :param opacity: float between 0 and 1
    :return: vtk actor
    """
    reader = vtk.vtkOBJReader()
    reader.SetFileName(str(obj_file))
    reader.Update()
    mapper = vtk.vtkPolyDataMapper()
    mapper.SetInputConnection(reader.GetOutputPort())
    actor = vtk.vtkActor()
    actor.SetMapper(mapper)
    actor.GetProperty().SetOpacity(opacity)
    actor.GetProperty().SetColor(color)
    fig.scene.add_actor(actor)
    fig.scene.render()
    return mapper, actor


def figure(grid=False, **kwargs):
    """
    Creates a mayavi figure with the brain atlas mesh
    :return: mayavi figure
    """
    fig = mlab.figure(bgcolor=(1, 1, 1), **kwargs)
    # engine = mlab.get_engine() # Returns the running mayavi engine.
    obj_file = Path(__file__).parent.joinpath("root.obj")
    mapper, actor = add_mesh(fig, obj_file)

    if grid:
        # https://vtk.org/Wiki/VTK/Examples/Python/Visualization/CubeAxesActor
        cubeAxesActor = vtk.vtkCubeAxesActor()
        cubeAxesActor.SetMapper(mapper)
        cubeAxesActor.SetBounds(mapper.GetBounds())
        cubeAxesActor.SetCamera(fig.scene.renderer._vtk_obj.GetActiveCamera())
        cubeAxesActor.SetXTitle("AP (um)")
        cubeAxesActor.SetYTitle("DV (um)")
        cubeAxesActor.SetZTitle("ML (um)")
        cubeAxesActor.GetTitleTextProperty(0).SetColor(1.0, 0.0, 0.0)
        cubeAxesActor.GetLabelTextProperty(0).SetColor(1.0, 0.0, 0.0)
        cubeAxesActor.GetTitleTextProperty(1).SetColor(0.0, 1.0, 0.0)
        cubeAxesActor.GetLabelTextProperty(1).SetColor(0.0, 1.0, 0.0)
        cubeAxesActor.GetTitleTextProperty(2).SetColor(0.0, 0.0, 1.0)
        cubeAxesActor.GetLabelTextProperty(2).SetColor(0.0, 0.0, 1.0)
        cubeAxesActor.DrawXGridlinesOn()
        cubeAxesActor.DrawYGridlinesOn()
        cubeAxesActor.DrawZGridlinesOn()
        fig.scene.add_actor(cubeAxesActor)

    mlab.view(azimuth=180, elevation=0)
    mlab.view(azimuth=210, elevation=210, reset_roll=False)

    return fig


def rotating_video(output_file, mfig, fps=12, secs=6):
    # ffmpeg -i input.webm -pix_fmt rgb24 output.gif
    # ffmpeg -i certification.webm -vf scale=640:-1 -r 10 -f image2pipe -vcodec ppm - | convert -delay 5 -loop 0 - certification.gif  # noqa

    file_video = Path(output_file)
    if file_video.suffix == '.avi':
        fourcc = cv2.VideoWriter_fourcc(*'XVID')
    elif file_video.suffix == '.webm':
        fourcc = cv2.VideoWriter_fourcc(*'VP80')
    else:
        NotImplementedError(f"Extension {file_video.suffix} not supported")

    mlab.view(azimuth=180, elevation=0)
    mfig.scene.render()
    mfig.scene._lift()
    frame = mlab.screenshot(figure=mfig, mode='rgb', antialiased=True)
    w, h, _ = frame.shape
    video = cv2.VideoWriter(str(file_video), fourcc, float(fps), (h, w))

    # import time
    for e in np.linspace(-180, 180, secs * fps):
        # frame = np.random.randint(0, 256, (w, h, 3), dtype=np.uint8)
        mlab.view(azimuth=0, elevation=e, reset_roll=False)
        mfig.scene.render()
        frame = mlab.screenshot(figure=mfig, mode='rgb', antialiased=True)
        print(e, (h, w), frame.shape)
        video.write(np.flip(frame, axis=2))  # bgr instead of rgb...
        # time.sleep(0.05)

    video.release()
    cv2.destroyAllWindows()
