from pathlib import Path

import numpy as np
import cv2
import vtk

import mayavi.mlab as mlab


def color_cycle(ind=None):
    """
    Gets the matplotlib color-cycle as RGB numpy array of floats between 0 and 1
    :return:
    """
    # import matplotlib as mpl
    # c = np.uint32(np.array([int(c['color'][1:], 16) for c in mpl.rcParams['axes.prop_cycle']]))
    # c = np.double(np.flip(np.reshape(c.view(np.uint8), (c.size, 4))[:, :3], 1)) / 255
    c = np.array([[0.12156863, 0.46666667, 0.70588235],
                  [1., 0.49803922, 0.05490196],
                  [0.17254902, 0.62745098, 0.17254902],
                  [0.83921569, 0.15294118, 0.15686275],
                  [0.58039216, 0.40392157, 0.74117647],
                  [0.54901961, 0.3372549, 0.29411765],
                  [0.89019608, 0.46666667, 0.76078431],
                  [0.49803922, 0.49803922, 0.49803922],
                  [0.7372549, 0.74117647, 0.13333333],
                  [0.09019608, 0.74509804, 0.81176471]])
    if ind is None:
        return c
    else:
        return tuple(c[ind % c.shape[0], :])


def figure(grid=False):
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
