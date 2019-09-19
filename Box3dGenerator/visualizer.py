import os
import math
import numpy as np
import cv2
import matplotlib
# matplotlib.use('TkAGG')
from matplotlib import pyplot as plt
from matplotlib.pyplot import cm as colormap
import vtk
from vtk_visualizer.plot3d import *
from vtk_visualizer import get_vtk_control


def visualize_bbox(scan_dir, obj):
    img_path = os.path.join(scan_dir, 'color', '{0}.jpg'.format(obj['frame_name']))

    img = cv2.imread(img_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    cv2.rectangle(img, (int(obj['dimension'][0]), int(obj['dimension'][1])),
                  (int(obj['dimension'][2]), int(obj['dimension'][3])),
                  (0, 255, 0), 15)
    cv2.putText(img, '%d %s' % (obj['instance_id'], obj['classname']),
                (max(int(obj['dimension'][0]), 15), max(int(obj['dimension'][1]) + 50, 15)),
                cv2.FONT_HERSHEY_SIMPLEX, 5, (255, 0, 0), 5)
    cv2.imshow('image', img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def visualize_epipolar_geometry(scan_dir, objects, src_frame_idx, src_bbox_idx, epipolar_line_forward,
                                dst_frame_idx, dst_bbox_idx, epipolar_line_backward):
    src_obj = objects[src_frame_idx][src_bbox_idx]
    src_img = cv2.imread(os.path.join(scan_dir, 'color', '{0}.jpg'.format(src_obj['frame_name'])))
    src_img = cv2.cvtColor(src_img, cv2.COLOR_BGR2RGB)

    dst_obj = objects[dst_frame_idx][dst_bbox_idx]
    dst_img = cv2.imread(os.path.join(scan_dir, 'color', '{0}.jpg'.format(dst_obj['frame_name'])))
    dst_img = cv2.cvtColor(dst_img, cv2.COLOR_BGR2RGB)

    img1 = src_img.copy()
    img2 = dst_img.copy()
    color = tuple(np.random.randint(0, 255, 3).tolist())
    cv2.circle(img1, (int(src_obj['center'][0]), int(src_obj['center'][1])), 30, color, -1)
    cv2.line(img2, (int(epipolar_line_forward[0][0]), int(epipolar_line_forward[0][1])),
                   (int(epipolar_line_forward[1][0]), int(epipolar_line_forward[1][1])), color, 20)
    dst_obj = objects[dst_frame_idx][dst_bbox_idx]
    cv2.rectangle(img2, (int(dst_obj['dimension'][0]), int(dst_obj['dimension'][1])),
                        (int(dst_obj['dimension'][2]), int(dst_obj['dimension'][3])), color, 15)
    cv2.putText(img2, '%s' % str(dst_bbox_idx),
                (max(int(dst_obj['dimension'][0]), 15), max(int(dst_obj['dimension'][1]), 15)),
                cv2.FONT_HERSHEY_SIMPLEX, 5, (255, 0, 0), 4)
    plt.subplot(221), plt.imshow(img1)
    plt.subplot(222), plt.imshow(img2)

    img3 = dst_img.copy()
    img4 = src_img.copy()
    sel_dst_obj = objects[dst_frame_idx][dst_bbox_idx]
    cv2.circle(img3, (int(sel_dst_obj['center'][0]), int(sel_dst_obj['center'][1])), 30, color, -1)
    cv2.line(img4, (int(epipolar_line_backward[0][0]), int(epipolar_line_backward[0][1])),
                   (int(epipolar_line_backward[1][0]), int(epipolar_line_backward[1][1])), color, 20)

    ref_obj = objects[src_frame_idx][src_bbox_idx]
    cv2.rectangle(img4, (int(ref_obj['dimension'][0]), int(ref_obj['dimension'][1])),
                        (int(ref_obj['dimension'][2]), int(ref_obj['dimension'][3])), color, 15)

    plt.subplot(223), plt.imshow(img4)
    plt.subplot(224), plt.imshow(img3)
    plt.show()
    # plt.waitforbuttonpress()


def visualize_trajectory(scan_dir, objects, trajectory):
    num_frames = len(trajectory)
    ncols = int(math.sqrt(num_frames))
    nrows = (num_frames // ncols) + 1

    for i, (frame_idx, bbox_idx) in enumerate(trajectory):
        obj = objects[frame_idx][bbox_idx]
        img_path = os.path.join(scan_dir, 'color', '{0}.jpg'.format(obj['frame_name']))

        img = cv2.imread(img_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        cv2.rectangle(img, (int(obj['dimension'][0]), int(obj['dimension'][1])),
                           (int(obj['dimension'][2]), int(obj['dimension'][3])),
                            (0, 255, 0), 15)
        cv2.putText(img, '%d %s' % (obj['instance_id'], obj['classname']),
                    (max(int(obj['dimension'][0]), 15), max(int(obj['dimension'][1])+50, 15)),
                    cv2.FONT_HERSHEY_SIMPLEX, 5, (255, 0, 0), 5)
        plt.subplot(nrows, ncols, i+1), plt.imshow(img)
    plt.show()


def write_video(video, fps, size, path):
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(path, cv2.CAP_FFMPEG, fourcc, fps, size)
    for frame in video:
        out.write(frame)
    out.release()


def visualize_trajectory_in_videos(scan_dir, frame_names, objects, trajectories):
    # read video
    video = []
    for frame_name in frame_names:
        frame_path = os.path.join(scan_dir, 'color', '{0}.jpg'.format(frame_name))
        video.append(cv2.imread(frame_path))

    for idx, trajectory in enumerate(trajectories):
        color = tuple(np.random.randint(0, 255, 3).tolist())
        for frame_idx, bbox_idx in trajectory:
            obj = objects[frame_idx][bbox_idx]
            cv2.rectangle(video[frame_idx], (int(obj['dimension'][0]), int(obj['dimension'][1])),
                          (int(obj['dimension'][2]), int(obj['dimension'][3])), color, 5)
            cv2.putText(video[frame_idx], '%d %s' % (idx, obj['classname']),
                        (max(int(obj['dimension'][0]), 15), max(int(obj['dimension'][1])+50, 15)),
                        cv2.FONT_HERSHEY_SIMPLEX, 2, color, 2)

    # show video
    while True:
        for frame in video:
            cv2.imshow('Frame', frame)
            # Press Q on keyboard to exit
            cv2.waitKey(500)
        if cv2.waitKey(0) & 0xFF == ord('q'):
            break
    cv2.destroyAllWindows()

    ## write_video
    # out_path = os.path.join('./visualize/{0}.mp4'.format(scan_dir.split('/')[-1]))
    # write_video(video, 2, (cfg.SCANNET.IMAGE_WIDTH, cfg.SCANNET.IMAGE_HEIGHT), out_path)


def visualize_rotation_angle(scan_dir, objects, traj_a, traj_b, angle):
    '''visualize the relative rotation angle between two frames in trajectory pair'''
    src_obj = objects[traj_a[-1][0]][traj_a[-1][1]]
    src_img = cv2.imread(os.path.join(scan_dir, 'color', '{0}.jpg'.format(src_obj['frame_name'])))
    src_img = cv2.cvtColor(src_img, cv2.COLOR_BGR2RGB)

    dst_obj = objects[traj_b[0][0]][traj_b[0][1]]
    dst_img = cv2.imread(os.path.join(scan_dir, 'color', '{0}.jpg'.format(dst_obj['frame_name'])))
    dst_img = cv2.cvtColor(dst_img, cv2.COLOR_BGR2RGB)

    plt.subplot(121), plt.imshow(src_img)
    plt.subplot(122), plt.imshow(dst_img)
    plt.gca().set_title(angle)
    plt.show()


def visualize_ptcloud_with_color(xyz, rgb):
    '''
    :param xyz: np.ndarray, shape (n,3), coordinate
    :param rgb: np.ndarray, shape (n,3), color information
    '''
    vtkControl = get_vtk_control(True)
    plotxyzrgb(np.hstack((xyz, rgb)))
    vtkControl.AddAxesActor(1.0)
    vtkControl.exec_()


def visualize_frustum_ptcloud_with_cam(frustum_ptcloud):
    '''
    :param frustum_ptcloud: np.ndarray, shape (n,4) [x,y,z,s], s is the class activation score \in (0,1)
    '''
    cm = colormap.get_cmap('inferno')
    color_mapper = colormap.ScalarMappable(
        norm=matplotlib.colors.Normalize(vmin=0, vmax=1, clip=True), cmap=cm)
    color = color_mapper.to_rgba(frustum_ptcloud[:, 3])[:, :3] * 255
    visualize_ptcloud_with_color(frustum_ptcloud[:,0:3], color)


def visualize_one_frustum(p_planes):

    colors = vtk.vtkNamedColors()

    planesArray = list(p_planes.flatten())
    planes = vtk.vtkPlanes()
    planes.SetFrustumPlanes(planesArray)

    frustumSource = vtk.vtkFrustumSource()
    frustumSource.ShowLinesOff()
    frustumSource.SetPlanes(planes)

    shrink = vtk.vtkShrinkPolyData()
    shrink.SetInputConnection(frustumSource.GetOutputPort())
    shrink.SetShrinkFactor(.95)

    mapper = vtk.vtkPolyDataMapper()
    mapper.SetInputConnection(shrink.GetOutputPort())

    back = vtk.vtkProperty()
    back.SetColor(colors.GetColor3d("Tomato"))

    actor = vtk.vtkActor()
    actor.SetMapper(mapper)
    actor.GetProperty().EdgeVisibilityOn()
    actor.GetProperty().SetColor(colors.GetColor3d("Banana"))
    actor.SetBackfaceProperty(back)

    # a renderer and render window
    renderer = vtk.vtkRenderer()
    renderWindow = vtk.vtkRenderWindow()
    renderWindow.SetWindowName("Frustum")
    renderWindow.AddRenderer(renderer)

    # an interactor
    renderWindowInteractor = vtk.vtkRenderWindowInteractor()
    renderWindowInteractor.SetRenderWindow(renderWindow)

    # add the actors to the scene
    renderer.AddActor(actor)
    renderer.SetBackground(colors.GetColor3d("Silver"))

    # Position the camera so that we can see the frustum
    renderer.GetActiveCamera().SetPosition(1, 0, 0)
    renderer.GetActiveCamera().SetFocalPoint(0, 0, 0)
    renderer.GetActiveCamera().SetViewUp(0, 1, 0)
    renderer.GetActiveCamera().Azimuth(30)
    renderer.GetActiveCamera().Elevation(30)
    renderer.ResetCamera()

    # render an image (lights and cameras are created automatically)
    renderWindow.Render()

    # begin mouse interaction
    renderWindowInteractor.Start()

    # frustumSource = vtk.vtkFrustumSource()
    # frustumSource.SetPlanes(planes)
    # frustumSource.Update()
    #
    # frustum = frustumSource.GetOutput()
    #
    # mapper = vtk.vtkPolyDataMapper()
    # if vtk.VTK_MAJOR_VERSION <= 5:
    #     mapper.SetInput(frustum)
    # else:
    #     mapper.SetInputData(frustum)
    #
    # actor = vtk.vtkActor()
    # actor.SetMapper(mapper)
    #
    # # a renderer and render window
    # renderer = vtk.vtkRenderer()
    # renderWindow = vtk.vtk.vtkRenderWindow()
    # renderWindow.AddRenderer(renderer)
    #
    # # an interactor
    # renderWindowInteractor = vtk.vtkRenderWindowInteractor()
    # renderWindowInteractor.SetRenderWindow(renderWindow)
    #
    # # add the actors to the scene
    # renderer.AddActor(actor)
    # renderer.SetBackground(.2, .1, .3)  # Background color dark purple
    #
    # # render an image (lights and cameras are created automatically)
    # renderWindow.Render()
    #
    # # begin mouse interaction
    # renderWindowInteractor.Start()


def test_visualize_frustum():
    colors = vtk.vtkNamedColors()

    camera = vtk.vtkCamera()
    camera.SetClippingRange(0.1, 4)
    planesArray = [0] * 24

    camera.GetFrustumPlanes(0.5, planesArray) # the first argument is the width/height aspect ratio for the viewpoint

    planes = vtk.vtkPlanes()
    planes.SetFrustumPlanes(planesArray)

    frustumSource = vtk.vtkFrustumSource()
    frustumSource.ShowLinesOff()
    frustumSource.SetPlanes(planes)

    shrink = vtk.vtkShrinkPolyData()
    shrink.SetInputConnection(frustumSource.GetOutputPort())
    shrink.SetShrinkFactor(1.0)

    mapper = vtk.vtkPolyDataMapper()
    mapper.SetInputConnection(shrink.GetOutputPort())

    back = vtk.vtkProperty()
    back.SetColor(colors.GetColor3d("Tomato"))

    actor = vtk.vtkActor()
    actor.SetMapper(mapper)
    actor.GetProperty().EdgeVisibilityOn()
    actor.GetProperty().SetColor(colors.GetColor3d("Banana"))
    actor.SetBackfaceProperty(back)

    # a renderer and render window
    renderer = vtk.vtkRenderer()
    renderWindow = vtk.vtkRenderWindow()
    renderWindow.SetWindowName("Frustum")
    renderWindow.AddRenderer(renderer)

    # an interactor
    renderWindowInteractor = vtk.vtkRenderWindowInteractor()
    renderWindowInteractor.SetRenderWindow(renderWindow)

    # add the actors to the scene
    renderer.AddActor(actor)
    renderer.SetBackground(colors.GetColor3d("Silver"))

    # Position the camera so that we can see the frustum
    renderer.GetActiveCamera().SetPosition(1, 0, 0)
    renderer.GetActiveCamera().SetFocalPoint(0, 0, 0)
    renderer.GetActiveCamera().SetViewUp(0, 1, 0)
    renderer.GetActiveCamera().Azimuth(30)
    renderer.GetActiveCamera().Elevation(30)
    renderer.ResetCamera()

    # render an image (lights and cameras are created automatically)
    renderWindow.Render()

    # begin mouse interaction
    renderWindowInteractor.Start()

if __name__ == '__main__':
    test_visualize_frustum()
