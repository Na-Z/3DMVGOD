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
    cv2.circle(img, (648,484),5,(0,0,255))
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
    shrink.SetShrinkFactor(1.)

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
    renderWindowInteractor.SetInteractorStyle(vtk.vtkInteractorStyleTrackballCamera())
    renderWindowInteractor.SetRenderWindow(renderWindow)

    # add the actors to the scene
    renderer.AddActor(actor)
    renderer.SetBackground(colors.GetColor3d("Silver"))

    transform = vtk.vtkTransform()
    transform.Translate(0,0,0)
    axes = vtk.vtkAxesActor()
    #  The axes are positioned with a user transform
    axes.SetAxisLabels(1)
    axes.SetUserTransform(transform)
    axes.AxisLabelsOff()
    axes.SetTotalLength(3.0, 3.0, 3.0)
    renderer.AddActor(axes)

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


def visualize_one_frustum_plus_points(p_planes, frustum_ptcloud):

    colors = vtk.vtkNamedColors()

    # set a renderer and a render window
    renderer = vtk.vtkRenderer()
    renderer.SetBackground(colors.GetColor3d("Silver"))
    renderWindow = vtk.vtkRenderWindow()
    renderWindow.SetSize(600, 600)
    renderWindow.SetWindowName("Frustum Intersection")
    renderWindow.AddRenderer(renderer)

    # an interactor
    renderWindowInteractor = vtk.vtkRenderWindowInteractor()
    renderWindowInteractor.SetInteractorStyle(vtk.vtkInteractorStyleTrackballCamera())
    renderWindowInteractor.SetRenderWindow(renderWindow)

    mappers = list()

    planesArray = list(p_planes.flatten())
    planes = vtk.vtkPlanes()
    planes.SetFrustumPlanes(planesArray)

    frustumSource = vtk.vtkFrustumSource()
    frustumSource.ShowLinesOff()
    frustumSource.SetPlanes(planes)

    shrink = vtk.vtkShrinkPolyData()
    shrink.SetInputConnection(frustumSource.GetOutputPort())
    shrink.SetShrinkFactor(1.)

    mappers.append(vtk.vtkPolyDataMapper())
    mappers[-1].SetInputConnection(shrink.GetOutputPort())

    # assin color to ptcloud (n,4)
    cm = colormap.get_cmap('inferno')
    color_mapper = colormap.ScalarMappable(
        norm=matplotlib.colors.Normalize(vmin=0, vmax=1, clip=True), cmap=cm)
    color = color_mapper.to_rgba(frustum_ptcloud[:, 3])[:, :3] * 255
    ptcloud = np.hstack((frustum_ptcloud[:,0:3], color))

    # Create the geometry of a point (the coordinate)
    points = vtk.vtkPoints()
    # Create the topology of the point (a vertex)
    vertices = vtk.vtkCellArray()
    # Setup colors
    Colors = vtk.vtkUnsignedCharArray()
    Colors.SetNumberOfComponents(3)
    Colors.SetName("Colors")
    # Add points
    for i in range(0, len(ptcloud)):
        p = ptcloud[i, :3]
        id = points.InsertNextPoint(p)
        vertices.InsertNextCell(1)
        vertices.InsertCellPoint(id)
        Colors.InsertNextTuple3(ptcloud[i,3], ptcloud[i,4], ptcloud[i,5])
    point = vtk.vtkPolyData()
    # Set the points and vertices we created as the geometry and topology of the polydata
    point.SetPoints(points)
    point.SetVerts(vertices)
    point.GetPointData().SetScalars(Colors)
    point.Modified()

    mappers.append(vtk.vtkPolyDataMapper())
    mappers[-1].SetInputData(point)

    for mapper in mappers:

        actor = vtk.vtkActor()
        actor.SetMapper(mapper)
        actor.GetProperty().EdgeVisibilityOn()
        actor.GetProperty().SetColor(colors.GetColor3d("Banana"))
        actor.GetProperty().SetOpacity(.5)

        renderer.AddActor(actor)

    transform = vtk.vtkTransform()
    transform.Translate(0,0,0)
    axes = vtk.vtkAxesActor()
    #  The axes are positioned with a user transform
    axes.SetAxisLabels(1)
    axes.SetUserTransform(transform)
    axes.AxisLabelsOff()
    axes.SetTotalLength(3.0, 3.0, 3.0)
    renderer.AddActor(axes)

    renderer.SetUseDepthPeeling(1)
    renderer.SetOcclusionRatio(0.1)
    renderer.SetMaximumNumberOfPeels(100)
    renderWindow.SetMultiSamples(0)
    renderWindow.SetAlphaBitPlanes(1)

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



def visualize_n_frustums(p_planes_list):

    colors = vtk.vtkNamedColors()

    # set a renderer and a render window
    renderer = vtk.vtkRenderer()
    renderer.SetBackground(colors.GetColor3d("Silver"))
    renderWindow = vtk.vtkRenderWindow()
    renderWindow.SetSize(600, 600)
    renderWindow.SetWindowName("Frustum Intersection")
    renderWindow.AddRenderer(renderer)

    # an interactor
    renderWindowInteractor = vtk.vtkRenderWindowInteractor()
    renderWindowInteractor.SetInteractorStyle(vtk.vtkInteractorStyleTrackballCamera())
    renderWindowInteractor.SetRenderWindow(renderWindow)

    for i, p_planes in enumerate(p_planes_list):

        planesArray = list(p_planes.flatten())
        planes = vtk.vtkPlanes()
        planes.SetFrustumPlanes(planesArray)

        frustum = vtk.vtkFrustumSource()
        frustum.ShowLinesOff()
        frustum.SetPlanes(planes)

        shrink = vtk.vtkShrinkPolyData()
        shrink.SetInputConnection(frustum.GetOutputPort())
        shrink.SetShrinkFactor(1.)

        mapper = vtk.vtkPolyDataMapper()
        mapper.SetInputConnection(shrink.GetOutputPort())

        actor = vtk.vtkActor()
        actor.SetMapper(mapper)
        actor.GetProperty().EdgeVisibilityOn()
        actor.GetProperty().SetColor(colors.GetColor3d("Banana"))
        actor.GetProperty().SetOpacity(.5)

        renderer.AddActor(actor)

    transform = vtk.vtkTransform()
    transform.Translate(0,0,0)
    axes = vtk.vtkAxesActor()
    #  The axes are positioned with a user transform
    axes.SetAxisLabels(1)
    axes.SetUserTransform(transform)
    axes.AxisLabelsOff()
    axes.SetTotalLength(3.0, 3.0, 3.0)
    renderer.AddActor(axes)

    renderer.SetUseDepthPeeling(1)
    renderer.SetOcclusionRatio(0.1)
    renderer.SetMaximumNumberOfPeels(100)
    renderWindow.SetMultiSamples(0)
    renderWindow.SetAlphaBitPlanes(1)

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


def visualize_frustums_plus_interior_point(p_planes_list, interior_point):

    colors = vtk.vtkNamedColors()

    # set a renderer and a render window
    renderer = vtk.vtkRenderer()
    renderer.SetBackground(colors.GetColor3d("Silver"))
    renderWindow = vtk.vtkRenderWindow()
    renderWindow.SetSize(600, 600)
    renderWindow.SetWindowName("Frustum Intersection plus interior point")
    renderWindow.AddRenderer(renderer)

    # an interactor
    renderWindowInteractor = vtk.vtkRenderWindowInteractor()
    renderWindowInteractor.SetInteractorStyle(vtk.vtkInteractorStyleTrackballCamera())
    renderWindowInteractor.SetRenderWindow(renderWindow)

    for i, p_planes in enumerate(p_planes_list):

        planesArray = list(p_planes.flatten())
        planes = vtk.vtkPlanes()
        planes.SetFrustumPlanes(planesArray)

        frustum = vtk.vtkFrustumSource()
        frustum.ShowLinesOff()
        frustum.SetPlanes(planes)

        shrink = vtk.vtkShrinkPolyData()
        shrink.SetInputConnection(frustum.GetOutputPort())
        shrink.SetShrinkFactor(1.)

        mapper = vtk.vtkPolyDataMapper()
        mapper.SetInputConnection(shrink.GetOutputPort())

        actor = vtk.vtkActor()
        actor.SetMapper(mapper)
        actor.GetProperty().EdgeVisibilityOn()
        if i % 2 == 0:
            actor.GetProperty().SetColor(colors.GetColor3d("Tomato"))
        else:
            actor.GetProperty().SetColor(colors.GetColor3d("Banana"))
        actor.GetProperty().SetOpacity(.5)

        renderer.AddActor(actor)

    # create a sphere for the interior point
    sphereSource = vtk.vtkSphereSource()
    sphereSource.SetCenter(interior_point[0], interior_point[1], interior_point[2])
    sphereSource.SetRadius(.05)
    # Make the surface smooth.
    sphereSource.SetPhiResolution(100)
    sphereSource.SetThetaResolution(100)

    mapper = vtk.vtkPolyDataMapper()
    mapper.SetInputConnection(sphereSource.GetOutputPort())

    actor = vtk.vtkActor()
    actor.SetMapper(mapper)
    actor.GetProperty().SetColor(colors.GetColor3d("DarkGreen"))

    renderer.AddActor(actor)

    transform = vtk.vtkTransform()
    transform.Translate(0,0,0)
    axes = vtk.vtkAxesActor()
    #  The axes are positioned with a user transform
    axes.SetAxisLabels(1)
    axes.SetUserTransform(transform)
    axes.AxisLabelsOff()
    axes.SetTotalLength(3.0, 3.0, 3.0)
    renderer.AddActor(axes)

    renderer.SetUseDepthPeeling(1)
    renderer.SetOcclusionRatio(0.1)
    renderer.SetMaximumNumberOfPeels(100)
    renderWindow.SetMultiSamples(0)
    renderWindow.SetAlphaBitPlanes(1)

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



def visualize_frustums_intersection(p_planes_list, intersections):

    colors = vtk.vtkNamedColors()

    # set a renderer and a render window
    renderer = vtk.vtkRenderer()
    renderer.SetBackground(colors.GetColor3d("Silver"))
    renderWindow = vtk.vtkRenderWindow()
    renderWindow.SetSize(600, 600)
    renderWindow.SetWindowName("Frustum Intersection plus intersection points")
    renderWindow.AddRenderer(renderer)

    # an interactor
    renderWindowInteractor = vtk.vtkRenderWindowInteractor()
    renderWindowInteractor.SetInteractorStyle(vtk.vtkInteractorStyleTrackballCamera())
    renderWindowInteractor.SetRenderWindow(renderWindow)

    for i, p_planes in enumerate(p_planes_list):

        planesArray = list(p_planes.flatten())
        planes = vtk.vtkPlanes()
        planes.SetFrustumPlanes(planesArray)

        frustum = vtk.vtkFrustumSource()
        frustum.ShowLinesOff()
        frustum.SetPlanes(planes)

        shrink = vtk.vtkShrinkPolyData()
        shrink.SetInputConnection(frustum.GetOutputPort())
        shrink.SetShrinkFactor(1.)

        mapper = vtk.vtkPolyDataMapper()
        mapper.SetInputConnection(shrink.GetOutputPort())

        actor = vtk.vtkActor()
        actor.SetMapper(mapper)
        actor.GetProperty().EdgeVisibilityOn()
        if i % 2 == 0:
            actor.GetProperty().SetColor(colors.GetColor3d("Tomato"))
            actor.GetProperty().SetOpacity(.5)
        else:
            actor.GetProperty().SetColor(colors.GetColor3d("Banana"))

        renderer.AddActor(actor)

    # create sphere for all the intersection point
    for i in range(intersections.shape[0]):
        point = intersections[i]
        sphereSource = vtk.vtkSphereSource()
        sphereSource.SetCenter(point[0], point[1], point[2])
        sphereSource.SetRadius(.05)
        # Make the surface smooth.
        sphereSource.SetPhiResolution(100)
        sphereSource.SetThetaResolution(100)

        mapper = vtk.vtkPolyDataMapper()
        mapper.SetInputConnection(sphereSource.GetOutputPort())

        actor = vtk.vtkActor()
        actor.SetMapper(mapper)
        actor.GetProperty().SetColor(colors.GetColor3d("DarkGreen"))

        renderer.AddActor(actor)

    transform = vtk.vtkTransform()
    transform.Translate(0,0,0)
    axes = vtk.vtkAxesActor()
    #  The axes are positioned with a user transform
    axes.SetAxisLabels(1)
    axes.SetUserTransform(transform)
    axes.AxisLabelsOff()
    axes.SetTotalLength(3.0, 3.0, 3.0)
    renderer.AddActor(axes)

    renderer.SetUseDepthPeeling(1)
    renderer.SetOcclusionRatio(0.1)
    renderer.SetMaximumNumberOfPeels(100)
    renderWindow.SetMultiSamples(0)
    renderWindow.SetAlphaBitPlanes(1)

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