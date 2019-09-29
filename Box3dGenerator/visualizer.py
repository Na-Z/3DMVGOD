import os
import math
import numpy as np
import cv2
from mayavi import mlab
import matplotlib
# matplotlib.use('TkAGG')
from matplotlib import pyplot as plt
from matplotlib.pyplot import cm as colormap
import vtk
from vtk_visualizer.plot3d import *
from vtk_visualizer import get_vtk_control
from scipy.spatial import Delaunay

import data.scannet.scannet_utils as utils


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


# =============================== Visualize Trajectories  ================================

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


# =============================== Visualize Point Clouds  ================================

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


def draw_gt_boxes3d(gt_boxe3d, instance_id, fig, color=(1, 1, 1), line_width=1, draw_text=False, text_scale=(1, 1, 1),
                    color_list=None):
    ''' Draw 3D bounding boxes
    Args:
        gt_boxe3d: numpy array (8,3) for XYZs of the box corners
        fig: mayavi figure handler
        color: RGB value tuple in range (0,1), box line color
        line_width: box line width
        draw_text: boolean, if true, write box indices beside boxes
        text_scale: three number tuple
        color_list: a list of RGB tuple, if not None, overwrite color.
    Returns:
        fig: updated fig
    '''
    b = gt_boxe3d
    if color_list is not None:
        color = color_list[0]
    if draw_text: mlab.text3d(b[4, 0], b[4, 1], b[4, 2], '%d' % instance_id, scale=text_scale, color=color, figure=fig)
    for k in range(0, 4):
        # http://docs.enthought.com/mayavi/mayavi/auto/mlab_helper_functions.html
        i, j = k, (k + 1) % 4
        mlab.plot3d([b[i, 0], b[j, 0]], [b[i, 1], b[j, 1]], [b[i, 2], b[j, 2]], color=color, tube_radius=None,
                    line_width=line_width, figure=fig)

        i, j = k + 4, (k + 1) % 4 + 4
        mlab.plot3d([b[i, 0], b[j, 0]], [b[i, 1], b[j, 1]], [b[i, 2], b[j, 2]], color=color, tube_radius=None,
                    line_width=line_width, figure=fig)

        i, j = k, k + 4
        mlab.plot3d([b[i, 0], b[j, 0]], [b[i, 1], b[j, 1]], [b[i, 2], b[j, 2]], color=color, tube_radius=None,
                    line_width=line_width, figure=fig)
    return fig


def visualize_bbox3d_in_whole_scene(scan_dir, scan_name, instance_id):
    '''visualize the whole point cloud and the ground truth 3D bounding box with respect to instance_id'''
    def compute_box_3d(obj_pc):
        xmin = np.min(obj_pc[:, 0])
        ymin = np.min(obj_pc[:, 1])
        zmin = np.min(obj_pc[:, 2])
        xmax = np.max(obj_pc[:, 0])
        ymax = np.max(obj_pc[:, 1])
        zmax = np.max(obj_pc[:, 2])
        bbox = np.array([(xmin + xmax) / 2, (ymin + ymax) / 2, (zmin + zmax) / 2,
                         xmax - xmin, ymax - ymin, zmax - zmin])
        return bbox

    def compute_box_corners(bbox):
        x, y, z = bbox[0:3]
        l, w, h = bbox[3:6]/2
        x_corners = [-l, l, l, -l, -l, l, l, -l]
        y_corners = [w, w, -w, -w, w, w, -w, -w]
        z_corners = [h, h, h, h, -h, -h, -h, -h]
        corners_3d = np.vstack([x_corners, y_corners, z_corners])
        corners_3d[0, :] += x
        corners_3d[1, :] += y
        corners_3d[2, :] += z
        return corners_3d.transpose()

    # Load point cloud and align it with axis
    mesh_file = os.path.join(scan_dir, '{0}_vh_clean_2.ply'.format(scan_name))
    mesh_vertices = utils.read_mesh_vertices_rgb(mesh_file)

    meta_file_path = os.path.join(scan_dir, '{0}.txt'.format(scan_name))
    axis_align_matrix = utils.read_axis_align_matrix(meta_file_path)
    pts = np.ones((mesh_vertices.shape[0], 4))
    pts[:,0:3] = mesh_vertices[:,0:3]
    pts = np.dot(pts, axis_align_matrix.transpose()) # Nx4
    mesh_vertices[:,0:3] = pts[:,0:3]

    # Load semantic and instance labels
    agg_file = os.path.join(scan_dir, '{0}.aggregation.json'.format(scan_name))
    seg_file = os.path.join(scan_dir, '{0}_vh_clean_2.0.010000.segs.json'.format(scan_name))
    object_id_to_segs, label_to_segs = utils.read_aggregation2(agg_file)
    seg_to_verts, num_verts = utils.read_segmentation(seg_file)
    instance_ids = np.zeros(shape=(num_verts), dtype=np.uint32)  # 0: unannotated
    for object_id, segs in object_id_to_segs.items():
        for seg in segs:
            verts = seg_to_verts[seg]
            instance_ids[verts] = object_id

    # Get the points inside the instance
    obj_pc = mesh_vertices[instance_ids == instance_id, 0:3]
    bbox = compute_box_3d(obj_pc) #[x,y,z,l,w,h]
    corners_3d = compute_box_corners(bbox) #(8,3)

    bgcolor=(0,0,0)
    fig = mlab.figure(figure=None, bgcolor=bgcolor, fgcolor=None, engine=None, size=(1600, 1000))
    mlab.points3d(mesh_vertices[:,0], mesh_vertices[:,1], mesh_vertices[:,2], mesh_vertices[:,2], mode='point',
                                colormap='gnuplot', figure=fig)
    mlab.points3d(obj_pc[:, 0], obj_pc[:, 1], obj_pc[:, 2], color=(1, 1, 1), mode='point', scale_factor=1,
                  figure=fig)
    mlab.points3d(0, 0, 0, color=(1,1,1), mode='sphere', scale_factor=0.2)
    draw_gt_boxes3d(corners_3d, instance_id, fig=fig)
    mlab.orientation_axes()
    mlab.show()


def visualize_convex_hull_plus_ptcloud_static(intersection_points, ptcloud, inside):
    hull =  Delaunay(intersection_points)

    # plot the convex hull
    fig = plt.figure()
    ax = fig.add_subplot(111, projection="3d")

    # Plot defining corner points
    ax.plot(intersection_points.T[0], intersection_points.T[1], intersection_points.T[2], "ko")

    # 12 = 2 * 6 faces are the simplices (2 simplices per square face)
    for s in hull.simplices:
        s = np.append(s, s[0])  # Here we cycle back to the first coordinate
        ax.plot(intersection_points[s, 0], intersection_points[s, 1], intersection_points[s, 2], "r-")

    # Make axis label
    for i in ["x", "y", "z"]:
        eval("ax.set_{:s}label('{:s}')".format(i, i))

    # plot tested points `p` - black are inside hull, red outside
    ax.plot(ptcloud[ inside,0],ptcloud[ inside,1],ptcloud[ inside,2],'.g')
    ax.plot(ptcloud[~inside,0],ptcloud[~inside,1],ptcloud[~inside,2],'.b')
    plt.show()


def visualize_convex_hull_plus_ptcloud_interactive(intersection_points, ptcloud, inside):
    pts_inside = ptcloud[inside, :]
    pts_outside = ptcloud[~inside, :]

    bgcolor=(0,0,0)
    fig = mlab.figure(figure=None, bgcolor=bgcolor, fgcolor=None, engine=None, size=(1600, 1000))

    mlab.points3d(pts_inside[:,0], pts_inside[:,1], pts_inside[:,2], color=(1,0,0), mode='point',
                  scale_factor=1, figure=fig) #red
    mlab.points3d(pts_outside[:, 0], pts_outside[:, 1], pts_outside[:, 2], color=(1, 1, 1), mode='point',
                  scale_factor=1, figure=fig) #white

    mlab.points3d(intersection_points[:,0], intersection_points[:,1], intersection_points[:,2], color=(0,0,1),
                  mode='cube', scale_factor=0.1) #blue
    mlab.points3d(0, 0, 0, color=(1,1,1), mode='sphere', scale_factor=0.2)
    mlab.orientation_axes()
    mlab.show()



# ========================== Visualize Frustums with VTK ==========================

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


def visualize_n_frustums_plus_ptclouds(p_planes_list, frustum_ptcloud):
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

    # add n frustums
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
        actor.GetProperty().SetOpacity(.2)

        renderer.AddActor(actor)

    # add ptclouds
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

    mapper = vtk.vtkPolyDataMapper()
    mapper.SetInputData(point)

    actor = vtk.vtkActor()
    actor.SetMapper(mapper)
    actor.GetProperty().EdgeVisibilityOn()
    actor.GetProperty().SetColor(colors.GetColor3d("Tomato"))
    actor.GetProperty().SetOpacity(.3)

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
            actor.GetProperty().SetOpacity(.2)
        else:
            actor.GetProperty().SetColor(colors.GetColor3d("Banana"))
            actor.GetProperty().SetOpacity(.4)

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