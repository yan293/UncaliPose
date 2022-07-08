"""
   This scipt contains functions for visualization.
   Author: Yan Xu
   Date: Nov 29, 2021
"""

from matplotlib import gridspec
import matplotlib.pyplot as plt
from matplotlib import patches
import pickle as pkl
import numpy as np
import glob
import copy
import cv2
import os


COLORS = ['r', 'g', 'b', 'y', 'm', 'c', 'lime', 'tab:blue', 'tab:orange',
          'tab:green', 'tab:red', 'tab:purple', 'tab:brown',
          'tab:pink', 'tab:gray', 'tab:olive', 'tab:cyan']
# COLORS = ['b', 'b', 'b', 'b', 'b']
# COLORS = ['y', 'y', 'y', 'y', 'y']
# COLORS = ['r', 'r', 'r', 'r', 'r']
# COLORS = ['c', 'c', 'c', 'c', 'c']
# COLORS = ['lime', 'lime', 'lime', 'lime', 'lime']
# COLORS = plt.cm.hsv(np.linspace(0, 1, 5))


# ------------------------- plot 2D ------------------------- #

def readImage(img_file, normalize=False):
    '''
    Load image using opencv.
        Input: img_file, the image file.
        Output: img, a 3 channel (WxHxC) or 2 channel (WxH) image
    '''
    img = plt.imread(img_file)
    if normalize:
        img = cv2.normalize(img, None, alpha=0, beta=1,
                            norm_type=cv2.NORM_MINMAX,
                            dtype=cv2.CV_32F)  # normalize
    return img


def plotSingleImage(image, title=None, cmap=None,
                       norm=None, interpolation=None):
    '''
    Display single image with matplotlib.
    
    Input: image, (WxHxC) or (WxH) image in the format of numpy array.
           title, title of image.
           cmap, color map.
           norm, instance used to scale scalar data to the [0, 1] range.
           interpolation, interpolation method.
    Output: img, a 3 channel (WxHxC) or 2 channel (WxH) image.
    '''
    title = title if title is not None else ""
    fig = plt.figure()
    ax = plt.subplot(1, 1, 1)
    ax.axis('off')
    ax.imshow(image, cmap=cmap, norm=norm, interpolation=interpolation)
    ax.set_title(title, fontsize=9)
    return fig, ax


def plotBoxOnImage(img, bboxes, color='r'):
    '''
    Display bounding boxes on image.
    
    Input: img,  (WxHxC) or (WxH) image in the format of numpy array.
           bboxes,  [Nx4] bounding boxes [xtl, ytl, xbr, ybr].
    Output: fig, ax,  return the figure and axis handlers.
    '''
    fig, ax = plotSingleImage(img)
    for i in range(len(bboxes)):
        xtl = bboxes[i, 0]
        ytl = bboxes[i, 1]
        xbr = bboxes[i, 2]
        ybr = bboxes[i, 3]
        if np.isnan(xtl):
            continue
        bbox = patches.Rectangle(
            (xtl, ytl), xbr-xtl, ybr-ytl, linewidth=2, edgecolor=color)
        ax.add_patch(bbox)
        ax.text(np.mean([xtl, xbr])*0.99, ytl*0.99, str(i+1),
                fontweight='bold',fontsize=12, c='w')
        fig.tight_layout()
    return fig, ax


def plotScattersOnImage(image, pts_2d, colors=None, cmaps=None,
                        title=None, cmap=None, marker='o',
                        norm=None, interpolation=None,
                        save_img_file=None, img_show=False,
                        texts=None, fig=None, ax=None):
    '''
    Display points with different colors and text descriptions.
    
    Input: image,  (WxHxC) or (WxH) image in the format of numpy array.
           pts_2d,  Nx2 2D points.
           colors,  color map for the points.
           marker,  the shape of the points.
           save_img_file,  the saved image file, don't save if "None".
           img_show,  show the image if not "None".
           texts,  the texts for the points.
           fig, ax,  the figure and axis handlers, create new if None.
    Output: fig, ax,  return the figure and axis handlers.
    '''

    if colors is None:
        colors = [None] * len(pts_2d)
    if fig is None and ax is None:
        fig, ax = plotSingleImage(image, title=title, cmap=cmap,
                                  norm=norm, interpolation=interpolation)
    for i in range(len(pts_2d)):
        if pts_2d[i, 0] or pts_2d[i, 1]:
            ax.scatter(pts_2d[i, 0], pts_2d[i, 1], c=colors[i],
                       s=100, marker=marker, linewidths=2)
            if texts is None:
                ax.text(pts_2d[i, 0]*1.01, pts_2d[i, 1]*1.01,
                        str(i + 1), fontweight='bold', fontsize=10, c='w')
            else:
                ax.text(pts_2d[i, 0]*1.01, pts_2d[i, 1]*1.01, texts[i],
                        fontweight='bold', fontsize=10, c='w')
    fig.tight_layout()
    if save_img_file:
        fig.savefig(save_img_file, bbox_inches='tight', pad_inches=0)
    if img_show:
        plt.show()
    plt.rcParams.update({'figure.max_open_warning': 0})
    return fig, ax


def plotMultipleImages(image, titles=None, cols=4, cmap=None, norm=None,
                       interpolation=None):
    """Display the given set of images, optionally with titles.
        Input: images,  list or array of image tensors in HWC format.
               titles,  optional. A list of titles to display with each image.
               cols,  number of images per row
               cmap,  optional. Color map to use. For example, "Blues".
               norm,  optional. A Normalize instance to map values to colors.
               interpolation,  image interpolation method.
        Output: fig,  return the figure handler.
    """
    titles = titles if titles is not None else [""] * len(images)
    cols = min(cols, len(images))
    rows = len(images) // cols + 1
    fig = plt.figure(figsize=(14, 14 * rows // cols))

    i = 1
    for image, title in zip(images, titles):
        ax_i = plt.subplot(rows, cols, i)
        ax_i.set_title(title, fontsize=9)
        # ax_i.axis('off')
        ax_i.imshow(image.astype(np.uint8), cmap=cmap,
                   norm=norm, interpolation=interpolation)
        i += 1
    fig.tight_layout()
    return fig


def plotMultiViewBoxClusters(image_files, titles=None, cols=1, cmap=None,
                             norm=None, interpolation=None):
    '''
    Plot boxes of the same cluster cropped from different views.
    '''
    images, titles = [], []
    for image_file in sorted(image_files):
        image = readImage(image_file)
        cam_id = image_file.split('/')[-1].split('_')[0]
        person_id = image_file.split('/')[-1].split('_')[1]
        images.append(image)
        titles.append('Camera {} | Person {}'.format(cam_id, person_id))
        
    cols = max(cols, len(images))
    rows = 1
    fig = plt.figure(figsize=(2*cols, 4))

    i = 1
    for image, title in zip(images, titles):
        ax_i = plt.subplot(rows, cols, i)
        ax_i.set_title(title, fontsize=9)
        # ax_i.axis('off')
        ax_i.imshow(image.astype(np.uint8), cmap=cmap,
                   norm=norm, interpolation=interpolation)
        i += 1
    fig.tight_layout()
    return fig


def plot2DJointsInImage(image_file, joints_dict, body_edges=None,
                        ax=None, title=None, figsize=(7, 7), colors=None,
                        plot_joint_id=True, save_file=None):
    '''
    Plot the 2D joints on the image.
    
    Input:
        image_file, the image file.
        joints_dict, {(frame_id, person_id): [2xN]/[4xN]}.
    ''' 
    im = plt.imread(image_file)
    
    if ax is None:
        fig = plt.figure(figsize=figsize)
        ax = fig.add_subplot(1, 1, 1)
    ax.set_title(title)
    ax.imshow(im)
    ax.set_autoscale_on(False)
    
    persons = list(joints_dict.keys())
    if colors is None:
        colors = COLORS
    
    person_to_color = {}
    for i in range(len(persons)):
        if persons[i] not in person_to_color:
            person_to_color[persons[i]] = colors[i]
    
    for person, joints in joints_dict.items():
        
        # --- check if joint valid or not
        if joints.shape[0] == 4:  # panoptic
            valid = joints[3, :] > 0.1
        elif joints.shape[0] == 3:  # with confidence
            valid = joints[2, :] > 0.1
        elif joints.shape[0] == 2:
            valid = np.arange(joints.shape[1])
        
        # --- plot bones
        if body_edges is not None:
            for edge in body_edges:
                if valid[edge[0]] or valid[edge[1]]:
                    ax.plot(joints[0,edge], joints[1,edge],
                            color=person_to_color[person], lw=3.0)
                    # ax.plot(joints[0,edge], joints[1,edge],
                    #         color=person_to_color[person], lw=7.0)
        
        # --- plot joints
        ax.plot(joints[0,valid], joints[1,valid], '.',
                color=person_to_color[person], ms=9.0)
        # ax.plot(joints[0,valid], joints[1,valid], '.',
        #         color='fuchsia', ms=20.0)
        # ax.plot(joints[0,valid], joints[1,valid], '.',
        #         color=person_to_color[person], ms=20.0)
        
        # --- plot joint numbers
        if plot_joint_id:
            n_joints = joints.shape[1]
            for j in range(n_joints):
                pt_inside_row = joints[0, j] >= 0 and joints[0, j] < im.shape[1]
                pt_inside_col = joints[1, j] >= 0 and joints[1,j] < im.shape[0]
                if pt_inside_row and pt_inside_col:
                    ax.text(joints[0, j], joints[1, j] - 5, '{0}'.format(j),
                            color=person_to_color[person])
        plt.draw()
    
    if save_file is not None:
        ax.axis('off')
        fig = ax.get_figure()
        fig.tight_layout()
        fig.savefig(save_file, bbox_inches='tight')
    
    return ax


def plotSingleFrameMultiView2dJoints(dataset, frame_id, n_fig_cols=3,
                                     figsize=(15, 7), plot_joint_id=True,
                                     color=None, save_plot=False):
    '''
    Plot the 2D skeleton on the image from different camera views.
    
    Input:
        dataset, the dataset.
        frame_id, frame id (integer).
    '''
    multiview_joints_dict = dataset.getSingleFrameMultiView2DJoints(frame_id)
    camera_name_list = sorted(list(multiview_joints_dict.keys()))
    n_cams = len(camera_name_list)
    ncols = min(n_cams, n_fig_cols)
    nrows = np.ceil(float(n_cams) / ncols).astype(int)
    fig = plt.figure(figsize=figsize)
    gs = gridspec.GridSpec(nrows, ncols)
    
    for i, camera_name in enumerate(camera_name_list):
        print('Plot joints in {} | camera {} | frame {}.'.format(
            dataset.data_name, camera_name, frame_id))
        image_file = dataset.fetchVideoFrameFile(camera_name, frame_id)
        joints_dict = multiview_joints_dict[camera_name]
        ax = fig.add_subplot(gs[i])
        
        title = 'Frame {0} | {1}'.format(frame_id, camera_name)
        
        colors = None
        if color is not None: colors = [color] * len(joints_dict.keys())
            
        plot2DJointsInImage(image_file, joints_dict, ax=ax,title=title,
            body_edges=dataset.body_edges, plot_joint_id=plot_joint_id,
            colors=colors)
        
        if save_plot:
            save_dir = os.path.join(dataset.data_dir, str(frame_id).zfill(8))
            save_dir = save_dir.replace('data', 'result')
            if not os.path.exists(save_dir): os.makedirs(save_dir)
            save_file = os.path.join(save_dir, camera_name+'_joints'+'.png')
            
            ax_save = plot2DJointsInImage(image_file, joints_dict,
                body_edges=dataset.body_edges, plot_joint_id=plot_joint_id,
                colors=colors, save_file=save_file)
            plt.close(ax_save.get_figure())
    fig.tight_layout()
    

# ------------------------- plot 3D ------------------------- #    

def plot3DPoints(pts_3d, colors=None, marker=None, size=100,
                 hide_zero=True, fig=None, ax=None, cmap=None,
                 norm=None, title=None, interpolation=None,
                 texts=None, save_img_file=None):
    '''
    Display 3D points in the form of plt.scatter(x, y, z).
    
    Input: pts_3d,  [Nx3] 3D points in the format of numpy array.
           colors,  color map for the points.
           size,  the size of the points.
           marker,  the shape of the points.
           save_img_file,  the saved image file, don't save if "None".
           img_show,  show the image if not "None".
           texts,  the texts for the points.
           fig, ax,  the figure and axis handlers, create new if None.
    Output: fig, ax,  return the figure and axis handlers.
    '''
    if fig is None:
        fig = plt.figure()
    if ax is None:
        ax = fig.add_subplot(1, 1, 1, projection='3d')
    if colors is None:
        colors = [None] * len(pts_3d)
    if title:
        ax.set_title(title)
    for i in range(len(pts_3d)):
        alpha = 1.
        if hide_zero and (i == 0 or i == 1) and np.sum(pts_3d[i]) == 0:
            alpha = 0.
        ax.scatter(pts_3d[i, 0], pts_3d[i, 1], pts_3d[i, 2], c=colors[i],
                   marker=marker, s=size, zorder=2, alpha=alpha)
        if texts is None:
            continue
            ax.text(pts_3d[i, 0]*1.05, pts_3d[i, 1]*1.05, pts_3d[i, 2]*1.05,
                    str(i + 1), fontweight='bold', zorder=1, alpha=alpha)
        else:
            continue
            ax.text(pts_3d[i, 0]*1.05, pts_3d[i, 1]*1.05, pts_3d[i, 2]*1.05,
                    texts[i], fontweight='bold', zorder=1, alpha=alpha)
    ax.set_xlabel('X (mm)')
    ax.set_ylabel('Y (mm)')
    ax.set_zlabel('Z (mm)')
    fig.tight_layout()
    if save_img_file:
        fig.savefig(save_img_file, bbox_inches='tight', pad_inches=0)
    return fig, ax


def plotSingle3DHumanPose(joints_3d, body_edges, plt_jnt_id=False,
                          color=None, marker=None, size=100, hide_zero=True,
                          fig=None,ax=None, cmap=None, norm=None, title=None,
                          interpolation=None,  save_img_file=None):
    '''
    Display 3D points in the form of plt.scatter(x, y, z).
    
    Input: joints_3d,  [Nx3] 3D points in the format of numpy array, invisible
               joints are denoted as "np.nan".
           body_edges, defines the connection relationship between joints.
           size,  the size of the points.
           marker,  the shape of the points.
           save_img_file,  the saved image file, don't save if "None".
           img_show,  show the image if not "None".
           plt_jnt_id,  if plot joint ids or not.
           fig, ax,  the figure and axis handlers, create new if None.
    Output: fig, ax,  return the figure and axis handlers.
    '''
    if fig is None:
        fig = plt.figure()
    if ax is None:
        ax = fig.add_subplot(1, 1, 1, projection='3d')
    if title:
        ax.set_title(title)
        
    x_span = np.nanmax(joints_3d[:, 0]) - np.nanmin(joints_3d[:, 0])
    y_span = np.nanmax(joints_3d[:, 1]) - np.nanmin(joints_3d[:, 1])
    z_span = np.nanmax(joints_3d[:, 2]) - np.nanmin(joints_3d[:, 2])
        
    # plot joints
    for i in range(len(joints_3d)):
        joint = joints_3d[i]
        if np.isnan(joint[0]): continue
        alpha = 1.
        if hide_zero and (i == 0 or i == 1) and np.sum(joint) == 0:
            alpha = 0.
        if not isinstance(color, str): color = color.reshape(1, -1)
        ax.scatter(joint[0], joint[1], joint[2], c=color,
                   marker=marker, s=size, zorder=2, alpha=alpha)
        if plt_jnt_id:
            loc = joint + np.array([x_span, y_span, z_span]) * 0.02
            ax.text(loc[0], loc[1], loc[2], str(i),
                    fontweight='bold', zorder=1, alpha=alpha)
    # plot edges
    joints_id = np.arange(len(joints_3d))
    for edge in body_edges:
        if edge[0] in joints_id and edge[1] in joints_id:
            p1 = np.argwhere(joints_id == edge[0])[0][0]
            p2 = np.argwhere(joints_id == edge[1])[0][0]
            line = joints_3d[np.array([p1, p2])]
            ax.plot(line[:, 0], line[:, 1], line[:, 2], color=color)
    return fig, ax


def plotMulti3DHumanPoses(jnts_3d_multi,n_persons,body_edges,plt_jnt_id=False,
                          color_list=None, marker=None, size=100, cmap=None,
                          hide_zero=True, norm=None, title=None,
                          interpolation=None, save_img_file=None,
                          show_label=True, xlim=None, ylim=None, zlim=None,
                          video_visual=False):
    '''
    Display 3D points in the form of plt.scatter(x, y, z).
    
    Input: jnts_3d_multi,  [Nx3] 3D points in the format of numpy array,
               invisible joints are denoted as "np.nan".
           n_persons, number of persons.
           body_edges, defines the connection relationship between joints.
           color,  color for the points.
           size,  the size of the points.
           marker,  the shape of the points.
           save_img_file,  the saved image file, don't save if "None".
           img_show,  show the image if not "None".
           plt_jnt_id,  if plot joint ids or not.
           fig, ax,  the figure and axis handlers, create new if None.
    Output: fig, ax,  return the figure and axis handlers.
    '''
    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1, projection='3d')
    if title:
        ax.set_title(title)
    n_joints = len(np.array(jnts_3d_multi)) // n_persons
    if color_list is None:
        color_list = COLORS
    
    person_joints_list = []
    for i in range(n_persons):
        person_joints_list.append(jnts_3d_multi[i*n_joints:(i+1)*n_joints])
    
    for i, person_joints in enumerate(person_joints_list):
        if np.sum(~np.isnan(person_joints)) == 0: continue
        plotSingle3DHumanPose(
            person_joints, body_edges, fig=fig, ax=ax, color=color_list[i],
            size=30, plt_jnt_id=plt_jnt_id)
    
    if xlim is not None:
        ax.set_xlim(xlim[0], xlim[1])
    if ylim is not None:
        ax.set_ylim(ylim[0], ylim[1])
    if zlim is not None:
        ax.set_zlim(zlim[0], zlim[1])
    if not show_label:
        ax.set_xticklabels([])
        ax.set_yticklabels([])
        ax.set_zticklabels([])
    fig.tight_layout()
    
    
    if video_visual:
        ax.get_xaxis().set_ticks([])
        ax.get_yaxis().set_ticks([])
        ax.get_zaxis().set_ticks([])
        ax.get_xaxis().line.set_linewidth(0)
        ax.get_yaxis().line.set_linewidth(0)
        ax.get_zaxis().line.set_linewidth(0)
        ax.w_xaxis.set_pane_color((1.0, 1.0, 1.0, 1.0)) # Hide YZ Plane
        ax.w_yaxis.set_pane_color((1.0, 1.0, 1.0, 1.0)) # Hide XZ Plane
        ax.grid(False)
    
    
    if save_img_file:
        img_name = save_img_file.split('/')[-1]
        save_dir = save_img_file[:-len(img_name)]
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        fig.savefig(
            save_img_file, bbox_inches='tight', pad_inches=0,transparent=True)
        plt.close()
    else:
        return fig, ax
    

# ------------------------- plot paper curves ------------------------- #

def normalizeToZeroOne(X):
    '''
    Normalize the data to [0, 1]
    '''
    X_ = copy.deepcopy(X)
    # X_ = np.array(X_) - min(X_)
    X_ = np.array(X_) / max(X_)
    return X_

def absSecondDerivative(X):
    '''
    Comput numerical 2nd-order derivative, pad 0s to the beginning and end.
    '''
    res = [0]
    for i in range(1, len(X) - 1):
        left_derivative = X[i] - X[i - 1]
        right_derivative = X[i + 1] - X[i]
        res.append(abs(right_derivative - left_derivative))
    res.append(0)
    return res

def plotElbowCurves(clustering_metrics, n_clusters=4):
    '''
    Plot sum square error, Silhouette Coefficient, Calinski Harabasz Score.
    '''
    x0 = n_clusters
    xs = np.array(clustering_metrics['cluster_number'])
    ys_sse = np.array(clustering_metrics['sum_square_error'])
    ys_sse_2nd_deriv = np.array(absSecondDerivative(ys_sse))
    ys_sc = np.array(clustering_metrics['silhouette_score'])
    ys_ch = np.array(clustering_metrics['calinski_harabasz_score'])
    
    ys_sse_norm = normalizeToZeroOne(ys_sse)
    ys_sse_2nd_deriv_norm = normalizeToZeroOne(ys_sse_2nd_deriv)
    ys_sc_norm = normalizeToZeroOne(ys_sc)    
    ys_ch_norm = normalizeToZeroOne(ys_ch)

    # --- plot raw data
    fig1, ((ax1, ax2), (ax3, ax4)) = plt.subplots(
        nrows=2, ncols=2, sharex=False, figsize=(12, 9))
    
    ax1.set_title('Sum Square Error (SSE)')
    ax1.plot(xs, ys_sse)
    ax1.vlines(x=xs, ymin=0, ymax=ys_sse, colors='purple', ls=':', lw=2)
    ys_sse_0 = ys_sse[np.where(np.array(xs)==x0)[0][0]]
    ax1.vlines(x=x0, ymin=0, ymax=ys_sse_0, colors='green', ls='--', lw=2)
    
    ax2.set_title('ABS 2nd-order derivative of SSE')
    ax2.plot(xs, ys_sse_2nd_deriv)
    ax2.vlines(x=xs, ymin=0,ymax=ys_sse_2nd_deriv,colors='purple',ls=':',lw=2)
    ys_sse_diff_0 = ys_sse_2nd_deriv[np.where(np.array(xs)==x0)[0][0]]
    ax2.vlines(x=x0, ymin=0, ymax=ys_sse_diff_0, colors='green', ls='--',lw=2)

    ax3.set_title('Silhouette Coefficient Score')
    ax3.plot(xs, ys_sc)
    ax3.vlines(x=xs, ymin=0, ymax=ys_sc, colors='purple', ls=':', lw=2)
    ys_sc_0 = ys_sc[np.where(np.array(xs)==x0)[0][0]]
    ax3.vlines(x=x0, ymin=0, ymax=ys_sc_0, colors='green', ls='--', lw=2)

    ax4.set_title('Calinski Harabasz Score')
    ax4.plot(xs, ys_ch)
    ax4.vlines(x=xs, ymin=0, ymax=ys_ch, colors='purple', ls=':', lw=2)
    ys_ch_0 = ys_ch[np.where(np.array(xs)==x0)[0][0]]
    ax4.vlines(x=x0, ymin=0, ymax=ys_ch_0, colors='green', ls='--', lw=2)
    

    # --- plot processed data
    fig2, ((ax5, ax6), (ax7, ax8)) = plt.subplots(
        nrows=2, ncols=2, sharex=False, figsize=(12, 9))
    
    yy = ys_sse_2nd_deriv_norm * ys_ch_norm * ys_ch_norm
    ax5.set_title('Weighted Combination Metric')
    ax5.plot(xs, yy)
    ax5.vlines(x=xs, ymin=0, ymax=yy, colors='purple', ls=':', lw=2)
    yy_0 = yy[np.where(np.array(xs)==x0)[0][0]]
    ax5.vlines(x=x0, ymin=0, ymax=yy_0, colors='green', ls='--',lw=2)
    
    ax6.set_title('ABS 2nd-order derivative of SSE')
    ax6.plot(xs, ys_sse_2nd_deriv_norm)
    ax6.vlines(x=xs, ymin=0,ymax=ys_sse_2nd_deriv_norm,colors='purple',ls=':',lw=2)
    ys_sse_diff_norm_0 = ys_sse_2nd_deriv_norm[np.where(np.array(xs)==x0)[0][0]]
    ax6.vlines(x=x0, ymin=0, ymax=ys_sse_diff_norm_0, colors='green', ls='--',lw=2)

    ax7.set_title('Silhouette Coefficient Score')
    ax7.plot(xs, ys_sc_norm)
    ax7.vlines(x=xs, ymin=0, ymax=ys_sc_norm, colors='purple', ls=':', lw=2)
    ys_sc_norm_0 = ys_sc_norm[np.where(np.array(xs)==x0)[0][0]]
    ax7.vlines(x=x0, ymin=0, ymax=ys_sc_norm_0, colors='green', ls='--', lw=2)

    ax8.set_title('Calinski Harabasz Score')
    ax8.plot(xs, ys_ch_norm)
    ax8.vlines(x=xs, ymin=0, ymax=ys_ch_norm, colors='purple', ls=':', lw=2)
    ys_ch_norm_0 = ys_ch_norm[np.where(np.array(xs)==x0)[0][0]]
    ax8.vlines(x=x0, ymin=0, ymax=ys_ch_norm_0, colors='green', ls='--', lw=2)
    
    plt.show()
    