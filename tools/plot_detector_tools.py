from __future__ import print_function
import numpy as np
import pandas as pd
from trackml.dataset import load_event
import matplotlib
import matplotlib.pyplot as plt
from tools.detector_v2 import Detector
import pickle
from itertools import cycle
import copy
import io
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
from matplotlib.figure import Figure
from mpl_toolkits.mplot3d.art3d import Poly3DCollection, Line3DCollection
from mpl_toolkits.mplot3d import Axes3D

class EventPlotter:
    def __init__(self, event_file, detector_file,
                 detector = None,
                 use_null_module = True,
                 import_track_hits = True):                      # Null module is an empty module to indicate no detector hit.
        self.hits, self.cells, self.particles, self.truth = load_event(event_file)
        if detector is None:
            self.detector = Detector(detector_file)
        else:
            self.detector = detector
        self.use_null_module = use_null_module
        self.track_hit_dict = pickle.load( open( event_file + "-trackdict.p", "rb" ) )
        
        self.init_detector_geometry()
        
    def init_detector_geometry(self):
        self.all_modules = self.detector.detector.values[:,:3]
        self.all_volumes = np.unique(self.all_modules[:,0],axis=0)
        if self.use_null_module:
            self.all_modules = np.append(self.all_modules,np.array([[0.,0.,0.]]),axis=0)
            
        self.module_verts = {}
        self.module_verts_cycle = {}
        self.module_lines = {}
        
        for module in self.all_modules[:-1]:
            vol = module[0]
            lay = module[1]
            mod = module[2]
            self.detector._load_element_info(vol,lay,mod)
            points = {}
            
            points['pp'] = np.array([self.detector.LocaluvwToGlobal_vector([self.detector.module_maxhu,self.detector.module_hv,0],vol,lay,mod)])
            points['pm'] = np.array([self.detector.LocaluvwToGlobal_vector([-self.detector.module_maxhu,self.detector.module_hv,0],vol,lay,mod)])
            points['mp'] = np.array([self.detector.LocaluvwToGlobal_vector([self.detector.module_minhu,-self.detector.module_hv,0],vol,lay,mod)])
            points['mm'] = np.array([self.detector.LocaluvwToGlobal_vector([-self.detector.module_minhu,-self.detector.module_hv,0],vol,lay,mod)])
            
            points_arr = np.concatenate((points['pp'],
                                         points['pm'],
                                         points['mm'],
                                         points['mp']),axis=0)

            
            x = points_arr[:,0]
            y = points_arr[:,2]
            z = points_arr[:,1]
            
            self.module_verts[tuple(module)] = [zip(x,y,z)]
            
            points_arr = np.concatenate((points['pp'],
                                         points['pm'],
                                         points['mm'],
                                         points['mp'],
                                         points['pp']),axis=0)
                                        
            x = points_arr[:,0]
            y = points_arr[:,2]
            z = points_arr[:,1]

            self.module_verts_cycle[tuple(module)] = [zip(x,y,z)]

            self.module_lines[tuple(module)] = np.concatenate((points['pp'],
                                                               points['pm'],
                                                               points['mm'],
                                                               points['mp'],
                                                               points['pp']),axis=0)

        if self.use_null_module:
            self.module_verts[tuple(self.all_modules[-1])] = [zip([0],[0],[0])]
            self.module_verts_cycle[tuple(self.all_modules[-1])] = [zip([0],[0],[0])]
            self.module_lines[tuple(self.all_modules[-1])] = np.array([[0,0,0]])

    '''
    This function plots zero or more tracks and a detector background onto a triplet of 2D projections in the
    xy, yz, zx planes. The detector background can be any of the following options:
 
    - 'noise_hits'
    - 'all_hits'
    - 'module_lines'.

    If it is set to anything else, there will be no background.

    Noise hits is the quickest to render. Module lines is the slowest, especially if you render all modules.
    There is the option 'volumes' to list a subset of the detector volumes to be rendered. This is the recommended usage.
    '''
    
    def prepare_2D_projection_plot(self, track_ids,
                                   bg_style = 'noise_hits',           # bg_style = 'noise_hits', 'all_hits', 'module_lines'.
                                   volumes = None,
                                   module_line_cmap = None,
                                   line_colors = None,
                                   hits = None,
                                   bg_points_size = None,
                                   bg_points_color = None):

#        bg_style_options = ['noise_hits', 'all_hits', 'module_lines']
#        if bg_style
        
        fig = plt.figure(figsize = (15,15))
        ax1 = plt.subplot2grid((4,4), (0, 0), colspan=1)
        ax2 = plt.subplot2grid((4,4), (0, 1), colspan=3)
        ax3 = plt.subplot2grid((4,4), (1, 0), rowspan=3)

        
        if bg_style == 'noise_hits':
            hits_xyz = self.get_track_hits(0)[0]
            if bg_points_size is None:
                size = 0.1
            else:
                size = bg_points_size
            if bg_points_color is None:
                color = '0.75'
            else:
                color = bg_points_color
            ax1.scatter(hits_xyz[:,1],hits_xyz[:,0],s=size,color=color)
            ax2.scatter(hits_xyz[:,2],hits_xyz[:,0],s=size,color=color)
            ax3.scatter(hits_xyz[:,1],hits_xyz[:,2],s=size,color=color)

        elif bg_style == 'all_hits':
            if bg_points_size is None:
                size = 0.002
            else:
                size = bg_points_size
            if bg_points_color is None:
                color = '0.65'
            else:
                color = bg_points_color
            hits_xyz = self.hits.values[:,1:4]
            ax1.scatter(hits_xyz[:,1],hits_xyz[:,0],s=size,color=color)
            ax2.scatter(hits_xyz[:,2],hits_xyz[:,0],s=size,color=color)
            ax3.scatter(hits_xyz[:,1],hits_xyz[:,2],s=size,color=color)
            
        elif bg_style == 'module_lines':
            if volumes is None:
                selected_modules = self.all_modules
            else:
                selected_modules = self.all_modules[np.array([np.any(volumes == module[0]) for module in self.all_modules])]
            if module_line_cmap is None:
                color_cycle = cycle(['0.5'])
                alpha = 0.2
                linewidth = 0.5
            else:
                cmap = matplotlib.cm.get_cmap(module_line_cmap)
                color_cycle = cycle([cmap(x/3.) for x in range(4)])
                alpha = 0.2
                linewidth = 0.5
            last_layer = 0
            for module in selected_modules:
                this_layer = module[1]
                if this_layer != last_layer:
                    color = color_cycle.next()
                last_layer = this_layer
                #for line in self.module_lines[tuple(module)]:
                line = self.module_lines[tuple(module)]
                ax1.plot(line[:,0],line[:,1],color=color,alpha=alpha,linewidth=linewidth)
                ax2.plot(line[:,2],line[:,1],color=color,alpha=alpha,linewidth=linewidth)
                ax3.plot(line[:,0],line[:,2],color=color,alpha=alpha,linewidth=linewidth)

        cmap = matplotlib.cm.get_cmap('tab10')
        color_cycle = cycle([cmap(i) for i in range(10)])
        for track_id in track_ids:
            linewidth = 1
            pointsize = 6
            track_hits = self.get_track_hits(track_id)[0]
            if hits is not None:
                track_hits = track_hits[hits]
            color = color_cycle.next()
            ax1.plot(track_hits[:,0],track_hits[:,1],color=color,linewidth=linewidth)
            ax2.plot(track_hits[:,2],track_hits[:,1],color=color,linewidth=linewidth)
            ax3.plot(track_hits[:,0],track_hits[:,2],color=color,linewidth=linewidth)
            ax1.scatter(track_hits[:,0],track_hits[:,1],color=color,s=pointsize)
            ax2.scatter(track_hits[:,2],track_hits[:,1],color=color,s=pointsize)
            ax3.scatter(track_hits[:,0],track_hits[:,2],color=color,s=pointsize)
            
        return fig, [ax1, ax2, ax3]

    def set_xyzlims_2D_projection_plot(self, axarr, xmin, xmax, ymin, ymax, zmin, zmax):
        axarr[0].set_xlim([xmin,xmax])
        axarr[0].set_ylim([ymin,ymax])
        axarr[1].set_xlim([zmin,zmax])
        axarr[1].set_ylim([ymin,ymax])
        axarr[2].set_xlim([xmin,xmax])
        axarr[2].set_ylim([zmin,zmax])

    def add_module_hit_outline_2D_projection_plot(self, fig, track_id, hitlist = None, copy_fig = False):
        if copy_fig:
            new_fig = self.copy_fig(fig)
        else:
            new_fig = fig
        all_hit_modules = self.get_track_hits(track_id)[1]
        if hitlist is None:
            hit_modules = all_hit_modules
        else:
            hit_modules = all_hit_modules[hitlist]

        linewidth = 2
        color = 'firebrick'
            
        ax1, ax2, ax3 = new_fig.axes
        for module in hit_modules:
            line = self.module_lines[tuple(module)]
            ax1.plot(line[:,0],line[:,1],color=color,linewidth=linewidth)
            ax2.plot(line[:,2],line[:,1],color=color,linewidth=linewidth)
            ax3.plot(line[:,0],line[:,2],color=color,linewidth=linewidth)

        return new_fig

    def add_module_hit_predictions_2D_projection_plot(self, fig, hit_modules, alphas, copy_fig = False):
        if copy_fig:
            new_fig = self.copy_fig(fig)
        else:
            new_fig = fig
        
        color = 'orangered'
        
        ax1, ax2, ax3 = new_fig.axes
        for i, module in enumerate(hit_modules):
            line = self.module_lines[tuple(module)]
            ax1.fill(line[:,0],line[:,1],color=color,alpha = alphas[i])
            ax2.fill(line[:,2],line[:,1],color=color, alpha = alphas[i])
            ax3.fill(line[:,0],line[:,2],color=color, alpha = alphas[i])
                
        return new_fig
    
    def copy_fig(self, fig):
        buf = io.BytesIO()
        pickle.dump(fig, buf)
        buf.seek(0)
        new_fig = pickle.load(buf)

        canvas = FigureCanvas(new_fig)
        canvas.print_figure('')

        return new_fig
    
    def get_track_hits(self, track_id):
        hitrows = self.hits.values[self.track_hit_dict[track_id]-1]
        track_xyz_coords = hitrows[:,1:4]
        track_module_ids = hitrows[:,4:]
        return track_xyz_coords, track_module_ids


    def prepare_3D_plot(self, track_id,
                        volumes = None,
                        module_line_cmap = None,
                        particlelinecolor = 'darkred',
                        particlepointcolor = 'darkred',
                        particlelinewidth = 3,
                        particlepointsize = 60,
                        hits = None):

        fig = plt.figure(figsize = (20,10))
        ax = fig.add_subplot(111, projection='3d')

        ax.view_init(0, 0)
        
        track_hits = self.get_track_hits(track_id)[0]
        if hits is not None:
            track_hits = track_hits[hits]

        ax.plot(track_hits[:,0],track_hits[:,2],track_hits[:,1],
                color=particlelinecolor,linewidth=particlelinewidth)
        ax.scatter(track_hits[:,0],track_hits[:,2],track_hits[:,1],
                   depthshade = False, s=particlepointsize,color=particlepointcolor)
        
        if module_line_cmap is not None:
            cmap = matplotlib.cm.get_cmap(module_line_cmap)
            color_cycle = cycle([cmap(x/3.) for x in range(4)])
        else:
            cmap = matplotlib.cm.get_cmap('viridis')
            color_cycle = cycle([cmap(x/3.) for x in range(4)])

        if volumes is None:
            selected_modules = self.all_modules
        else:
            selected_modules = self.all_modules[np.array([np.any(volumes == module[0]) for module in self.all_modules])]
        alpha = 0.2
        linewidth = 1
        last_layer = 0
        for module in selected_modules:
            this_layer = module[1]
            if this_layer != last_layer:
                color = color_cycle.next()
            last_layer = this_layer
                
            module_cell = Line3DCollection(self.module_verts_cycle[tuple(module)],
                                               edgecolor = color, alpha = alpha, linewidth = linewidth)
                
            ax.add_collection3d(module_cell)

        ax.set_xlabel('x')
        ax.set_ylabel('z')
        ax.set_zlabel('y')
        
        return fig, ax

    def add_module_hit_outline_3D_plot(self, fig, track_id, hitlist = None, copy_fig = False,
                                                  edgecolor = 'firebrick',
                                                  linewidth = 5):

        if copy_fig:
            new_fig = self.copy_fig(fig)
        else:
            new_fig = fig

        hit_modules = self.get_track_hits(track_id)[1]
        if hitlist is not None:
            hit_modules = hit_modules[hitlist]
            
        ax = new_fig.axes[0]
        for module in hit_modules:
            module_cell = Line3DCollection(self.module_verts_cycle[tuple(module)],linewidth=linewidth,edgecolor=edgecolor)
            ax.add_collection3d(module_cell)
                    
        return new_fig


    def add_module_hit_predictions_3D_plot(self, fig, hit_modules, alphas, copy_fig = False,
                                                      color = 'orangered'):
        if copy_fig:
            new_fig = self.copy_fig(fig)
        else:
            new_fig = fig
            
        color = 'orangered'
        
        ax = new_fig.axes[0]
        for i, module in enumerate(hit_modules):
            module_cell = Poly3DCollection(self.module_verts[tuple(module)],alpha=alphas[i])
            module_cell.set_facecolor(color)
            ax.add_collection3d(module_cell)

        return new_fig
