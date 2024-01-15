"""
Code for the project Machu-Picchu written by Théo Verhelst
Supervisors at Orange: Denis Mercier, Jeevan Shrestha
Academic supervision: Gianluca Bontempi
"""

import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np

#      z
#   +------+
# y |      | w
#   |      |
#   +------+
#      x

class HalfAxes(mpl.axes.Axes):
    def setup_bottom_left(self):
        self.setup(True)

    def setup_top_right(self):
        self.setup(False)

    def setup(self, bottom_left):
        if bottom_left:
            vertical = "bottom"
            opp_vertical = "top"
            opp_horizontal = "right"
            grid_clip = [[0, 0], [0, 1], [1, 0]]
        else:
            vertical = "top"
            opp_vertical = "bottom"
            opp_horizontal = "left"
            self.xaxis.tick_top()
            self.yaxis.tick_right()
            self.xaxis.set_label_position("top")
            self.yaxis.set_label_position("right")
            self.invert_xaxis()
            self.invert_yaxis()
            grid_clip = [[1, 1], [1, 0], [0, 1]]

        self.spines[opp_vertical].set_visible(False)
        self.spines[opp_horizontal].set_visible(False)
        self.add_line(mpl.lines.Line2D(
            [0, 1], [1, 0],
            transform=self.transAxes,
            color="black",
            linewidth=self.spines[vertical].get_linewidth()
        ))
        # To ensure the Axes is square even with data imbalance
        self.set_box_aspect(1)
        # To make sure the grid is below data patches
        self.set_axisbelow(True)
        # Add a grid but clip it
        self.grid(
            visible=True,
            clip_on=True,
            clip_path=mpl.patches.Polygon(
                grid_clip,
                transform=self.transAxes
            )
        )
        # Clip the patch as well
        self.patch.set_clip_path(mpl.patches.Polygon(
                grid_clip,
                transform=self.transAxes
            )
        )

    def update_datalim(self, xys, updatex=True, updatey=True):
        super().update_datalim(xys, updatex=updatex, updatey=updatey)
        eps = 0.01 # A small margin to stay away from the diagonal

        if self.dataLim.width == 0 and self.dataLim.height == 0:
            # If we are here, the plot contains zero or one point.
            # We want to put the current center of dataLim in the lower left
            # quarter of the Axes. We artificially expand dataLim to
            # that end.
            x0, x1 = self.get_xbound()
            y0, y1 = self.get_ybound()
            self.dataLim.x0 -= eps
            self.dataLim.y0 -= eps
            self.dataLim.x1 += 3 * eps
            self.dataLim.y1 += 3 * eps
            return

        # If the dataLim is already a non-null rectangle,
        # we want to make sure nothing is in the upper-right
        # triangle. This is done with some basic algebra.
        x0 = self.dataLim.xmin
        x1 = self.dataLim.xmax
        y0 = self.dataLim.ymin
        y1 = self.dataLim.ymax
        a = (y1 - y0) / (x0 - x1)
        b = y1 - a * x0

        for xp, yp in self.dataLim.get_points():
            bp = yp - a * xp
            # If the point is above the diagonal line
            if np.isfinite(bp) and bp + eps >= b:
                # Update x1 and y1 so that the point is lower
                # than the diagonal line, plus a small margin
                new_y1 = a * x0 + bp + eps
                new_x1 = (y0 - bp - eps) / a
                self.dataLim.x1 = new_x1
                self.dataLim.y1 = new_y1
                b = new_y1 - a * x0



class Simplex4DAxes:
    def __init__(self, fig, sep_margin=0.02, side_margin=0.18):
        l = side_margin
        b = side_margin
        w = 1 - side_margin - l
        h = 1 - side_margin - b

        self.xy_ax = fig.add_axes(
            [l, b, w, h],
            aspect="equal",
            axes_class=HalfAxes
        )
        self.xy_ax.setup_bottom_left()

        self.zw_ax = fig.add_axes(
            [l + sep_margin, b + sep_margin, w, h],
            aspect="equal",
            axes_class=HalfAxes
        )
        self.zw_ax.setup_top_right()

    def scatter(self, x, y, z, w, *args, **kwargs):
        self.xy_ax.scatter(x, y, *args, **kwargs)
        return self.zw_ax.scatter(z, w, *args, **kwargs)

    def plot_bounds(self, ub, lb, *args, **kwargs):
        # Trick to have a line in the legend instead of a Patch
        # (which is the default with FancyArrowPatch), we remove
        # the label from the kwargs, and we re-insert it in
        # dummy lines
        label = None
        if "label" in kwargs:
            label = kwargs["label"]
            del kwargs["label"]
            self.xy_ax.add_line(mpl.lines.Line2D(
                [], [], label=label, *args, **kwargs
            ))
            self.zw_ax.add_line(mpl.lines.Line2D(
                [], [], label=label, *args, **kwargs
            ))

        self.xy_ax.add_patch(mpl.patches.FancyArrowPatch(
            [lb[1], ub[0]], [ub[1], lb[0]],
            mutation_scale=5,
            arrowstyle="|-|",
            *args, **kwargs
        ))
        self.zw_ax.add_patch(mpl.patches.FancyArrowPatch(
            [lb[3], ub[2]], [ub[3], lb[2]],
            mutation_scale=5,
            arrowstyle="|-|",
            *args, **kwargs
        ))

    def set_xlabel(self, label, *args, **kwargs):
        self.xy_ax.set_xlabel(label, *args, **kwargs)

    def set_ylabel(self, label, *args, **kwargs):
        self.xy_ax.set_ylabel(label, *args, **kwargs)

    def set_zlabel(self, label, *args, **kwargs):
        self.zw_ax.set_xlabel(label, *args, **kwargs)

    def set_wlabel(self, label, *args, **kwargs):
        self.zw_ax.set_ylabel(label, *args, **kwargs)

    def set_xlim(self, lb, ub, *args, **kwargs):
        self.xy_ax.set_xlim(lb, ub, *args, **kwargs)

    def set_ylim(self, lb, ub, *args, **kwargs):
        self.xy_ax.set_ylim(lb, ub, *args, **kwargs)

    def set_zlim(self, lb, ub, *args, **kwargs):
        self.zw_ax.set_xlim(ub, lb, *args, **kwargs)

    def set_wlim(self, lb, ub, *args, **kwargs):
        self.zw_ax.set_ylim(ub, lb, *args, **kwargs)
