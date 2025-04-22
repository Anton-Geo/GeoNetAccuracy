import os
import numpy as np
import math
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from points import BasePoints, RefinedPoints, EstimatedPoints
from measurements import LinearMeasurements, AngularMeasurements


def plot_geodetic_network(points, measurements, instrument_stations,
                          ellipse_scale=1.0, plan_scale=1000, show_stations=True,
                          project_title='Geodetic Network Project'):
    fig, ax = plt.subplots()

    # Size of symbols and labels in mm
    point_size_mm = 2
    arrow_size_mm = 2
    text_size_mm = 3
    axis_label_size_mm = 3
    cross_size_mm = 3
    coord_text_size_mm = 2

    # Converting sizes to terrain scale
    point_size = point_size_mm / 1000 * plan_scale
    arrow_size = arrow_size_mm / 1000 * plan_scale
    text_size = text_size_mm / 1000 * plan_scale
    axis_label_size = axis_label_size_mm / 1000 * plan_scale
    cross_size = cross_size_mm / 1000 * plan_scale
    coord_text_size = coord_text_size_mm / 1000 * plan_scale

    # Visualization of network points with labels
    for point in points:
        if isinstance(point, BasePoints):
            marker = '^'
            edgecolor = 'black'
            facecolor = 'none'
            size = 1.25 * point_size
        elif isinstance(point, RefinedPoints):
            marker = 's'
            edgecolor = 'blue'
            facecolor = 'none'
            size = point_size
        elif isinstance(point, EstimatedPoints):
            marker = 's'
            edgecolor = 'red'
            facecolor = 'none'
            size = point_size
        else:
            marker = 'o'
            edgecolor = 'black'
            facecolor = 'none'
            size = point_size

        for station in instrument_stations:
            if station.point == point:
                facecolor = edgecolor

        ax.scatter(point.y, point.x, edgecolor=edgecolor, facecolor=facecolor,
                   marker=marker, s=size**2, linewidth=0.5, zorder=8)
        ax.text(point.y - 1.5 * point_size, point.x, point.name, fontsize=text_size,
                ha='right', zorder=7,
                bbox=dict(facecolor='white', alpha=0.75,
                          edgecolor='none', boxstyle='round,pad=0.3'))

    # Visualization of mean square error ellipses
    for point in points:
        if isinstance(point, RefinedPoints) or isinstance(point, EstimatedPoints):
            a = point.rmse_major * ellipse_scale * plan_scale
            b = point.rmse_minor * ellipse_scale * plan_scale
            angle = float(np.degrees(math.pi / 2 - point.ellipse_angle))  # left to right !
            ellipse = patches.Ellipse((point.y, point.x), 2 * a, 2 * b, angle=angle,
                                      edgecolor='purple', facecolor='none', linewidth=0.5, zorder=6)
            ax.add_patch(ellipse)

    # Visualization of measurements
    for measurement in measurements:
        x_values = [measurement.start_point.y, measurement.end_point.y]
        y_values = [measurement.start_point.x, measurement.end_point.x]
        if isinstance(measurement, LinearMeasurements):
            line_style = 'r-'
            arrow_color = 'black'
            arrow_fc = 'white'
            layer_order = 3
        elif isinstance(measurement, AngularMeasurements):
            line_style = 'r--'
            arrow_color = 'black'
            arrow_fc = 'r'
            layer_order = 4
        else:
            line_style = 'g-'
            arrow_color = 'black'
            arrow_fc = 'none'
            layer_order = 3
        ax.plot(x_values, y_values, line_style, linewidth=0.5, zorder=2)

        mid_x = (measurement.end_point.y + 2 * measurement.start_point.y) / 3
        mid_y = (measurement.end_point.x + 2 * measurement.start_point.x) / 3
        dx = measurement.end_point.y - measurement.start_point.y
        dy = measurement.end_point.x - measurement.start_point.x
        ax.arrow(mid_x, mid_y, dx * 0.001, dy * 0.001, head_width=arrow_size,
                 head_length=arrow_size, fc=arrow_fc, ec=arrow_color,
                 linewidth=0.25, zorder=layer_order)

    # Visualization of instrument stations
    if show_stations:
        station_dict = {}
        for station in instrument_stations:
            if station.point in station_dict:
                station_dict[station.point].append(station.name)
            else:
                station_dict[station.point] = [station.name]

        for point, station_names in station_dict.items():
            ax.text(point.y + 1.5 * point_size, point.x, ', '.join(station_names),
                    color='green', fontsize=text_size, ha='left', zorder=7,
                    bbox=dict(facecolor='white', alpha=0.75,
                              edgecolor='none', boxstyle='round,pad=0.3'))

    # Setting up axis labels
    tick_spacing = 100 / 1000 * plan_scale
    x_min = np.ceil((ax.get_xlim()[0] - 50) / tick_spacing) * tick_spacing
    x_max = np.floor((ax.get_xlim()[1] + 50) / tick_spacing) * tick_spacing
    y_min = np.ceil((ax.get_ylim()[0] - 50) / tick_spacing) * tick_spacing
    y_max = np.floor((ax.get_ylim()[1] + 50) / tick_spacing) * tick_spacing

    ax.xaxis.set_major_locator(plt.MultipleLocator(tick_spacing))
    ax.yaxis.set_major_locator(plt.MultipleLocator(tick_spacing))
    ax.tick_params(axis='both', which='major', labelsize=axis_label_size)

    # Adding crosses at the intersections of coordinate lines
    x_ticks = np.arange(x_min, x_max + tick_spacing, tick_spacing)
    y_ticks = np.arange(y_min, y_max + tick_spacing, tick_spacing)
    for x in x_ticks:
        for y in y_ticks:
            ax.plot([x - cross_size / 2, x + cross_size / 2], [y, y],
                    color='darkgreen', linewidth=0.25, zorder=1)
            ax.plot([x, x], [y - cross_size / 2, y + cross_size / 2],
                    color='darkgreen', linewidth=0.25, zorder=1)
            ax.text(x + cross_size / 6, y + cross_size / 6, f'X={0 if y==0 else y:.0f}',
                    color='darkgreen', fontsize=coord_text_size,
                    ha='left', zorder=1)
            ax.text(x - cross_size, y + cross_size / 6, f'Y={0 if x==0 else x:.0f}',
                    color='darkgreen', fontsize=coord_text_size,
                    va='bottom', rotation=90, zorder=1)

    legend_elements = [
        plt.Line2D([0], [0], marker='^', color='w', markerfacecolor='none',
                   markeredgecolor='black', markersize=(1.25 * point_size),
                   linewidth=0.05, label='BP - Base Points'),
        plt.Line2D([0], [0], marker='^', color='w', markerfacecolor='black',
                   markeredgecolor='black', markersize=(1.25 * point_size),
                   linewidth=0.05, label='Instrument Station at BP'),
        plt.Line2D([0], [0], marker='s', color='w', markerfacecolor='none',
                   markeredgecolor='blue', markersize=point_size,
                   linewidth=0.05, label='RF - Refined Points'),
        plt.Line2D([0], [0], marker='s', color='w', markerfacecolor='blue',
                   markeredgecolor='blue', markersize=point_size,
                   linewidth=0.05, label='Instrument Station at RP'),
        plt.Line2D([0], [0], marker='s', color='w', markerfacecolor='none',
                   markeredgecolor='red', markersize=point_size,
                   linewidth=0.05, label='EP - Estimated Points'),
        plt.Line2D([0], [0], marker='s', color='w', markerfacecolor='red',
                   markeredgecolor='red', markersize=point_size,
                   linewidth=0.05, label='Instrument Station at EP'),
        plt.Line2D([0], [0], color='red', linestyle='-',
                   linewidth=0.5, label='Linear Measurements'),
        plt.Line2D([0], [0], color='red', linestyle='--',
                   linewidth=0.5, label='Angular Measurements')

    ]

    legend = ax.legend(handles=legend_elements, loc='best', fontsize=coord_text_size)
    for legend_handle in legend.legendHandles:
        legend_handle._sizes = [point_size * 10]

    ax.axis('off')

    ax.set_aspect('equal')
    plt.title(f'{project_title}', fontsize=2*axis_label_size)
    # plt.suptitle(f'scale plan 1:{plan_scale}', fontsize=axis_label_size)
    ax.text(0.5, -0.1, f'scale plan 1:{plan_scale}',
            transform=ax.transAxes, fontsize=axis_label_size, ha='center')

    folder_path = 'Images'
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)

    plt.savefig(f'../output/{folder_path}/{project_title.replace(" ", "_")}.png', dpi=800)
    plt.show()
