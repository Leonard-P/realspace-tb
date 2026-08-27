"""
Animation utilities for plotting and rendering animations of 2D Lattice geometries.
"""

import warnings
from dataclasses import dataclass, fields as _dc_fields
from typing import cast, Any
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import ScalarFormatter
from matplotlib import animation
from matplotlib.colorbar import Colorbar
from matplotlib.patches import Arc, RegularPolygon, FancyArrowPatch
from matplotlib.collections import PathCollection
from matplotlib import cm
from matplotlib.colors import Normalize
import matplotlib.lines as mlines

from realspace_tb.orbitronics_2d.honeycomb_geometry import HoneycombLatticeGeometry

from .lattice_2d_geometry import Lattice2DGeometry
from .. import backend as B
from .biot_savart import biot_savart_on_plane
from .observables import LatticeFrameObservable


@dataclass
class PlotConfig:
    """Visual style configuration for `save_simulation_animation` and `show_simulation_frame`.

    Collect all visual/style parameters into one object and pass it as the
    *config* argument rather than listing each keyword individually.  Old-style
    flat keyword arguments are still accepted by the public functions but will
    raise a `DeprecationWarning`; constructing a `PlotConfig` directly is the
    preferred way going forward.

    Parameters:
        density_cmap: Colormap name for site-occupation scatter.
        density_vmin: Lower bound of the density colormap.
        density_vmax: Upper bound of the density colormap.
        current_max: Absolute maximum used to normalise bond-current arrows.
            Derived from data when `None`.
        site_marker_size: Scatter marker area (pts²) for site-occupation circles.
        show_flow_arrows: Draw flow-direction arrows along bonds.
        arrows_per_edge: Number of arrows distributed along each bond.
        arrow_scale: Arrow-length scaling factor relative to current magnitude.
        arrow_width: Arrow stem width in data units.
        arrow_color: Colour of flow-direction arrows.
        show_oam_indicators: Draw OAM-value scatter at plaquette centres.
        oam_cmap: Colormap name for orbital angular momentum values.
        oam_vmax: Absolute maximum for the OAM colormap. Derived from data when `None`.
        oam_marker_size: Scatter marker area (pts²) for OAM indicator circles.
        show_oam_direction_arrows: Draw circular arrows indicating OAM handedness.
        oam_arrow_color_mode: Color mode for circular OAM arrows.
            ``"discrete"`` uses positive/negative colors; ``"continuous"`` samples
            the selected colormap according to the vorticity value.
        oam_arrow_cmap: Colormap name used when ``oam_arrow_color_mode`` is
            ``"continuous"``. If `None`, `oam_cmap` is used.
        oam_arrow_radius: Radius of the circular OAM arrows in data units.
        oam_arrow_lw: Line width of the circular OAM arrows.
        oam_arrow_positive_color: Colour for counter-clockwise (positive OAM) arrows.
        oam_arrow_negative_color: Colour for clockwise (negative OAM) arrows.
        oam_arrow_threshold: Relative threshold below which OAM direction arrows are hidden.
        show_vorticity_sources: Draw vorticity-source markers at plaquette centres.
            Positive source is shown as a cross, negative as a dot.
            If `None`, this is auto-enabled when a source observable is passed.
        vorticity_source_marker_size: Maximum marker size (pts²) used for
            source symbols.
        vorticity_source_max_linewidth: Maximum line width used for source symbols.
        vorticity_source_color: Marker color used for source symbols.
        vorticity_source_max: Optional absolute normalization value for source
            magnitudes. Derived from data when `None`.
        show_vorticity_flow: Draw directed arrows between plaquettes using
            vorticity-flow tensors.
            If `None`, this is auto-enabled when a flow observable is passed.
        vorticity_flow_color: Base color of vorticity-flow arrows.
        vorticity_flow_threshold: Relative threshold below which arrows are hidden.
        vorticity_flow_min_linewidth: Minimum line width for visible flow arrows.
        vorticity_flow_max_linewidth: Maximum line width for flow arrows.
        vorticity_flow_alpha: Alpha value for flow arrows.
        vorticity_flow_mutation_scale: Arrow-head scale for vorticity-flow arrows.
        vorticity_flow_shrink: Padding around plaquette centers that shortens
            flow arrows near start and end points.
        vorticity_flow_curvature: Signed curvature magnitude used for
            vorticity-flow arrows. `0.0` gives straight arrows.
        vorticity_flow_max: Optional absolute normalization value for
            vorticity-flow magnitudes. Derived from the full animation when `None`.
        show_legend: Draw an in-axes legend.
        legend_frameon: Whether to draw a box around the legend.
        legend_show_bond_current: Include bond-current proxy in legend.
        legend_show_site_occupation: Include site-occupation proxy in legend.
        legend_show_oam: Include OAM/vorticity proxy in legend.
        legend_show_oam_arrows: Use curved circular arrow proxy for OAM.
        legend_show_vorticity_sources: Include source proxy in legend.
        legend_show_vorticity_flow: Include flow proxy in legend.
        legend_vorticity_source_label: Legend entry for source markers.
        legend_vorticity_flow_label: Legend entry for vorticity-flow arrows.
        legend_location: Matplotlib legend location string.
        show_density_colorbar: Toggle density colorbar.
        show_oam_colorbar: Toggle OAM colorbar.
        show_bfield_colorbar: Toggle B-field colorbar. If `None`, follows
            `show_bfield_underlay`.
        colorbar_bfield_label: Colorbar axis label for magnetic field Bz.
        show_bfield_underlay: Render a Biot-Savart B-field raster behind the lattice.
        bfield_z_height: Plane height, in lattice units, at which the B-field is sampled.
        bfield_x_pixels: Number of raster columns for the B-field image.
        bfield_y_pixels: Number of raster rows for the B-field image.
        bfield_x_margin: Extra x-padding added around the outermost sites for the B-field plane.
        bfield_y_margin: Extra y-padding added around the outermost sites for the B-field plane.
        bfield_n_images: Periodic-image range forwarded to the Biot-Savart helper.
        bfield_block_size: Chunk size used by the Biot-Savart helper.
        bfield_cmap: Colormap name for the Bz raster.
        bfield_alpha: Alpha used when drawing the B-field raster.
        bfield_vmin: Lower color limit for the B-field raster. Derived from the first frame when `None`.
        bfield_vmax: Upper color limit for the B-field raster. Derived from the first frame when `None`.
        field_arrow_type: Placement of the electric-field arrow.  ``"vertical"``
            draws it to the left of the lattice; ``"horizontal"`` draws it above.
        field_arrow_label: Optional label placed beside the electric-field arrow.
        field_arrow_color: Colour for the electric-field arrow and its label.
        frame_texts: Per-frame title strings shown at the top-left corner.  When
            `None` the default ``"frame i/F"`` counter is used.
        electric_field_vectors: Per-frame 2-D vectors for an external electric-field
            arrow overlay.  Pass `None` entries for frames where no arrow should
            appear.
        legend_bond_current_label: Legend entry for the bond-current arrow proxy artist.
        legend_site_occupation_label: Legend entry for site-occupation markers.
        legend_oam_label: Legend entry for plaquette-OAM markers.
        colorbar_site_occupation_label: Colorbar axis label for site occupation.
        colorbar_oam_label: Colorbar axis label for plaquette OAM.
        colorbar_layout_direction: Layout direction for the default colorbars.
            'vertical' (default) stacks them below each other, 'horizontal' places
            the OAM colorbar next to the site occupation colorbar.
        colorbar_width: Absolute colorbar width in figure coordinates.
            When `None`, a sensible default is used.
        colorbar_height: Absolute colorbar height in figure coordinates.
            When `None`, a sensible default is used.
    """

    # --- Density -----------------------------------------------------------------
    density_cmap: str = "Greys"
    density_vmin: float = 0.0
    density_vmax: float = 1.0
    current_max: float | None = None
    site_marker_size: float = 320.0

    # --- Flow-direction arrows ---------------------------------------------------
    show_flow_arrows: bool = True
    arrows_per_edge: int = 3
    arrow_scale: float = 0.55
    arrow_width: float = 0.04
    arrow_color: str = "black"

    # --- OAM indicators ----------------------------------------------------------
    show_oam_indicators: bool = True
    oam_cmap: str = "RdBu"
    oam_vmax: float | None = None
    oam_marker_size: float = 180.0

    # --- Circular OAM arrows -----------------------------------------------------
    show_oam_direction_arrows: bool = True
    oam_arrow_color_mode: str = "discrete"
    oam_arrow_cmap: str | None = None
    oam_arrow_radius: float = 0.6
    oam_arrow_lw: float = 1.5
    oam_arrow_positive_color: str = "blue"
    oam_arrow_negative_color: str = "red"
    oam_arrow_threshold: float = 0.01

    # --- Vorticity source markers ------------------------------------------------
    show_vorticity_sources: bool | None = None
    vorticity_source_marker_size: float = 220.0
    vorticity_source_max_linewidth: float = 2.2
    vorticity_source_color: str = "black"
    vorticity_source_max: float | None = None

    # --- Vorticity flow arrows ---------------------------------------------------
    show_vorticity_flow: bool | None = None
    vorticity_flow_color: str = "#2B7A78"
    vorticity_flow_threshold: float = 0.05
    vorticity_flow_min_linewidth: float = 0.3
    vorticity_flow_max_linewidth: float = 3.0
    vorticity_flow_alpha: float = 1.0
    vorticity_flow_mutation_scale: float = 14.0
    vorticity_flow_shrink: float = 11.0
    vorticity_flow_curvature: float = 0.12
    vorticity_flow_max: float | None = None

    # --- Legend ------------------------------------------------------------------
    show_legend: bool = False
    legend_frameon: bool = True
    legend_show_bond_current: bool = True
    legend_show_site_occupation: bool = True
    legend_show_oam: bool = True
    legend_show_oam_arrows: bool = True
    legend_show_vorticity_sources: bool = True
    legend_show_vorticity_flow: bool = True
    legend_vorticity_source_label: str = "Vorticity Source (+ / -)"
    legend_vorticity_flow_label: str = "Vorticity Flow"
    legend_location: str = "upper left"
    legend_fontsize: float = 12.0

    # --- B-field underlay -------------------------------------------------------
    show_density_colorbar: bool = True
    show_oam_colorbar: bool = True
    show_bfield_colorbar: bool | None = None
    colorbar_bfield_label: str = "$B_z$ [T]"
    colorbar_bfield_fontsize: float = 16
    colorbar_bfield_textsize: float = 12
    show_bfield_underlay: bool = False
    bfield_z_height: float = 0.0
    bfield_x_pixels: int = 200
    bfield_y_pixels: int = 200
    bfield_x_margin: float = 0.6
    bfield_y_margin: float = 0.6
    bfield_n_images: int = 2
    bfield_block_size: int = 256
    bfield_cmap: str = "RdBu_r"
    bfield_alpha: float = 0.8
    bfield_vmin: float | None = None
    bfield_vmax: float | None = None

    # --- Electric-field arrow ----------------------------------------------------
    field_arrow_type: str = "vertical"
    field_arrow_label: str | None = None
    field_arrow_color: str = "green"

    # --- Per-frame data ----------------------------------------------------------
    frame_texts: list[str] | None = None
    electric_field_vectors: list[np.ndarray | None] | None = None

    # --- Legend / colorbar labels ------------------------------------------------
    legend_bond_current_label: str = "Bond Current"
    legend_site_occupation_label: str = "Site Occupation $\\langle \\hat n_i\\rangle $"
    legend_oam_label: str = "Vorticity"
    colorbar_site_occupation_label: str = "Site Occupation"
    colorbar_site_occupation_fontsize: float = 16
    colorbar_site_occupation_textsize: float = 12
    colorbar_oam_label: str = "Vorticity [a.u.]"
    colorbar_oam_fontsize: float = 16
    colorbar_oam_textsize: float = 12
    colorbar_layout_direction: str = "horizontal"
    colorbar_width: float | None = None
    colorbar_height: float | None = None


# Frozen set of all PlotConfig field names — used by the **kwargs deprecation shim.
_PLOTCONFIG_FIELDS: frozenset[str] = frozenset(f.name for f in _dc_fields(PlotConfig))


def _resolve_config(
    config: PlotConfig | None,
    kwargs: dict[str, Any],
    stacklevel: int = 2,
) -> PlotConfig:
    """Return *config*, building one from *kwargs* with a deprecation warning if needed."""
    if config is not None and kwargs:
        raise TypeError(
            "Cannot mix 'config=' with individual style kwargs: "
            + ", ".join(repr(k) for k in sorted(kwargs))
        )
    if config is not None:
        return config
    if kwargs:
        unknown = sorted(set(kwargs) - _PLOTCONFIG_FIELDS)
        if unknown:
            raise TypeError(f"Unexpected keyword argument(s): {unknown}")
        warnings.warn(
            "Passing style parameters as flat keyword arguments is deprecated. "
            "Use config=PlotConfig(...) instead.",
            DeprecationWarning,
            stacklevel=stacklevel + 1,
        )
        return PlotConfig(**kwargs)
    return PlotConfig()


def append_colorbar(
    fig: plt.Figure,
    ax: plt.Axes,
    mappable: Any,
    label: str | None = None,
    direction: str = "vertical",
    colorbar_width: float | None = None,
    colorbar_height: float | None = None,
    **kwargs: Any,
) -> Colorbar:
    """Append a colorbar to the figure dynamically using absolute positioned axes.

    This avoids stealing space from the main axes (which can ruin 'equal' aspect plots).
    Subsequent calls to this function will stack colorbars in a grid format.

    Args:
        fig: The matplotlib Figure.
        ax: The main matplotlib Axes.
        mappable: The ScalarMappable to draw the colorbar for.
        label: Optional label for the colorbar.
        direction: 'vertical' (starts a new row below the lowest colorbar) or
            'horizontal' (starts a new column to the right of the last appended colorbar).
        colorbar_width: Optional absolute colorbar width in figure coordinates.
            If omitted and previous appended colorbars exist, their width is reused.
        colorbar_height: Optional absolute colorbar height in figure coordinates.
            If omitted and previous appended colorbars exist, their height is reused.
        **kwargs: Additional keyword arguments passed to `fig.colorbar()`.

    Returns:
        The created Colorbar instance.
    """
    fig.canvas.draw()
    ax_bbox = ax.get_position()

    cbar_axes = [a for a in fig.axes if a.get_label() == "appended_colorbar"]
    if cbar_axes:
        # Keep geometry stable across subsequent calls (e.g. after figure resizing).
        ref_bbox = cbar_axes[0].get_position()
        cbar_w = float(ref_bbox.width)
        cbar_h = float(ref_bbox.height)
    else:
        cbar_w = 0.02
        cbar_h = min(0.35, max(0.12, ax_bbox.height * 0.6))

    if colorbar_width is not None:
        cbar_w = float(colorbar_width)
    if colorbar_height is not None:
        cbar_h = float(colorbar_height)

    cbar_spacing_y = cbar_h * 0.2
    cbar_spacing_x = cbar_w * 5

    if not cbar_axes:
        # First colorbar: align top with main ax, pad right
        cax_x = min(0.98 - cbar_w, ax_bbox.x1 + 0.02)
        cax_y = min(0.95 - cbar_h, ax_bbox.y0 + ax_bbox.height - cbar_h)
        lgd = ax.get_legend()
        if lgd:
            # Shift the first colorbar below the legend if it exists
            lgd_bbox = lgd.get_window_extent().transformed(fig.transFigure.inverted())
            cax_x = lgd_bbox.x0
            cax_y = lgd_bbox.y0 - 0.02 - cbar_h
        cax_y = max(0.05, cax_y)
    else:
        if direction == "vertical":
            # Stack below the lowest colorbar in the grid
            min_y0 = min(a.get_position().y0 for a in cbar_axes)
            # Re-use the starting x-coordinate of the first appended colorbar
            first_x0 = cbar_axes[0].get_position().x0
            cax_x = first_x0
            cax_y = max(0.05, min_y0 - cbar_spacing_y - cbar_h)
        elif direction == "horizontal":
            # Stack to the right of the *last* appended colorbar
            last_cax_bbox = cbar_axes[-1].get_position()
            cax_x = last_cax_bbox.x1 + cbar_spacing_x
            cax_y = last_cax_bbox.y0
        else:
            raise ValueError("direction must be 'vertical' or 'horizontal'")

    cax = fig.add_axes((cax_x, cax_y, cbar_w, cbar_h), label="appended_colorbar")
    cb = fig.colorbar(mappable, cax=cax, orientation="vertical", **kwargs)
    if label:
        cb.set_label(label, size="small")
    cb.ax.tick_params(labelsize="small")
    return cb


def _build_geometry_segments(geometry: Lattice2DGeometry) -> np.ndarray:
    """Build line segments array for nearest-neighbor bonds using ``nn_bond_vectors``.

    For PBC geometries ``geometry.nn_bond_vectors`` stores the *short* displacement
    vector ``r_j - r_i`` pointing to the nearest periodic image, so wrapped bonds
    are drawn as short stubs rather than lines that cross the entire lattice.
    The second endpoint of each segment is therefore ``r_i + bond_vector``, which
    may lie outside the physical simulation cell – the viewer can imagine the
    periodic image site at that position.

    Parameters:
        geometry: Lattice2DGeometry with nearest_neighbors, site_positions and
                  nn_bond_vectors defined.

    Returns:
        array of shape (E, 2, 2): [ [ (x_i, y_i), (x_i+dx, y_i+dy) ], ... ].
    """
    rows = geometry.nearest_neighbors[:, 0]
    pos = geometry.site_positions  # (N, 2)
    bv = geometry.nn_bond_vectors  # (E, 2)
    segs = np.empty((len(rows), 2, 2), dtype=float)
    segs[:, 0, :] = pos[rows]  # start: r_i
    segs[:, 1, :] = pos[rows] + bv  # end:   r_i + (r_j − r_i)  [short vector]
    return segs


def _build_bfield_plane(
    geometry: Lattice2DGeometry,
    currents: np.ndarray,
    *,
    z_height: float,
    x_margin: float,
    y_margin: float,
    x_pixels: int,
    y_pixels: int,
    n_images: int,
    block_size: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, tuple[float, float, float, float]]:
    pos = np.asarray(geometry.site_positions, dtype=float)
    x_min = float(np.min(pos[:, 0])) - float(x_margin)
    x_max = float(np.max(pos[:, 0])) + float(x_margin)
    y_min = float(np.min(pos[:, 1])) - float(y_margin)
    y_max = float(np.max(pos[:, 1])) + float(y_margin)

    x_values = np.linspace(x_min, x_max, max(2, int(x_pixels)))
    y_values = np.linspace(y_min, y_max, max(2, int(y_pixels)))
    _, _, b_grid = biot_savart_on_plane(
        x_values,
        y_values,
        z_height,
        currents,
        geometry,
        block_size=block_size,
        n_images=n_images,
    )
    return x_values, y_values, b_grid[..., 2], (x_min, x_max, y_min, y_max)


def _resolve_bfield_clim(
    bfield_image_data: np.ndarray,
    vmin: float | None,
    vmax: float | None,
) -> tuple[float, float]:
    if vmin is None and vmax is None:
        max_abs = (
            float(np.max(np.abs(bfield_image_data)))
            if np.size(bfield_image_data)
            else 0.0
        )
        if max_abs == 0.0:
            max_abs = 1.0
        return -max_abs, max_abs
    if vmin is None:
        vmin = float(np.min(bfield_image_data))
    if vmax is None:
        vmax = float(np.max(bfield_image_data))
    return float(vmin), float(vmax)


def _honeycomb_plaquette_centers(
    geometry: Lattice2DGeometry,
    lattice_frame_obs: LatticeFrameObservable,
) -> tuple[np.ndarray, np.ndarray]:
    if not isinstance(geometry, HoneycombLatticeGeometry):
        raise NotImplementedError(
            "Vorticity plaquette overlays are only implemented for HoneycombLatticeGeometry."
        )
    _, curl_pos, _, _ = lattice_frame_obs.geometry.plaquettes
    cx = curl_pos[:, 0]
    cy = curl_pos[:, 1]
    return cx, cy


def _update_vorticity_source_markers(
    source_pos_sc: PathCollection,
    source_neg_sc: PathCollection,
    source_vals: np.ndarray,
    source_max_abs: float,
    base_size: float,
    max_linewidth: float,
) -> None:
    denom = source_max_abs if source_max_abs > 0 else 1.0
    rel = np.clip(np.abs(source_vals) / denom, 0.0, 1.0)

    sizes = base_size * rel
    linewidths = (0.15 + rel) * max_linewidth

    is_pos = source_vals > 0
    is_neg = source_vals < 0

    pos_sizes = np.where(is_pos, sizes, 0.0)
    neg_sizes = np.where(is_neg, sizes, 0.0)
    pos_lw = np.where(is_pos, linewidths, 0.0)
    neg_lw = np.where(is_neg, linewidths, 0.0)

    source_pos_sc.set_sizes(pos_sizes)
    source_neg_sc.set_sizes(neg_sizes)
    source_pos_sc.set_linewidths(pos_lw)
    source_neg_sc.set_linewidths(neg_lw)


def _build_vorticity_flow_arrows(
    ax: plt.Axes,
    centers: np.ndarray,
    flow_values: np.ndarray,
    *,
    rel_threshold: float,
    min_lw: float,
    max_lw: float,
    color: str,
    alpha: float,
    mutation_scale: float,
    shrink: float,
    curvature: float,
    max_abs: float,
) -> list[FancyArrowPatch]:
    arrows: list[FancyArrowPatch] = []
    if flow_values.size == 0:
        return arrows

    n_cells = centers.shape[0]
    if max_abs == 0.0:
        return arrows

    abs_threshold = max(0.0, float(rel_threshold)) * max_abs

    for i in range(n_cells):
        for j in range(n_cells):
            if i == j:
                continue
            val = float(flow_values[i, j])
            if val <= 0.0:
                continue
            if abs(val) < abs_threshold:
                continue

            rel = min(1.0, abs(val) / max_abs)
            lw = float(min_lw + (max_lw - min_lw) * rel)
            start = centers[i]
            end = centers[j]
            arrow = FancyArrowPatch(
                (float(start[0]), float(start[1])),
                (float(end[0]), float(end[1])),
                arrowstyle="-|>",
                connectionstyle=f"arc3,rad={(curvature if j >= i else -curvature):.6f}",
                mutation_scale=mutation_scale,
                lw=lw,
                color=color,
                alpha=alpha,
                zorder=3.5,
                shrinkA=shrink,
                shrinkB=shrink,
            )
            ax.add_patch(arrow)
            arrows.append(arrow)

    return arrows


def _create_scene(
    lattice_frame_obs: LatticeFrameObservable,
    config: PlotConfig,
    vort_source_obs: Any | None = None,
    vort_flow_obs: Any | None = None,
    include_colorbars: bool = True,
) -> tuple[plt.Figure, plt.Axes, dict[str, Any]]:
    """Builds the static scene (figure, artists, legend, colorbars) and returns a context dict for updating per-frame."""
    # Unpack config fields into local names so the existing body code is unchanged.
    density_cmap = config.density_cmap
    density_vmin = config.density_vmin
    density_vmax = config.density_vmax
    current_max = config.current_max  # may be updated below when None
    site_marker_size = config.site_marker_size
    show_flow_arrows = config.show_flow_arrows
    arrows_per_edge = config.arrows_per_edge
    arrow_scale = config.arrow_scale
    arrow_width = config.arrow_width
    arrow_color = config.arrow_color
    show_oam_indicators = config.show_oam_indicators
    oam_cmap = config.oam_cmap
    oam_vmax = config.oam_vmax
    oam_marker_size = config.oam_marker_size
    show_oam_direction_arrows = config.show_oam_direction_arrows
    oam_arrow_color_mode = config.oam_arrow_color_mode
    oam_arrow_cmap = config.oam_arrow_cmap
    oam_arrow_radius = config.oam_arrow_radius
    oam_arrow_lw = config.oam_arrow_lw
    oam_arrow_positive_color = config.oam_arrow_positive_color
    oam_arrow_negative_color = config.oam_arrow_negative_color
    oam_arrow_threshold = config.oam_arrow_threshold
    show_vorticity_sources_cfg = config.show_vorticity_sources
    vorticity_source_marker_size = config.vorticity_source_marker_size
    vorticity_source_max_linewidth = config.vorticity_source_max_linewidth
    vorticity_source_color = config.vorticity_source_color
    vorticity_source_max = config.vorticity_source_max
    show_vorticity_flow_cfg = config.show_vorticity_flow
    vorticity_flow_color = config.vorticity_flow_color
    vorticity_flow_threshold = config.vorticity_flow_threshold
    vorticity_flow_min_linewidth = config.vorticity_flow_min_linewidth
    vorticity_flow_max_linewidth = config.vorticity_flow_max_linewidth
    vorticity_flow_alpha = config.vorticity_flow_alpha
    vorticity_flow_mutation_scale = config.vorticity_flow_mutation_scale
    vorticity_flow_shrink = config.vorticity_flow_shrink
    vorticity_flow_curvature = config.vorticity_flow_curvature
    vorticity_flow_max = config.vorticity_flow_max
    show_bfield_underlay = config.show_bfield_underlay
    show_density_colorbar = config.show_density_colorbar
    show_oam_colorbar = config.show_oam_colorbar
    show_bfield_colorbar = (
        bool(show_bfield_underlay)
        if config.show_bfield_colorbar is None
        else bool(config.show_bfield_colorbar)
    )
    bfield_z_height = config.bfield_z_height
    bfield_x_pixels = config.bfield_x_pixels
    bfield_y_pixels = config.bfield_y_pixels
    bfield_x_margin = config.bfield_x_margin
    bfield_y_margin = config.bfield_y_margin
    bfield_n_images = config.bfield_n_images
    bfield_block_size = config.bfield_block_size
    bfield_cmap = config.bfield_cmap
    bfield_alpha = config.bfield_alpha
    bfield_vmin = config.bfield_vmin
    bfield_vmax = config.bfield_vmax
    field_arrow_type = config.field_arrow_type
    field_arrow_label = config.field_arrow_label
    field_arrow_color = config.field_arrow_color
    frame_texts = config.frame_texts
    electric_field_vectors = config.electric_field_vectors

    animation_values = cast(dict[str, B.FCPUArray], lattice_frame_obs.values)
    densities = animation_values["densities"]  # (F, N)
    bond_currents = animation_values["currents"]  # (F, E)

    source_all = (
        np.asarray(vort_source_obs.values) if vort_source_obs is not None else None
    )
    flow_all = np.asarray(vort_flow_obs.values) if vort_flow_obs is not None else None

    show_vorticity_sources = (
        source_all is not None
        if show_vorticity_sources_cfg is None
        else bool(show_vorticity_sources_cfg)
    )
    show_vorticity_flow = (
        flow_all is not None
        if show_vorticity_flow_cfg is None
        else bool(show_vorticity_flow_cfg)
    )

    F, N = densities.shape
    _, E = bond_currents.shape

    if show_vorticity_sources and source_all is None:
        raise ValueError(
            "show_vorticity_sources=True but no source data was provided. Pass vort_source_obs."
        )

    if show_vorticity_flow and flow_all is None:
        raise ValueError(
            "show_vorticity_flow=True but no flow data was provided. Pass vort_flow_obs."
        )

    has_current_vorts = "current_vorts" in animation_values
    if show_oam_indicators and not has_current_vorts:
        warnings.warn(
            "No 'current_vorts' data found in lattice_frame_obs.values. OAM indicators are disabled.",
            UserWarning,
            stacklevel=2,
        )
        show_oam_indicators = False
        show_oam_direction_arrows = False

    geometry = lattice_frame_obs.geometry

    # Coordinates
    xs = geometry.site_positions[:, 0]
    ys = geometry.site_positions[:, 1]
    x_min = float(np.min(xs)) if xs.size else 0.0
    x_max = float(np.max(xs)) if xs.size else 0.0
    y_min = float(np.min(ys)) if ys.size else 0.0
    y_max = float(np.max(ys)) if ys.size else 0.0
    span_x = x_max - x_min if x_max > x_min else 1.0
    span_y = y_max - y_min if y_max > y_min else 1.0
    span_max = max(span_x, span_y, 1.0)
    field_quiv = None
    field_label_artist: plt.Text | None = None
    field_scale_factor = 0.0
    field_base = np.array([0.0, 0.0])
    field_arrow_type_norm = (field_arrow_type or "vertical").lower()
    if field_arrow_type_norm not in {"vertical", "horizontal"}:
        field_arrow_type_norm = "vertical"
    label_offset_value = 0.1

    segments = _build_geometry_segments(geometry)
    seg_x = segments[:, :, 0] if segments.size else np.empty((0,), dtype=float)
    seg_y = segments[:, :, 1] if segments.size else np.empty((0,), dtype=float)

    # Normalizations
    if current_max is None:
        current_max = float(np.max(np.abs(bond_currents)))
        if current_max == 0:
            current_max = 1.0

    # Figure
    Lx, Ly = geometry.Lx, geometry.Ly
    fig_width = max(1, 0.7 * int(Lx)) + 2.0  # reserve right space for legend/colorbars
    fig_height = max(1, 1 * int(Ly))
    fig, ax = plt.subplots(figsize=(fig_width, fig_height))
    ax.set_aspect("equal")
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(False)
    fig.subplots_adjust(left=0.04, right=1 - 1 / Lx, top=0.96, bottom=0.04)

    # Explicit axis limits with padding so edge sites/bonds are not clipped.
    # The padding is at least 0.6 data units (≈ one bond length for the
    # honeycomb) and grows mildly with system size so large systems stay tight.
    _pad = max(0.6, 0.04 * span_max)
    x_left, x_right = x_min - _pad, x_max + _pad
    y_bottom, y_top = y_min - _pad, y_max + _pad
    if seg_x.size:
        x_left = min(x_left, float(np.min(seg_x)) - _pad)
        x_right = max(x_right, float(np.max(seg_x)) + _pad)
    if seg_y.size:
        y_bottom = min(y_bottom, float(np.min(seg_y)) - _pad)
        y_top = max(y_top, float(np.max(seg_y)) + _pad)

    need_plaquette_centers = (
        show_oam_indicators or show_vorticity_sources or show_vorticity_flow
    )
    cx = cy = None
    centers = None
    if need_plaquette_centers:
        cx, cy = _honeycomb_plaquette_centers(geometry, lattice_frame_obs)
        centers = np.column_stack((cx, cy))
        if centers.size:
            x_left = min(x_left, float(np.min(centers[:, 0])) - _pad)
            x_right = max(x_right, float(np.max(centers[:, 0])) + _pad)
            y_bottom = min(y_bottom, float(np.min(centers[:, 1])) - _pad)
            y_top = max(y_top, float(np.max(centers[:, 1])) + _pad)

    bfield_image = None
    if show_bfield_underlay:
        currents0 = np.asarray(bond_currents[0], dtype=float)
        _, _, b_z, bfield_extent = _build_bfield_plane(
            geometry,
            currents0,
            z_height=bfield_z_height,
            x_margin=bfield_x_margin,
            y_margin=bfield_y_margin,
            x_pixels=bfield_x_pixels,
            y_pixels=bfield_y_pixels,
            n_images=bfield_n_images,
            block_size=bfield_block_size,
        )
        bfield_vmin_f, bfield_vmax_f = _resolve_bfield_clim(
            b_z,
            bfield_vmin,
            bfield_vmax,
        )
        bfield_image = ax.imshow(
            b_z,
            origin="lower",
            extent=[
                bfield_extent[0],
                bfield_extent[1],
                bfield_extent[2],
                bfield_extent[3],
            ],
            aspect="equal",
            cmap=bfield_cmap,
            vmin=bfield_vmin_f,
            vmax=bfield_vmax_f,
            alpha=bfield_alpha,
            zorder=-1,
            interpolation="none",
        )
        x_left = min(x_left, bfield_extent[0])
        x_right = max(x_right, bfield_extent[1])
        y_bottom = min(y_bottom, bfield_extent[2])
        y_top = max(y_top, bfield_extent[3])
    ax.set_xlim(x_left, x_right)
    ax.set_ylim(y_bottom, y_top)

    # Densities as scatter
    dens0 = densities[0]
    sc = ax.scatter(
        xs,
        ys,
        c=dens0,
        cmap=density_cmap,
        vmin=density_vmin,
        vmax=density_vmax,
        s=site_marker_size,
        edgecolor="black",
        linewidths=0.6,
        zorder=2,
    )

    # Electric field arrow
    field_vectors_processed: list[np.ndarray | None] | None = None
    if electric_field_vectors is not None:
        processed: list[np.ndarray | None] = []
        for vec in electric_field_vectors:
            if vec is None:
                processed.append(None)
                continue
            arr = np.asarray(vec, dtype=float).flatten()
            if arr.size >= 2:
                processed.append(arr[:2])
            else:
                processed.append(None)
        if processed:
            field_vectors_processed = processed

    if field_vectors_processed is not None and any(
        v is not None for v in field_vectors_processed
    ):
        non_null_vectors = np.array(
            [v for v in field_vectors_processed if v is not None]
        )
        mags = np.linalg.norm(non_null_vectors, axis=1)
        max_mag = float(np.max(mags)) if mags.size else 0.0
        if max_mag == 0.0:
            max_mag = 1.0
        margin = 0.02 * span_max + 5 / Lx
        if field_arrow_type_norm == "vertical":
            field_base = np.array([x_min - margin, (y_min + y_max) / 2.0])
        else:
            field_base = np.array([(x_min + x_max) / 2.0, y_max + margin])
        field_scale_factor = 0.2 * span_max / max_mag

        vec0 = next((v for v in field_vectors_processed if v is not None), np.zeros(2))
        scaled0 = vec0 * field_scale_factor
        field_quiv = ax.quiver(
            [field_base[0]],
            [field_base[1]],
            [scaled0[0]],
            [scaled0[1]],
            angles="xy",
            scale_units="xy",
            scale=1,
            pivot="middle",
            color=field_arrow_color,
            width=0.03 / Lx,
            headwidth=10,
            headlength=10,
            headaxislength=10,
            zorder=6,
        )
        if not np.any(vec0):
            field_quiv.set_alpha(0.0)

        # Place label at a fixed position relative to field_base (not the arrow tip)
        if field_arrow_label is not None:
            if field_arrow_type_norm == "vertical":
                # Label to the left of the arrow base
                label_pos = (field_base[0] - label_offset_value, field_base[1])
                ha, va = "right", "center"
            else:
                # Label above the arrow base
                label_pos = (field_base[0], field_base[1] + label_offset_value)
                ha, va = "center", "bottom"
            field_label_artist = ax.text(
                label_pos[0],
                label_pos[1],
                field_arrow_label,
                color=field_arrow_color,
                fontsize=2 * Lx if field_arrow_type_norm == "vertical" else 2 * Ly,
                ha=ha,
                va=va,
                visible=True,
                zorder=6,
            )

        # Expand axis limits to include the arrow
        cur_xlim = ax.get_xlim()
        cur_ylim = ax.get_ylim()
        all_tips = [
            field_base + field_scale_factor * v
            for v in field_vectors_processed
            if v is not None
        ]
        if all_tips:
            tips_arr = np.array(all_tips)
            new_xmin = min(
                cur_xlim[0],
                field_base[0] - 0.1 * span_max,
                np.min(tips_arr[:, 0]) - 0.1 * span_max,
            )
            new_xmax = max(
                cur_xlim[1],
                field_base[0] + 0.1 * span_max,
                np.max(tips_arr[:, 0]) + 0.1 * span_max,
            )
            new_ymin = min(
                cur_ylim[0],
                field_base[1] - 0.1 * span_max,
                np.min(tips_arr[:, 1]) - 0.1 * span_max,
            )
            new_ymax = max(
                cur_ylim[1],
                field_base[1] + 0.1 * span_max,
                np.max(tips_arr[:, 1]) + 0.1 * span_max,
            )
            ax.set_xlim(new_xmin, new_xmax)
            ax.set_ylim(new_ymin, new_ymax)

    # Flow-direction arrows via Quiver
    quiv = None
    dirx = diry = None
    if show_flow_arrows and E > 0 and arrows_per_edge > 0:
        P0 = segments[:, 0, :]
        P1 = segments[:, 1, :]
        dP = P1 - P0  # r_k - r_l points to r_k
        lengths = np.linalg.norm(dP, axis=1)
        safe_lengths = np.where(lengths == 0, 1.0, lengths)
        dirs = dP / safe_lengths[:, None]  # (E,2)

        fracs = (np.arange(1, arrows_per_edge + 1) / (arrows_per_edge + 1)).astype(
            float
        )
        Px = (P0[:, 0:1] + fracs * dP[:, 0:1]).reshape(-1)
        Py = (P0[:, 1:2] + fracs * dP[:, 1:2]).reshape(-1)
        dirx = np.repeat(dirs[:, 0], arrows_per_edge)
        diry = np.repeat(dirs[:, 1], arrows_per_edge)
        J0 = bond_currents[0]
        s0 = -np.sign(J0)
        L0 = arrow_scale * np.repeat(np.abs(J0) / current_max, arrows_per_edge)
        U0 = L0 * np.repeat(s0, arrows_per_edge) * dirx
        V0 = L0 * np.repeat(s0, arrows_per_edge) * diry

        quiv = ax.quiver(
            Px,
            Py,
            U0,
            V0,
            units="xy",
            angles="xy",
            scale_units="xy",
            scale=1,
            pivot="middle",
            color=arrow_color,
            width=arrow_width,
            headwidth=7,
            headlength=8,
            headaxislength=7,
            linewidth=0,
            zorder=1,
        )

    # OAM indicators and optional circular arrows
    curl_sc = None
    curl_all = None
    curl_ccw_arcs: list[Arc] = []
    curl_ccw_heads: list[RegularPolygon] = []
    curl_cw_arcs: list[Arc] = []
    curl_cw_heads: list[RegularPolygon] = []
    oam_vmax_f: float = 1.0
    oam_arrow_cmap_obj = plt.get_cmap(oam_arrow_cmap or oam_cmap)
    oam_arrow_color_mode_norm = (oam_arrow_color_mode or "discrete").lower()
    if oam_arrow_color_mode_norm not in {"discrete", "continuous"}:
        warnings.warn(
            "oam_arrow_color_mode must be 'discrete' or 'continuous'. Falling back to 'discrete'.",
            UserWarning,
            stacklevel=2,
        )
        oam_arrow_color_mode_norm = "discrete"

    # Vorticity-source markers and vorticity-flow arrows
    source_pos_sc = None
    source_neg_sc = None
    source_max_abs = 1.0
    flow_arrows: list[FancyArrowPatch] = []

    if show_vorticity_sources and source_all is not None and centers is not None:
        source_vals0 = np.asarray(source_all[0], dtype=float)
        if vorticity_source_max is None:
            source_max_abs = (
                float(np.max(np.abs(source_all))) if np.size(source_all) else 1.0
            )
            if source_max_abs == 0.0:
                source_max_abs = 1.0
        else:
            source_max_abs = float(abs(vorticity_source_max))
            if source_max_abs == 0.0:
                source_max_abs = 1.0

        source_pos_sc = ax.scatter(
            centers[:, 0],
            centers[:, 1],
            marker="x",
            s=np.zeros(centers.shape[0], dtype=float),
            linewidths=0.0,
            c=vorticity_source_color,
            zorder=4.2,
        )
        source_neg_sc = ax.scatter(
            centers[:, 0],
            centers[:, 1],
            marker="o",
            s=np.zeros(centers.shape[0], dtype=float),
            linewidths=0.0,
            facecolors=vorticity_source_color,
            edgecolors=vorticity_source_color,
            alpha=0.9,
            zorder=4.1,
        )
        _update_vorticity_source_markers(
            source_pos_sc,
            source_neg_sc,
            source_vals0,
            source_max_abs,
            vorticity_source_marker_size,
            vorticity_source_max_linewidth,
        )

    if show_vorticity_flow and flow_all is not None and centers is not None:
        if vorticity_flow_max is None:
            flow_max_abs = float(np.max(np.abs(flow_all))) if np.size(flow_all) else 1.0
            if flow_max_abs == 0.0:
                flow_max_abs = 1.0
        else:
            flow_max_abs = float(abs(vorticity_flow_max))
            if flow_max_abs == 0.0:
                flow_max_abs = 1.0
        flow0 = np.asarray(flow_all[0], dtype=float)
        flow_arrows = _build_vorticity_flow_arrows(
            ax,
            centers,
            flow0,
            rel_threshold=vorticity_flow_threshold,
            min_lw=vorticity_flow_min_linewidth,
            max_lw=vorticity_flow_max_linewidth,
            color=vorticity_flow_color,
            alpha=vorticity_flow_alpha,
            mutation_scale=vorticity_flow_mutation_scale,
            shrink=vorticity_flow_shrink,
            curvature=vorticity_flow_curvature,
            max_abs=flow_max_abs,
        )
    else:
        flow_max_abs = 1.0

    draw_oam_dots = show_oam_indicators and not show_vorticity_sources

    if show_oam_indicators:
        curl_all = animation_values["current_vorts"]  # (F, C)
        curl_vals0 = np.asarray(curl_all[0])
        assert cx is not None and cy is not None
        assert centers is not None
        if curl_vals0.shape[0] != centers.shape[0]:
            raise ValueError(
                "Vorticity plaquette count does not match lattice geometry."
            )

        if oam_vmax is None:
            oam_vmax_f = float(np.max(np.abs(curl_all))) if np.size(curl_all) else 1.0
            if oam_vmax_f == 0:
                oam_vmax_f = 1.0
        else:
            oam_vmax_f = float(oam_vmax)

        if draw_oam_dots:
            curl_sc = ax.scatter(
                cx,
                cy,
                c=curl_vals0,
                cmap=oam_cmap,
                vmin=-oam_vmax_f,
                vmax=oam_vmax_f,
                s=oam_marker_size,
                edgecolor="none",
                zorder=3,
                alpha=1.0,
            )

        if show_oam_direction_arrows:
            angle_ = 125
            theta2_ = 310
            for i in range(len(cx)):
                x = cx[i]
                y = cy[i]
                # CCW
                arc_ccw = Arc(
                    (x, y),
                    oam_arrow_radius,
                    oam_arrow_radius,
                    angle=angle_,
                    theta1=0,
                    theta2=theta2_,
                    capstyle="round",
                    linestyle="-",
                    lw=oam_arrow_lw,
                    color=oam_arrow_positive_color,
                    zorder=4,
                    alpha=1.0,
                )
                endX_ccw = x + (oam_arrow_radius / 2.0) * np.cos(
                    np.radians(theta2_ + angle_)
                )
                endY_ccw = y + (oam_arrow_radius / 2.0) * np.sin(
                    np.radians(theta2_ + angle_)
                )
                orient_ccw = np.radians(angle_ + theta2_)
                head_ccw = RegularPolygon(
                    (endX_ccw, endY_ccw),
                    3,
                    radius=oam_arrow_radius / 7.0,
                    orientation=orient_ccw,
                    color=oam_arrow_positive_color,
                    zorder=5,
                    alpha=1.0,
                )
                ax.add_patch(arc_ccw)
                ax.add_patch(head_ccw)
                curl_ccw_arcs.append(arc_ccw)
                curl_ccw_heads.append(head_ccw)
                # CW
                arc_cw = Arc(
                    (x, y),
                    oam_arrow_radius,
                    oam_arrow_radius,
                    angle=angle_,
                    theta1=0,
                    theta2=theta2_,
                    capstyle="round",
                    linestyle="-",
                    lw=oam_arrow_lw,
                    color=oam_arrow_negative_color,
                    zorder=4,
                    alpha=1.0,
                )
                endX_cw = x + (oam_arrow_radius / 2.0) * np.cos(np.radians(angle_))
                endY_cw = y + (oam_arrow_radius / 2.0) * np.sin(np.radians(angle_))
                orient_cw = np.radians(angle_) + np.pi
                head_cw = RegularPolygon(
                    (endX_cw, endY_cw),
                    3,
                    radius=oam_arrow_radius / 7.0,
                    orientation=orient_cw,
                    color=oam_arrow_negative_color,
                    zorder=5,
                    alpha=1.0,
                )
                ax.add_patch(arc_cw)
                ax.add_patch(head_cw)
                curl_cw_arcs.append(arc_cw)
                curl_cw_heads.append(head_cw)

            # Initialize visibility from frame 0
            norm0 = curl_vals0 / (oam_vmax_f if oam_vmax_f else 1.0)
            for i, v in enumerate(norm0):
                show_cw = v <= -oam_arrow_threshold
                show_ccw = v >= oam_arrow_threshold
                curl_ccw_arcs[i].set_visible(show_ccw)
                curl_ccw_heads[i].set_visible(show_ccw)
                curl_cw_arcs[i].set_visible(show_cw)
                curl_cw_heads[i].set_visible(show_cw)
                if oam_arrow_color_mode_norm == "continuous":
                    rgba = oam_arrow_cmap_obj(
                        0.5 * (float(np.clip(v, -1.0, 1.0)) + 1.0)
                    )
                    curl_ccw_arcs[i].set_color(rgba)
                    curl_ccw_heads[i].set_color(rgba)
                    curl_cw_arcs[i].set_color(rgba)
                    curl_cw_heads[i].set_color(rgba)
                else:
                    curl_ccw_arcs[i].set_color(oam_arrow_positive_color)
                    curl_ccw_heads[i].set_color(oam_arrow_positive_color)
                    curl_cw_arcs[i].set_color(oam_arrow_negative_color)
                    curl_cw_heads[i].set_color(oam_arrow_negative_color)

    # Title / per-frame text
    title = ax.text(0.02, 0.98, "", transform=ax.transAxes, va="top", ha="left")
    if frame_texts is not None and len(frame_texts) > 0:
        title.set_text(frame_texts[0])
    # else:
    #    title.set_text("frame 1/1")

    # Legend and colorbars
    handles: list[mlines.Line2D] = []
    labels: list[str] = []
    if config.legend_show_bond_current:
        handles.append(
            mlines.Line2D(
                [],
                [],
                color="black",
                linestyle="-",
                marker=">",
                markersize=6,
                mfc="black",
                mec="black",
                lw=1.8,
            )
        )
        labels.append(config.legend_bond_current_label)
    if config.legend_show_site_occupation:
        occ_color = cm.get_cmap(density_cmap)(0.6)
        handles.append(
            mlines.Line2D(
                [],
                [],
                color=occ_color,
                marker="o",
                linestyle="None",
                markersize=9,
                markeredgecolor="black",
                mew=1.0,
            )
        )
        labels.append(config.legend_site_occupation_label)
    if show_oam_indicators and config.legend_show_oam:
        if getattr(config, "legend_show_oam_arrows", False):
            # Show a circular arrow in the legend
            handles.append(
                mlines.Line2D(
                    [],
                    [],
                    color=config.oam_arrow_positive_color,
                    marker=r"$\circlearrowleft$",
                    linestyle="None",
                    markersize=10,
                    mew=0.8,
                )
            )
            labels.append(config.legend_oam_label)
        else:
            oam_color = cm.get_cmap(oam_cmap)(0.75)
            handles.append(
                mlines.Line2D(
                    [],
                    [],
                    color=oam_color,
                    marker="o",
                    linestyle="None",
                    markersize=10,
                    markeredgecolor=oam_color,
                )
            )
            labels.append(config.legend_oam_label)
    if show_vorticity_sources and config.legend_show_vorticity_sources:
        handles.append(
            mlines.Line2D(
                [],
                [],
                color=vorticity_source_color,
                marker="x",
                linestyle="None",
                markersize=8,
                markeredgewidth=1.8,
            )
        )
        labels.append(config.legend_vorticity_source_label)
    if show_vorticity_flow and config.legend_show_vorticity_flow:
        handles.append(
            mlines.Line2D(
                [],
                [],
                color=vorticity_flow_color,
                linestyle="-",
                marker=">",
                markersize=6,
                lw=2.2,
            )
        )
        labels.append(config.legend_vorticity_flow_label)
    handles_labels = (handles, labels)

    if config.show_legend and handles and labels:
        ax.legend(
            handles,
            labels,
            loc=config.legend_location,
            bbox_to_anchor=(1.01, 0.7),
            frameon=config.legend_frameon,
            fontsize=config.legend_fontsize,
            handletextpad=0.6,
            handlelength=1.8,
            borderpad=0.4,
        )

    colorbar_specs: list[dict[str, Any]] = []
    if show_density_colorbar:
        occ_norm = Normalize(
            vmin=density_vmin if density_vmin is not None else np.nanmin(densities),
            vmax=density_vmax if density_vmax is not None else np.nanmax(densities),
        )
        occ_sm = cm.ScalarMappable(norm=occ_norm, cmap=plt.get_cmap(density_cmap))
        occ_sm.set_array([])
        colorbar_specs.append(
            {
                "mappable": occ_sm,
                "label": config.colorbar_site_occupation_label,
                "formatter": None,
            }
        )
        if include_colorbars:
            cb_occ = append_colorbar(
                fig,
                ax,
                occ_sm,
                label=config.colorbar_site_occupation_label,
                direction=config.colorbar_layout_direction,
                colorbar_width=config.colorbar_width,
                colorbar_height=config.colorbar_height,
            )
            cb_occ.set_label(
                config.colorbar_site_occupation_label,
                fontsize=config.colorbar_site_occupation_fontsize,
            )
            cb_occ.ax.tick_params(labelsize=config.colorbar_site_occupation_textsize)

    draw_oam_cb = (
        show_oam_indicators
        and show_oam_colorbar
        and (curl_sc is not None or oam_arrow_color_mode_norm == "continuous")
    )
    if draw_oam_cb:
        oam_norm = Normalize(vmin=-oam_vmax_f, vmax=oam_vmax_f)
        # Use the specific arrow cmap if rendering arrows continuously, else fallback to base oam_cmap
        cmap_to_use = (
            oam_arrow_cmap_obj if (curl_sc is None) else plt.get_cmap(oam_cmap)
        )
        oam_sm = cm.ScalarMappable(norm=oam_norm, cmap=cmap_to_use)
        oam_sm.set_array([])
        colorbar_specs.append(
            {
                "mappable": oam_sm,
                "label": config.colorbar_oam_label,
                "formatter": {
                    "kind": "scalar",
                    "use_math_text": True,
                    "power_limits": (-2, 2),
                },
            }
        )
        if include_colorbars:
            cb_oam = append_colorbar(
                fig,
                ax,
                oam_sm,
                label=config.colorbar_oam_label,
                direction=config.colorbar_layout_direction,
                colorbar_width=config.colorbar_width,
                colorbar_height=config.colorbar_height,
            )
            formatter = ScalarFormatter(useMathText=True)
            formatter.set_powerlimits((-2, 2))
            cb_oam.set_label(
                config.colorbar_oam_label, fontsize=config.colorbar_oam_fontsize
            )
            cb_oam.ax.tick_params(labelsize=config.colorbar_oam_textsize)
            cb_oam.ax.yaxis.set_major_formatter(formatter)
            cb_oam.update_ticks()

    if bfield_image is not None and show_bfield_colorbar:
        bfield_norm = Normalize(
            vmin=float(bfield_image.norm.vmin), vmax=float(bfield_image.norm.vmax)
        )
        bfield_sm = cm.ScalarMappable(norm=bfield_norm, cmap=plt.get_cmap(bfield_cmap))
        bfield_sm.set_array([])
        colorbar_specs.append(
            {
                "mappable": bfield_sm,
                "label": config.colorbar_bfield_label,
                "formatter": None,
            }
        )
        if include_colorbars:
            cb_bfield = append_colorbar(
                fig,
                ax,
                bfield_sm,
                label=config.colorbar_bfield_label,
                direction=config.colorbar_layout_direction,
                colorbar_width=config.colorbar_width,
                colorbar_height=config.colorbar_height,
            )
            cb_bfield.set_label(
                config.colorbar_bfield_label, fontsize=config.colorbar_bfield_fontsize
            )
            cb_bfield.ax.tick_params(labelsize=config.colorbar_bfield_textsize)

    ctx: dict[str, Any] = {
        "F": F,
        "densities": densities,
        "bond_currents": bond_currents,
        "arrows_per_edge": arrows_per_edge,
        "arrow_scale": arrow_scale,
        "current_max": current_max,
        "dirx": dirx,
        "diry": diry,
        "sc": sc,
        "quiv": quiv,
        "title": title,
        "frame_texts": frame_texts,
        # OAM
        "curl_all": curl_all,
        "curl_sc": curl_sc,
        "show_oam_direction_arrows": show_oam_direction_arrows,
        "oam_vmax_f": oam_vmax_f,
        "curl_ccw_arcs": curl_ccw_arcs,
        "curl_ccw_heads": curl_ccw_heads,
        "curl_cw_arcs": curl_cw_arcs,
        "curl_cw_heads": curl_cw_heads,
        "oam_arrow_color_mode": oam_arrow_color_mode_norm,
        "oam_arrow_cmap_obj": oam_arrow_cmap_obj,
        "legend_handles": handles_labels[0],
        "legend_labels": handles_labels[1],
        "show_vorticity_sources": show_vorticity_sources,
        "source_all": source_all,
        "source_pos_sc": source_pos_sc,
        "source_neg_sc": source_neg_sc,
        "source_max_abs": source_max_abs,
        "show_vorticity_flow": show_vorticity_flow,
        "flow_all": flow_all,
        "flow_centers": centers,
        "flow_arrows": flow_arrows,
        "flow_max_abs": flow_max_abs,
        "field_quiv": field_quiv,
        "field_vectors": field_vectors_processed,
        "field_scale": field_scale_factor,
        "field_label_artist": field_label_artist,
        "bfield_image": bfield_image,
        "bfield_geometry": geometry,
        "colorbar_specs": colorbar_specs,
        "config": config,
    }

    return fig, ax, ctx


def _update_scene(ctx: dict[str, Any], frame: int) -> tuple[plt.Artist, ...]:
    densities = ctx["densities"]
    bond_currents = ctx["bond_currents"]
    arrows_per_edge = ctx["arrows_per_edge"]
    arrow_scale = ctx["arrow_scale"]
    current_max = ctx["current_max"]
    dirx = ctx["dirx"]
    diry = ctx["diry"]
    sc = ctx["sc"]
    quiv = ctx["quiv"]
    title = ctx["title"]
    frame_texts = ctx["frame_texts"]
    config: PlotConfig = ctx["config"]
    field_quiv = ctx.get("field_quiv")
    field_vectors = ctx.get("field_vectors")
    field_scale = ctx.get("field_scale", 0.0)
    field_label_artist = ctx.get("field_label_artist")
    bfield_image = ctx.get("bfield_image")
    bfield_geometry = ctx.get("bfield_geometry")
    show_vorticity_sources = ctx.get("show_vorticity_sources", False)
    source_all = ctx.get("source_all")
    source_pos_sc = ctx.get("source_pos_sc")
    source_neg_sc = ctx.get("source_neg_sc")
    source_max_abs = ctx.get("source_max_abs", 1.0)
    show_vorticity_flow = ctx.get("show_vorticity_flow", False)
    flow_all = ctx.get("flow_all")
    flow_centers = ctx.get("flow_centers")
    flow_arrows = ctx.get("flow_arrows", [])
    flow_max_abs = float(ctx.get("flow_max_abs", 1.0))

    artists: list[plt.Artist] = [sc, title]

    # Densities
    d = densities[frame]
    sc.set_array(d)

    # Currents arrows
    if (
        quiv is not None
        and dirx is not None
        and diry is not None
        and arrows_per_edge > 0
    ):
        J = bond_currents[frame]
        sgn = -np.sign(J)
        L = arrow_scale * np.repeat(np.abs(J) / current_max, arrows_per_edge)
        U = L * np.repeat(sgn, arrows_per_edge) * dirx
        V = L * np.repeat(sgn, arrows_per_edge) * diry
        quiv.set_UVC(U, V)
        artists.append(quiv)

    # OAM indicators
    curl_sc = ctx["curl_sc"]
    curl_all = ctx["curl_all"]
    vals = None
    if curl_all is not None:
        vals = np.asarray(curl_all[frame])

    if curl_sc is not None and vals is not None:
        curl_sc.set_array(vals)
        artists.append(curl_sc)

    if vals is not None and ctx["show_oam_direction_arrows"]:
        oam_vmax_f = ctx["oam_vmax_f"]
        oam_arrow_threshold = config.oam_arrow_threshold
        denom = oam_vmax_f if oam_vmax_f else (np.max(np.abs(vals)) or 1.0)
        normv = vals / denom
        for i, v in enumerate(normv):
            show_cw = v <= -oam_arrow_threshold
            show_ccw = v >= oam_arrow_threshold
            ctx["curl_ccw_arcs"][i].set_visible(show_ccw)
            ctx["curl_ccw_heads"][i].set_visible(show_ccw)
            ctx["curl_cw_arcs"][i].set_visible(show_cw)
            ctx["curl_cw_heads"][i].set_visible(show_cw)
            if ctx.get("oam_arrow_color_mode") == "continuous":
                cmap_obj = ctx.get("oam_arrow_cmap_obj", plt.get_cmap("RdBu"))
                rgba = cmap_obj(0.5 * (float(np.clip(v, -1.0, 1.0)) + 1.0))
                ctx["curl_ccw_arcs"][i].set_color(rgba)
                ctx["curl_ccw_heads"][i].set_color(rgba)
                ctx["curl_cw_arcs"][i].set_color(rgba)
                ctx["curl_cw_heads"][i].set_color(rgba)
            else:
                pos_color = config.oam_arrow_positive_color
                neg_color = config.oam_arrow_negative_color
                ctx["curl_ccw_arcs"][i].set_color(pos_color)
                ctx["curl_ccw_heads"][i].set_color(pos_color)
                ctx["curl_cw_arcs"][i].set_color(neg_color)
                ctx["curl_cw_heads"][i].set_color(neg_color)

    if (
        show_vorticity_sources
        and source_all is not None
        and source_pos_sc is not None
        and source_neg_sc is not None
    ):
        source_vals = np.asarray(source_all[frame], dtype=float)
        _update_vorticity_source_markers(
            source_pos_sc,
            source_neg_sc,
            source_vals,
            float(source_max_abs),
            float(config.vorticity_source_marker_size),
            float(config.vorticity_source_max_linewidth),
        )
        artists.append(source_pos_sc)
        artists.append(source_neg_sc)

    if show_vorticity_flow and flow_all is not None and flow_centers is not None:
        for arrow in flow_arrows:
            arrow.remove()
        flow_frame = np.asarray(flow_all[frame], dtype=float)
        flow_ax = title.axes
        new_flow_arrows = _build_vorticity_flow_arrows(
            flow_ax,
            np.asarray(flow_centers),
            flow_frame,
            rel_threshold=float(config.vorticity_flow_threshold),
            min_lw=float(config.vorticity_flow_min_linewidth),
            max_lw=float(config.vorticity_flow_max_linewidth),
            color=str(config.vorticity_flow_color),
            alpha=float(config.vorticity_flow_alpha),
            mutation_scale=float(config.vorticity_flow_mutation_scale),
            shrink=float(config.vorticity_flow_shrink),
            curvature=float(config.vorticity_flow_curvature),
            max_abs=flow_max_abs,
        )
        ctx["flow_arrows"] = new_flow_arrows
        artists.extend(new_flow_arrows)

    # Electric field arrow
    if field_quiv is not None:
        vec = None
        if field_vectors is not None and frame < len(field_vectors):
            vec = field_vectors[frame]
            if vec is not None:
                vec = np.asarray(vec, dtype=float).flatten()
                if vec.size >= 2:
                    vec = vec[:2]
                else:
                    vec = None
        if vec is None or not np.any(vec):
            field_quiv.set_UVC([0.0], [0.0])
            field_quiv.set_alpha(0.0)
        else:
            scaled = field_scale * vec
            field_quiv.set_UVC([scaled[0]], [scaled[1]])
            field_quiv.set_alpha(1.0)
        artists.append(field_quiv)
        if field_label_artist is not None:
            artists.append(field_label_artist)

    if bfield_image is not None and bfield_geometry is not None:
        currents = np.asarray(bond_currents[frame], dtype=float)
        _, _, b_z, _ = _build_bfield_plane(
            bfield_geometry,
            currents,
            z_height=config.bfield_z_height,
            x_margin=config.bfield_x_margin,
            y_margin=config.bfield_y_margin,
            x_pixels=config.bfield_x_pixels,
            y_pixels=config.bfield_y_pixels,
            n_images=config.bfield_n_images,
            block_size=config.bfield_block_size,
        )
        bfield_image.set_data(b_z)
        artists.append(bfield_image)

    # Title text
    if frame_texts is not None and frame < len(frame_texts):
        title.set_text(frame_texts[frame])

    return tuple(artists)


def save_simulation_animation(
    lattice_frame_obs: LatticeFrameObservable,
    out_path: str,
    fps: int = 10,
    dpi: int = 150,
    vort_source_obs: Any | None = None,
    vort_flow_obs: Any | None = None,
    config: PlotConfig | None = None,
    export_legend: bool = False,
    **kwargs: Any,
) -> None:
    """Save an animation visualising onsite densities and bond currents over frames.

    Parameters:
        lattice_frame_obs: Observable that recorded ``densities``, ``currents``,
            and ``plaquette_oam`` during the simulation, with ``geometry`` set.
        out_path: Destination file path (e.g. ``"anim.mp4"`` or ``"anim.gif"``).
        fps: Frames per second in the output animation.
        dpi: Output resolution.
        vort_source_obs: Optional vorticity-source observable. If provided,
            source markers are
            automatically enabled.
        vort_flow_obs: Optional vorticity-flow observable. If provided, flow arrows
            are automatically enabled.
        config: Visual-style configuration object.  All style, label, and
            per-frame data options live here.  See `PlotConfig` for the full
            list of fields.  When omitted a default `PlotConfig` is used.
        export_legend: When `True`, a standalone ``<stem>_legend.pdf`` file
            containing the legend and colorbars is saved alongside the animation
            (colorbars are then omitted from the animation itself).
        **kwargs: *Deprecated.*  Flat `PlotConfig` field values passed as plain
            keyword arguments (old calling style).  A `DeprecationWarning` is
            raised; wrap them in ``config=PlotConfig(...)`` instead.
    """
    config = _resolve_config(config, kwargs, stacklevel=2)
    fig, ax, ctx = _create_scene(
        lattice_frame_obs=lattice_frame_obs,
        config=config,
        vort_source_obs=vort_source_obs,
        vort_flow_obs=vort_flow_obs,
        include_colorbars=not export_legend,
    )

    anim = animation.FuncAnimation(
        fig,
        lambda i: _update_scene(ctx, i),
        frames=ctx["F"],
        interval=1000 // max(1, fps),
        blit=False,
    )

    try:
        anim.save(out_path, writer="ffmpeg", fps=fps, dpi=dpi)
    except Exception:
        from matplotlib.animation import PillowWriter

        anim.save(out_path, writer=PillowWriter(fps=fps), dpi=dpi)

    if export_legend:
        handles = ctx.get("legend_handles")
        labels = ctx.get("legend_labels")
        colorbar_specs = ctx.get("colorbar_specs", [])
        if handles and labels:
            legend_path = Path(out_path)
            legend_pdf_path = legend_path.with_name(f"{legend_path.stem}_legend.pdf")
            legend_fig = plt.figure(figsize=(3.6, 2.1))
            legend_ax = legend_fig.add_axes((0.05, 0.58, 0.9, 0.37))
            legend_ax.axis("off")
            legend_ax.legend(
                handles,
                labels,
                loc="center",
                frameon=False,
                fontsize="medium",
                handletextpad=0.8,
                handlelength=1.8,
                borderpad=0.5,
            )
            if colorbar_specs:
                cbar_width = 0.09
                cbar_height = 0.50
                total_width = 0.9
                spacing = (total_width - len(colorbar_specs) * cbar_width) / (
                    len(colorbar_specs) + 1
                )
                for idx, spec in enumerate(colorbar_specs):
                    x = 0.05 + spacing * (idx + 1) + cbar_width * idx
                    cax = legend_fig.add_axes((x, 0.05, cbar_width, cbar_height))
                    cb = legend_fig.colorbar(
                        spec["mappable"], cax=cax, orientation="vertical"
                    )
                    cb.set_label(spec["label"], size="small")
                    fmt_info = spec.get("formatter")
                    if fmt_info and fmt_info.get("kind") == "scalar":
                        formatter = ScalarFormatter(
                            useMathText=fmt_info.get("use_math_text", False)
                        )
                        power_limits = fmt_info.get("power_limits")
                        if power_limits:
                            formatter.set_powerlimits(power_limits)
                        cb.ax.yaxis.set_major_formatter(formatter)
                        cb.update_ticks()
                    cb.ax.tick_params(labelsize="small")
            legend_fig.savefig(legend_pdf_path, format="pdf", bbox_inches="tight")
            plt.close(legend_fig)
    plt.close(fig)


def show_simulation_frame(
    lattice_frame_obs: LatticeFrameObservable,
    frame: int = 0,
    vort_source_obs: Any | None = None,
    vort_flow_obs: Any | None = None,
    config: PlotConfig | None = None,
    show: bool = True,
    **kwargs: Any,
) -> tuple[plt.Figure, plt.Axes]:
    """Render a single frame to the current figure (useful for notebooks).

    Parameters:
        lattice_frame_obs: Observable that recorded ``densities``, ``currents``,
            and ``plaquette_oam`` during the simulation, with ``geometry`` set.
        frame: Index of the frame to render.
        vort_source_obs: Optional vorticity-source observable. If provided, source markers are
            automatically enabled.
        vort_flow_obs: Optional vorticity-flow observable. If provided, flow arrows
            are automatically enabled.
        config: Visual-style configuration object.  All style, label, and
            per-frame data options live here.  See `PlotConfig` for the full
            list of fields.  When omitted a default `PlotConfig` is used.
        show: Whether to call ``plt.show()``.  Set to `False` when you intend
            to save the figure yourself.
        **kwargs: *Deprecated.*  Flat `PlotConfig` field values passed as plain
            keyword arguments (old calling style).  A `DeprecationWarning` is
            raised; wrap them in ``config=PlotConfig(...)`` instead.

    Returns:
        The matplotlib `Figure` and `Axes` objects.
    """
    config = _resolve_config(config, kwargs, stacklevel=2)
    fig, ax, ctx = _create_scene(
        lattice_frame_obs=lattice_frame_obs,
        config=config,
        vort_source_obs=vort_source_obs,
        vort_flow_obs=vort_flow_obs,
    )

    frame_clamped = int(np.clip(frame, 0, ctx["F"] - 1))
    _update_scene(ctx, frame_clamped)

    if show:
        plt.show()

    return fig, ax
