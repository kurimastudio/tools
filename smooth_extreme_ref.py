import os
import numpy as np
import matplotlib.pyplot as plt

from astropy.io import fits
from astropy.wcs import WCS
from astropy.convolution import Gaussian2DKernel, convolve_fft
from astropy.stats import sigma_clip

from scipy.ndimage import binary_closing, binary_fill_holes
from skimage.morphology import disk
from skimage.measure import label, find_contours


# ============================================================
# USER SETTINGS
# ============================================================
os.chdir("/home/kurima/Data/002")

# IRAC already reprojected onto the 13CO grid (recommended)
irac_fits = "IRAC_on_13COgrid.fits"

# Optional: overlay outline onto another WCS map (moment-0, etc.)
# Set to None if you only want the IRAC reference plot.
overlay_fits = "IRAC_on_13COgrid.fits"   # e.g. "mom0_13co.fits" or "co13_m0_cut.fits"
overlay_cmap = "magma"
overlay_title = r"$^{13}$CO moment-0 with IRAC outline"

# Outputs
out_ref_fits = "irac_ref_smooth.fits"
out_irac_png = "irac_ref_smooth.png"
out_overlay_png = "overlay_with_irac_outline.png"
out_outline_npy = "m33_outline_radec.npy"

# Smoothing / contour parameters
smooth_pix = 12             # try 10–20
do_star_clip = True
clip_sigma = 3.0

smooth_bg_pix = 40  # big smoothing scale = background


contour_percentile = 50      # try ~25–45 depending on how much envelope you want
closing_radius = 2            # disk radius for binary_closing; try 4, 6, 8

plot_outline_color_irac = "red"
plot_outline_color_overlay = "white"
outline_lw = 2.0

# Display stretch (for plotting only)
vmin_pct, vmax_pct = 5, 99.5


# ============================================================
# HELPERS
# ============================================================
def compute_smoothed_irac(irac_data, smooth_pix, do_star_clip=True, clip_sigma=3.0):
    """Star-suppress (optional) then heavily smooth IRAC image."""
    work = irac_data.astype(float).copy()
    work[~np.isfinite(work)] = np.nan

    if do_star_clip:
        clipped = sigma_clip(work, sigma=clip_sigma, maxiters=5)
        med = np.nanmedian(clipped.data[~clipped.mask])
        work[clipped.mask] = med

    kernel = Gaussian2DKernel(x_stddev=smooth_pix)
    sm = convolve_fft(work, kernel, nan_treatment="interpolate", normalize_kernel=True)
    return sm

def extract_multi_contours(sm, wcs, contour_percentile, closing_radius, N_KEEP=5, min_vertices=80):
    """
    Build a binary mask from a percentile threshold on the smoothed (or arm-enhanced) image,
    fill holes, close gaps, keep the largest N connected components, and extract ALL contours.

    Returns:
      conts_world: list of (N_i,2) arrays [RA,Dec] for each contour polyline
      level: threshold used
      mask_keep: final boolean mask used for contouring
    """
    finite = np.isfinite(sm)
    if not np.any(finite):
        raise RuntimeError("Smoothed image has no finite pixels.")

    level = np.nanpercentile(sm[finite], contour_percentile)
    mask = sm > level

    # Fill holes + close gaps
    mask = binary_fill_holes(mask)
    mask = binary_closing(mask, structure=disk(closing_radius))

    # Connected components
    lab = label(mask)
    sizes = np.bincount(lab.ravel())
    sizes[0] = 0  # background

    if sizes.max() == 0:
        return [], level, None

    # Keep largest N components
    keep_labels = np.argsort(sizes)[-N_KEEP:]
    mask_keep = np.isin(lab, keep_labels)

    # Extract contours from the combined mask
    conts_pix = find_contours(mask_keep.astype(float), 0.5)

    conts_world = []
    for c in conts_pix:
        if c.shape[0] < min_vertices:
            continue
        x = c[:, 1]  # col
        y = c[:, 0]  # row
        ra, dec = wcs.wcs_pix2world(x, y, 0)
        conts_world.append(np.column_stack([ra, dec]))

    # Sort contours by size (largest first)
    conts_world = sorted(conts_world, key=lambda a: a.shape[0], reverse=True)

    return conts_world, level, mask_keep

# ============================================================
# MAIN
# ============================================================
# Load IRAC
h_irac = fits.getheader(irac_fits)
w_irac = WCS(h_irac)
irac = fits.getdata(irac_fits).astype(float)

# Smooth IRAC
sm = compute_smoothed_irac(irac, smooth_pix=smooth_pix, do_star_clip=do_star_clip, clip_sigma=clip_sigma)

# --- arm enhancement: unsharp masking / high-pass ---
kernel_bg = Gaussian2DKernel(x_stddev=smooth_bg_pix)
bg = convolve_fft(sm, kernel_bg, nan_treatment="interpolate", normalize_kernel=True)

arm_img = sm - bg   # highlights arms / structures over smooth disk

# Save smoothed IRAC reference FITS
fits.writeto(out_ref_fits, sm, h_irac, overwrite=True)
print("Wrote:", out_ref_fits)

# Extract outline contour (world coords)

conts_world, level, mask_keep = extract_multi_contours(
    arm_img, w_irac,
    contour_percentile=contour_percentile,
    closing_radius=closing_radius,
    N_KEEP=5,            # increase if needed (e.g. 8)
    min_vertices=60      # lower if islands are small
)

print(f"Contours found: {len(conts_world)} | p{contour_percentile} -> level={level:.4g}")

#ra, dec, level, mask_big = extract_largest_closed_contour(
#    arm_img, w_irac,
#    contour_percentile=contour_percentile,
#    closing_radius=closing_radius
#)

#ra, dec, level, mask_big = extract_largest_closed_contour(
#    sm, w_irac,
#    contour_percentile=contour_percentile,
#    closing_radius=closing_radius
#)


def plot_wcs_image_with_outlines(data, header, conts_world, title, cmap,
                                vmin_pct=5, vmax_pct=99.5,
                                outline_color="red", outline_lw=2.0,
                                max_paths=None, out_png=None,
                                show_colorbar=False,
                                cbar_label=""):
    
    w = WCS(header)
    finite = np.isfinite(data)
    vmin = np.nanpercentile(data[finite], vmin_pct) if np.any(finite) else None
    vmax = np.nanpercentile(data[finite], vmax_pct) if np.any(finite) else None

    fig = plt.figure(figsize=(7, 6))
    ax = plt.subplot(projection=w)

    im = ax.imshow(data, origin="lower", cmap=cmap, vmin=vmin, vmax=vmax)

    ax.set_title(title)

    # ---- axis labels ----
    ax.set_xlabel("RA")
    ax.set_ylabel("Dec")

    # ---- nicer RA/Dec formatting ----
    ax.coords[0].set_major_formatter('hh:mm:ss')
    ax.coords[1].set_major_formatter('dd:mm')

    # ---- optional colorbar ----
    if show_colorbar:
        cbar = plt.colorbar(im, ax=ax, pad=0.02)
        cbar.set_label(cbar_label)

    if conts_world:
        if max_paths is not None:
            conts_to_plot = conts_world[:max_paths]
        else:
            conts_to_plot = conts_world

        for xy in conts_to_plot:
            ax.plot(xy[:, 0], xy[:, 1],
                    transform=ax.get_transform("world"),
                     color=outline_color, lw=outline_lw)

    plt.tight_layout()

    if out_png is not None:
        plt.savefig(out_png, dpi=200)
        print("Wrote:", out_png)

    plt.show()


# Optional: overlay on another WCS image
if overlay_fits is not None:
    h_map = fits.getheader(overlay_fits)
    m = fits.getdata(overlay_fits).astype(float)

 

plot_wcs_image_with_outlines(
    data=irac,
    header=h_irac,
    conts_world=conts_world,
    title=f"IRAC reference (smoothed: {smooth_pix} pix), outline p{contour_percentile}",
    cmap="magma",
    vmin_pct=vmin_pct,
    vmax_pct=vmax_pct,
    outline_color=plot_outline_color_irac,
    outline_lw=outline_lw,
    max_paths=None,
    out_png=out_irac_png,
    show_colorbar=False
)

