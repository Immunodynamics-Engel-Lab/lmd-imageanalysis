# %% Imports
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import cv2
import numpy as np
import polars as pl
from bioio import BioImage
from bioio.writers import OmeTiffWriter
from cellpose import core, models
from lmd.lib import SegmentationLoader
from scipy.ndimage import distance_transform_edt
from scipy.signal import fftconvolve
from skimage import exposure
from skimage.feature import peak_local_max
from skimage.measure import regionprops_table
from tqdm import tqdm
# %% [markdown]
# # 🛠️ Channel Configuration
#
# Adjust the channel indices below to match your microscopy image configuration.
# * Channels are 0-indexed (e.g., 0, 1, 2, ...).
# * The default mapping is: Marker (Ch 0), Autofluorescence (Ch 1), DAPI (Ch 2).
# %%
# Define the default channel indices
CHANNEL_MAP = {
    "Marker": 0,          # e.g., Ly6g, GFP, etc.
    "Autofluorescence": 1,
    "DAPI": 2,
}
# %% [markdown]
# # Step 1: Segment Neutrophils
# %% Use GPU if possible
if not core.use_gpu():
    model = models.CellposeModel(gpu=False)
    print("No GPU access, continuing with CPU...")
else:
    model = models.CellposeModel(gpu=True)


# %%
def contrast_stretching(img: np.ndarray, percentile: tuple) -> np.ndarray:
    lower_v = np.percentile(img, percentile[0])
    upper_v = np.percentile(img, percentile[1])
    img = np.clip(img, lower_v, upper_v)
    img = (img - lower_v) / (upper_v - lower_v)

    return img


def segment_with_cellpose(
    image_array, cellprob_threshold, batch_size, tile_overlap, flow_threshold
):
    masks, _, _ = model.eval(
        image_array,
        normalize=False,
        resample=True,
        cellprob_threshold=cellprob_threshold,
        flow_threshold=flow_threshold,
        batch_size=batch_size,
        tile_overlap=tile_overlap,
    )
    return masks

# %% [markdown]
# # 🛠️ Segmentation Configuration
#
# Adjust the input and output folders, filter thresholds and cellpose parameters.
# %% Adjust paths and parameters
# relative path to the input folder with the *.tiff images
input_path = "input"
# relative path for the output
output_path = "output"
# save unfiltered segmentation
save_unfiltered = False
# filter cells that have a mean normalised marker intensity below
th_mean = 0.35
# filter cells that have a normalised marker intensity below th_pos
# for more than th_ratio_pos[%] amount of pixels
th_pos = 0.2
th_ratio_pos = 0.95

# cellpose machine learning parameters
# https://cellpose.readthedocs.io/en/latest/api.html#cellpose.models.CellposeModel.eval
# bigger cellprob_threshold results in smaller cells
cellprob_threshold = 0.8
# flow error threshold (all cells with errors below threshold are kept)
flow_threshold = 0.4
# bigger batch_size means faster segmentation but more RAM/VRAM needed
batch_size = 64
# % of overlap between individual tiles
tile_overlap = 0.2

# %% Segment each image
src_root = Path(input_path)
dst_root = Path(output_path)
files = [f for f in src_root.rglob("*.tiff")]
if not files:
    print("No Input files found, please put your .tiff files in the input folder")
else:
    for src_path in tqdm(files):
        dst_path = dst_root / src_path.relative_to(src_root)
        dst_path.parent.mkdir(parents=True, exist_ok=True)

        # import image and preprocess
        reader = BioImage(src_path)
        img = reader.get_image_data("CYX").astype(np.float64)
        img = np.delete(img, CHANNEL_MAP["Autofluorescence"], axis=0)
        for c in range(img.shape[0]):
            img[c] = contrast_stretching(img[c], (0.1, 99.9))

        # predict cells
        img_segmented = segment_with_cellpose(
            img, cellprob_threshold, batch_size, tile_overlap, flow_threshold
        )

        if save_unfiltered:
            stem = str(dst_path.name).split(".", 1)[0]
            suffix = "".join(dst_path.suffixes)
            dst_full_path = dst_path.parent / f"{stem}_full{suffix}"
            OmeTiffWriter.save(
                img_segmented,
                dst_full_path,
                dim_order="YX",
                channel_names=["DAPI"],
                physical_pixel_sizes=reader.physical_pixel_sizes,
            )

        marker_img = img[CHANNEL_MAP["Marker"]]
        props = regionprops_table(
            img_segmented,
            intensity_image=marker_img,
            properties=["label", "mean_intensity", "area"],
        )

        # filter cells with low mean marker intensity
        df = pl.DataFrame(props).filter(pl.col("mean_intensity") > th_mean)

        # compute ratio of positive pixels for the marker
        results = []
        for label in df["label"].to_numpy():
            cell_mask = img_segmented == label
            values = marker_img[cell_mask]
            marker_pixels_over_half = int((values > th_pos).sum())
            results.append(
                {
                    "label": label,
                    "marker_pixels_over_0.2": marker_pixels_over_half,
                }
            )

        # filter out cells with too few positive pixels
        neutrophil_labels = (
            df.join(pl.DataFrame(results), on="label", how="left")
            .with_columns(
                (
                    pl.col("marker_pixels_over_0.2") / pl.col("area").cast(pl.Int64)
                ).alias("overlap")
            )
            .filter(pl.col("overlap") > th_ratio_pos)["label"]
            .to_numpy()
        )
        neutrophil_mask = np.isin(img_segmented, neutrophil_labels)
        img_segmented_filtered = img_segmented[~neutrophil_mask] = 0
        OmeTiffWriter.save(
            img_segmented_filtered,
            dst_path,
            dim_order="YX",
            channel_names=["DAPI"],
            physical_pixel_sizes=reader.physical_pixel_sizes,
            compression=None,
        )

# %% [markdown]
# # Step 2: Generate Shape-XML
# ### Step 2.1 Generate the T-Template
# %% T-Template
STEM_LEN = 180  # stem length of the T
BAR_LEN = 120  # bar length of the T
WIDTH = 15  # width of bar and stem
PADDING = 8  # bright halo around the T
XDIST = 9  # distance from the left side
TILT_DEG = 2.5  # degree of upward tilt
# note: only affects stem, the bar will be centered on the tilted T and will be straight


def make_T_template(
    width: int = WIDTH,
    stem_len: int = STEM_LEN,
    bar_len: int = BAR_LEN,
    padding: int = PADDING,
    xdist: int = XDIST,
    tilt_deg: float = TILT_DEG,
):
    # canvas Size
    h = max(width, bar_len) + 2 * padding + 40
    w = stem_len + width + 2 * padding + xdist
    canvas = np.zeros((h, w), dtype=np.float32)

    # build stem as a separate image
    rect_h = width
    rect_w = stem_len
    diag = int(np.hypot(rect_h, rect_w)) + 4
    stem_img = np.zeros((diag, diag), dtype=np.uint8)

    cx, cy_s = diag // 2, diag // 2
    x1 = cx - rect_w // 2
    x2 = x1 + rect_w
    y1 = cy_s - rect_h // 2
    y2 = y1 + rect_h
    stem_img[y1:y2, x1:x2] = 255  # stem rectangle

    # rotate so stem rises from bottom-left to top-right
    M = cv2.getRotationMatrix2D((cx, cy_s), tilt_deg, 1.0)
    stem_rot = cv2.warpAffine(
        stem_img, M, (diag, diag), flags=cv2.INTER_NEAREST, borderValue=0
    )
    stem_mask_rot = stem_rot > 0

    # place stem into main canvas
    ys, xs = np.where(stem_mask_rot)
    s_x1, s_x2 = xs.min(), xs.max() + 1
    s_y1, s_y2 = ys.min(), ys.max() + 1
    stem_crop = stem_mask_rot[s_y1:s_y2, s_x1:s_x2]
    crop_h, crop_w = stem_crop.shape

    place_x1 = padding + xdist  # Distance to 0 on a-axis
    place_x2 = place_x1 + crop_w
    place_cy = h // 2
    place_y1 = place_cy - crop_h // 2
    place_y2 = place_y1 + crop_h

    sx1 = 0
    sy1 = 0
    dx1 = place_x1
    dy1 = place_y1
    if place_x1 < 0:
        sx1 = -place_x1
        dx1 = 0
    if place_y1 < 0:
        sy1 = -place_y1
        dy1 = 0
    dx2 = min(w, place_x2)
    dy2 = min(h, place_y2)
    sx2 = sx1 + (dx2 - dx1)
    sy2 = sy1 + (dy2 - dy1)

    t_mask = np.zeros((h, w), dtype=bool)
    t_mask[dy1:dy2, dx1:dx2] |= stem_crop[sy1:sy2, sx1:sx2]

    # create vertical bar centered on the stem's right end
    stem_right = np.where(t_mask.any(axis=0))[0].max()
    stem_center_y = h // 2

    # place bar so its horizontal center aligns with stem_right
    bar_x_center = stem_right + 0.5

    bar_x1 = int(np.floor(bar_x_center - width / 2.0))
    bar_x2 = bar_x1 + width
    bar_y1 = stem_center_y - bar_len // 2
    bar_y2 = bar_y1 + bar_len

    bar_x1c = max(0, bar_x1)
    bar_x2c = min(w, bar_x2)
    bar_y1c = max(0, bar_y1)
    bar_y2c = min(h, bar_y2)
    if bar_x1c < bar_x2c and bar_y1c < bar_y2c:
        t_mask[bar_y1c:bar_y2c, bar_x1c:bar_x2c] = True

    canvas[:] = 1.0
    canvas[t_mask] = 0.0

    # define values for convolution
    dist = distance_transform_edt(~t_mask)
    band_mask = (dist > 0) & (dist <= padding)
    out_mask = (dist > 0) & (dist > padding)
    canvas[band_mask] = 1.0
    canvas[canvas == 0.0] = -1
    canvas[out_mask] = 0.0
    return canvas


# %% Possibility to inspect the T
# import matplotlib.pyplot as plt
# template = make_T_template()
# plt.imshow(template, cmap="grey", vmin=-1, vmax=1)
# %% [markdown]
# ### Step 2.2 Configure pyLMD and generate the XML
# %% Adjust paths and parameters
# details: https://mannlabs.github.io/py-lmd/pages/segmentation_loader.html#overview-of-configuration
loader_config = {
    "orientation_transform": np.array([[0, -1], [1, 0]]),  # Fixed
    "binary_smoothing": 14,
    "convolution_smoothing": 15,
    "poly_compression_factor": 30,
    "shape_dilation": 2,
    "shape_erosion": 0,
    "distance_heuristic": 300,
    "path_optimization": "none",
}

# relative path to the input folder with the *.tiff images
input_path = "input"
# relative path for the output
output_path = "output"

# details: https://scikit-image.org/docs/stable/api/skimage.exposure.html#skimage.exposure.equalize_adapthist
CLIP_LIMIT = 0.01
KERNEL_SIZE = (256, 256)

# cosmetic marker sizes in the "_calibdilated" images to help locate the exact position of the reference point
MARKER_RADIUS = 5  # radius of the circle
MARKER_LINELEN = 3  # len of the lines at the cardinals of the circle

MAX_WORKERS = 6  # mumber of parallel processes; depends on CPU-Cores and RAM


# %% Generate XML
def xml_shape_generator(src_path_str):
    src_path = Path(src_path_str)

    dst_path = dst_root / src_path.relative_to(src_root)

    filename = dst_path.name
    # remove the .ome.tiff ending to get "filename"
    base = filename[: -len(".ome.tiff")]
    calib_path = dst_path.parent / f"{base}_calib.ome.tiff"
    calibdilated_path = dst_path.parent / f"{base}_calibdilated.ome.tiff"
    shapes_path = dst_path.parent / f"{base}_shapes.xml"

    reader = BioImage(src_path)
    img = reader.get_image_data("CYX").astype(np.float32)[CHANNEL_MAP["Autofluorescence"]]
    img_inverted = 1 - contrast_stretching(img, (0, 100))
    img_inverted_rescaled = exposure.rescale_intensity(
        img_inverted.copy(), in_range=(0, 1), out_range=(-1, 1)
    )
    img_normalized = 1 - exposure.equalize_adapthist(
        img_inverted_rescaled, clip_limit=CLIP_LIMIT, kernel_size=KERNEL_SIZE
    )

    template = make_T_template()

    template_flip = template[::-1, ::-1]
    corr = fftconvolve(img_normalized, template_flip, mode="same")  # same size as image
    corr = exposure.rescale_intensity(corr, in_range="image", out_range=(0, 1))
    coords = peak_local_max(corr, num_peaks=3, min_distance=1000)
    corr_coords = []
    for y, x in coords:
        corr_coords.append(
            np.array(
                [
                    int(y - STEM_LEN * np.tan(np.radians(TILT_DEG))),  # correct tilt
                    int(
                        x + ((STEM_LEN - WIDTH - PADDING) / 2)
                    ),  # correct from stem center to stem-bar intersection
                ]
            )
        )

    img_segmented = BioImage(dst_path).get_image_data("YX")

    mask = np.zeros_like(img_normalized, dtype=np.uint8)
    mask_dilated = np.zeros_like(img_normalized, dtype=np.uint8)
    Y, X = np.ogrid[: img_normalized.shape[0], : img_normalized.shape[1]]
    radius = MARKER_RADIUS
    line_len = MARKER_LINELEN
    for i, (y, x) in enumerate(np.round(corr_coords[:3]).astype(int)):
        temp_mask = np.zeros_like(img_normalized, dtype=np.bool)
        circle = (X - x) ** 2 + (Y - y) ** 2 <= radius**2
        dist = np.sqrt((X - x) ** 2 + (Y - y) ** 2)
        circle = np.abs(dist - radius) < 0.5

        # cardinal lines outward from the circle boundary; Helps finding the exact location
        north = (X == x) & (Y >= y - radius - line_len) & (Y <= y - radius + 1)
        south = (X == x) & (Y <= y + radius + line_len) & (Y >= y + radius - 1)
        west = (Y == y) & (X >= x - radius - line_len) & (X <= x - radius + 1)
        east = (Y == y) & (X <= x + radius + line_len) & (X >= x + radius - 1)

        temp_mask |= circle | north | south | east | west
        mask_dilated[temp_mask] = 255 - (i + 1)
        mask_dilated[y, x] = 255 - (i + 1)
        mask[y, x] = 255 - (i + 1)

    OmeTiffWriter.save(
        mask_dilated,
        calibdilated_path,
        dim_order="YX",
        channel_names=["DAPI"],
        compression=None,
    )
    OmeTiffWriter.save(
        mask,
        calib_path,
        dim_order="YX",
        channel_names=["DAPI"],
        compression=None,
    )
    all_classes = np.unique(img_segmented)
    cell_sets = [{"classes": all_classes, "well": "A1"}]
    calibration_points = np.array(corr_coords)
    sl = SegmentationLoader(config=loader_config)
    shape_collection = sl(img_segmented, cell_sets, calibration_points)
    shape_collection.save(shapes_path)
    return str(src_path)


src_root = Path(input_path)
dst_root = Path(output_path)
# gather files
files = [str(p) for p in src_root.rglob("*.tiff") if "16" in str(p.name)]

if not files:
    print("No Input files found, please put your .tiff files in the input folder")
else:
    # run in parallel
    with ProcessPoolExecutor(max_workers=MAX_WORKERS) as exe:
        futures = {exe.submit(xml_shape_generator, f): f for f in files}
        for fut in tqdm(as_completed(futures), total=len(futures)):
            try:
                res = fut.result()
            except Exception as e:
                src = futures[fut]
                print(f"Error processing {src}: {e}")
