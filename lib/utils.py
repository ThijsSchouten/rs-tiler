import rasterio as rio
import geopandas as gpd
import numpy as np

from shapely.geometry import box, Polygon

def get_tif_bbox_geom(tif_path):
    with rio.open(tif_path) as src:
        bounds = src.bounds
        bbox = box(bounds.left, bounds.bottom, bounds.right, bounds.top)
        return bbox
    
def chw_to_hwc(arr):
    return np.moveaxis(arr, 0, -1)

def hwc_to_chw(arr):
    return np.moveaxis(arr, -1, 0)

def check_nodata(arr):
    # if the alpha channel is all 0, then there is no data
    if arr.shape[-1] == 4:
        return arr[:,:,-1].sum() == 0


def remove_alpha_if_exists(arr):
    if arr.shape[-1] == 4:
        return arr[:,:,:3]
    else:
        return arr

def distribute_indices(values, ratios):
    # Indices of nonzero ratios
    nonzero_ratio_indices = np.where(np.array(ratios) > 0)[0]

    # Find indices of zero-valued and non-zero-valued elements
    all_indices = np.arange(len(values))
    
    # Shuffle all indices
    np.random.shuffle(all_indices)

    # Calculate the size of each class
    class_sizes = (np.array(ratios)[nonzero_ratio_indices] * len(values)).astype(int)
    
    # Initialize lists to hold the distributed indices
    distributed_indices = [[] for _ in range(len(ratios))]

    # Distribute the shuffled indices
    start = 0
    for i, class_size in enumerate(class_sizes):
        end = start + class_size
        distributed_indices[nonzero_ratio_indices[i]] = all_indices[start:end].tolist()
        start = end

    # Distribute remaining indices to classes, if any remain
    if start < len(all_indices):
        for i in range(start, len(all_indices)):
            for j in range(len(distributed_indices)):
                if len(distributed_indices[j]) < class_sizes[j]:
                    distributed_indices[j].append(all_indices[i])
                    break

    return distributed_indices

