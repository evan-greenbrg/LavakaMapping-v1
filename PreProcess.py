import os
import json

import gdal
import rasterio
from rasterio.mask import mask
from shapely.geometry import box
import geopandas as gpd
from fiona.crs import from_epsg
import pycrs

from RasterHelper import RasterHelper


def clip_raster(ipath,  epsg, outroot, mask_path=None, bounding=None):
    """
    Clips raster file based on bounding box coordinates
    """
    rh = RasterHelper()
    outfn = rh.rename_files(ipath, '_clip').split('/')[-1]
    opath = os.path.join(outroot, outfn)

    data = rasterio.open(ipath)

    if mask_path:
        dsmask = gdal.Open(mask_path)
        minx0, miny0, maxx0, maxy0 = rh.bounding_coordinates(dsmask)
        dsmask = None
    if bounding:
        minx0, miny0, maxx0, maxy0 = bounding

    bbox = box(minx0, miny0, maxx0, maxy0)
    geo = gpd.GeoDataFrame(
        {'geometry': bbox},
        index=[0],
        crs=from_epsg(epsg)
    )
    coords = [json.loads(geo.to_json())['features'][0]['geometry']]

    out_img, out_transform = mask(
        dataset=data,
        shapes=coords,
        crop=True
    )

    out_meta = data.meta.copy()
    data = None

    out_meta.update(
        {
            "driver": "GTiff",
            "height": out_img.shape[1],
            "width": out_img.shape[2],
            "transform": out_transform,
            "crs": pycrs.parse.from_epsg_code(epsg).to_proj4()
        }
    )

    with rasterio.open(opath, "w", **out_meta) as dest:
        dest.write(out_img)


# PreProcessing Workflow
rh = RasterHelper()

# Set SAR Projection
inputs = {
    'vhPath': '',
    'vvPath': '',
    'vv_vhPath': '',
}
rh.set_epsg(inputs)

# Reproject images
inputs = {
    'b1Path': '',
    'b2Path': '',
    'b3Path': '',
    'b4Path': '',
    'b5Path': '',
    'vhPath': '',
    'vvPath': '',
    'vv_vhPath': '',
    'demPath': '',
    'slopePath': '',
}
# Path to the output directory
outroot = ''
paths, epsg = rh.reproject_files(inputs, outroot)

# Clip to size
inputs = {
    'demPath': '',
    'slopePath': '',
    'b1Path': '',
    'b2Path': '',
    'b3Path': '',
    'b4Path': '',
    'b5Path': '',
    'vhPath': '',
    'vvPath': '',
    'vv_vhPath': '',
}
outroot = ''
bounding = (637036.54, -2125264.53, 701309.38, -2047680.35)
# Hard coded EPSG of UTM zone 38
for key in inputs.keys():
    clip_raster(inputs[key], 32638, outroot, mask_path=None, bounding=bounding)

# Median Filter for SAR images
inputs = {
    'vhPath': '',
    'vvPath': '',
    'vv_vhPath': '',

}
# Output location the median filters
outroot = ''
for key in inputs.keys():
    print(key)
    rh.median_filter(inputs[key], outroot)

# Find coarsest DEM
inputs = {
    'demPath': '',
    'slopePath': '',
    'b1Path': '',
    'b2Path': '',
    'b3Path': '',
    'b4Path': '',
    'b5Path': '',
    'vhPath': '',
    'vvPath': '',
    'vv_vhPath': '',
}
coarsest = rh.find_coarsest_resolution(inputs)

# Resample to same image size
outroot = ''    # Output directory for resampled files
paths = [i for i in rh.resample_images(inputs, coarsest, outroot)]

# Stack the files into multi-band file
outpath = ''    # Output for the stacked files
rh.stack_bands(paths, outpath)
