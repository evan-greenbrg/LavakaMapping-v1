import os
import json

import gdal
import numpy as np
import osr
import rasterio
import utm
from rasterio.mask import mask
from rasterio.merge import merge
from scipy.signal import medfilt
from scipy.ndimage.measurements import label
from scipy.ndimage import minimum_filter
from skimage.measure import label as sklabel
from shapely.geometry import box
import geopandas as gpd
from fiona.crs import from_epsg


class RasterHelper():

    def __init__(self):
        pass

    def rename_files(self, inpath, str_add):
        """
        Renames file in file path with an _str_add
        """
        pathc = inpath.split('/')
        fnc = pathc[-1].split('.')
        fn = fnc[0] + '_{}'.format(str_add)
        fnc[0] = fn
        fn = '.'.join(fnc)
        pathc[-1] = fn

        return '/'.join(pathc)

    def value_from_coordinates(self, ds, northing, easting,
                               xOrigin, yOrigin, pixelWidth, pixelHeight):
        """
        Finds DEM values from set of coordinates
        """
        i = np.floor((yOrigin - easting) / pixelHeight).astype('int')
        j = np.floor((northing - xOrigin) / pixelWidth).astype('int')

        return ds[i, j]

    def bounding_coordinates(self, ds):
        """
        Finds bounding coordinates from a geoTif file
        """
        width = ds.RasterXSize
        height = ds.RasterYSize
        gt = ds.GetGeoTransform()
        minx = gt[0]
        miny = gt[3] + width*gt[4] + height*gt[5]
        maxx = gt[0] + width*gt[1] + height*gt[2]
        maxy = gt[3]

        return minx, miny, maxx, maxy

    def get_pixel_size(self, ds):
        """
        Gets the pixel size in whatever units the raster is using
        """
        gt = ds.transform
        pixelSizeX = gt[0]
        pixelSizeY = -gt[4]

        return pixelSizeX, pixelSizeY

    def reproject_raster(self, ds, inpath, outpath, outEPSG):
        """
        Reprojects raster to a specified EPSG
        """
        try:
            gdal.Warp(
                outpath,
                ds,
                dstSRS='EPSG:{}'.format(outEPSG)
            )
        except:
            raise Exception(
                'Could not convert raster to EPSG: {}'.format(outEPSG)
            )

        return outpath

    def reproject_files(self, inputs, outroot, envi=False):
        """
        Takes the file list and checks to see if reprojectons are created
        This is specific to reproject form non UTM to UTM
        This is the alternative version that is specifically used in
        PreProcess
        """
        rh = RasterHelper()
        outpaths = []
        for key in inputs.keys():
            ds = gdal.Open(inputs[key])
            srs = osr.SpatialReference(wkt=ds.GetProjection())
            epsg = int(srs.GetAttrValue('AUTHORITY', 1))
            # If the projection is already UTM
            if epsg == 32638:
                continue

            print('Reprojection: {}'.format(key))
            # Open Raster

            # Find bounding coordinates
            minx, miny, maxx, maxy = rh.bounding_coordinates(ds)

            # Get the UTM information for the location
            y, x, zone, letter = utm.from_latlon(miny, minx)
            outEPSG = int('326' + str(zone))

            # Create reprojected files if they don't already exist
            outfn = rh.rename_files(inputs[key], str(outEPSG)).split('/')[-1]
            outpath = os.path.join(outroot, outfn)

            if not os.path.exists(outpath):
                rh.reproject_raster(ds, inputs[key], outpath, outEPSG)

            # Close Gdal file + Open rasterio object
            ds = None

            outpaths.append(outpath)

        return outpaths, outEPSG

    def stack_bands(self, files, outpath):
        """
        Stacks a file list into a single file with multiple bands
        """
        outvrt = '/vsimem/stacked.vrt'
        outds = gdal.BuildVRT(outvrt, files, separate=True)
        outds = gdal.Translate(outpath, outds)
        outds = None

        return os.path.exists(outpath)

    def files_to_mosaic(self, fps, outpath,
                        search_regex=None, dem_fps=None, write=True):
        """
        Takes list of file paths and merges them into a single file
        """

        src_files_to_mosaic = []
        for fp in fps:
            src = rasterio.open(fp)
            src_files_to_mosaic.append(src)

        mosaic, out_trans = merge(src_files_to_mosaic)

        out_meta = src.meta.copy()
        out_meta.update(
            {
                "driver": "GTiff",
                "height": mosaic.shape[1],
                "width": mosaic.shape[2],
                "transform": out_trans
            }
        )

        if write:
            with rasterio.open(outpath, "w", **out_meta) as dest:
                dest.write(mosaic)

        return mosaic

    def filter_on_class_size(self, path, class_int, pixel_size=5):
        """
        Filter the classified pixels by a minimum pixel size
        """
        ds = rasterio.open(path)
        ar = ds.read(1)

        # Filter for Gully class
        ar[ar != class_int] = 0

        # Set up structure for connectivity and find connectivity
        structure = np.ones((3, 3), dtype=np.int)
        labeled, ncomponents = label(ar, structure)

        # use sklabel to find the sizes of the labels
        labels = sklabel(ar)
        unique, counts = np.unique(labels, return_counts=True)
        label_filt = unique[(counts > 5) & (counts < max(counts))]

        # Filter based on label size
        array_filt = labeled == label_filt[0]
        for l in label_filt:
            array_filt += labeled == l

        array_filt = array_filt.astype(int)

        return array_filt

    def clip_raster(self, ipath, mask_path, epsg, outroot):
        """
        Clips raster file based on bounding box coordinates
        """
        outfn = self.rename_files(ipath, '_clip').split('/')[-1]
        opath = os.path.join(outroot, outfn)

        data = rasterio.open(ipath)
        dsmask = gdal.Open(mask_path)

        minx0, miny0, maxx0, maxy0 = self.bounding_coordinates(dsmask)
        dsmask = None

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
                "crs": rasterio.crs.CRS.from_epsg(epsg)
            }
        )

        with rasterio.open(opath, "w", **out_meta) as dest:
            dest.write(out_img)

        return opath

    def minimum_filter_array(self, ipath, outpath):
        """
        Takes a median filter of the image
        """
        ds = rasterio.open(ipath)
        meta = ds.meta.copy()

        array = ds.read(1)
        filt_array = minimum_filter(array, (3, 3))

        meta.update({
            'height': filt_array.shape[0],
            'width': filt_array.shape[1]
        })

        with rasterio.open(outpath, 'w', **meta) as dst:
            dst.write(filt_array.astype(rasterio.float32), 1)

    def resample_images(self, i, coarsest, outpath):
        """
        Resamples the images based on the coarsest image found
        """
        # Get Coarsest shape for resampling
        coarse_ds = rasterio.open(coarsest)
        coarse_array = coarse_ds.read(1)
        coarse_meta = coarse_ds.meta.copy()

        ds = rasterio.open(i)
        dsarray = ds.read(1)

        # Check if file exists
        if os.path.exists(outpath):
            print('File already exists: {}'.format(outpath))
            return outpath

        out_array = np.empty(coarse_array.shape)
        # Iterate through all training, find values at the lowest resolution
        for (row, col), x in np.ndenumerate(coarse_array):
            search_coords = coarse_ds.xy(row, col)
            try:
                out_array[row, col] = float(dsarray[
                    ds.index(search_coords[0], search_coords[1])
                ])
            except:
                out_array[row, col] = None

        # Update the meta data to save geotif
        out_meta = ds.meta.copy()
        out_meta.update(
            {
                "driver": "GTiff",
                "height": coarse_meta['height'],
                "width": coarse_meta['width'],
                "transform": coarse_meta['transform'],
                "crs": coarse_meta['crs'],
                "nodata": None,
                "dtype": 'float32'
            }
        )
        # Stop holding data
        ds = None
        dsarray = None

        # Save the file
        with rasterio.open(outpath, "w", **out_meta) as dest:
            dest.write(out_array.astype(rasterio.float32), 1)
        out_array = None

        # Return the outpath for the output file
        return outpath

    def median_filter(self, ipath, outroot):
        """
        Takes a median filter of the image
        From PreProcess
        """
        outfn = self.rename_files(ipath, '_median').split('/')[-1]
        outpath = os.path.join(outroot, outfn)

        ds = rasterio.open(ipath)
        meta = ds.meta.copy()

        array = ds.read(1)
        filt_array = medfilt(array, 3)

        with rasterio.open(outpath, 'w', **meta) as dst:
            dst.write(filt_array.astype(rasterio.float32), 1)

    def find_coarsest_resolution(self, inputs):
        """
        Takes the file list and finds the coarsest resolution
        Used in PreProcess
        """
        rh = RasterHelper()
        print('Finding Coarsest Resolution')
        coarsest = ('None', 0)
        for key in inputs.keys():
            # Open Dataset
            ds = rasterio.open(inputs[key])

            # Get the coarsest resolution
            pixelsizeX, pixelsizeY = rh.get_pixel_size(ds)
            if pixelsizeX > coarsest[1]:
                coarsest = (key, pixelsizeX)

            # Close Rasterio object
            ds = None

        return coarsest

    def set_epsg(self, inputs):
        """
        Sets the EPSG for the file
        From PreProcess
        """
        for key in inputs.keys():
            ds = rasterio.open(inputs[key])
            ar = ds.read(1)
            meta = ds.meta.copy()
            crs = rasterio.crs.CRS.from_epsg(4326)
            wkt = crs.to_wkt()
            meta.update(
                {
                    "driver": "GTiff",
                    'height': ar.shape[0],
                    'width': ar.shape[1],
                    'dtype': 'float32',
                    'transform': ds.transform,
                    'crs': wkt
                }
            )

            opath = inputs[key]
            with rasterio.open(opath, "w", **meta) as dest:
                dest.write(ar.astype(rasterio.float32), 1)
