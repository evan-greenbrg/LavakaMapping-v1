import numpy as np
from osgeo import gdal, gdal_array
import rasterio
from sklearn.externals import joblib


def create_multi_image(ds):
    """
    From the multispectral image will create a multi-level array of the image
    """
    img = np.zeros(
        (ds.RasterYSize, ds.RasterXSize, ds.RasterCount),
        gdal_array.GDALTypeCodeToNumericTypeCode(ds.GetRasterBand(1).DataType)
    )

    for b in range(1, img.shape[2]):
        img[:, :, b] = ds.GetRasterBand(b + 1).ReadAsArray()

    return img


# Load the model
path = 'TrainedRF_20200311.pkl'
rf = joblib.load(path)

# Path to the classification stack (input features)
img_path = ''
ds = gdal.Open(img_path)
img = create_multi_image(ds)
img = img[:-1, :, 1:]
ds = None

# Predict the whole image
new_shape = (img.shape[0] * img.shape[1], img.shape[2])
img_as_array = img[:, :, :].reshape(new_shape)
print('Reshaped from {o} to {n}'.format(
    o=img.shape,
    n=img_as_array.shape)
)

# Now predict for each pixel
class_prediction = rf.predict(img_as_array)

# Reshape our classification map
class_prediction = class_prediction.reshape(img[:, :, 0].shape)

# Output Class Predictions
dsr = rasterio.open(img_path)
meta = dsr.meta.copy()
# Note: Hard coded epsg here to UTM zone 38
meta.update({'crs': rasterio.crs.CRS.from_epsg(32638)})
dsr = None

# Path to where you want to put the classified himage
opath = 'classified.tif'
with rasterio.open(opath, "w", **meta) as dest:
    dest.write(class_prediction.astype(rasterio.float32), 1)
