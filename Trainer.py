from datetime import datetime
import os

import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import ImageGrid
import numpy as np
from osgeo import gdal, gdal_array
import pandas as pd
import rasterio
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.externals import joblib


def create_multi_image(ds):
    """
    From the multispectral image will create a multi-level array of the image
    """
    img = np.zeros(
        (ds.RasterYSize, ds.RasterXSize, ds.RasterCount - 1),
        gdal_array.GDALTypeCodeToNumericTypeCode(ds.GetRasterBand(1).DataType)
    )

    for b in range(1, img.shape[2]):
        img[:, :, b] = ds.GetRasterBand(b + 1).ReadAsArray()

    return img


def create_roi(ds, roi_idx):
    """
    Creates the roi array with the classes
    """
    roi = ds.GetRasterBand(ds.RasterCount).ReadAsArray().astype(np.uint8)
    roi[roi == 128] = 0

    return roi


def fit_hyperparameters(x, y):
    params = {
        'n_estimators': [
            int(x) for x in np.linspace(start=400, stop=600, num=3)
        ],
        'max_depth': [90, 100, 110, None]
    }
    rf = RandomForestClassifier(oob_score=True)
    clf = GridSearchCV(
        estimator=rf,
        param_grid=params,
        cv=3,
        n_jobs=-1,
        verbose=2
    )
    # Fit the random search model
    clf.fit(x, y)

    return clf.best_estimator_, max(clf.cv_results_['mean_test_score'])


def color_stretch(image, index, minmax=(0, 10000)):
    colors = image[:, :, index].astype(np.float64)

    max_val = minmax[1]
    min_val = minmax[0]

    # Enforce maximum and minimum values
    colors[colors[:, :, :] > max_val] = max_val
    colors[colors[:, :, :] < min_val] = min_val

    for b in range(colors.shape[2]):
        colors[:, :, b] = colors[:, :, b] * 1 / (max_val - min_val)

    return colors


# Set the data up
img_path = ''    # Path to the stack of training features
ds = gdal.Open(img_path)

# Set up arrays
img = create_multi_image(ds)
img = img[:, :, 1:]
roi = create_roi(ds, ds.RasterCount)
ds = None

# Create X matrix containing features and Y matrix containing labels
x = img[roi > 0, :]
y = roi[roi > 0]

# Sample the holdout data
x_train, x_test, y_train, y_test = train_test_split(
    x,
    y,
    test_size=0.2,
    random_state=0
)

rf, score = fit_hyperparameters(x_train, y_train)
# Test the Holdout
rf = rf.fit(x_test, y_test)

# Go through some descriptive stuff
print('Our OOB prediction of accuracy is: {oob}%'.format(
    oob=rf.oob_score_ * 100
))

# Feature importance
bands = [1, 2, 3, 4, 5, 6, 7]
for b, imp in zip(bands, rf.feature_importances_):
    print('Band {b} importance: {imp}'.format(b=b, imp=imp))

# Confusion Matrix
df = pd.DataFrame(data={
    'truth': y,
    'predict': rf.predict(x)
})

output_root = ''    # Root if where to put the ouput data
df_name = 'prediction_matrix.csv'
out = os.path.join(output_root, df_name)
df.to_csv(out)

# Cross-tabulate predictions
print(pd.crosstab(df['truth'], df['predict'], margins=True))

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
meta.update({'crs': rasterio.crs.CRS.from_epsg(32638)})

opath = 'classified_311.tif'
with rasterio.open(opath, "w", **meta) as dest:
    dest.write(class_prediction.astype(rasterio.float32), 1)

# Save the model
now = datetime.now().strftime('%Y%m%d')
fn = 'TrainedRF_{}.pkl'.format(now)
joblib.dump(rf, fn)

# Visualize Training
# Path to the correlation file
cor_path = ''
cords = rasterio.open(cor_path)
cor_ar = cords.read(1)

imgLandsat = color_stretch(img, [5, 4, 3], (0, 25000))
imgRadar = color_stretch(img, [6, 8, 7], (0, 500))
imgDEM = color_stretch(img, [0, 0, 0], (0, 100))
imgCor = color_stretch(img, [0, 0, 0], (0, 1))
plt.imshow(imgCor)
plt.show()

n = class_prediction.max()
# Next setup a colormap for our map
colors = dict((
    (0, (255, 255, 255, 255)),  # Nodata
    (1, (108, 122, 137, 255)),  # Bedrock
    (2, (145, 61, 136, 255)),  # Farmland
    (3, (30, 130, 76, 255)),  # Forest
    (4, (242, 120, 75, 255)),  # Grassland
    (5, (207, 0, 15, 255)),  # Gully
    (6, (31, 58, 147, 255)),  # Stream
    (7, (0, 181, 204, 255)),  # Urban
))
# Put 0 - 255 as float 0 - 1
for k in colors:
    v = colors[k]
    _v = [_v / 255.0 for _v in v]
    colors[k] = _v

index_colors = [
    colors[key] if key in colors
    else (255, 255, 255, 0)
    for key in range(1, n + 1)
]
pred_cmap = plt.matplotlib.colors.ListedColormap(
    index_colors,
    'Classification',
    n
)

index_colors = [
    colors[key] if key in colors
    else (255, 255, 255, 0)
    for key in range(0, n)
]
roi_cmap = plt.matplotlib.colors.ListedColormap(
    index_colors,
    'Classification',
    n
)

# TRAINING FIGURE
fig = plt.figure(figsize=(4., 4.))
grid = ImageGrid(
    fig, 111,
    nrows_ncols=(2, 2),
    axes_pad=0.1,
)

grid[0].imshow(imgDEM)
grid[2].imshow(imgLandsat)
grid[3].imshow(imgRadar)

plt.show()

fig.savefig('Classification_figure.svg', format='svg')


fig2 = plt.figure()
grid = ImageGrid(
    fig2,
    111,
    nrows_ncols=(1, 2),
    axes_pad=0.1
)
grid[0].imshow(class_prediction, cmap=pred_cmap, interpolation='none')
grid[1].imshow(cor_ar, cmap='Greys_r')

fig2.savefig('Correlation_figure.svg', format='svg')
