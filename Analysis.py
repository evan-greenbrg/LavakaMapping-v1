import os

import pandas
import gdal
import numpy as np
import rasterio
from matplotlib import pyplot as plt

from RasterHelper import RasterHelper


def get_stats(array, cor_array, classi):
    """
    Get the stats from the correlation arrays
    """
    class_ar = array[classi, :, :]

    # filter the correlation array for just that class
    values = cor_ar[class_ar > 0]
    values = values[~np.isnan(values)]
    n = len(values)
    q5 = np.quantile(values, .05)
    q1 = np.quantile(values, .01)
    pct_le_5 = len(values[values < q5]) / n
    pct_le_1 = len(values[values < q1]) / n

    return values, {
        'n': n,
        'median': np.median(values),
        'std': np.std(values),
        'mini': np.min(values),
        'maxi': np.max(values),
        'q25': np.quantile(values, .25),
        'q75': np.quantile(values, .75),
        'q5': q5,
        'pct_le_5': pct_le_5,
        'pct_le_1': pct_le_1,
    }


# Analysis
rh = RasterHelper()

# Path to the classification image
path = ''

ds = rasterio.open(path)
meta = ds.meta.copy()
meta.update({
    'count': 1,
    'dtype': 'float32'
})

# Create individual rasters for each land cover type
classes = [
    'bedrock',
    'farmland',
    'forest',
    'grassland',
    'gully',
    'stream',
    'urban'
]
paths = []
for i in range(0, 7):
    array_filt = rh.filter_on_class_size(path, i + 1, 10)
    print(array_filt.shape)
    opath = rh.rename_files(path, '_{}'.format(classes[i]))
    paths.append(opath)
    if os.path.exists(opath):
        continue
    else:
        with rasterio.open(opath, "w", **meta) as dest:
            dest.write(array_filt.astype(rasterio.dtypes.float32), 1)

# Iterate through all the date files to get all of the data
dates = [
    '20160111_20160204',
    '20160204_20160228',
]
class_paths = paths
raw_data = {}
for date in dates:
    paths = class_paths
    print(date)
    # This points to the root folder for all of the correlation files
    cor_root = ''
    cor_name = 'phsig.cor.geo'
    cor_path = os.path.join(cor_root, date, cor_name)

    # Reproject
    print('reprojecting')
    ds = gdal.Open(cor_path)
    outpath = os.path.join(cor_root, date, '')
    outpath = rh.reproject_raster(ds, cor_path, outpath, 32638)

    # Cut the Correlation diagram to the size of the classification
    print('clipping')
    clippath = rh.clip_raster(
        outpath,
        paths[0],
        32638,
        os.path.join(cor_root, date)
    )

    # Take a minimum filter
    print('Minimum filter')
    minpath = os.path.join(cor_root, date, '')
    rh.minimum_filter_array(clippath, minpath)

    # Resample
    print('Resample')
    respath = os.path.join(cor_root, date, '')
    respath = rh.resample_images(minpath, paths[0], respath)

    # Stack the two images and output to specified path
    outpath = ''
    rh.stack_bands(paths + [respath], outpath)

    # Calculate statistics
    ds = rasterio.open(outpath)
    array = ds.read()

    # Set up to only lok at actual values
    cor_ar = array[7, :, :]
    cor_ar[cor_ar == 0] = np.NaN

    ns = []
    medians = []
    stds = []
    q5s = []
    minis = []
    maxis = []
    q25s = []
    q75s = []
    pct_le_5s = []
    pct_le_1s = []
    classes = [
        'bedrock',
        'farmland',
        'forest',
        'grassland',
        'gully',
        'stream',
        'urban'
    ]
    class_data = {}
    for i in range(0, 7):
        print(classes[i])
        raw_, values = get_stats(array, cor_ar, i)
        ns.append(values['n'])
        medians.append(values['median'])
        stds.append(values['std'])
        minis.append(values['mini'])
        maxis.append(values['maxi'])
        q5s.append(values['q5'])
        q25s.append(values['q25'])
        q75s.append(values['q75'])
        pct_le_5s.append(values['pct_le_5'])
        pct_le_1s.append(values['pct_le_1'])

        class_data[classes[i]] = raw_
    raw_data[date] = class_data

    # Stats Dataframe
    stats_df = pandas.DataFrame(data={
        'n': ns,
        'class': classes,
        'median': medians,
        'std': stds,
        'min': minis,
        'max': maxis,
        'q5': q5s,
        'q25': q25s,
        'q75': q75s,
        'pct_le_5': pct_le_5s,
        'pct_le_1': pct_le_1s,
    })
    print(stats_df)
    cor_path = ''
    df_name = '{}_stats_df_training.csv'.format(date)
    outp = os.path.join(cor_path, df_name)
    stats_df.to_csv(outp)

fn = ''
np.save(fn, raw_data)


# Histogram
xs = []
for i, key in enumerate(raw_data.keys()):
    for j, c in enumerate(raw_data[key]):
        xs.append(raw_data[key][c])

figsize = (16, 10)
cols = 11
rows = 7
bins = 50
fig1, axs = plt.subplots(
    rows,
    cols,
    sharex=True,
    figsize=figsize
)
# Column Names:
col_names = [
    '1/11-2/04',
    '2/04-2/28',
    '2/28-3/23',
    '3/23-4/16',
    '4/16-5/10',
    '5/10-6/03',
    '6/03-7/21',
    '7/21-8/14',
    '8/14-10/01',
    '10/01-10/25',
    '1/11-10/25',
]

row_names = [i for i in raw_data['20160111_20160204'].keys()]
for ax, col in zip(axs[0], col_names):
    ax.set_title(col, fontsize=7)

for ax, row in zip(axs[:, 0], row_names):
    ax.set_ylabel(row)

for i, key in enumerate(raw_data.keys()):
    for j, c in enumerate(raw_data[key]):
        axs[j, i].hist(raw_data[key][c], bins=bins, color='black')
        axs[j, i].get_yaxis().set_ticks([])

for ax, col in zip(axs[-1], col_names):
    ax.set_xticks([0, .5, 1])

fig1.tight_layout()
na = ''
fig1.savefig(na)
plt.close()

# Boxplot
for classi in row_names:
    box_data = {}
    for key, values in raw_data.items():
        data = values[classi]
        box_data[key] = data

    flierprops = dict(
        marker='o',
        markerfacecolor='black',
        markersize=3,
        linestyle='none',
        markeredgecolor='black'
    )
    data = [data for data in box_data.values()]
    figsize = (16, 5)
    fig2, ax1 = plt.subplots(figsize=figsize)
    ax1.set_title(classi)
    ax1.boxplot(data, flierprops=flierprops)

    plt.setp(ax1, xticks=[y + 1 for y in range(len(data))],
             xticklabels=col_names)
    ax1.tick_params(axis='both', which='major', labelsize=7)
    ax1.set_ylim(1, 0)
    ax1.set_ylabel('Phase Correlation')
    ax1.set_xlabel('Interferogram Date Range')

    na = ''
    fig2.savefig(na, format='jpg')
    plt.close()
