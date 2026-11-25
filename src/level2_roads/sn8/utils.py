# Copied from https://github.com/SpaceNetChallenge/SpaceNet8 (Apache-2.0),
# commit 30ca30c2530c6a234d040f60a99a7366ff62d756, file baseline/utils/utils.py.
# Changes: none.
# Copyright of the original: the SpaceNet 8 authors; licence: third_party/spacenet8/LICENSE.
# ruff: noqa
from osgeo import gdal

def write_geotiff(output_tif, ncols, nrows,
                  xmin, xres,ymax, yres,
                 raster_srs, label_arr):
    driver = gdal.GetDriverByName('GTiff')
    out_ds = driver.Create(output_tif, ncols, nrows, len(label_arr), gdal.GDT_Byte)
    out_ds.SetGeoTransform((xmin, xres, 0, ymax, 0, yres))
    out_ds.SetProjection(raster_srs.ExportToWkt())
    for i in range(len(label_arr)):
        outband = out_ds.GetRasterBand(i+1)
        outband.WriteArray(label_arr[i])
        #outband.SetNoDataValue(0)
        outband.FlushCache()
    out_ds = None