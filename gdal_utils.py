# -*- coding: utf-8 -*-
"""
Created on Wed Jul 18 10:25:09 2018

@author: rmgu
"""

import math
import numpy as np
import tempfile
from osgeo import ogr, osr, gdal
from pyproj import Proj, Transformer, CRS
import xarray
import logging
import multiprocessing as mp
import pandas as pd
import xarray as xr

log = logging.getLogger(__name__)

def calculate_geotransform(center_x, center_y, pixel_size, width, height):
    xmin = center_x - pixel_size * width / 2
    ymax = center_y + pixel_size * height / 2
    geotransform = (xmin, pixel_size, 0, ymax, 0, -pixel_size)
    return geotransform

def rasterlist2dict(raster_list):

    out_dict = {}
    for im in raster_list:
        name = im.name
        type = name.split('_')[1]
        if type == 'IP':
            year = name.split('_')[2]
            month = name.split('_')[-1][:-4]
            date_dt = pd.to_datetime(f'{year}-{month}', format='%Y-%m')
        else:
            month = name.split('_')[-1][:-4]
            date_dt = int(month)

        fid = gdal.Open(str(im))
        ar = fid.GetRasterBand(1).ReadAsArray()
        out_dict[date_dt] = ar
    return out_dict

def rasterlist2xarray(raster_list):
    times = []
    out_ds = xr.Dataset()
    for im in raster_list:
        name = im.name
        year = name.split('_')[-2]
        month = name.split('_')[-1][:-4]
        date_dt = pd.to_datetime(f'{year}-{month}', format='%Y-%m')
        times.append(date_dt)
        fid = gdal.Open(str(im))
        ar = fid.GetRasterBand(1).ReadAsArray()
        out_ds[date_dt] = (['y', 'x'], ar)

    # Assign coordinates
    out_ds.coords['time'] = np.array(times)
    out_ds.coords['y'] = np.arange(ar.shape[1])
    out_ds.coords['x'] = np.arange(ar.shape[0])

    return out_ds


def get_raster_data(input_file_path, bands=None):
    """
    Helper to read a GDAL image file

    Parameters
    ----------
    input_file_path : str of Path object
        Path to the input GDAL image file

    Returns
    -------
    array : 2D array
        Output numpy array
    """
    fid = gdal.Open(str(input_file_path), gdal.GA_ReadOnly)
    if isinstance(bands, type(None)):
        bands = range(1, fid.RasterCount + 1)
    array = []
    for band in bands:
        array.append(fid.GetRasterBand(band).ReadAsArray())
    del fid
    array = np.squeeze(np.array(array))
    return array


def raster_info(input_file_path):

    fid = gdal.Open(str(input_file_path), gdal.GA_ReadOnly)
    gt = fid.GetGeoTransform()
    proj = fid.GetProjection()
    x_size = fid.RasterXSize
    y_size = fid.RasterYSize
    bands = fid.RasterCount
    lr_x = gt[0] + x_size * gt[1] + y_size * gt[2]
    lr_y = gt[3] + x_size * gt[4] + y_size * gt[5]
    extent = [gt[0], lr_y, lr_x, gt[3]]
    del fid
    center = np.mean([gt[0], lr_x]), np.mean([gt[3], lr_y])
    p = Proj(proj)
    center_geo = p(center[0], center[1], inverse=True)

    return proj, gt, x_size, y_size, extent, center_geo, bands


def save_image(in_array, gt, proj, output_file, no_data_value=np.nan):
    """
    Helper to save a GDAL Cloud Optimized GeoTiff

    Parameters
    ----------
    in_array : 2D array
        Numpy image array
    gt : list of floats
        Output GDAL geotransform
    proj : str
        Output GDAL projection
    output_file : str or Path object
        Path to the output GeoTiff image
    """
    memDriver = gdal.GetDriverByName("MEM")
    if in_array.dtype == bool:
        gdal_type = gdal.GDT_Byte
        no_data_value = None
    elif in_array.dtype == int:
        gdal_type = gdal.GDT_Int32
    else:
        gdal_type = gdal.GDT_Float32

    shape = in_array.shape
    if len(shape) > 2:
        ds = memDriver.Create("MEM", shape[1], shape[0], shape[2], gdal_type)
        ds.SetProjection(proj)
        ds.SetGeoTransform(gt)
        for i in range(shape[2]):
            ds.GetRasterBand(i+1).WriteArray(in_array[:, :, i])

    else:
        ds = memDriver.Create("MEM", shape[1], shape[0], 1, gdal_type)
        ds.SetProjection(proj)
        ds.SetGeoTransform(gt)
        ds.GetRasterBand(1).WriteArray(in_array)
    ds.FlushCache()
    if output_file == "MEM":
        ds.GetRasterBand(1).SetNoDataValue(no_data_value)
        out_ds = ds
    else:
        driver_opt = ['COMPRESS=DEFLATE',
                      'PREDICTOR=YES',
                      'BIGTIFF=IF_SAFER',
                      'NUM_THREADS=ALL_CPUS']
        
        out_ds = gdal.Translate(str(output_file), 
                                ds, 
                                format="COG", 
                                creationOptions=driver_opt,
                                noData=no_data_value, 
                                stats=False)
    # If GDAL driers for other formats do not exist then default to GeoTiff
    if out_ds is None:
        log.warning("Selected GDAL driver is not supported!"
                    "Saving as GeoTiff!")
        driver_opt = ['COMPRESS=DEFLATE',
                      'PREDICTOR=1',
                      'BIGTIFF=IF_SAFER',
                      'NUM_THREADS=ALL_CPUS']

        ds = gdal.Translate(str(output_file), 
                            ds, 
                            format="GTiff", 
                            creationOptions=driver_opt,
                            noData=no_data_value, 
                            stats=True)
    else:
        ds = out_ds

    return ds


def resample_with_gdalwarp(src,
                           template,
                           resample_alg,
                           out_file="",
                           out_format="MEM",
                           warp_options=None):

    # Get template projection, extent and resolution
    proj, gt, _, _, extent, *_ = raster_info(template)
    if warp_options is None:
        warp_options = {"multithread": True,
                        "warpOptions": ["NUM_THREADS=%i"%mp.cpu_count()]}
    # Resample with GDAL warp
    ds = gdal.Warp(str(out_file),
                   src,
                   format=out_format,
                   dstSRS=proj,
                   xRes=gt[1],
                   yRes=gt[5],
                   outputBounds=extent,
                   resampleAlg=resample_alg,
                   **warp_options)

    return ds


def merge_raster_layers(input_list, output_filename, separate=False):
    merge_list = []
    for input_file in input_list:
        bands = raster_info(input_file)[-1]
        # GDAL Build VRT cannot stack multiple multi-band images, so they have to be split into
        # multiple singe-band images first.
        if bands > 1:
            for band in range(1, bands+1):
                temp_filename = tempfile.mkstemp(suffix="_"+str(band)+".vrt")[1]
                gdal.BuildVRT(temp_filename, [input_file], bandList=[band])
                merge_list.append(temp_filename)
        else:
            merge_list.append(input_file)
    fp = gdal.BuildVRT(output_filename, merge_list, separate=separate)
    return fp


def get_subset(roi_shape, raster_proj_wkt, raster_geo_transform):

    # Find extent of ROI in roiShape projection
    roi = ogr.Open(roi_shape)
    roi_layer = roi.GetLayer()
    roi_extent = roi_layer.GetExtent()

    # Convert the extent to raster projection
    roi_proj = roi_layer.GetSpatialRef()
    raster_proj = osr.SpatialReference()
    raster_proj.ImportFromWkt(raster_proj_wkt)
    transform = osr.CoordinateTransformation(roi_proj, raster_proj)
    point_UL = ogr.CreateGeometryFromWkt("POINT (" +
                                         str(min(roi_extent[0], roi_extent[1])) + " " +
                                         str(max(roi_extent[2], roi_extent[3])) + ")")
    point_UL.Transform(transform)
    point_UL = point_UL.GetPoint()
    point_LR = ogr.CreateGeometryFromWkt("POINT (" +
                                         str(max(roi_extent[0], roi_extent[1])) + " " +
                                         str(min(roi_extent[2], roi_extent[3])) + ")")
    point_LR.Transform(transform)
    point_LR = point_LR.GetPoint()

    # Get pixel location of this extent
    ulX = raster_geo_transform[0]
    ulY = raster_geo_transform[3]
    pixel_size = raster_geo_transform[1]
    pixel_UL = [max(int(math.floor((ulY - point_UL[1]) / pixel_size)), 0),
                max(int(math.floor((point_UL[0] - ulX) / pixel_size)), 0)]
    pixel_LR = [int(round((ulY - point_LR[1]) / pixel_size)),
                int(round((point_LR[0] - ulX) / pixel_size))]

    # Get projected extent
    point_proj_UL = (ulX + pixel_UL[1]*pixel_size, ulY - pixel_UL[0]*pixel_size)
    point_proj_LR = (ulX + pixel_LR[1]*pixel_size, ulY - pixel_LR[0]*pixel_size)

    # Create a subset from the extent
    subset_proj = [point_proj_UL, point_proj_LR]
    subset_pix = [pixel_UL, pixel_LR]

    return subset_pix, subset_proj


def read_subset(source, subset_pix):
    if type(source) is np.ndarray:
        data = source[subset_pix[0][0]:subset_pix[1][0], subset_pix[0][1]:subset_pix[1][1]]
    elif type(source) == int or type(source) == float:
        data = np.zeros((subset_pix[1][0] - subset_pix[0][0],
                         subset_pix[1][1] - subset_pix[0][1])) + source
    # Otherwise it should be a file path
    else:
        fp = gdal.Open(source)
        data = fp.GetRasterBand(1).ReadAsArray(subset_pix[0][1],
                                               subset_pix[0][0],
                                               subset_pix[1][1] - subset_pix[0][1],
                                               subset_pix[1][0] - subset_pix[0][0])
        del fp
    return data


# Save pyTSEB input dataset to an NetCDF file
def save_dataset(dataset, gt, proj, output_filename, roi_vector=None, attrs={},
                 compression={'zlib': True, 'complevel': 6}):

    # Get the raster subset extent
    if roi_vector is not None:
        subset_pix, subset_proj = get_subset(roi_vector, proj, gt)
        # NetCDF and GDAL geocoding are off by half a pixel so need to take this into account.
        pixel_size = gt[1]
        subset_proj = [[subset_proj[0][0] + 0.5*pixel_size, subset_proj[0][1] - 0.5*pixel_size],
                       [subset_proj[1][0] + 0.5*pixel_size, subset_proj[1][1] - 0.5*pixel_size]]
    else:
        shape = dataset["LAI"].shape
        subset_pix = [[0, 0], [shape[0], shape[1]]]
        subset_proj = [[gt[0] + gt[1]*0.5, gt[3] + gt[5]*0.5],
                       [gt[0] + gt[1]*(shape[1]+0.5), gt[3]+gt[5]*(shape[0]+0.5)]]

    # Create xarray DataSet
    x = np.linspace(subset_proj[0][1], subset_proj[1][1], subset_pix[1][0] - subset_pix[0][0],
                    endpoint=False)
    y = np.linspace(subset_proj[0][0], subset_proj[1][0], subset_pix[1][1] - subset_pix[0][1],
                    endpoint=False)
    ds = xarray.Dataset({}, coords={'x': (['x'], x),
                                    'y': (['y'], y),
                                    'crs': (['crs'], [])})
    ds.crs.attrs['spatial_ref'] = proj

    # Save the data in the DataSet
    encoding = {}
    for name in dataset:
        data = read_subset(dataset[name], subset_pix)
        ds = ds.assign(temporary=xarray.DataArray(data, coords=[ds.coords['x'], ds.coords['y']],
                                                  dims=('x', 'y')))
        ds["temporary"].attrs['grid_mapping'] = 'crs'
        ds = ds.rename({'temporary': name})
        encoding[name] = compression

    ds.attrs = attrs

    # Save dataset to file
    ds.to_netcdf(output_filename, encoding=encoding)


def prj_to_src(prj):
    src = osr.SpatialReference()
    src.ImportFromWkt(prj)
    return src


def prj_to_epsg(prj):
    src = osr.SpatialReference()
    src.ImportFromWkt(prj)
    epsg = int(src.GetAttrValue("AUTHORITY", 1))
    return epsg


def get_map_coordinates(row, col, geoTransform):
    X = geoTransform[0]+geoTransform[1]*col+geoTransform[2]*row
    Y = geoTransform[3]+geoTransform[4]*col+geoTransform[5]*row
    return X, Y


def get_pixel_coordinates(X, Y, geoTransform):
    row = (Y - geoTransform[3]) / geoTransform[5]
    col = (X - geoTransform[0]) / geoTransform[1]
    return int(row), int(col)


def convert_coordinate(input_coordinate, input_src, output_src=None):
    ''' Coordinate conversion between two coordinate systems

    Parameters
    ----------
    input_coordinate : tuple
        input coordinate (x,y)
    input_src : proj
        input project coordinates
    outputEPSG : proj
        output project coordinates

    Returns
    -------
    X_out : float
        output X coordinate
    Y_out : float
        output X coordinate
    '''

    if not output_src:
        output_crs = Proj('epsg:4326')
    else:
        output_crs = CRS(output_src)
    
    input_crs = CRS(input_src)
    transformer = Transformer.from_crs(input_crs, output_crs, always_xy=True)
    x, y = transformer.transform(*input_coordinate)
    # print point in EPSG 4326
    return x, y


def update_nan(file, src_nodata_value=-32768, dst_nodata_value=np.nan):
    fid = gdal.Open(file, gdal.GA_Update)
    bands = fid.RasterCount
    for band in range(bands):
        band_ds = fid.GetRasterBand(band + 1)
        data = band_ds.ReadAsArray()
        data[data == src_nodata_value] = dst_nodata_value
        band_ds.WriteArray(data)
        band_ds.SetNoDataValue(dst_nodata_value)
        band_ds.FlushCache()

    del fid

def convert_to_cog(input_file, output_file=None, no_data_value=np.nan):
    if not output_file:
        source_file = input_file.replace(input_file.parent / "MEM")
        output_file = input_file
    else:
        source_file = input_file
    ds = gdal.Open(str(source_file), gdal.GA_ReadOnly)
    driver_opt = ['COMPRESS=DEFLATE',
                  'PREDICTOR=YES',
                  'BIGTIFF=IF_SAFER',
                  'NUM_THREADS=ALL_CPUS']



    out_ds = gdal.Translate(str(output_file),
                            ds,
                            format="COG",
                            creationOptions=driver_opt,
                            noData=no_data_value,
                            stats=True)
    del ds
    source_file.unlink()
    return out_ds
