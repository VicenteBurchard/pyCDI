import rasterio
import gdal_utils as gu
import pandas as pd
import numpy as np
import xarray as xr
import geopandas as gpd
from rasterio.mask import mask as riomask
import matplotlib.pyplot as plt
from osgeo import ogr, osr, gdal

def compute_ltm(da, time_steps, gt, proj, outfolder, unit_scaler=1, deficit = False, gee = False):
    '''
    Computes long-term-mean (LTM) for each month over the entire
    available time period. Also computes the LTM of run lengths of
    excess/deficit with respect to LTM for each month

    Args:
        da: xarray dataset with variable of interest
        time_steps: array with datetime objects of LTM period
        unit_scaler: conversion factor if needed to convert to appropriate units
        deficit: True if RL refers to a deficit over LTM averages (i.e. Precip).
                 False if RL refers to an excess over LTM averages (i.e. Temp).

    Returns:

    '''

    # get all months and years within LTM period
    months = np.array(time_steps.month)
    years = np.array(time_steps.year)
    # initialize empty dictionary to store average values and run lenghts (RL) for LTM
    ltm_dict = {}
    RL_ltm_dict = {}

    for month in np.unique(months):
        print(f'Processing Long-Term-Mean (LTM) for Month {month}')
        # for each month get back three months conditions
        if month == 1:
            months_ip = [11, 12, month]
        elif month == 2:
            months_ip = [12, month - 1, month]
        else:
            months_ip = [month - 2, month - 1, month]

        ip_mask = np.logical_or.reduce((months == months_ip[0],
                                        months == months_ip[1],
                                        months == months_ip[2]))

        # filter dataset for 3-month period in question
        ds_ip = da[ip_mask, :, :]
        # get LTM of 3-month period
        mean_data = ds_ip.mean(dim='time', keep_attrs=True)
        # convert to units
        mean_ar = mean_data.values * unit_scaler

        # seems that ERA-5 GEE datasets are flipped
        if gee:
            mean_ar = np.flip(mean_ar, axis=0)

        ltm_dict[month] = mean_ar
        # save to file
        output_mean = outfolder/'mean'
        if not output_mean.exists():
            output_mean.mkdir(parents=True)

        output_file = output_mean / f'Mean_LTM_{month}.tif'
        print(f'\tsaving result to {str(output_file)}')
        gu.save_image(mean_ar, gt, proj, str(output_file))

        # now get LTM Run Lengths (RL)
        # initialize tempoeary empty list
        rl_ip = []
        for year in np.unique(years):
            print(f'\tProcessing deficit run length for year {year}')
            rl_month = np.zeros_like(mean_ar)
            for i in months_ip:
                # get month in question
                month_mask = np.logical_and(months == i, years == year)
                ds_ip = da[month_mask, :, :]
                values = ds_ip.values[0, :, :] * unit_scaler
                # check which values are lower than LTM mean
                new_values = np.zeros_like(mean_ar)
                # all values below ltm assigned 1
                if deficit:
                    condition = values < mean_ar
                else:
                    condition = values > mean_ar

                new_values[condition] = 1
                # add new values to pixels which were previously below LTM
                #rl_month = rl_month + new_values

                rl_accum = np.logical_and(rl_month > 0, condition)
                new_values[rl_accum] = rl_month[rl_accum] + new_values[rl_accum]
                # assign 1 to pixels which were not previously below LTM
                rl_new = np.logical_and(rl_month == 0, condition)
                new_values[rl_new] = 1
                rl_month = new_values

            # store in list
            rl_ip.append(rl_month)
        # get mean RL for all years during IP
        rl_mean = np.nanmean(rl_ip, axis=0)
        # store LTM mean
        RL_ltm_dict[month] = rl_mean

        # save to file
        output_rl = outfolder / 'RL'
        if not output_rl.exists():
            output_rl.mkdir(parents=True)
        output_file = output_rl / f'RL_LTM_{month}.tif'
        print(f'\tsaving result to {str(output_file)}\n')
        gu.save_image(rl_mean, gt, proj, str(output_file))

    return ltm_dict, RL_ltm_dict


def compute_ltm_xr(da, time_steps, gt, proj, outfolder, unit_scaler=1, deficit = False, gee = False):
    '''
    Computes long-term-mean (LTM) for each month over the entire
    available time period. Also computes the LTM of run lengths of
    excess/deficit with respect to LTM for each month

    Args:
        da: xarray dataset with variable of interest
        time_steps: array with datetime objects of LTM period
        unit_scaler: conversion factor if needed to convert to appropriate units
        deficit: True if RL refers to a deficit over LTM averages (i.e. Precip).
                 False if RL refers to an excess over LTM averages (i.e. Temp).

    Returns:

    '''

    # get all months and years within LTM period
    months = np.array(time_steps.month)
    years = np.array(time_steps.year)

    # convert dictionary keys of timestamps into array
    x = np.arange(0, da.shape[1])
    y = np.arange(0, da.shape[2])
    coords = {'time': months, 'x': x, 'y': y}
    dims = ('time', 'x', 'y')

    # initialize empty data arays to store average values
    # and run lengths (RL) for LTM
    ltm_ds = xr.DataArray(np.empty((months.size, len(x), len(y))), coords=coords, dims=dims)
    rl_ltm_ds = xr.DataArray(np.empty((months.size, len(x), len(y))), coords=coords, dims=dims)


    for month in np.unique(months):
        print(f'Processing Long-Term-Mean (LTM) for Month {month}')
        # for each month get back three months conditions
        if month == 1:
            months_ip = [11, 12, month]
        elif month == 2:
            months_ip = [12, month - 1, month]
        else:
            months_ip = [month - 2, month - 1, month]

        ip_mask = np.logical_or.reduce((months == months_ip[0],
                                        months == months_ip[1],
                                        months == months_ip[2]))

        # filter dataset for 3-month period in question
        ds_ip = da[ip_mask, :, :]
        # get LTM of 3-month period
        mean_data = ds_ip.mean(dim='time', keep_attrs=True)
        # convert to units
        mean_ar = mean_data.values * unit_scaler

        # seems that ERA-5 GEE datasets are flipped
        if gee:
            mean_ar = np.flip(mean_ar, axis=0)

        #ltm_dict[month] = mean_ar
        ltm_ds.loc[month, :, :] = mean_ar

        # save to file
        output_mean = outfolder/'mean'
        if not output_mean.exists():
            output_mean.mkdir(parents=True)

        output_file = output_mean / f'Mean_LTM_{month}.tif'
        print(f'\tsaving result to {str(output_file)}')
        gu.save_image(mean_ar, gt, proj, str(output_file))

        # now get LTM Run Lengths (RL)
        # initialize tempoeary empty list
        rl_ip = []
        for year in np.unique(years):
            print(f'\tProcessing deficit run length for year {year}')
            rl_month = np.zeros_like(mean_ar)
            for i in months_ip:
                # get month in question
                month_mask = np.logical_and(months == i, years == year)
                ds_ip = da[month_mask, :, :]
                values = ds_ip.values[0, :, :] * unit_scaler
                # check which values are lower than LTM mean
                new_values = np.zeros_like(mean_ar)
                # all values below ltm assigned 1
                if deficit:
                    condition = values < mean_ar
                else:
                    condition = values > mean_ar

                new_values[condition] = 1
                # add new values to pixels which were previously below LTM
                #rl_month = rl_month + new_values

                rl_accum = np.logical_and(rl_month > 0, condition)
                new_values[rl_accum] = rl_month[rl_accum] + new_values[rl_accum]
                # assign 1 to pixels which were not previously below LTM
                rl_new = np.logical_and(rl_month == 0, condition)
                new_values[rl_new] = 1
                rl_month = new_values


            # store in list
            rl_ip.append(rl_month)
        # get mean RL for all years during IP
        rl_mean = np.nanmean(rl_ip, axis=0)
        # store LTM mean
        #RL_ltm_dict[month] = rl_mean
        rl_ltm_ds.loc[month, :, :] = rl_mean

        # save to file
        output_rl = outfolder / 'RL'
        if not output_rl.exists():
            output_rl.mkdir(parents=True)
        output_file = output_rl / f'RL_LTM_{month}.tif'
        print(f'\tsaving result to {str(output_file)}\n')
        gu.save_image(rl_mean, gt, proj, str(output_file))

    return ltm_ds, rl_ltm_ds



def compute_ip(da, time_steps, gt, proj, outfolder, unit_scaler=1, deficit = False, gee = False):
    '''
    Computes the actual mean and run length of excess/deficit
    against long term mean (LTM) for the interest period (IP)
    (i.e. 3-month period).

    Args:
        da: xarray dataset with variable of interest
        time_steps: array with datetime objects of LTM period
        unit_scaler: conversion factor if needed to convert to appropriate units
        deficit: True = if deficit promotes drought and False = if excess promotes drought
        gee: True if using google earth engine STAC dataset


    Returns:

    '''

    # get all months and years within LTM period
    months = np.array(time_steps.month)
    years = np.array(time_steps.year)

    # initialize empty dictionary to store average values
    # and run lengths (RL) for LTM
    ip_dict = {}
    RL_ip_dict = {}

    for year in np.unique(years):
        print(f'Processing IP periods for year {year}')
        for month in np.unique(months):
            print(f'\tmonth {month}')
            year_month_dt = pd.to_datetime(f'{year}-{month}', format='%Y-%m')
            ts_mask = np.logical_and(months == month, years == year)
            time_step = time_steps[ts_mask][0]

            # special condition for first month (cannot compute 3-month period)
            if time_step == time_steps[0]:
                time_step_ip = time_steps == time_step
                months_ip = [time_step]
            # special condition for second month (cannot compute 3-month period)
            elif time_step == time_steps[1]:
                time_step_minus1 = time_step - pd.DateOffset(months=1)
                time_step_ip = np.logical_or(time_steps == time_step,
                                             time_steps == time_step_minus1)
                months_ip = [time_step_minus1, time_step]

            else:
                time_step_minus1 = time_step - pd.DateOffset(months=1)
                time_step_minus2 = time_step - pd.DateOffset(months=2)
                time_step_ip = np.logical_or.reduce((time_steps == time_step,
                                                     time_steps == time_step_minus1,
                                                     time_steps == time_step_minus2))
                months_ip = [time_step_minus2, time_step_minus1, time_step]

            ds_ip = da[time_step_ip, :, :]

            mean_data = ds_ip.mean(dim='time', keep_attrs=True)

            # convert to units
            mean_ar = mean_data.values * unit_scaler

            # seems that ERA-5 GEE datasets are flipped over x axis (vertically)
            if gee:
                mean_ar = np.flip(mean_ar, axis=0)

            ip_dict[year_month_dt] = mean_ar

            # save to file
            output_mean = outfolder / 'mean'
            if not output_mean.exists():
                output_mean.mkdir(parents=True)

            output_file = output_mean / f'Mean_IP_{year}_{month}.tif'
            print(f'\tsaving result to {str(output_file)}')
            gu.save_image(mean_ar, gt, proj, str(output_file))

            # now calculate RL
            rl_month = np.zeros_like(mean_ar)
            for m in months_ip:
                month_mask = time_steps == m
                ds_ip_m1 = da[month_mask, :, :]

                values = ds_ip_m1.values[0, :, :] * unit_scaler

                # check which values are lower than LTM mean
                new_values = np.zeros_like(mean_ar)
                # all values below ltm assigned 1
                if deficit:
                    condition = values < mean_ar
                else:
                    condition = values > mean_ar

                # all values below ltm assigned 1
                new_values[condition] = 1
                # addition of new values to pixels which were previously below LTM (consecutive)
                rl_accum = np.logical_and(rl_month > 0, condition)
                new_values[rl_accum] = rl_month[rl_accum] + new_values[rl_accum]
                # assign 1 to pixels which were not previously below LTM
                rl_new = np.logical_and(rl_month == 0, condition)
                new_values[rl_new] = 1
                rl_month = new_values
            # store results in dict
            RL_ip_dict[year_month_dt] = rl_month

            # save to file
            output_rl = outfolder / 'RL'
            if not output_rl.exists():
                output_rl.mkdir(parents=True)
            output_file = output_rl / f'RL_IP_{year}_{month}.tif'
            print(f'\tsaving result to {str(output_file)}\n')
            gu.save_image(rl_month, gt, proj, str(output_file))

    return ip_dict, RL_ip_dict

def compute_ip_xr(da, time_steps, gt, proj, outfolder, unit_scaler=1, deficit = False, gee = False):
    '''
    Computes the actual mean and run length of excess/deficit
    against long term mean (LTM) for the interest period (IP)
    (i.e. 3-month period).

    Args:
        da: xarray dataset with variable of interest
        time_steps: array with datetime objects of LTM period
        ltm_dict: dictionary with stored LTM monthly values
        unit_scaler: conversion factor if needed to convert to appropriate units


    Returns:

    '''
    # convert dictionary keys of timestamps into array
    x = np.arange(0, da.shape[1])
    y = np.arange(0, da.shape[2])
    coords = {'time': time_steps, 'x': x, 'y': y}
    dims = ('time', 'x', 'y')

    # initialize empty data arays to store average values
    # and run lengths (RL) for LTM
    ip_ds = xr.DataArray(np.empty((time_steps.size, len(x), len(y))), coords=coords, dims=dims)
    rl_ip_ds = xr.DataArray(np.empty((time_steps.size, len(x), len(y))), coords=coords, dims=dims)

    # get all months and years within LTM period
    months = np.array(time_steps.month)
    years = np.array(time_steps.year)

    for year in np.unique(years):
        print(f'Processing IP periods for year {year}')
        for month in np.unique(months):
            print(f'\tmonth {month}')
            year_month_dt = pd.to_datetime(f'{year}-{month}', format='%Y-%m')
            ts_mask = np.logical_and(months == month, years == year)
            time_step = time_steps[ts_mask][0]

            # special condition for first month (cannot compute 3-month period)
            if time_step == time_steps[0]:
                time_step_ip = time_steps == time_step
                months_ip = [time_step]
            # special condition for second month (cannot compute 3-month period)
            elif time_step == time_steps[1]:
                time_step_minus1 = time_step - pd.DateOffset(months=1)
                time_step_ip = np.logical_or(time_steps == time_step,
                                             time_steps == time_step_minus1)
                months_ip = [time_step_minus1, time_step]

            else:
                time_step_minus1 = time_step - pd.DateOffset(months=1)
                time_step_minus2 = time_step - pd.DateOffset(months=2)
                time_step_ip = np.logical_or.reduce((time_steps == time_step,
                                                     time_steps == time_step_minus1,
                                                     time_steps == time_step_minus2))
                months_ip = [time_step_minus2, time_step_minus1, time_step]

            ds_ip = da[time_step_ip, :, :]

            mean_data = ds_ip.mean(dim='time', keep_attrs=True)

            # convert to units
            mean_ar = mean_data.values * unit_scaler

            # seems that ERA-5 GEE datasets are flipped over x axis (vertically)
            if gee:
                mean_ar = np.flip(mean_ar, axis=0)

            ip_ds.loc[year_month_dt, :, :] = mean_ar

            # save to file
            output_mean = outfolder / 'mean'
            if not output_mean.exists():
                output_mean.mkdir(parents=True)

            output_file = output_mean / f'Mean_IP_{year}_{month}.tif'
            print(f'\tsaving result to {str(output_file)}')
            gu.save_image(mean_ar, gt, proj, str(output_file))

            # now calculate RL
            rl_month = np.zeros_like(mean_ar)
            for m in months_ip:
                month_mask = time_steps == m
                ds_ip_m1 = da[month_mask, :, :]

                values = ds_ip_m1.values[0, :, :] * unit_scaler

                # check which values are lower than LTM mean
                new_values = np.zeros_like(mean_ar)
                # all values below ltm assigned 1
                if deficit:
                    condition = values < mean_ar
                else:
                    condition = values > mean_ar

                # all values below ltm assigned 1
                new_values[condition] = 1
                # addition of new values to pixels which were previously below LTM (consecutive)
                rl_accum = np.logical_and(rl_month > 0, condition)
                new_values[rl_accum] = rl_month[rl_accum] + new_values[rl_accum]
                # assign 1 to pixels which were not previously below LTM
                rl_new = np.logical_and(rl_month == 0, condition)
                new_values[rl_new] = 1
                rl_month = new_values
            # store results in dict
            rl_ip_ds.loc[year_month_dt, :, :] = rl_month

            # save to file
            output_rl = outfolder / 'RL'
            if not output_rl.exists():
                output_rl.mkdir(parents=True)
            output_file = output_rl / f'RL_IP_{year}_{month}.tif'
            print(f'\tsaving result to {str(output_file)}\n')
            gu.save_image(rl_month, gt, proj, str(output_file))

    return ip_ds, rl_ip_ds


def calc_pdi(ip_dict, RL_ip_dict, ltm_dict, RL_ltm_dict, gt, proj, outfolder, RL_max = 3):
    '''

    Args:
        ip_dict: actual mean values for IPs (interest periods)
        RL_ip_dict: actual deficit run lengths for IPs (interest periods)
        ltm_dict: LTM values for study area
        RL_ltm_dict: LTM deficit values for study area

    Returns: dictionary with monthly PDI results

    '''
    # convert dictionary keys of timestamps into array
    time_steps = pd.to_datetime(np.array([timestamp for timestamp in ip_dict.keys()]))
    months = np.array(time_steps.month)
    years = np.array(time_steps.year)

    pdi_dict = {}

    for year in np.unique(years):
        print(year)
        for month in np.unique(months):
            print(f'\tMonth: {month}')
            date_id = pd.to_datetime(f'{year}-{month}', format='%Y-%m')
            P_ip = ip_dict[date_id]
            # normalize IP Precip and deficit run lengths
            P_ip_star = P_ip + 1
            RL_ip = RL_ip_dict[date_id]
            RL_ip_star = (RL_max + 1) - RL_ip

            # normalize LTM Precip and deficit run lengths
            P_ltm = ltm_dict[month]
            P_ltm_star = P_ltm + 1
            # run lengths
            RL_ltm = RL_ltm_dict[month]
            RL_ltm_star = (RL_max + 1) - RL_ltm
            # compute PDI (precipitation drought index)
            pdi = (P_ip_star / P_ltm_star) * np.sqrt(RL_ip_star / RL_ltm_star)
            pdi_dict[date_id] = pdi

            #save to file
            output_pdi = outfolder / 'PDI'
            if not output_pdi.exists():
                output_pdi.mkdir(parents=True)
            output_file = output_pdi / f'PDI_IP_{year}_{month}.tif'

            print(f'\tsaving PDI result for {year}-{month} to {str(output_file)}\n')
            gu.save_image(pdi, gt, proj, str(output_file))


    return pdi_dict

def calc_pdi_xr(ip_ds, RL_ip_ds, ltm_ds, RL_ltm_ds, gt, proj, outfolder):
    '''

    Args:
        ip_dict: actual mean values for IPs (interest periods)
        RL_ip_dict: actual deficit run lengths for IPs (interest periods)
        ltm_dict: LTM values for study area
        RL_ltm_dict: LTM deficit values for study area

    Returns: dictionary with monthly PDI results

    '''
    # convert dictionary keys of timestamps into array
    time_steps = pd.to_datetime(ip_ds['time'].values)
    months_ltm = ltm_ds['time'].values.astype('int')
    x = np.arange(0, ip_ds.values.shape[1])
    y = np.arange(0, ip_ds.values.shape[2])
    coords = {'time':time_steps, 'x':x, 'y':y}
    dims = ('time', 'x', 'y')
    out_ds = xr.DataArray(np.empty((time_steps.size, len(x), len(y))), coords=coords, dims=dims)

    # get max RL in 2D
    RL_ip_ar = RL_ip_ds.values
    RL_max = np.nanmax(RL_ip_ar, axis=0)

    for ts in time_steps:
        print(ts)
        month = ts.month
        year = ts.year

        P_ip = ip_ds[time_steps == ts, :, :].values
        RL_ip = RL_ip_ds[time_steps == ts, :, :].values

        # normalize IP Precip and deficit run lengths
        P_ip_star = P_ip + 1
        RL_ip_star = (RL_max + 1) - RL_ip

        # normalize LTM Precip and deficit run lengths
        P_ltm = ltm_ds[months_ltm == month, :, :].values
        P_ltm_star = P_ltm + 1
        # run lengths
        RL_ltm = RL_ltm_ds[months_ltm == month, :, :].values
        RL_ltm_star = (RL_max + 1) - RL_ltm

        # compute PDI (precipitation drought index)
        pdi = (P_ip_star / P_ltm_star) * np.sqrt(RL_ip_star / RL_ltm_star)
        out_ds.loc[ts, :, :] = pdi[0, :, :]

        # save to file
        output_pdi = outfolder / 'PDI'
        if not output_pdi.exists():
            output_pdi.mkdir(parents=True)
        output_file = output_pdi / f'PDI_IP_{year}_{month}.tif'

        print(f'\tsaving PDI result for {year}-{month} to {str(output_file)}\n')
        gu.save_image(pdi[0, :, :], gt, proj, str(output_file))

    return out_ds

def calc_tdi(ip_dict, RL_ip_dict, ltm_dict, RL_ltm_dict, Ta_max, gt, proj, outfolder, RL_max = 3):
    '''

    Args:
        ip_dict: actual mean values for IPs (interest periods)
        RL_ip_dict: actual deficit run lengths for IPs (interest periods)
        ltm_dict: LTM values for study area
        RL_ltm_dict: LTM deficit values for study area

    Returns: dictionary with monthly TDI results

    '''
    # convert dictionary keys of timestamps into array
    time_steps = pd.to_datetime(np.array([timestamp for timestamp in ip_dict.keys()]))
    months = np.array(time_steps.month)
    years = np.array(time_steps.year)

    tdi_dict = {}

    for year in np.unique(years):
        print(year)
        for month in np.unique(months):
            print(f'\tMonth: {month}')
            date_id = pd.to_datetime(f'{year}-{month}', format='%Y-%m')
            T_ip = ip_dict[date_id]
            # normalize IP Precip and deficit run lengths
            T_ip_star = T_ip - (Ta_max + 1)

            RL_ip = RL_ip_dict[date_id]
            RL_ip_star = (RL_max + 1) - RL_ip

            # normalize LTM Precip and deficit run lengths
            T_ltm = ltm_dict[month]
            T_ltm_star = T_ltm - (Ta_max + 1)

            # run lengths
            RL_ltm = RL_ltm_dict[month]
            RL_ltm_star = (RL_max + 1) - RL_ltm
            # compute PDI (precipitation drought index)
            tdi = (T_ip_star / T_ltm_star) * np.sqrt(RL_ip_star / RL_ltm_star)
            tdi_dict[date_id] = tdi

            # save to file
            output_pdi = outfolder / 'TDI'
            if not output_pdi.exists():
                output_pdi.mkdir(parents=True)
            output_file = output_pdi / f'TDI_IP_{year}_{month}.tif'

            print(f'\tsaving TDI result for {year}-{month} to {str(output_file)}\n')
            gu.save_image(tdi, gt, proj, str(output_file))

    return tdi_dict


def calc_tdi_xr(ip_ds, RL_ip_ds, ltm_ds, RL_ltm_ds, Ta_max, gt, proj, outfolder):
    '''

    Args:
        ip_dict: actual mean values for IPs (interest periods)
        RL_ip_dict: actual deficit run lengths for IPs (interest periods)
        ltm_dict: LTM values for study area
        RL_ltm_dict: LTM deficit values for study area

    Returns: dictionary with monthly TDI results

    '''
    # convert dictionary keys of timestamps into array
    time_steps = pd.to_datetime(ip_ds['time'].values)
    months_ltm = ltm_ds['time'].values.astype('int')

    x = np.arange(0, ip_ds.values.shape[1])
    y = np.arange(0, ip_ds.values.shape[2])
    coords = {'time': time_steps, 'x': x, 'y': y}
    dims = ('time', 'x', 'y')
    out_ds = xr.DataArray(np.empty((time_steps.size, len(x), len(y))), coords=coords, dims=dims)

    # get max RL in 2D
    RL_ip_ar = RL_ip_ds.values
    #RL_max = np.nanmax(RL_ip_ar, axis=0)
    RL_max = np.ones_like(RL_ip_ar) * 3
    for ts in time_steps:
        month = ts.month
        year = ts.year

        T_ip = ip_ds[time_steps == ts, :, :].values
        RL_ip = RL_ip_ds[time_steps == ts, :, :].values

        # normalize IP Precip and deficit run lengths
        T_ip_star = (Ta_max + 1) - T_ip
        RL_ip_star = (RL_max + 1) - RL_ip

        # normalize LTM Precip and deficit run lengths
        T_ltm = ltm_ds[months_ltm == month, :, :].values
        T_ltm_star = (Ta_max + 1) - T_ltm

        # run lengths
        RL_ltm = RL_ltm_ds[months_ltm == month, :, :].values
        RL_ltm_star = (RL_max + 1) - RL_ltm

        # compute PDI (precipitation drought index)
        tdi = (T_ip_star / T_ltm_star) * np.sqrt(RL_ip_star / RL_ltm_star)
        out_ds.loc[ts, :, :] = tdi[0, :, :]

        # save to file
        output_tdi = outfolder / 'TDI'
        if not output_tdi.exists():
            output_tdi.mkdir(parents=True)
        output_file = output_tdi / f'TDI_IP_{year}_{month}.tif'

        print(f'\tsaving TDI result for {year}-{month} to {str(output_file)}\n')
        gu.save_image(tdi[0, :, :], gt, proj, str(output_file))
    return out_ds



def calc_vdi(ip_dict, RL_ip_dict, ltm_dict, RL_ltm_dict, ndvi_min, gt, proj, outfolder, RL_max = 3):
    '''

    Args:
        ip_dict: actual mean values for IPs (interest periods)
        RL_ip_dict: actual deficit run lengths for IPs (interest periods)
        ltm_dict: LTM values for study area
        RL_ltm_dict: LTM deficit values for study area

    Returns: dictionary with monthly TDI results

    '''
    # convert dictionary keys of timestamps into array
    time_steps = pd.to_datetime(np.array([timestamp for timestamp in ip_dict.keys()]))
    months = np.array(time_steps.month)
    years = np.array(time_steps.year)

    vdi_dict = {}

    for year in np.unique(years):
        print(year)
        for month in np.unique(months):
            print(f'\tMonth: {month}')
            date_id = pd.to_datetime(f'{year}-{month}', format='%Y-%m')
            ndvi_ip = ip_dict[date_id]
            # normalize IP Precip and deficit run lengths
            ndvi_ip_star = ndvi_ip - ndvi_min
            RL_ip = RL_ip_dict[date_id]
            RL_ip_star = (RL_max + 1) - RL_ip

            # normalize LTM ndvi and deficit run lengths
            ndvi_ltm = ltm_dict[month]
            ndvi_ltm_star = ndvi_ltm - ndvi_min
            # run lengths
            RL_ltm = RL_ltm_dict[month]
            RL_ltm_star = (RL_max + 1) - RL_ltm
            # compute PDI (precipitation drought index)
            vdi = (ndvi_ip_star / ndvi_ltm_star) * np.sqrt(RL_ip_star / RL_ltm_star)
            vdi_dict[date_id] = vdi

            # save to file
            output_pdi = outfolder / 'VDI'
            if not output_pdi.exists():
                output_pdi.mkdir(parents=True)
            output_file = output_pdi / f'VDI_IP_{year}_{month}.tif'

            print(f'\tsaving VDI result for {year}-{month} to {str(output_file)}\n')
            gu.save_image(vdi, gt, proj, str(output_file))

    return vdi_dict
def calc_vdi_xr(ip_ds, RL_ip_ds, ltm_ds, RL_ltm_ds, ndvi_min, gt, proj, outfolder):
    '''

    Args:
        ip_dict: actual mean values for IPs (interest periods)
        RL_ip_dict: actual deficit run lengths for IPs (interest periods)
        ltm_dict: LTM values for study area
        RL_ltm_dict: LTM deficit values for study area

    Returns: dictionary with monthly TDI results

    '''
    # convert dictionary keys of timestamps into array
    time_steps = pd.to_datetime(ip_ds['time'].values)
    months_ltm = ltm_ds['time'].values.astype('int')


    x = np.arange(0, ip_ds.values.shape[1])
    y = np.arange(0, ip_ds.values.shape[2])
    coords = {'time': time_steps, 'x': x, 'y': y}
    dims = ('time', 'x', 'y')
    out_ds = xr.DataArray(np.empty((time_steps.size, len(x), len(y))), coords=coords, dims=dims)

    # get max RL in 2D
    RL_ip_ar = RL_ip_ds.values
    RL_max = np.nanmax(RL_ip_ar, axis=0)

    for ts in time_steps:
        month = ts.month
        year = ts.year

        ndvi_ip = ip_ds[time_steps == ts, :, :].values
        RL_ip = RL_ip_ds[time_steps == ts, :, :].values

        # normalize IP Precip and deficit run lengths
        ndvi_ip_star = ndvi_ip - (ndvi_min - 1)
        RL_ip_star = (RL_max + 1) - RL_ip

        # normalize LTM Precip and deficit run lengths
        ndvi_ltm = ltm_ds[months_ltm == month, :, :].values
        ndvi_ltm_star = ndvi_ltm - (ndvi_min - 1)

        # run lengths
        RL_ltm = RL_ltm_ds[months_ltm == month, :, :].values
        RL_ltm_star = (RL_max + 1) - RL_ltm

        # compute PDI (precipitation drought index)
        vdi = (ndvi_ip_star / ndvi_ltm_star) * np.sqrt(RL_ip_star / RL_ltm_star)
        out_ds.loc[ts, :, :] = vdi[0, :, :]

        # save to file
        output_vdi = outfolder / 'VDI'
        if not output_vdi.exists():
            output_vdi.mkdir(parents=True)
        output_file = output_vdi / f'VDI_IP_{year}_{month}.tif'

        print(f'\tsaving VDI result for {year}-{month} to {str(output_file)}\n')
        gu.save_image(vdi[0, :, :], gt, proj, str(output_file))
    return out_ds

def calc_ndvi(nir, red):
    num = nir - red
    denum = nir + red
    ndvi = num/denum
    return ndvi

def compute_ndvi_monthly(da, start_date, end_date, gt, proj, outfolder):
    '''
    function to calculate the monthly average NDVI from 8-day product
    Args:
        da:

    Returns:

    '''

    start_year = pd.to_datetime(start_date).year
    end_year = pd.to_datetime(end_date).year
    years = pd.to_datetime(da['start_datetime'].values).year
    years_mask = np.logical_and(years >= start_year, years <= end_year)
    da = da[years_mask, :, :, :]

    years = years[years_mask]
    time_steps = pd.to_datetime(da['time'])
    months_start = time_steps.month
    months_end = pd.to_datetime(da['end_datetime'].values).month


    times = []
    out_ds = xr.Dataset()
    for year in np.unique(years):
        for month in np.unique(months_start):
            print(f'Processing monthly mean NDVI for {year}-{month}')
            date_dt = pd.to_datetime(f'{year}-{month}', format='%Y-%m')

            #if date_dt < pd.to_datetime(start_date) or date_dt > pd.to_datetime(end_date):
            #    continue

            month_mask = np.logical_and.reduce((months_start == month,
                                                months_end == month,
                                                years == year))

            ds_month = da[month_mask, :, :, :]
            ndvi_list = []
            for i in range(ds_month.shape[0]):
                ds_day = ds_month[i, :, :, :]
                band1 = ds_day.sel(band='sur_refl_b01')
                band2 = ds_day.sel(band='sur_refl_b02')
                qc = ds_day.sel(band='sur_refl_qc_250m')
                ndvi = calc_ndvi(band2.values, band1.values)
                # f'{4096:016b}' = '0001000000000000' = 'best quality data'
                vi_mask = qc.values == 4096
                # only use best quality data
                ndvi[~vi_mask] = None
                ndvi_list.append(ndvi)

            ndvi_mean = np.nanmean(ndvi_list, axis=0)
            times.append(date_dt)
            #out_ds[date_dt] = ndvi_mean
            out_ds[date_dt] = (['y', 'x'], ndvi_mean)

            # save to file
            output_ndvi = outfolder / 'monthly_rasters'
            if not output_ndvi.exists():
                output_ndvi.mkdir(parents=True)
            output_file = output_ndvi / f'ndvi_monthly_mean_{year}_{month}.tif'

            print(f'\tsaving ndvi monthly mean for {year}-{month} to {str(output_file)}\n')
            gu.save_image(ndvi_mean, gt, proj, str(output_file))

    # Assign coordinates
    out_ds.coords['time'] = np.array(times)
    out_ds.coords['y'] = np.arange(ndvi_mean.shape[1])
    out_ds.coords['x'] = np.arange(ndvi_mean.shape[0])


    return out_ds


def resample_indices(outfolder, roi_shapefile, template_file, start_date, end_date, indices=['PDI', 'TDI', 'VDI']):
    '''

    Args:
        outfolder:
        roi_shapefile:
        template_file:
        indices:

    Returns:

    '''

    # open shapefile with geopandas
    clip_gdf = gpd.read_file(str(roi_shapefile))
    # Use the bounds of the GeoDataFrame to get the geometry for clipping
    clip_geometry = clip_gdf.geometry
    times_pdi = []
    times_tdi = []
    times_vdi = []

    pdi_ds = xr.Dataset()
    tdi_ds = xr.Dataset()
    vdi_ds = xr.Dataset()

    for di in indices:
        print(f'resampling {di} images')
        #out_ds[di] = {}
        index_folder = outfolder / di
        img_list = list(index_folder.glob('*.tif'))
        roi_folder = index_folder / 'ROI'
        if not roi_folder.exists():
            roi_folder.mkdir(parents=True)

        for img in img_list:
            filename = img.name

            outname = filename[:-4] + '_roi.tif'
            tempname = filename[:-4] + '_temp.tif'
            outfile = roi_folder / outname
            tempfile = roi_folder / tempname

            month = outname.split('_')[3]
            year = outname.split('_')[2]
            date_dt = pd.to_datetime(f'{year}-{month}', format = '%Y-%m')

            if date_dt < pd.to_datetime(start_date) or date_dt >  pd.to_datetime(end_date):
                continue

            print(date_dt)
            if di == 'PDI':
                times_pdi.append(date_dt)
            elif di == 'TDI':
                times_tdi.append(date_dt)
            else:
                times_vdi.append(date_dt)


            fid = gu.resample_with_gdalwarp(str(img),
                                            str(template_file),
                                            "bilinear",
                                            out_file=str(tempfile),
                                            out_format="GTiff"
                                            )
            del fid
            # open raster with rasterio
            input_raster = rasterio.open(str(tempfile))
            # Clip the raster using rasterio.mask.mask
            clipped_raster, clipped_transform = riomask(dataset=input_raster, shapes=clip_geometry, crop=True,
                                                        indexes=1,
                                                        nodata=None)
            #out_ds[di][date_dt] = clipped_raster
            if di == 'PDI':
                pdi_ds[date_dt] = (['y', 'x'], clipped_raster)
            elif di == 'TDI':
                tdi_ds[date_dt] = (['y', 'x'], clipped_raster)
            else:
                vdi_ds[date_dt] = (['y', 'x'], clipped_raster)



            # Update metadata for the clipped raster
            clipped_meta = input_raster.meta.copy()
            clipped_meta.update({
                "driver": "GTiff",
                "height": clipped_raster.shape[0],
                "width": clipped_raster.shape[1],
                "transform": clipped_transform,
                "count": 1
            })
            print(f'Saving as {str(outfile)}')

            # Write the clipped raster to a new GeoTIFF file
            with rasterio.open(str(outfile), "w", **clipped_meta) as dest:
                dest.write(clipped_raster, 1)

            # Close the datasets
            input_raster.close()
            # delete tempfile
            tempfile.unlink()
    # Assign coordinates
    pdi_ds.coords['time'] = np.array(times_pdi)
    pdi_ds.coords['x'] = np.arange(clipped_raster.shape[1])
    pdi_ds.coords['y'] = np.arange(clipped_raster.shape[0])
    pdi_ds = pdi_ds.sortby('time')

    tdi_ds.coords['time'] = np.array(times_tdi)
    tdi_ds.coords['x'] = np.arange(clipped_raster.shape[1])
    tdi_ds.coords['y'] = np.arange(clipped_raster.shape[0])
    tdi_ds = tdi_ds.sortby('time')

    vdi_ds.coords['time'] = np.array(times_vdi)
    vdi_ds.coords['x'] = np.arange(clipped_raster.shape[1])
    vdi_ds.coords['y'] = np.arange(clipped_raster.shape[0])
    vdi_ds = vdi_ds.sortby('time')

    return pdi_ds, tdi_ds, vdi_ds

def calc_cdi_img(start_date, end_date, gt, proj, outfolder):

    # folder directory with drought indices
    pdi_dir = outfolder / 'PDI'/ 'ROI'
    tdi_dir = outfolder / 'TDI'/ 'ROI'
    vdi_dir = outfolder / 'VDI'/ 'ROI'
    outdict = {}
    time_steps = pd.date_range(start=start_date, end=end_date, freq='MS')

    for time_step in time_steps:
        print(time_step)
        month = time_step.month
        year = time_step.year


        # PDI
        pdi_file = pdi_dir / f'PDI_IP_{year}_{month}_roi.tif'
        pdi_fid = gdal.Open(str(pdi_file))
        pdi_ar = pdi_fid.GetRasterBand(1).ReadAsArray()
        del pdi_fid

        # TDI
        tdi_file = tdi_dir / f'TDI_IP_{year}_{month}_roi.tif'
        tdi_fid = gdal.Open(str(tdi_file))
        tdi_ar = tdi_fid.GetRasterBand(1).ReadAsArray()
        del tdi_fid

        # VDI
        vdi_file = vdi_dir / f'VDI_IP_{year}_{month}_roi.tif'
        vdi_fid = gdal.Open(str(vdi_file))
        vdi_ar = vdi_fid.GetRasterBand(1).ReadAsArray()
        del vdi_fid


        cdi = (0.5*pdi_ar) + (0.25*tdi_ar) + (0.25*vdi_ar)
        cdi[cdi>3] = np.nan
        cdi[cdi<0] = np.nan

        # store result in dict
        outdict[time_step] = cdi

        # save to file
        output_dir = outfolder / 'CDI'
        if not output_dir.exists():
            output_dir.mkdir(parents=True)
        output_file = output_dir / f'CDI_IP_{year}_{month}.tif'

        print(f'\tsaving final CDI image for {year}-{month} to {str(output_file)}\n')
        gu.save_image(cdi, gt, proj, str(output_file))

    return outdict

def calc_cdi(pdi_ds, tdi_ds, vdi_ds, gt, proj, outfolder):
    time_steps = pd.to_datetime(pdi_ds['time'])

    cdi_ds = xr.Dataset()
    for time_step in time_steps:
        month = time_step.month
        year = time_step.year
        date_mask = time_steps == time_step
        pdi = pdi_ds.to_array()
        pdi_values = pdi[date_mask, :, :].values[0,:,:]

        #plt.imshow(pdi_values)

        tdi = tdi_ds.to_array()
        tdi_values = tdi[date_mask, :, :].values[0,:,:]
        plt.imshow(tdi_values)

        vdi = vdi_ds.to_array()
        vdi_values = vdi[date_mask, :, :].values[0,:,:]
        #plt.imshow(vdi_values)


        cdi = (0.5*pdi_values) + (0.25*tdi_values) + (0.25*vdi_values)

        #plt.imshow(cdi)

        cdi_ds[time_step] = (['y', 'x'], cdi)
        # save to file
        output_dir = outfolder / 'CDI'
        if not output_dir.exists():
            output_dir.mkdir(parents=True)
        output_file = output_dir / f'CDI_IP_{year}_{month}.tif'

        print(f'\tsaving final CDI image for {year}-{month} to {str(output_file)}\n')
        gu.save_image(cdi, gt, proj, str(output_file))

    # Assign coordinates
    cdi_ds.coords['time'] = np.array(time_steps)
    cdi_ds.coords['x'] = np.arange(cdi.shape[1])
    cdi_ds.coords['y'] = np.arange(cdi.shape[0])
    cdi_ds = pdi_ds.sortby('time')


    return cdi_ds














