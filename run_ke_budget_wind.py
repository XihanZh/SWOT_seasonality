#!/usr/bin/env python3
import MITgcmutils as mit
import sys, os, glob
from matplotlib import pyplot as plt
import pandas as pd
import pickle
from scipy.interpolate import interp2d
import numpy as np
from scipy.interpolate import griddata
from scipy.optimize import curve_fit
from math import sin, cos, sqrt, atan2, radians
from scipy.ndimage import gaussian_filter
import xarray as xr
from netCDF4 import Dataset
import kebudget as ke
import argparse

def spatialfilter_2D(data,l):
    delta = 2e3
    truncate = 4
    radius_in_pixels = l/delta
    sigma_pixels = radius_in_pixels/truncate
    lowpass = gaussian_filter(data, sigma=(sigma_pixels, sigma_pixels))  
    highpass = data - lowpass
    return lowpass,highpass

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--t1", type=int, required=True, help="start time index (inclusive)")
    p.add_argument("--t2", type=int, required=True, help="end time index (exclusive)")
    p.add_argument("--savedir", type=str, required=True, help="output directory")
    p.add_argument("--prefix", type=str, default="windstress_KE_full_", help="output filename prefix")
    args = p.parse_args()

    t0 = 7073572
    ts = 45
    tl = len(range(args.t1, args.t2))

    path="/resnick/groups/oceanphysics/xihanz/NA_energy"
    os.chdir(path)
    l_ls = np.logspace(2,6,21)
    # dzf = np.squeeze(mit.rdmds('DRF'))
    # dzc = np.squeeze(mit.rdmds('DRC'))
    # dyc = mit.rdmds('DYC')
    # dxc = mit.rdmds('DXC')
    # dyg = mit.rdmds('DYG')
    # dxg = mit.rdmds('DXG')
    rac = mit.rdmds('RAC')
    
    dz = 5
    y_south_indx = 486
    y_north_indx = 774
    x_west_indx = 726
    x_east_indx = 1274
    ysidx = 3
    ynidx = -3
    xwidx = 3
    xeidx = -3
    # dxg_3d = np.broadcast_to(dxg, (45, *dxg.shape))
    # dxc_3d = np.broadcast_to(dxc, (45, *dxc.shape))
    rac_region = rac[y_south_indx:y_north_indx,x_west_indx:x_east_indx]
    # rac_region_3d = np.broadcast_to(rac_region, (45, *rac_region.shape))
    # hfacregion = np.load('/home/xihanz/my-jupyter-notebooks/hfacregion.npy',allow_pickle='TRUE').item()
    # for key, value in hfacregion.items():
    #     globals()[key] = value


    # --- allocate ---
    KE_wind_sum = np.zeros((tl,len(l_ls)))
    ax = (0, 1)
    # --- compute ---
    for n, m in enumerate(range(args.t1, args.t2)):
        file_index = t0 + ts * m
        interior = xr.open_dataset(f"diags_sub_new/interior.{file_index:010d}.nc")['var']
        tendencies = xr.open_dataset(f"diags_sub_new/tendencies.{file_index:010d}.nc")['var']
        tendencies_med = xr.open_dataset(f"diags_sub_new/tendencies_med.{file_index:010d}.nc")['var']
        Um_ext = tendencies[8,0,ysidx:ynidx,xwidx:xeidx+1] - tendencies[8,1,ysidx:ynidx,xwidx:xeidx+1]
        Vm_ext = tendencies_med[4,0,ysidx:ynidx+1,xwidx:xeidx] - tendencies_med[4,1,ysidx:ynidx+1,xwidx:xeidx]
        u = interior[0,0,ysidx:ynidx,xwidx:xeidx+1].values
        v = interior[1,0,ysidx:ynidx+1,xwidx:xeidx].values


        for i, ll in enumerate(l_ls):
            u_l,_ = spatialfilter_2D(u,ll)  
            uextmap_l,_ = spatialfilter_2D(Um_ext,ll)
            v_l,_ = spatialfilter_2D(v,ll)  
            vextmap_l,_ = spatialfilter_2D(Vm_ext,ll)
            vvextmap_avgy = 0.5*(v_l[0:-1,:]*vextmap_l[0:-1,:] + v_l[1:,:]*vextmap_l[1:,:])
            uuextmap_avgx = 0.5*(u_l[:,0:-1]*uextmap_l[:,0:-1] + u_l[:,1:]*uextmap_l[:,1:])
            KE_wind_l = vvextmap_avgy + uuextmap_avgx
            KE_wind_sum[n,i] = np.nansum(KE_wind_l* dz * rac_region, axis=ax) 
    # --- save ---
    os.makedirs(args.savedir, exist_ok=True)
    fn = os.path.join(args.savedir, f"{args.prefix}{args.t1}_to_{args.t2-1}.npy")


    d = {
        "t1": args.t1, "t2": args.t2,
        "l_ls": l_ls,
        "KE_wind_sum": KE_wind_sum
    }

    np.save(fn, d, allow_pickle=True)
    print("Saved:", fn)

if __name__ == "__main__":
    main()

 



