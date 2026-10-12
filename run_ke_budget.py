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

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--t1", type=int, required=True, help="start time index (inclusive)")
    p.add_argument("--t2", type=int, required=True, help="end time index (exclusive)")
    p.add_argument("--savedir", type=str, required=True, help="output directory")
    p.add_argument("--prefix", type=str, default="GS_KE_full_", help="output filename prefix")
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
    # rac = mit.rdmds('RAC')
    
    
    # y_south_indx = 486
    # y_north_indx = 774
    # x_west_indx = 726
    # x_east_indx = 1274
    # dxg_3d = np.broadcast_to(dxg, (45, *dxg.shape))
    # dxc_3d = np.broadcast_to(dxc, (45, *dxc.shape))
    # rac_region = rac[y_south_indx:y_north_indx,x_west_indx:x_east_indx]
    # rac_region_3d = np.broadcast_to(rac_region, (45, *rac_region.shape))
    hfacregion = np.load('/home/xihanz/my-jupyter-notebooks/hfacregion.npy',allow_pickle='TRUE').item()
    for key, value in hfacregion.items():
        globals()[key] = value


    # --- allocate ---
    KE_tend_ls = np.zeros((tl, 43, len(l_ls)))
    KE_cori_ls = np.zeros_like(KE_tend_ls)
    pi_ls      = np.zeros_like(KE_tend_ls)
    KE_adv_ls  = np.zeros_like(KE_tend_ls)
    KE_phi_ls  = np.zeros_like(KE_tend_ls)
    KE_diss_ls = np.zeros_like(KE_tend_ls)
    KE_ext_ls  = np.zeros_like(KE_tend_ls)
    KE_imp_ls  = np.zeros_like(KE_tend_ls)
    vertvis_ls = np.zeros_like(KE_tend_ls)
    wdpdz_ls   = np.zeros_like(KE_tend_ls)
    res_ls     = np.zeros_like(KE_tend_ls)

    w = rac_region_3d[1:-1, :, :]
    ax = (1, 2)

    # --- compute ---
    for n, m in enumerate(range(args.t1, args.t2)):
        file_index = t0 + ts * m
        
        (uvel, uvel_sub, uvel_subsub,
         vvel, vvel_sub, vvel_subsub,
         wvel, wvel_sub, wvel_subsub,
         tendencies, tendencies_med,
         KPPviscAup, KPPviscAbt, phi) = ke.readfiles(file_index,hacw_region,hacc_region,hacs_region)


        for i, ll in enumerate(l_ls):
            KE_tend_l, KE_cori_l, KE_adv_l, KE_phi_l, KE_diss_l, KE_ext_l, KE_imp_l, KEres = \
                ke.onlineterms(ll, tendencies, tendencies_med, uvel_sub, vvel_sub)

            pi = ke.compute_pi(ll, uvel, uvel_sub, uvel_subsub,
                               vvel, vvel_sub, vvel_subsub,
                               wvel_sub, wvel_subsub,
                               dyg, dyc, dxg_3d, dxc_3d, dzc)

            vertvis_avg = ke.computevis(ll, uvel, uvel_sub, uvel_subsub,
                                        vvel, vvel_sub, vvel_subsub,
                                        KPPviscAup, KPPviscAbt, dzc)

            wdpdzbar = ke.verticalterm(ll, wvel_sub, wvel_subsub, phi, dzc)
            
            KE_tend_ls[n, :, i] = np.nansum(KE_tend_l * w, axis=ax)
            KE_cori_ls[n, :, i] = np.nansum(KE_cori_l * w, axis=ax)
            pi_ls[n, :, i]      = np.nansum(pi * w, axis=ax)
            KE_adv_ls[n, :, i]  = np.nansum(KE_adv_l * w, axis=ax)
            KE_phi_ls[n, :, i]  = np.nansum(KE_phi_l * w, axis=ax)
            KE_diss_ls[n, :, i] = np.nansum(KE_diss_l * w, axis=ax)
            vertvis_ls[n, :, i] = np.nansum(vertvis_avg * w, axis=ax)
            KE_ext_ls[n, :, i]  = np.nansum(KE_ext_l * w, axis=ax)
            KE_imp_ls[n, :, i]  = np.nansum(KE_imp_l * w, axis=ax)
            res_ls[n, :, i]     = np.nansum(KEres * w, axis=ax)
            wdpdz_ls[n, :, i]   = np.nansum(wdpdzbar * w, axis=ax)
    
    # --- save ---
    os.makedirs(args.savedir, exist_ok=True)
    fn = os.path.join(args.savedir, f"{args.prefix}{args.t1}_to_{args.t2-1}.npy")


    d = {
        "t1": args.t1, "t2": args.t2,
        "l_ls": l_ls,
        "KE_tend_ls": KE_tend_ls,
        "KE_cori_ls": KE_cori_ls,
        "pi_ls": pi_ls,
        "KE_adv_ls": KE_adv_ls,
        "KE_phi_ls": KE_phi_ls,
        "wdpdz_ls": wdpdz_ls,
        "KE_diss_ls": KE_diss_ls,
        "vertvis_ls": vertvis_ls,
        "KE_ext_ls": KE_ext_ls,
        "KE_imp_ls": KE_imp_ls,
        "res_ls": res_ls,
    }

    np.save(fn, d, allow_pickle=True)
    print("Saved:", fn)

if __name__ == "__main__":
    main()

 



