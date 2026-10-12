import MITgcmutils as mit
import sys, os, glob
from scipy.ndimage import gaussian_filter
import xarray as xr
from netCDF4 import Dataset
import numpy as np

path="/resnick/groups/oceanphysics/xihanz/NA_energy"
os.chdir(path)

y_south_indx = 486
y_north_indx = 774
x_west_indx = 726
x_east_indx = 1274


def spatialfilter(data,l):
    delta = 2e3
    truncate = 4
    radius_in_pixels = l/delta
    sigma_pixels = radius_in_pixels/truncate
    lowpass = gaussian_filter(data, sigma=(0,sigma_pixels, sigma_pixels))  
    highpass = data - lowpass
    return lowpass,highpass

def readfiles(file_index,hacw_region,hacc_region,hacs_region):
    
    KPPviscA_regional = xr.open_dataset(f"diags_sub_new/KPPviscA.{file_index:010d}.nc")['var']
    tendencies_regional = xr.open_dataset(f"diags_sub_new/tendencies.{file_index:010d}.nc")['var']
    tendencies_med_regional = xr.open_dataset(f"diags_sub_new/tendencies_med.{file_index:010d}.nc")['var']
    phi_regional = xr.open_dataset(f"diags_sub_new/PHIHYD.{file_index:010d}.nc")['var']
    interior_regional = xr.open_dataset(f"diags_sub_new/interior.{file_index:010d}.nc")['var']

    u = interior_regional[0,:,:,:].values
    v = interior_regional[1,:,:,:].values
    w = interior_regional[2,:,:,:].values
    u[hacw_region == 0] = np.nan
    v[hacs_region == 0] = np.nan
    w[hacc_region == 0] = np.nan

    KPPviscAup = KPPviscA_regional[1:-1,:,:]
    KPPviscAbt = KPPviscA_regional[2:,:,:]
    
    uvel = u[0:-2,:,:]
    uvel_sub = u[1:-1,:,:]
    uvel_subsub = u[2:,:,:]
    
    vvel = v[0:-2,:,:]
    vvel_sub = v[1:-1,:,:]
    vvel_subsub = v[2:,:,:]
    
    wvel = w[0:-2,:,:]
    wvel_sub = w[1:-1,:,:]
    wvel_subsub = w[2:,:,:]

    return uvel,uvel_sub,uvel_subsub,vvel,vvel_sub,vvel_subsub,wvel,wvel_sub,wvel_subsub,\
    tendencies_regional,tendencies_med_regional,KPPviscAup,KPPviscAbt,phi_regional

def onlineterms(l,tendencies,tendencies_med,uvel_sub,vvel_sub,ysidx = 3,ynidx = -3,xwidx = 3,xeidx = -3):
    # zonal term
    Um_diss = tendencies[0,:,ysidx:ynidx,xwidx:xeidx+1]
    Um_pg = tendencies[2,:,ysidx:ynidx,xwidx:xeidx+1]
    
    Um_tend = tendencies[4,:,ysidx:ynidx,xwidx:xeidx+1]
    Um_adv = tendencies[5,:,ysidx:ynidx,xwidx:xeidx+1]
    Um_impl = tendencies[6,:,ysidx:ynidx,xwidx:xeidx+1]
    Um_cori = tendencies[7,:,ysidx:ynidx,xwidx:xeidx+1]
    Um_ext = tendencies[8,:,ysidx:ynidx,xwidx:xeidx+1]

    Vm_pg = tendencies[3,:,ysidx:ynidx+1,xwidx:xeidx]
    Vm_pg = tendencies[3,:,ysidx:ynidx+1,xwidx:xeidx]
    Vm_tend = tendencies_med[0,:,ysidx:ynidx+1,xwidx:xeidx]
    Vm_adv = tendencies_med[1,:,ysidx:ynidx+1,xwidx:xeidx]
    Vm_impl = tendencies_med[2,:,ysidx:ynidx+1,xwidx:xeidx]
    Vm_cori = tendencies_med[3,:,ysidx:ynidx+1,xwidx:xeidx]
    Vm_ext = tendencies_med[4,:,ysidx:ynidx+1,xwidx:xeidx]
    Vm_diss = tendencies[1,:,ysidx:ynidx+1,xwidx:xeidx]

    uvel_sub_region = uvel_sub[:,ysidx:ynidx,xwidx:xeidx+1]
    vvel_sub_region = vvel_sub[:,ysidx:ynidx+1,xwidx:xeidx]

    u_l,_ = spatialfilter(uvel_sub_region,l)  
    utendmap_l,_ = spatialfilter(Um_tend/86400,l)  
    ucorimap_l,_ = spatialfilter(Um_cori,l)
    uadvmap_l,_ = spatialfilter(Um_adv - Um_cori,l)
    uphixmap_l,_ = spatialfilter(Um_pg,l)
    udissmap_l,_ = spatialfilter(Um_diss,l)
    uextmap_l,_ = spatialfilter(Um_ext,l)
    uimpmap_l,_ = spatialfilter(Um_impl,l)

    v_l,_ = spatialfilter(vvel_sub_region,l)  
    vtendmap_l,_ = spatialfilter(Vm_tend/86400,l)  
    vcorimap_l,_ = spatialfilter(Vm_cori,l)
    vadvmap_l,_ = spatialfilter(Vm_adv - Vm_cori,l)
    vphixmap_l,_ = spatialfilter(Vm_pg,l)
    vdissmap_l,_ = spatialfilter(Vm_diss,l)
    vextmap_l,_ = spatialfilter(Vm_ext,l)
    vimpmap_l,_ = spatialfilter(Vm_impl,l)


    vvtendmap_avgy = 0.5*(v_l[:,0:-1,:]*vtendmap_l[1:-1,0:-1,:] + v_l[:,1:,:]*vtendmap_l[1:-1,1:,:])
    vvcorimap_avgy = 0.5*(v_l[:,0:-1,:]*vcorimap_l[1:-1,0:-1,:] + v_l[:,1:,:]*vcorimap_l[1:-1,1:,:])
    vvadvmap_avgy = 0.5*(v_l[:,0:-1,:]*vadvmap_l[1:-1,0:-1,:] + v_l[:,1:,:]*vadvmap_l[1:-1,1:,:])
    vvphixmap_avgy = 0.5*(v_l[:,0:-1,:]*vphixmap_l[1:-1,0:-1,:] + v_l[:,1:,:]*vphixmap_l[1:-1,1:,:])
    vvdissmap_avgy = 0.5*(v_l[:,0:-1,:]*vdissmap_l[1:-1,0:-1,:] + v_l[:,1:,:]*vdissmap_l[1:-1,1:,:])
    vvextmap_avgy = 0.5*(v_l[:,0:-1,:]*vextmap_l[1:-1,0:-1,:] + v_l[:,1:,:]*vextmap_l[1:-1,1:,:])
    vvimpmap_avgy = 0.5*(v_l[:,0:-1,:]*vimpmap_l[1:-1,0:-1,:] + v_l[:,1:,:]*vimpmap_l[1:-1,1:,:])
    
    uutendmap_avgx = 0.5*(u_l[:,:,0:-1]*utendmap_l[1:-1,:,0:-1] + u_l[:,:,1:]*utendmap_l[1:-1,:,1:])
    uucorimap_avgx = 0.5*(u_l[:,:,0:-1]*ucorimap_l[1:-1,:,0:-1] + u_l[:,:,1:]*ucorimap_l[1:-1,:,1:])
    uuadvmap_avgx = 0.5*(u_l[:,:,0:-1]*uadvmap_l[1:-1,:,0:-1] + u_l[:,:,1:]*uadvmap_l[1:-1,:,1:])
    uuphixmap_avgx = 0.5*(u_l[:,:,0:-1]*uphixmap_l[1:-1,:,0:-1] + u_l[:,:,1:]*uphixmap_l[1:-1,:,1:])
    uudissmap_avgx = 0.5*(u_l[:,:,0:-1]*udissmap_l[1:-1,:,0:-1] + u_l[:,:,1:]*udissmap_l[1:-1,:,1:])
    uuextmap_avgx = 0.5*(u_l[:,:,0:-1]*uextmap_l[1:-1,:,0:-1] + u_l[:,:,1:]*uextmap_l[1:-1,:,1:])
    uuimpmap_avgx = 0.5*(u_l[:,:,0:-1]*uimpmap_l[1:-1,:,0:-1] + u_l[:,:,1:]*uimpmap_l[1:-1,:,1:])
    
    KE_tend_l = vvtendmap_avgy + uutendmap_avgx
    KE_cori_l = vvcorimap_avgy + uucorimap_avgx
    KE_adv_l = vvadvmap_avgy + uuadvmap_avgx
    KE_phi_l = vvphixmap_avgy + uuphixmap_avgx
    KE_diss_l = vvdissmap_avgy + uudissmap_avgx
    KE_ext_l = vvextmap_avgy + uuextmap_avgx
    KE_imp_l = vvimpmap_avgy + uuimpmap_avgx
    KEres = KE_phi_l + KE_cori_l + KE_adv_l+ KE_diss_l + KE_ext_l+ KE_imp_l -KE_tend_l
    return KE_tend_l,KE_cori_l,KE_adv_l,KE_phi_l,KE_diss_l,KE_ext_l,KE_imp_l,KEres

def compute_pi(l,uvel,uvel_sub,uvel_subsub,vvel,vvel_sub,vvel_subsub,wvel_sub,wvel_subsub,dyg, dyc,dxg_3d,dxc_3d, dzc,\
               ysidx = 3,ynidx = -3,xwidx = 3,xeidx = -3):
    u_pi1 = uvel_sub[:,ysidx:ynidx,xwidx:xeidx+1]
    dxg_pi1 = dxg_3d[1:-1,y_south_indx:y_north_indx,x_west_indx:x_east_indx]
    ubar1,_ = spatialfilter(u_pi1,l)
    uubar1,_ = spatialfilter(u_pi1**2,l)
    dudx = (ubar1[:,:,1:] - ubar1[:,:,0:-1])/(dxg_pi1)
    uu_subgrid = uubar1 - ubar1*ubar1
    pi1_avg = dudx * 0.5*(uu_subgrid[:,:,0:-1]+uu_subgrid[:,:,1:])

    u_pi2 = uvel_sub[:,ysidx-1:ynidx+1,xwidx:xeidx+1]
    v_pi2 = vvel_sub[:,ysidx:ynidx+1,xwidx:xeidx]
    ubar2,_ = spatialfilter(u_pi2,l)
    vbar2,_ = spatialfilter(v_pi2,l)
    uv = (0.5*(u_pi2[:,1:-1,0:-1]+u_pi2[:,1:-1,1:]))*(0.5*(v_pi2[:,0:-1,:]+v_pi2[:,1:,:]))
    uvbar2,_ = spatialfilter(uv,l)
    dy = dyc[0,0]
    dudy = (ubar2[:,2:,0:-1] - ubar2[:,0:-2,0:-1])/(2*dy)
    uv_sub = uvbar2 - (0.5*(ubar2[:,1:-1,0:-1] + ubar2[:,1:-1,1:]))*(0.5*(vbar2[:,0:-1,:]+vbar2[:,1:,:]))
    pi2_avg = dudy*uv_sub

    u_surf_pi3 = uvel[:,ysidx:ynidx,xwidx:xeidx+1] # u1
    u_pi3 = uvel_sub[:,ysidx:ynidx,xwidx:xeidx+1] # u2
    u_sub_pi3 = uvel_subsub[:,ysidx:ynidx,xwidx:xeidx+1]# u3
    w_pi3 = wvel_sub[:,ysidx:ynidx,xwidx:xeidx]
    w_sub_pi3 = wvel_subsub[:,ysidx:ynidx,xwidx:xeidx]
    usurfbar3,_ = spatialfilter(u_surf_pi3,l)# u1
    ubar3,_ = spatialfilter(u_pi3,l)# u2
    usubar3,_ = spatialfilter(u_sub_pi3,l)# u3
    wbar3,_ = spatialfilter(w_pi3,l)
    wsubar3,_ = spatialfilter(w_sub_pi3,l)
    uw = 0.5*(u_pi3[:,:,0:-1]+u_pi3[:,:,1:])*(0.5*(w_pi3+w_sub_pi3))
    uwbar,_ = spatialfilter(uw,l)
    dzc_3d3 = dzc[0:45, None, None] * np.ones(usurfbar3.shape[1:])
    dudz = (usurfbar3 - usubar3)/(dzc_3d3[1:-1,:,:]+dzc_3d3[2:,:,:])
    uw_sub = uwbar - 0.5*(ubar3[:,:,0:-1]+ubar3[:,:,1:])*0.5*(wbar3+wsubar3)
    pi3_avg = 0.5*(dudz[:,:,0:-1]+dudz[:,:,1:])*uw_sub


    v_pi4 = vvel_sub[:,ysidx:ynidx+1,xwidx-1:xeidx+1]
    u_pi4 = uvel_sub[:,ysidx:ynidx,xwidx:xeidx+1]
    dxc_pi4 = dxc_3d[:,y_south_indx:y_north_indx+1,x_west_indx:x_east_indx]
    ubar4,_ = spatialfilter(u_pi4,l)
    vbar4,_ = spatialfilter(v_pi4,l)
    uv4 = (0.5*(u_pi4[:,:,0:-1]+u_pi4[:,:,1:]))*(0.5*(v_pi4[:,0:-1,1:-1]+v_pi4[:,1:,1:-1]))
    uv_bar4,_ = spatialfilter(uv4,l)
    dvdx = (vbar4[:,:,2:]-vbar4[:,:,0:-2])/(2*dxc_pi4[1:-1,:,:])
    uv_sub_pi4 = uv_bar4 - (0.5*(ubar4[:,:,0:-1]+ubar4[:,:,1:]))*(0.5*(vbar4[:,0:-1,1:-1]+vbar4[:,1:,1:-1]))
    pi4_avg = 0.5*(dvdx[:,0:-1,:]+dvdx[:,1:,:])*uv_sub_pi4

    v_pi5 = vvel_sub[:,ysidx:ynidx+1,xwidx:xeidx]
    vbar5,_ = spatialfilter(v_pi5,l)
    dyg0 = dyg[0,0]
    dvdy = (vbar5[:,1:,:]-vbar5[:,0:-1,:])/dyg0
    vv = v_pi5**2
    vvbar,_ = spatialfilter(vv,l)
    vv_sub = vvbar - vbar5**2
    pi5_avg = dvdy*0.5*(vv_sub[:,1:,:]+vv_sub[:,0:-1,:])


    v_surfpi6 = vvel[:,ysidx:ynidx+1,xwidx:xeidx] # v1
    v_pi6 = vvel_sub[:,ysidx:ynidx+1,xwidx:xeidx] # v2
    v_subpi6 = vvel_subsub[:,ysidx:ynidx+1,xwidx:xeidx]# v3
    w_pi6 = wvel_sub[:,ysidx:ynidx,xwidx:xeidx]
    w_subpi6 = wvel_subsub[:,ysidx:ynidx,xwidx:xeidx]
    
    vbar_surf6,_ = spatialfilter(v_surfpi6,l)
    vbar6,_ = spatialfilter(v_pi6,l)
    vbar_sub6,_ = spatialfilter(v_subpi6,l)
    wbar6,_ = spatialfilter(w_pi6,l)
    wbar_sub6,_ = spatialfilter(w_subpi6,l)
    dzc_3d6 = dzc[0:45, None, None] * np.ones(vbar_surf6.shape[1:])
    dvdz = (vbar_surf6 - vbar_sub6)/(dzc_3d6[1:-1,:,:]+dzc_3d6[2:,:,:])
    vw = 0.5*(v_pi6[:,0:-1,:] + v_pi6[:,1:,:])*0.5*(w_pi6 + w_subpi6)
    vwbar,_ = spatialfilter(vw,l)
    vw_sub = vwbar - 0.5*(wbar6 + wbar_sub6)*0.5*(vbar6[:,0:-1,:]+vbar6[:,1:,:])
    pi6_avg = 0.5*(dvdz[:,0:-1,:]+dvdz[:,1:,:])*vw_sub
    pi = - (pi1_avg + pi2_avg + pi3_avg + pi4_avg + pi5_avg + pi6_avg)
    return pi    


def computevis(l,uvel,uvel_sub,uvel_subsub,vvel,vvel_sub,vvel_subsub,KPPviscAup,KPPviscAbt,dzc,\
              ysidx = 3,ynidx = -3,xwidx = 3,xeidx = -3):
    
    u0_vertvis = uvel[:,ysidx:ynidx,xwidx:xeidx+1]
    v0_vertvis = vvel[:,ysidx:ynidx+1,xwidx:xeidx]
    u_vertvis = uvel_sub[:,ysidx:ynidx,xwidx:xeidx+1]
    v_vertvis = vvel_sub[:,ysidx:ynidx+1,xwidx:xeidx]
    usub_vertvis = uvel_subsub[:,ysidx:ynidx,xwidx:xeidx+1]
    vsub_vertvis = vvel_subsub[:,ysidx:ynidx+1,xwidx:xeidx]
    vermu = KPPviscAup[:,ysidx:ynidx,xwidx:xeidx]
    vermu_bt = KPPviscAbt[:,ysidx:ynidx,xwidx:xeidx]
    
    # u_up = 0.5*(u0_vertvis + u_vertvis) # u_1.5
    # v_up = 0.5*(v0_vertvis + v_vertvis) # v_1.5
    # u_bt = 0.5*(u_vertvis + usub_vertvis) # u_2.5
    # v_bt = 0.5*(v_vertvis + vsub_vertvis) # v_2.5
    # u_upbar,_ = spatialfilter(u_up,l)
    # u_btbar,_ = spatialfilter(u_bt,l)
    # v_upbar,_ = spatialfilter(v_up,l)
    # v_btbar,_ = spatialfilter(v_bt,l)
    
    u0bar,_ =spatialfilter(u0_vertvis,l)
    u1bar,_ = spatialfilter(u_vertvis,l)
    u2bar,_ = spatialfilter(usub_vertvis,l)
    v0bar,_ = spatialfilter(v0_vertvis,l)
    v1bar,_ = spatialfilter(v_vertvis,l)
    v2bar,_ = spatialfilter(vsub_vertvis,l)
    
    dzc_3du = dzc[0:45, None, None] * np.ones(u0bar.shape[1:])
    dudz = (u0bar - u2bar)/(dzc_3du[1:-1,:,:]+dzc_3du[2:,:,:])
    dzc_3dv = dzc[0:45, None, None] * np.ones(v0bar.shape[1:])
    dvdz = (v0bar - v2bar)/(dzc_3dv[1:-1,:,:]+dzc_3dv[2:,:,:])
    dudzbar = 0.5*(dudz[:,:,1:]+dudz[:,:,0:-1])
    dvdzbar = 0.5*(dvdz[:,1:,:]+dvdz[:,0:-1,:])

    dudz_uncg = (u0_vertvis - usub_vertvis)/(dzc_3du[1:-1,:,:]+dzc_3du[2:,:,:])
    dvdz_uncg = (v0_vertvis - vsub_vertvis)/(dzc_3dv[1:-1,:,:]+dzc_3dv[2:,:,:])
    dudzbar_uncg = 0.5*(dudz_uncg[:,:,1:]+dudz_uncg[:,:,0:-1])
    dvdzbar_uncg = 0.5*(dvdz_uncg[:,1:,:]+dvdz_uncg[:,0:-1,:])

    nududz = 0.5*(vermu + vermu_bt)*dudzbar_uncg
    nudvdz = 0.5*(vermu + vermu_bt)*dvdzbar_uncg

    nududzbar,_ = spatialfilter(nududz,l)
    nudvdzbar,_ = spatialfilter(nudvdz,l)
    
    # vertvis_x = 0.5*(vermu + vermu_bt)*(dudzbar**2)
    # vertvis_y = 0.5*(vermu + vermu_bt)*(dvdzbar**2)
    vertvis_x = nududzbar*dudzbar
    vertvis_y = nudvdzbar*dvdzbar
    vertvis_avg = vertvis_x + vertvis_y
    return vertvis_avg

def verticalterm(l,wvel_sub,wvel_subsub,phi,dzc,ysidx = 3,ynidx = -3,xwidx = 3,xeidx = -3):
    phi_region = phi[:,ysidx:ynidx,xwidx:xeidx]
    wsub_region = wvel_sub[:,ysidx:ynidx,xwidx:xeidx]
    wsubsub_region = wvel_subsub[:,ysidx:ynidx,xwidx:xeidx]
    
    phi1bar,_ = spatialfilter(phi_region[0:-2,:,:],l)
    phi2bar,_ = spatialfilter(phi_region[2:,:,:],l)
    w2bar,_ = spatialfilter(wsub_region,l)
    w3bar,_ = spatialfilter(wsubsub_region,l)
    wl = 0.5*(w2bar+w3bar)
    dzc_3dp = dzc[0:45, None, None] * np.ones(phi1bar.shape[1:])
    wdpdzbar = wl*(phi1bar-phi2bar)/(dzc_3dp[1:-1,:,:]+dzc_3dp[2:,:,:])
    return wdpdzbar