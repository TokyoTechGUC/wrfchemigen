from glob import glob
import numpy as np
import jismesh.utils as ju
import geopandas as gpd
import xarray as xr
import fnmatch
import time  # Make sure to import the time module if not already imported
import pandas as pd
import os
import f90nml  
import calendar
from shapely.geometry import box
from pyproj import Transformer
import xesmf as xe
import cf_xarray
from warnings import simplefilter
simplefilter(action="ignore", category=pd.errors.PerformanceWarning)
import mpi4py.MPI

### User inputs
namelist_file = '/gs/bs/tga-guc-lab/users/chenz/simulations/210924_s1_t1/namelist.input' #full path recommended
wrfinput_reference_path = '/gs/bs/tga-guc-lab/users/chenz/simulations/210924_s1_t1/'
outfolder = './out/'
localtime = +9 # Local time of Japan from UTC
jstream_source_path = '/gs/bs/tga-guc-lab/compilation/wrf-chem-boundaries/chatani/j-stream/emisconv_dataset/v202407/input/J-STREAM_v202401/emis/EMIS_MESH/EMIS_MESH_J-STREAM_v202401_' #Not a file, but a prefix for the files
PM25EI_GS_source_path = '/gs/bs/tga-guc-lab/compilation/wrf-chem-boundaries/chatani/j-stream/emisconv_dataset/v202407/input/H29_PM25EI_GS/emis/growth2/EMIS_H29_PM25EI_GS_' #Not a file, but a prefix for the files
PM25EI_AS_source_path = '/gs/bs/tga-guc-lab/compilation/wrf-chem-boundaries/chatani/j-stream/emisconv_dataset/v202407/input/H30_PM25EI_AS/emis/growth2/EMIS_MESH_H30_PM25EI_AS_' #Not a file, but a prefix for the files
meshfile = '/gs/bs/tga-guc-lab/compilation/wrf-chem-boundaries/chatani/j-stream/emisconv_dataset/v202407/input/J-STREAM_v202401/shpin/jstream_mesh_area.pkl' #Make sure that this path exists.
jstream_shp_path = '/gs/bs/tga-guc-lab/compilation/wrf-chem-boundaries/chatani/j-stream/emisconv_dataset/v202407/input/J-STREAM_v202401/shpin/shpin_CITY_2020_MESH3_enc.shp'
jstream_tref_week = '/gs/bs/tga-guc-lab/compilation/wrf-chem-boundaries/chatani/j-stream/emisconv_dataset/v202407/input/J-STREAM_v202401/tfac/tfac_week_J-STREAM_v202401.csv'
### 

    
### Details of emissions from J-Stream converted to Mozart. Do not modify.
aerosol = ['PM25','NA','CL','EC','ORG','SO4','NO3','NH4','PM_10'] # Need to change to per second.
cb6tomozart = {"E_MACR":{"coefs":[1.0,1.0],"vars":["ACROLEIN","BUTADIENE13"],"constants":[],"units":"mol km^-2 hr^-1"},
               "E_CH3CHO":{"coefs":[1.0,1.0],"vars":["ALD2","ALDX"],"constants":[],"units":"mol km^-2 hr^-1"},
               "E_BENZENE":{"coefs":[1.0],"vars":["BENZ"],"constants":[],"units":"mol km^-2 hr^-1"},
               "E_CH4":{"coefs":[1.0],"vars":["CH4"],"constants":[],"units":"mol km^-2 hr^-1"},
               "E_CO":{"coefs":[1.0],"vars":["CO"],"constants":[],"units":"mol km^-2 hr^-1"},
               "E_C2H6":{"coefs":[1.0],"vars":["ETHA"],"constants":[],"units":"mol km^-2 hr^-1"},
               "E_C2H4":{"coefs":[1.0],"vars":["ETH"],"constants":[],"units":"mol km^-2 hr^-1"},
               "E_C2H5OH":{"coefs":[1.0],"vars":["ETOH"],"constants":[],"units":"mol km^-2 hr^-1"},
               "E_CH2O":{"coefs":[1.0],"vars":["FORM"],"constants":[],"units":"mol km^-2 hr^-1"},
               "E_ISOP":{"coefs":[1.0],"vars":["ISOP"],"constants":[],"units":"mol km^-2 hr^-1"},
               "E_CH3OH":{"coefs":[1.0],"vars":["MEOH"],"constants":[],"units":"mol km^-2 hr^-1"},
               "E_NH3":{"coefs":[1.0],"vars":["NH3"],"constants":[],"units":"mol km^-2 hr^-1"},
               "E_NO2":{"coefs":[1.0],"vars":["NO2"],"constants":[],"units":"mol km^-2 hr^-1"},
               "E_NO":{"coefs":[1.0],"vars":["NO"],"constants":[],"units":"mol km^-2 hr^-1"},
               "E_HONO":{"coefs":[1.0],"vars":["HONO"],"constants":[],"units":"mol km^-2 hr^-1"},
               "E_SO2":{"coefs":[1.0],"vars":["SO2"],"constants":[],"units":"mol km^-2 hr^-1"},
               "E_SULF":{"coefs":[1.0],"vars":["SULF"],"constants":[],"units":"mol km^-2 hr^-1"},
               "E_C10H16":{"coefs":[1.0],"vars":["TERP"],"constants":[],"units":"mol km^-2 hr^-1"},
               "E_TOLUENE":{"coefs":[1.0],"vars":["TOL"],"constants":[],"units":"mol km^-2 hr^-1"},
               "E_XYLENE":{"coefs":[1.0],"vars":["XYL"],"constants":[],"units":"mol km^-2 hr^-1"},
               "E_C3H6":{"coefs":[0.14,0.14],"vars":["OLE","PAR"],"constants":[],"units":"mol km^-2 hr^-1"},
               "E_BIGENE":{"coefs":[1.0,2.0],"vars":["OLE","PAR"],"constants":[],"units":"mol km^-2 hr^-1"}, #IOLE + 2 × PAR
               "E_BIGALK":{"coefs":[5.0],"vars":["PAR"],"constants":[],"units":"mol km^-2 hr^-1"},
               "E_C3H8":{"coefs":[1.12*10**-2],"vars":["CO"],"constants":[],"units":"mol km^-2 hr^-1"}, #1.12 × 10-2 × CO (Borbon et al., 2013)
               "E_CH3COCH3":{"coefs":[1.18*10**-2],"vars":["CO"],"constants":[],"units":"mol km^-2 hr^-1"}, #1.18 × 10-2 × CO (Borbon et al., 2013)
               "E_MVK":{"coefs":[2.40*10**-4],"vars":["CO"],"constants":[],"units":"mol km^-2 hr^-1"}, #2.40 × 10-4 × CO (Borbon et al., 2013)
               "E_C2H2":{"coefs":[5.87*10**-3],"vars":["CO"],"constants":[],"units":"mol km^-2 hr^-1"}, #5.87 × 10-3 × CO (Borbon et al., 2013)
               "E_PM25i":{"coefs":[0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1],"vars":["PAL","PCA","PFE","PK","PMFINE","PMG","PMN","PMOTHR","PSI","PTI"],"constants":[],"units":"ug/m3 m/s"},
               "E_NAi":{"coefs":[0.1],"vars":["PNA"],"constants":[],"units":"ug/m3 m/s"},
               "E_CLi":{"coefs":[0.1],"vars":["PCL"],"constants":[],"units":"ug/m3 m/s"},
               "E_ECi":{"coefs":[0.1],"vars":["PEC"],"constants":[],"units":"ug/m3 m/s"},
               "E_ORGi":{"coefs":[0.1],"vars":["POC"],"constants":[],"units":"ug/m3 m/s"},
               "E_SO4i":{"coefs":[0.1],"vars":["PSO4"],"constants":[],"units":"ug/m3 m/s"},
               "E_NO3i":{"coefs":[0.1],"vars":["PNO3"],"constants":[],"units":"ug/m3 m/s"},
               "E_NH4i":{"coefs":[0.1],"vars":["PNH4"],"constants":[],"units":"ug/m3 m/s"},
               "E_PM25j":{"coefs":[0.9,0.9,0.9,0.9,0.9,0.9,0.9,0.9,0.9,0.9],"vars":["PAL","PCA","PFE","PK","PMFINE","PMG","PMN","PMOTHR","PSI","PTI"],"constants":[],"units":"ug/m3 m/s"},
               "E_NAj":{"coefs":[0.9],"vars":["PNA"],"constants":[],"units":"ug/m3 m/s"},
               "E_CLj":{"coefs":[0.9],"vars":["PCL"],"constants":[],"units":"ug/m3 m/s"},
               "E_ECj":{"coefs":[0.9],"vars":["PEC"],"constants":[],"units":"ug/m3 m/s"},
               "E_ORGj":{"coefs":[0.9],"vars":["POC"],"constants":[],"units":"ug/m3 m/s"},
               "E_SO4j":{"coefs":[0.9],"vars":["PSO4"],"constants":[],"units":"ug/m3 m/s"},
               "E_NO3j":{"coefs":[0.9],"vars":["PNO3"],"constants":[],"units":"ug/m3 m/s"},
               "E_NH4j":{"coefs":[0.9],"vars":["PNH4"],"constants":[],"units":"ug/m3 m/s"},
               "E_PM_10":{"coefs":[1.0],"vars":["PMC"],"constants":[],"units":"ug/m3 m/s"},
}
saprctomozart = {"E_MACR":{"coefs":[1.0],"vars":["MACR"],"constants":[],"units":"mol km^-2 hr^-1"},
                 "E_BIGENE":{"coefs":[1.0],"vars":["OLE2"],"constants":[],"units":"mol km^-2 hr^-1"},
                 "E_C3H6":{"coefs":[1.0],"vars":["OLE1"],"constants":[],"units":"mol km^-2 hr^-1"},
                 "E_BIGALK":{"coefs":[1.0,1.0,1.0],"vars":["ALK3","ALK4","ALK5"],"constants":[],"units":"mol km^-2 hr^-1"},
}
### End of details for Mozart-Mosaic in WRFCHEM

def get_datetime_list(yearstart, monthstart, daystart, hourstart, yearend, monthend, dayend, hourend,localtime=0):
    datestart = pd.to_datetime(f"{yearstart}-{monthstart:02d}-{daystart:02d} {hourstart:02d}:00:00") + pd.Timedelta(hours=localtime)
    dateend = pd.to_datetime(f"{yearend}-{monthend:02d}-{dayend:02d} {hourend:02d}:00:00") + pd.Timedelta(hours=localtime)
    daterange = pd.date_range(start=datestart, end=dateend, freq='h')
    return daterange.to_list()

def obtain_mesh_file():
    if not os.path.exists(meshfile):
        meshgridarea = gpd.read_file(f'{os.path.dirname(jstream_shp_path)}/qgis_mesh_japan.shp').drop(columns=['geometry'])
        meshes = gpd.read_file(jstream_shp_path,encoding='utf-8')
        #meshes_projected = meshes.to_crs(epsg=6677) # < - needed for checking only.

        meshes = meshes[['cloc2', 'area_in']]
        meshes = pd.merge(meshes, meshgridarea, left_on='cloc2', right_on='code', how='left').drop_duplicates(subset=['cloc2'])
        meshes['lat'], meshes['lon'] = ju.to_meshpoint(meshes['cloc2'], lat_multiplier=0.5, lon_multiplier=0.5) 
        meshes = meshes[['lat','lon','area_in','area']]
        meshes = meshes.drop_duplicates(subset=['lat', 'lon'])
        meshes['area_in'] = meshes['area_in']/1e6 # Convert area from m^2 to km^2
        #meshes['actual_area'] = meshes_projected.geometry.area/1000./1000.
        meshes[['lat','lon','area']].to_pickle(meshfile, compression='gzip')
    else:
        meshes = pd.read_pickle(meshfile, compression='gzip')
    return meshes

def obtain_dates_from_namelist():
    dayweights = pd.read_csv(jstream_tref_week).drop(columns=['#']).transpose()[0].values #jstream has values from 1 to 7. Necessary to determine the dates later.
    namelist = f90nml.read(namelist_file)
    yearstart = namelist['time_control']['start_year'][0]
    monthstart = namelist['time_control']['start_month'][0]
    daystart = namelist['time_control']['start_day'][0]
    hourstart = namelist['time_control']['start_hour'][0]
    yearend = namelist['time_control']['end_year'][0]
    monthend = namelist['time_control']['end_month'][0]
    dayend = namelist['time_control']['end_day'][0]
    hourend = namelist['time_control']['end_hour'][0]
    maxdom = namelist['domains']['max_dom']
    daterange = get_datetime_list(yearstart, monthstart, daystart, hourstart, yearend, monthend, dayend, hourend,localtime=localtime)
    return daterange, maxdom, dayweights

def date_breakdown(idate):
    """Break down the date into year, month, day, hour."""
    year = idate.year
    month = idate.month
    day = idate.day
    hour = idate.hour
    return year, month, day, hour

def construct_dataarray_lat_lon(lons,lats):
    uniquelats = np.unique(lats)
    uniquelons = np.unique(lons)
    gridlon,gridlat = np.meshgrid(uniquelons,uniquelats)
    df_lat = xr.DataArray(gridlat,dims=['south_north','west_east'],coords={'south_north':uniquelats,'west_east':uniquelons})
    df_lon = xr.DataArray(gridlon,dims=['south_north','west_east'],coords={'south_north':uniquelats,'west_east':uniquelons})
    return df_lon, df_lat

def extract_mozart_mosaic(source,dayweight,month_days,status='jstream'):
    df_reference = pd.read_pickle(source, compression='gzip')
    df_reference = pd.merge(df_reference,meshes,how='inner',on=['lat','lon'])#.dropna()
    df_emission_reference = pd.DataFrame()
    df_emission_reference['lat'] = df_reference['lat']
    df_emission_reference['lon'] = df_reference['lon']
    spec_columns = [icol for icol in df_reference.columns if 'cb6' in icol or 'saprc' in icol]
    for spec in spec_columns:
        if status == 'jstream':
            df_emission_reference[spec] = df_reference[spec]*dayweight*7.0/month_days / df_reference['area']
        else:
            df_emission_reference[spec] = df_reference[spec] / df_reference['area']
    df_emission_reference = df_emission_reference.groupby(['lat', 'lon']).sum().reset_index()
    cb6_spec_columns = [icol for icol in df_emission_reference.columns if 'cb6' in icol]
    saprc_spec_columns = [icol for icol in df_emission_reference.columns if 'saprc' in icol]

    ### Main algorithm for calculating Mozart-mosaic emissions
    df_out = pd.DataFrame()
    df_out['lat'] = df_emission_reference['lat']
    df_out['lon'] = df_emission_reference['lon']
    mozart_variables = list(cb6tomozart.keys())
    for mozart_var in mozart_variables:
        aero = any(entry in mozart_var for entry in aerosol)
        if aero: 
            aerounits = (1.0/3600.0)
        else:
            aerounits = 1.0
        coefs = cb6tomozart[mozart_var]['coefs']
        vars = cb6tomozart[mozart_var]['vars']
        constants = cb6tomozart[mozart_var]['constants']
        df_out[mozart_var] = 0.0
        for varcount,ivar in enumerate(vars):
            cb6_col_selected = [cb6 for cb6 in cb6_spec_columns if f'_{ivar}' in cb6]
            if len(cb6_col_selected) > 0:
                df_out[mozart_var] = df_out[mozart_var] + coefs[varcount] * df_emission_reference[cb6_col_selected[0]]
        if len(constants) > 0:
            df_out[mozart_var] = df_out[mozart_var] + sum(constants)
        df_out[mozart_var] = df_out[mozart_var] * aerounits  # Convert to per second if necessary

    # Override selected species by extracting from the SAPRC estimates
    mozart_variables = list(saprctomozart.keys())
    for mozart_var in mozart_variables:
        aero = any(entry in mozart_var for entry in aerosol)
        if aero: 
            aerounits = (1.0/3600.0)
        else:
            aerounits = 1.0
        coefs = saprctomozart[mozart_var]['coefs']
        vars = saprctomozart[mozart_var]['vars']
        constants = saprctomozart[mozart_var]['constants']
        df_out[mozart_var] = 0.0 #Override CB values
        for varcount,ivar in enumerate(vars):
            saprc_col_selected = [saprc for saprc in saprc_spec_columns if ivar in f'_{saprc}']
            if len(saprc_col_selected) > 0:
                df_out[mozart_var] = df_out[mozart_var] + coefs[varcount] * df_emission_reference[saprc_col_selected[0]]
        if len(constants) > 0:
            df_out[mozart_var] = df_out[mozart_var] + sum(constants)
        df_out[mozart_var] = df_out[mozart_var] * aerounits
    return df_out

def master_construct(idate,meshes):
    year,month,day,hour = date_breakdown(idate)
    month_days = calendar.monthrange(year,month)[1]# Get the number of days in the month
    day_of_week = idate.weekday()  # Monday=0, Sunday=6
    dayweight = dayweights[day_of_week]  # J-Stream has values from 1 to 7, so we use the weekday index directly

    #Checking if processed Jstreams already exist.
    source = glob(jstream_source_path + f'*y{year}_m{month:02d}_h{hour:02d}_*full.pkl')[0]
    df_out_jstream = extract_mozart_mosaic(source,dayweight,month_days,status='jstream')
    source = glob(PM25EI_GS_source_path + f'*y{year}_m{month:02d}_h{hour:02d}_*full.pkl')[0]
    df_out_PM25EI_GS = extract_mozart_mosaic(source,dayweight,month_days,status='PM25EI_GS')
    source = glob(PM25EI_AS_source_path + f'*y{year}_m{month:02d}_h{hour:02d}_*full.pkl')[0]
    df_out_PM25EI_AS = extract_mozart_mosaic(source,dayweight,month_days,status='PM25EI_AS')
    
    df_out = pd.concat([df_out_jstream, df_out_PM25EI_GS, df_out_PM25EI_AS]).groupby(['lat','lon'], as_index=False)[[col for col in df_out_jstream.columns if col not in ['lat','lon']]].sum()
    df_out = df_out.reset_index(drop=True)
    
    ### Generate for all domains.
    # Refers to the wrfinputs for the XLAT and XLONG variables.
    # Fill the path above.
    for idom in range(1,maxdom+1):
        wrfinput = glob(f'{wrfinput_reference_path}wrfinput_d{idom:02d}')[0] #full path recommended
        outdate = idate- pd.Timedelta(hours=localtime)
        outyear, outmonth, outday, outhour = date_breakdown(outdate)
        wrfchemout = f'{outfolder}wrfchemi_d{idom:02d}_{outyear:04d}-{outmonth:02d}-{outday:02d}_{outhour:02d}:00:00'
        if not os.path.exists(wrfchemout):
            df = xr.open_dataset(wrfinput)
            df_xr_out = df[['Times']]
            df_xr_out.attrs = df.attrs
            df_xr_out['longitude'], df_xr_out['latitude'] = construct_dataarray_lat_lon(df_out['lon'].values, df_out['lat'].values)
            df_xr_out = df_xr_out.cf.add_bounds('latitude')
            df_xr_out = df_xr_out.cf.add_bounds('longitude')
            df_xr_out = df_xr_out.set_coords('latitude')
            df_xr_out = df_xr_out.set_coords('longitude')
            for ivar in list(cb6tomozart.keys()):
                values = df_out[['lat','lon',ivar]].pivot_table(index='lat',columns='lon',values=ivar).reindex(index=np.unique(df_out['lat'].values),columns=np.unique(df_out['lon'].values)).fillna(0).values
                df_xr_out[ivar] = xr.DataArray(values.astype('float32'), 
                                                    dims=['south_north', 'west_east'], 
                                                    coords={'latitude': df_xr_out['latitude'], 'longitude': df_xr_out['longitude']})
            df['lon'] = df['XLONG'][0,:,:]
            df['lat'] = df['XLAT'][0,:,:]
            df = df.cf.add_bounds('lon')
            df = df.cf.add_bounds('lat')
            dimensions = {
                    "lat": (('y','x'),df['XLAT'][0,:,:].values),
                    "lon": (('y','x'),df['XLONG'][0,:,:].values),
                    "lat_b": (('y_b','x_b'),cf_xarray.bounds_to_vertices(df['lat_bounds'],bounds_dim='bounds').values),
                    "lon_b": (('y_b','x_b'),cf_xarray.bounds_to_vertices(df['lon_bounds'],bounds_dim='bounds').values)
                }
            df_regridder_target = xr.Dataset(coords=dimensions)
            df_regridder_from = df_xr_out[['latitude','longitude']].copy()
            df_regridder_from = df_regridder_from.cf.add_bounds('latitude')
            df_regridder_from = df_regridder_from.cf.add_bounds('longitude')
            df_regridder_from = df_regridder_from.set_coords('latitude')
            df_regridder_from = df_regridder_from.set_coords('longitude')
            df_regridder_from['lat_b']= cf_xarray.bounds_to_vertices(df_regridder_from['latitude_bounds'],bounds_dim='bounds')
            df_regridder_from = df_regridder_from.set_coords('lat_b')
            df_regridder_from['lon_b']= cf_xarray.bounds_to_vertices(df_regridder_from['longitude_bounds'],bounds_dim='bounds')
            df_regridder_from = df_regridder_from.set_coords('lon_b')#xe.Regridder(df_regridder_from, df_target, resampler)#,ignore_degenerate=True)
            regridder_file = f'{outfolder}/regridder_d{idom:02d}.nc'
            #Recycle regridders to save time.
            if not os.path.exists(regridder_file):
                regridder = xe.Regridder(df_regridder_from, df_regridder_target, 'conservative', periodic=False, ignore_degenerate=True)
                regridder.to_netcdf(regridder_file)
            else:
                regridder = xe.Regridder(df_regridder_from, df_regridder_target, 'conservative', periodic=False, ignore_degenerate=True, weights=regridder_file)
            df_wrfchemi_final = regridder(df_xr_out)
            df_wrfchemi_final = df_wrfchemi_final.rename({'y':'south_north','x':'west_east'})
            df_wrfchemi_final = df_wrfchemi_final.expand_dims(dim='emissions_zdim')
            #Add the time information        
            df_wrfchemi_final['Times'] = df['Times'].copy()
            df_wrfchemi_final['Times'][:] = np.array([(f'{outyear}-{outmonth:02d}-{outday:02d}_{outhour:02d}:00:00').encode('utf-8')])
            #Add the attributes and fix dimensions of the emissions.
            df_wrfchemi_final.transpose('Time','emissions_zdim','south_north','west_east') #Reorder the dimensions
            for ivar in df_wrfchemi_final.keys():
                if "E_" in ivar:
                    df_wrfchemi_final[ivar] = df_wrfchemi_final[ivar].expand_dims(dim='Time')
                    df_wrfchemi_final[ivar].attrs = {}
                    df_wrfchemi_final[ivar].attrs['FieldType'] = np.int32(104)
                    df_wrfchemi_final[ivar].attrs['MemoryOrder'] = "XYZ"
                    df_wrfchemi_final[ivar].attrs['description'] = "Emissions"
                    df_wrfchemi_final[ivar].attrs['units'] = cb6tomozart[ivar]['units']
                    df_wrfchemi_final[ivar].attrs['stagger'] = ""
                    df_wrfchemi_final[ivar].attrs['coordinates'] = "XLONG XLAT XTIME"
            #Standardize the lat/lon information to WRF nomenclature
            df_wrfchemi_final = df_wrfchemi_final.rename({'lat':'XLAT'})
            df_wrfchemi_final['XLAT'].attrs = {}
            df_wrfchemi_final['XLAT'].attrs['MemoryOrder'] = "XY "
            df_wrfchemi_final['XLAT'].attrs['description'] = "LATITUDE, SOUTH IS NEGATIVE"
            df_wrfchemi_final['XLAT'].attrs['units'] = "degree north"
            df_wrfchemi_final['XLAT'].attrs['stagger'] = ""
            df_wrfchemi_final['XLAT'].attrs['FieldType'] = np.int32(104)
            df_wrfchemi_final = df_wrfchemi_final.rename({'lon':'XLONG'})
            df_wrfchemi_final['XLONG'].attrs = {}
            df_wrfchemi_final['XLONG'].attrs['MemoryOrder'] = "XY "
            df_wrfchemi_final['XLONG'].attrs['description'] = "LONGITUDE, WEST IS NEGATIVE"
            df_wrfchemi_final['XLONG'].attrs['units'] = "degree east"
            df_wrfchemi_final['XLONG'].attrs['stagger'] = ""
            df_wrfchemi_final['XLONG'].attrs['FieldType'] = np.int32(104)
            df_wrfchemi_final.to_netcdf(wrfchemout)
        else:
            print(f'{wrfchemout} already exists. Skipping this date.')
    return

if not os.path.exists(outfolder):
    os.makedirs(outfolder)
meshes = obtain_mesh_file()
daterange, maxdom, dayweights = obtain_dates_from_namelist()

rank = mpi4py.MPI.COMM_WORLD.Get_rank()
size = mpi4py.MPI.COMM_WORLD.Get_size()

for i,idate in enumerate(daterange):
    if i % size != rank:
        continue
    master_construct(idate, meshes)
