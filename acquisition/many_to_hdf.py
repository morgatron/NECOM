import numpy as np
import h5py as h5
from ruamel import yaml
import xarray as xa
import os
from os import path




def filter_base_names(names):
    """ From a list of file names related to the old data style, identify the base dataset names"""
    return {name[:-6] for name in names if name.endswith("_t.npy")}

def remove_old_files(basePath):
    """ Remove the all the old-style files associated with that path name"""
    endings = [ ".npy", "_t.npy", "_signatures.npz", "_params.yaml", "_meta.yaml", "_flags.npy"]
    for ending in endings:
        try:
            os.remove(basePath +ending)
        except FileNotFoundError as e:
            print(e)



def replace_dataset(basePath):
    """ Probably only this should actually be called from a script"""
    try:
        ds = write_new_hdf_from_old_files(basePath)
        if ds:
            print("removing old files...")
            remove_old_files(basePath)
    except Exception as e:
        print(f"Skipping {basePath} because:\n {e}")
        return None
    return ds

def write_new_hdf_from_old_files(basePath):

    d = {
        "t":np.load(basePath+"_t.npy"),
        "signal":np.load(basePath+".npy", mmap_mode="r"),
        "signatures":np.load(basePath+"_signatures.npz"),
        "params": yaml.load(open(basePath+"_params.yaml")),
        "meta": yaml.load(open(basePath+"_meta.yaml"))
    }
    Nds = np.round(4.0/(d['signal'].shape[-1]/1000) ) # Based on the assumption that the original sample rate was 1MS/s and the record time was ~4ms

    da_sig = xa.DataArray(d['signal'], 
                  dims=["T","t"], 
                  coords={"T":d["t"], 't':np.arange(d['signal'].shape[1])*1e-3*Nds } )

    try:
        flags = np.load(basePath+"_flags.npy")
        da_flags = xa.DataArray(flags, 
                    dims=["T"], 
                    coords={"T":d["t"]} 
                )
        ds = xa.Dataset({'signal': da_sig, 'flags':da_flags})
    except FileNotFoundError:
        ds = xa.Dataset({'signal': da_sig})


    ds.attrs['params'] = yaml.dump(d['params'])
    ds.attrs['meta'] = yaml.dump(d['meta'])
    for key,val in dict(d['signatures']).items():
        ds.attrs['signature_'+key] = val

    enc_opts = {"signal":{
                    "zlib": True,
                    "complevel": 3,
                    #"fletcher32": True,
                    #"chunksizes": tuple(map(lambda x: x//2, ds['signal'].shape))
            }}
    ds.to_netcdf(basePath+".h5", format="NETCDF4", engine="netcdf4", encoding=enc_opts)
    return ds
    



if __name__ == "__main__":
    #search_path = "../recorded"
    #search_path = "/home/morgan/Insync/u4284055@anu.edu.au/OneDrive Biz/NECOM_DATA/2023"
    search_path = "/home/morgan/opt/onedrive/NECOM_DATA/2023"
    for root, dirs, files in os.walk(search_path):
        #print(root, dirs, files)
        for fileName in list(filter_base_names(files))[:]:
            baseName = path.join(root, fileName)
            ds= replace_dataset(baseName)
        print("\n\n")

