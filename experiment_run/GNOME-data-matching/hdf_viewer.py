import dmt_files as dmt
import numpy as np
import h5py
import pylab as pl

if __name__ == "__main__":
    N = 70
    #slc = slice(-20, None)
    #slc = slice(5000,6000)
    L = []
    names = ["MagneticFields", "MagneticFields2"]#, "MagneticFields_old", "MagneticFields2_old"]
    #fileList = dmt._sortedListOfDmtFilesIn(dmt.uploadDir)
    #fileList = dmt._sortedListOfDmtFilesIn(r"C:\Users\gnome\datatest\upload\2023\08\15\Uploaded")
    fileList = dmt._sortedListOfDmtFilesIn(r"C:\Users\gnome\datatest\incoming\2023\10\12")
    #fileList = dmt._sortedListOfDmtFilesIn(r"C:\Users\gnome\datatest\incoming\2023\09\25")

    if 0:
        #for path in fileList[-N:]:
        for path in fileList[slc]:

            with h5py.File(path, 'r') as f:
                L.append([f[key][:] for key in names])
                L[-1][1][-1] = -50
        
        A = np.hstack(L)
        fig, axs = pl.subplots(4,1, sharex=True)
        for ax,y,name in zip(axs, A, names):
            ax.plot(y)
            ax.set_title(name)
    idxL = []
    tL = []
    t_createL = []
    for path in fileList:
        with h5py.File(path, 'r') as f:
            idxL.append( np.diff(f["MagneticFields"][:]).argmax() )
            tL.append(f['MagneticFields'].attrs['t0'])
            t_createL.append(f.attrs['LocalFileCreationTime'])