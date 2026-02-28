from astropy.io import fits

# --- open original cube ---
hdu = fits.open("13CO21.fits")[0]
data = hdu.data
hdr  = hdu.header.copy()

# --- define cut ---
x1, x2 = 109, 811
y1, y2 = 133, 894

subcube = data[:, y1:y2, x1:x2]

# --- FIX WCS ---
hdr["CRPIX1"] -= x1
hdr["CRPIX2"] -= y1

# --- update axis size ---
hdr["NAXIS1"] = subcube.shape[2]
hdr["NAXIS2"] = subcube.shape[1]
hdr["NAXIS3"] = subcube.shape[0]

# --- save ---
fits.writeto(
    "13CO_emission_cut_fixed.fits",
    subcube,
    hdr,
    overwrite=True
)