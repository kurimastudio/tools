from spectral_cube import SpectralCube

cube = SpectralCube.read("13CO21.fits")

sub = cube[:, y1:y2, x1:x2]

sub.write("13CO_cut.fits", overwrite=True)