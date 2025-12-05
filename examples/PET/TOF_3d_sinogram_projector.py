"""
Example of 3D Time of Flight sinogram projector.
"""

import numpy as np
import torch
import odl
import matplotlib.pyplot as plt

from odl.applications.PET.geometry import PolygonPETGeometry
from odl.applications.PET.operators import PETProjector


# reconstruction space

impl = "pytorch"    # or "numpy"
device = "cuda:0"   # or "cpu"


reco_space = odl.uniform_discr(
    min_pt=[-40, -40, -8],
    max_pt=[40, 40, 8],
    shape=[40, 40, 8],
    dtype='float32',
    impl=impl,
    device=device
)


# Define a polygon pet geometry
num_rings = 3
num_tof_bins = 9
geometry = PolygonPETGeometry(
    radius = 65.0,
    num_sides = 12,
    num_lor_endpoints_per_side = 15,
    lor_spacing = 2.0,
    ring_positions = np.linspace(-10, 10, num_rings),
    radial_trim = 10, # number of LORs to trim at the edges of the sinogram
    max_ring_difference = 2,
    symmetry_axis = 2,
    tof_bins = num_tof_bins
)

# Forward operator (PETProjector)

pet_proj = PETProjector(
    reco_space,
    geometry
)

# Visualize the geometry
pet_proj.show_geometry(force_show=True)


# create a discrete phantom: 3 hot rods
phantom_arr = np.zeros(reco_space.shape)
# central rod
phantom_arr[reco_space.shape[0] // 2, reco_space.shape[1] // 2, :] = 1.0
# off-center rods
phantom_arr[5, reco_space.shape[1] // 2, :] = 1.0
phantom_arr[reco_space.shape[0] // 2, 5, :] = 1.0

phantom = reco_space.element(phantom_arr)

# Forward projection
proj_data = pet_proj(phantom)

# Back-projection
backproj = pet_proj.adjoint(proj_data)


# Visualizations
phantom.show(title="Phantom - Middle Z Slice", coords=[None, None, reco_space.shape[2] // 2])

# Visualize the sinograms for several time of flight bins
num_planes = backproj.shape[2]
fig, ax = plt.subplots(num_planes, num_tof_bins, figsize=(1.4 * num_tof_bins, 1.4 * num_planes), sharex=True, sharey=True)
vmax = float(torch.max(proj_data.data))
for i in range(num_planes):
    for j in range(num_tof_bins):
        ax[i, j].imshow(
            proj_data.data[:, :, i, j].cpu().numpy().T,
            cmap="Greys",
            vmin=0,
            vmax=vmax,
        )
        if i == 0:
            ax[i, j].set_title(
                f"tof bin {j - num_tof_bins//2}", fontsize="medium"
            )
        if j == 0:
            ax[i, j].set_ylabel(f"sino pl. {i}", fontsize="medium")
        # ax[i,j].set_axis_off()
fig.tight_layout()
fig.show()


proj_data.show(title="Sinogram - Middle non-oblique plane", coords = [None, None, 3, num_tof_bins//2])

print(backproj.shape)
backproj.show(title="Backprojection - Middle Z slice", coords = [None, None, reco_space.shape[2] // 2], force_show=True)
