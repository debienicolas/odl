"""
Example of 2D non Time of Flight sinogram projector.
"""

import numpy as np
import odl
import matplotlib.pyplot as plt

from odl.applications.PET.geometry import PolygonPETGeometry
from odl.applications.PET.operators import PETProjector


# reconstruction space

reco_space = odl.uniform_discr(
    min_pt=[-40, -40, -8],
    max_pt=[40, 40, 8],
    shape=[40, 40, 8],
    dtype='float32',
    impl='numpy'
)


# Define a polygon pet geometry
num_rings = 5
geometry = PolygonPETGeometry(
    radius = 65.0,
    num_sides = 12,
    num_lor_endpoints_per_side = 15,
    lor_spacing = 2.0,
    ring_positions = np.linspace(-10, 10, num_rings),
    radial_trim = 10, # number of LORs to trim at the edges of the sinogram
    max_ring_difference = 2,
    symmetry_axis = 2
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
proj_data.show(title="Sinogram - Middle non-oblique plane", coords = [None, None, 3])
proj_data.show(title="Sinogram - Ring difference = 2, ring 0 to ring 2", coords = [None, None, 14])
backproj.show(title="Backprojection - Middle Z slice", coords = [None, None, reco_space.shape[2] // 2], force_show=True)
