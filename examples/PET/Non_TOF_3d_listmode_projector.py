"""
Example of 3D non Time of Flight List-mode projector.
"""

import numpy as np
import odl
import matplotlib.pyplot as plt

from odl.applications.PET.geometry import PolygonPETGeometry
from odl.applications.PET.operators import PETListmodeProjector

# define the reconstruction space
reco_space = odl.uniform_discr(
    min_pt=[-40, -40, -8],
    max_pt=[40, 40, 8],
    shape=[40, 40, 8],
    dtype='float32',
    impl='numpy'
)


# define a polygon pet geometry
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


# define a listmode projector
num_lors = 100
start_coords, end_coords = geometry.sample_random_lors(100)


lm_proj = PETListmodeProjector(
    reco_space,
    geometry,
    start_coords,
    end_coords
)

# Create a phantom
# create a discrete phantom: 3 hot rods
phantom_arr = np.zeros(reco_space.shape)
# central rod
phantom_arr[reco_space.shape[0] // 2, reco_space.shape[1] // 2, :] = 1.0
# off-center rods
phantom_arr[5, reco_space.shape[1] // 2, :] = 1.0
phantom_arr[reco_space.shape[0] // 2, 5, :] = 1.0

phantom = reco_space.element(phantom_arr)

proj_data = lm_proj(phantom)
print(proj_data.shape)


# backprojection the data

backproj = lm_proj.adjoint(proj_data)

print(backproj.shape)

backproj.show(title="Backprojection", force_show=True)