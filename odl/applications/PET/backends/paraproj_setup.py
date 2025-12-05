# Copyright 2014-2025 The ODL contributors
#
# This file is part of ODL.
#
# This Source Code Form is subject to the terms of the Mozilla Public License,
# v. 2.0. If a copy of the MPL was not distributed with this file, You can
# obtain one at https://mozilla.org/MPL/2.0/.

"""
Converting ODL PET geometry to parallelproj
"""

#from odl.applications.PET.geometry import PolygonPETGeometry

try:
    import parallelproj
    PARALLELPROJ_AVAILABLE = True
except ImportError:
    PARALLELPROJ_AVAILABLE = False

__all__ = (
    'PARALLELPROJ_AVAILABLE',
    'create_scanner_geometry',
    'create_lor_descriptor',
    'create_projector',
)


def get_array_module(space):
    """Get the appropriate array module and device from reconstruction space.
    
    Parameters
    ----------
    space : DiscretizedSpace
        Reconstruction space.
        
    Returns
    -------
    xp : module
        Array module (numpy, pytorch or cupy).
    dev : str
        Device string for parallelproj
    """

    device = space.device
    impl = space.impl

    if impl == 'numpy':
        try:
            import array_api_compat.numpy as np
            return np, device
        except ImportError:
            raise ImportError(
                "NumPy is required for numpy implementation. ")
    elif impl == 'pytorch':
        try:
            import array_api_compat.torch as torch
            return torch, device
        except ImportError:
            raise ImportError(
                "PyTorch is required for pytorch implementation. ")
    elif impl == 'cupy':
        try:
            import array_api_compat.cupy as cp
            return cp, device
        except ImportError:
            raise ImportError(
                "CuPy is required for cupy implementation. ")
    else:
        raise ValueError(f"Unsupported implementation: {impl}")

def create_scanner_geometry(odl_geometry, xp, dev):
    """Create a parallelproj scanner geometry from an ODL PET geometry.
    
    Parameters
    ----------
    odl_geometry : PETGeometry
        ODL PET geometry object.
    xp : module
        Array module.
    dev : str
        Device string.
        
    Returns
    -------
    scanner : parallelproj scanner geometry
        The parallelproj scanner geometry object.
    """
    if not PARALLELPROJ_AVAILABLE:
        raise ImportError("parallelproj has to be installed")
    
    if isinstance(xp, str):
        import_str = f"import array_api_compat.{xp} as xp"
        exec(import_str)
    
    from odl.applications.PET.geometry import PolygonPETGeometry
    if isinstance(odl_geometry, PolygonPETGeometry):
        scanner = parallelproj.RegularPolygonPETScannerGeometry(
            xp,
            dev,
            radius=odl_geometry.radius,
            num_sides=odl_geometry.num_sides,
            num_lor_endpoints_per_side=odl_geometry.num_lor_endpoints_per_side,
            lor_spacing=odl_geometry.lor_spacing,
            ring_positions=xp.asarray(
                odl_geometry.ring_positions, 
                device=dev
            ),
            symmetry_axis=odl_geometry.symmetry_axis,
        )
    else:
        raise NotImplementedError(
            f"Unsupported PET geometry type: {type(odl_geometry)}"
        )
    
    return scanner

def create_lor_descriptor(odl_geometry, scanner):
    """Create a parallelproj LOR descriptor from an ODL PET geometry.
    
    Parameters
    ----------
    odl_geometry : PETGeometry
        ODL PET geometry object.
    scanner : parallelproj scanner geometry
        The parallelproj scanner geometry.
    xp : module
        Array module (numpy or cupy).
        
    Returns
    -------
    lor_desc : parallelproj LOR descriptor
        The parallelproj LOR descriptor object.
    """
    if not PARALLELPROJ_AVAILABLE:
        raise ImportError("parallelproj is not available")
    
    from odl.applications.PET.geometry import PolygonPETGeometry
    if isinstance(odl_geometry, PolygonPETGeometry):
        lor_desc = parallelproj.RegularPolygonPETLORDescriptor(
            scanner,
            radial_trim=odl_geometry.radial_trim,
            max_ring_difference=odl_geometry.max_ring_difference,
            sinogram_order=parallelproj.SinogramSpatialAxisOrder.RVP,
        )
    else:
        raise NotImplementedError(
            f"Geometry type {type(odl_geometry)} is not yet supported"
        )
    
    return lor_desc

def create_projector(lor_descriptor, img_shape, voxel_size, img_origin=None):
    """Create a parallelproj projector.
    
    Parameters
    ----------
    lor_descriptor : parallelproj LOR descriptor
        The LOR descriptor defining the sinogram geometry.
    img_shape : tuple of int
        Shape of the image volume (nx, ny, nz).
    voxel_size : tuple of float
        Size of each voxel (dx, dy, dz).
    img_origin : tuple of float, optional
        Origin of the image volume. If None, the image is centered
        at (0, 0, 0).
        
    Returns
    -------
    proj : parallelproj projector
        The parallelproj projector object.
    """
    if not PARALLELPROJ_AVAILABLE:
        raise ImportError("parallelproj is not available")
    
    kwargs = {
        'lor_descriptor': lor_descriptor,
        'img_shape': img_shape,
        'voxel_size': voxel_size,
    }
    
    if img_origin is not None:
        kwargs['img_origin'] = img_origin
    
    # Determine projector type based on LOR descriptor type
    if isinstance(lor_descriptor, parallelproj.RegularPolygonPETLORDescriptor):
        proj = parallelproj.RegularPolygonPETProjector(**kwargs)
    else:
        raise NotImplementedError(
            f"LOR descriptor type {type(lor_descriptor)} is not yet supported"
        )
    
    return proj

def get_sinogram_shape(lor_descriptor):
    """Get the shape of the sinogram from a LOR descriptor.
    
    Parameters
    ----------
    lor_descriptor : parallelproj LOR descriptor
        The LOR descriptor.
        
    Returns
    -------
    shape : tuple of int
        Shape of the sinogram.
    """
    # parallelproj sinogram shape depends on the LOR descriptor
    # currently the default is set to RVP order
    # shape is (num_rad, num_views, num_planes)
    # TODO: add support for fetching plane descriptions from the LOR descriptor
    return lor_descriptor.spatial_sinogram_shape

def get_sinogram_shape_from_geometry(geometry, vol_space):

    xp,dev = get_array_module(vol_space)
    scanner = create_scanner_geometry(geometry, xp, dev)
    lor_desc = create_lor_descriptor(geometry, scanner)

    sinogram_shape = get_sinogram_shape(lor_desc)
    if geometry.tof_bins is not None:
        sinogram_shape = (*sinogram_shape, geometry.tof_bins)
    return sinogram_shape

def create_tof_parameters(geometry):
    """Create time of flight parameters from a geometry."""

    if geometry.tof_bins is None:
        raise ValueError("Time of flight bins are not set")

    return parallelproj.TOFParameters(num_tofbins=geometry.tof_bins)

def create_listmode_projector(start_coords, end_coords, img_shape, voxel_size):


    lm_proj = parallelproj.ListmodePETProjector(
        start_coords.data,
        end_coords.data,
        img_shape,
        voxel_size,
        img_origin=None
    )
    
    return lm_proj

if __name__ == '__main__':
    from odl.core.util.testutils import run_doctests
    run_doctests()