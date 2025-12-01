# Copyright 2014-2025 The ODL contributors
#
# This file is part of ODL.
#
# This Source Code Form is subject to the terms of the Mozilla Public License,
# v. 2.0. If a copy of the MPL was not distributed with this file, You can
# obtain one at https://mozilla.org/MPL/2.0/.

"""Backend for PET projections using parallelproj."""

import numpy as np
import odl

from odl.applications.PET.backends.paraproj_setup import (
    PARALLELPROJ_AVAILABLE,
    get_array_module,
    create_scanner_geometry,
    create_lor_descriptor,
    create_projector,
)

if PARALLELPROJ_AVAILABLE:
    import parallelproj

__all__ = (
    'ParallelprojPETImpl',
)


class ParallelprojPETImpl:
    """Implementation of PET projectors using parallelproj.
    
    This class wraps parallelproj's PET projectors to provide forward
    and backward projection operations compatible with ODL's operator
    interface.
    
    Parameters
    ----------
    geometry : PETGeometry
        ODL PET geometry defining the scanner configuration.
    vol_space : DiscretizedSpace
        ODL discretized space for the image volume.
    proj_space : DiscretizedSpace
        ODL discretized space for the projection data (sinogram).
    device : str, optional
        Device to use ('cpu' or 'cuda'). Default: 'cpu'.
        
    Attributes
    ----------
    geometry : PETGeometry
        The PET geometry.
    vol_space : DiscretizedSpace
        The volume (image) space.
    proj_space : DiscretizedSpace
        The projection (sinogram) space.
    projector : parallelproj projector
        The underlying parallelproj projector.
    """
    
    def __init__(self, geometry, vol_space, proj_space, device='cpu'):
        if not PARALLELPROJ_AVAILABLE:
            raise ImportError(
                "parallelproj is not available. "
                "Install with: pip install parallelproj"
            )
        
        self.geometry = geometry
        self._vol_space = vol_space
        self._proj_space = proj_space
        self._device = device
        
        # Get array module and device
        self._xp, self._dev = get_array_module(vol_space)

        # Create parallelproj objects: scanner, lor_descriptor and projector
        self._scanner = create_scanner_geometry(geometry, self._xp, self._dev)
        self._lor_desc = create_lor_descriptor(geometry, self._scanner)
        
        # Create parallelproj projector
        img_shape = vol_space.shape
        voxel_size = tuple(
            (vol_space.max_pt[i] - vol_space.min_pt[i]) / vol_space.shape[i]
            for i in range(vol_space.ndim)
        )
        
        img_origin = tuple(
            vol_space.min_pt[i] + voxel_size[i] / 2
            for i in range(vol_space.ndim)
        )
        
        self._projector = create_projector(
            self._lor_desc,
            img_shape=img_shape,
            voxel_size=voxel_size,
            img_origin=img_origin,
        )


    
    @property
    def vol_space(self):
        """The volume (image) space."""
        return self._vol_space
    
    @property
    def proj_space(self):
        """The projection (sinogram) space."""
        return self._proj_space
    
    @property
    def projector(self):
        """The underlying parallelproj projector."""
        return self._projector
    
    @property
    def scanner(self):
        """The parallelproj scanner geometry."""
        return self._scanner
    
    @property
    def lor_descriptor(self):
        """The parallelproj LOR descriptor."""
        return self._lor_desc
    
    def call_forward(self, vol_data, out=None):
        """Perform forward projection.
        
        Parameters
        ----------
        vol_data : vol_space element
            The image volume to project.
        out : proj_space element, optional
            Output array. If None, a new array is created.
            
        Returns
        -------
        out : proj_space element
            The forward projection (sinogram).
        """
        # Convert ODL element to array on the appropriate device
        # if self._device == 'cuda':
        #     x = self._xp.asarray(vol_data.asarray(), dtype=self._xp.float32)
        # else:
        #     x = np.asarray(vol_data.asarray(), dtype=np.float32)

        x = vol_data.asarray().copy()
        
        # Perform forward projection
        sino = self._projector(x)
        
        # Convert back to numpy if needed
        # if self._device == 'cuda':
        #     sino = parallelproj.to_numpy_array(sino)
        
        # Store in output
        if out is None:
            out = self._proj_space.element(sino)
        else:
            out[:] = sino
        
        return out
    
    def call_backward(self, proj_data, out=None):
        """Perform backward projection (sinogram -> image).
        
        Parameters
        ----------
        proj_data : proj_space element
            The sinogram to backproject.
        out : vol_space element, optional
            Output array. If None, a new array is created.
            
        Returns
        -------
        out : vol_space element
            The backprojection (image).
        """
        # Convert ODL element to array on the appropriate device
        # if self._device == 'cuda':
        #     y = self._xp.asarray(proj_data.asarray(), dtype=self._xp.float32)
        # else:
        #     y = np.asarray(proj_data.asarray(), dtype=np.float32)
        
        y = proj_data.asarray().copy()
        
        # Perform backward projection (adjoint)
        img = self._projector.adjoint(y)
        
        # Convert back to numpy if needed
        if self._device == 'cuda':
            img = parallelproj.to_numpy_array(img)
        
        # Store in output
        if out is None:
            out = self._vol_space.element(img)
        else:
            out[:] = img
        
        return out
    
    def show_geometry(self, ax=None):
        """Visualize the scanner and image geometry.
        
        Parameters
        ----------
        ax : matplotlib 3D axis, optional
            Axis to plot on. If None, a new figure is created.
            
        Returns
        -------
        fig : matplotlib figure
            The figure containing the plot.
        """
        import matplotlib.pyplot as plt
        
        if ax is None:
            fig = plt.figure(figsize=(10, 10))
            ax = fig.add_subplot(111, projection='3d')
        else:
            fig = ax.get_figure()
        
        self._projector.show_geometry(ax)
        
        return fig
