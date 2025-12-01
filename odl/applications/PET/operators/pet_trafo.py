# Copyright 2014-2025 The ODL contributors
#
# This file is part of ODL.
#
# This Source Code Form is subject to the terms of the Mozilla Public License,
# v. 2.0. If a copy of the MPL was not distributed with this file, You can
# obtain one at https://mozilla.org/MPL/2.0/.

"""PET projection operators."""

from __future__ import annotations

from odl import Operator

from odl.applications.PET.backends.paraproj_impl import ParallelprojPETImpl
from odl.applications.PET.backends.paraproj_setup import get_sinogram_shape_from_geometry
__all__ = (
    'PETProjector',
)


def _default_proj_space(geometry, vol_space):
    import odl

    sino_shape = get_sinogram_shape_from_geometry(geometry, vol_space)
    proj_space = odl.uniform_discr(
        min_pt=[0] * len(sino_shape),
        max_pt=list(sino_shape),
        shape=sino_shape,
        dtype=vol_space.dtype,
        impl=vol_space.impl,
        device=vol_space.device
    )

    return proj_space


class PETProjector(Operator):
    """PET forward and backward projector.
    
    This operator performs forward projection (image -> sinogram) and
    provides an adjoint for backward projection (sinogram -> image).
    
    Parameters
    ----------
    vol_space : DiscretizedSpace
        The discretized image volume space (domain of forward projection).
    geometry : PETGeometry
        PET scanner geometry.
    proj_space : DiscretizedSpace, optional
        The discretized projection (sinogram) space. If not provided,
        it is inferred from the geometry.
    impl : str, optional
        Implementation backend. Currently only 'parallelproj' is supported.
        Default: 'parallelproj'.
    device : str, optional
        Device to use ('cpu' or 'cuda'). Default: 'cpu'.
    use_cache : bool, optional
        If True, cache the backend implementation. Default: True.
    """
    
    def __init__(
        self,
        vol_space,
        geometry,
        proj_space=None,
        impl='parallelproj',
        use_cache=True,
    ):
        self._geometry = geometry
        self._impl_name = impl
        self._use_cache = use_cache
        self._cached_impl = None
        
        self._proj_space = proj_space

        if self._proj_space is None:
            self._proj_space = _default_proj_space(geometry, vol_space)


        # Initialize operator
        super().__init__(domain=vol_space, range=self._proj_space, linear=True)
        
        # Store adjoint reference
        self._adjoint = None
    
    @property
    def geometry(self):
        """The PET geometry."""
        return self._geometry
    
    @property
    def impl(self):
        """Name of the implementation backend."""
        return self._impl_name
    
    @property
    def device(self):
        """Device used for computation."""
        return self._device
    
    @property
    def use_cache(self):
        """Whether the implementation is cached."""
        return self._use_cache
    
    def get_impl(self, use_cache=True):
        """Get or create the implementation backend.
        
        Parameters
        ----------
        use_cache : bool
            If True, return cached implementation if available.
            
        Returns
        -------
        impl : ParallelprojPETImpl
            The implementation backend.
        """
        if not use_cache or self._cached_impl is None:
            if self._impl_name == 'parallelproj':
                self._cached_impl = ParallelprojPETImpl(
                    self._geometry,
                    self.domain,
                    self.range
                )
            else:
                raise ValueError(
                    f"Unknown implementation: {self._impl_name}. "
                    f"Supported: 'parallelproj'"
                )
        
        return self._cached_impl
    
    def _call(self, x, out=None):
        """Forward projection.
        
        Parameters
        ----------
        x : domain element
            Image volume to project.
        out : range element, optional
            Output sinogram.
            
        Returns
        -------
        out : range element
            Forward projection (sinogram).
        """
        impl = self.get_impl(self._use_cache)
        return impl.call_forward(x, out)

    @property
    def adjoint(self):

        if self._adjoint is None:
            forward_op = self

            class PETBackProjection(Operator):
                def _call(self, x, out=None, **kwargs):
                    """Backprojection.

                    Parameters
                    ----------
                    x : DiscretizedSpaceElement
                        A sinogram. Must be an element of
                        `RayTransform.range` (domain of `RayBackProjection`).
                    out : `RayBackProjection.domain` element, optional
                        A volume to which the result of this evaluation is
                        written.

                    Returns
                    -------
                    DiscretizedSpaceElement
                        Result of the transform in the domain
                        of `RayProjection`.
                    """
                    return forward_op.get_impl(
                        forward_op.use_cache
                    ).call_backward(x, out)

                @property
                def geometry(self):
                    return forward_op.geometry

                @property
                def adjoint(self):
                    return forward_op

            self._adjoint = PETBackProjection(domain = self.range, range=self.domain, linear=True)
        return self._adjoint

    
    def show_geometry(self, ax=None, force_show=False):
        """Visualize the scanner and image geometry.
        
        Parameters
        ----------
        ax : matplotlib 3D axis, optional
            Axis to plot on.
            
        Returns
        -------
        fig : matplotlib figure
            The figure.
        """
        impl = self.get_impl(self._use_cache)
        fig = impl.show_geometry(ax)
        if force_show:
            fig.show()
        return fig
