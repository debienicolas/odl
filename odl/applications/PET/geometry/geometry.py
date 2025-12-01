# Copyright 2014-2025 The ODL contributors
#
# This file is part of ODL.
#
# This Source Code Form is subject to the terms of the Mozilla Public License,
# v. 2.0. If a copy of the MPL was not distributed with this file, You can
# obtain one at https://mozilla.org/MPL/2.0/.

"""PET geometry classes for ODL."""

from builtins import object
import numpy as np



__all__ = (
    'PolygonPETGeometry',
)

class PolygonPETGeometry(object):
    """PET geometry based on a regular polygon scanner.
    
    This geometry represents a PET scanner as a regular polygon with
    detectors placed along the edges.
    
    Parameters
    ----------
    radius : float
        Radius of the scanner.
    num_sides : int
        Number of sides of the polygon.
    num_lor_endpoints_per_side : int
        Number of detector elements (LOR endpoints) per polygon side.
    lor_spacing : float
        Spacing between adjacent LOR endpoints.
    ring_positions : array-like
        Axial positions of the detector rings.
    radial_trim : int, optional
        Number of LORs to trim at the edges of the sinogram.
        Default: 65.
    max_ring_difference : int or None, optional
        Maximum ring difference for oblique LORs. If None, all ring
        combinations are allowed. Default: None.
    symmetry_axis : int, optional
        Axis of symmetry (0, 1, or 2). Default: 2 (z-axis).
        
    Examples
    --------
    Create a simple 5-ring PET scanner:
    
    >>> ring_positions = np.linspace(-10, 10, 5)
    >>> geom = RegularPolygonPETGeometry(
    ...     radius=65.0,
    ...     num_sides=12,
    ...     num_lor_endpoints_per_side=15,
    ...     lor_spacing=2.3,
    ...     ring_positions=ring_positions
    ... )
    """
    
    def __init__(
        self,
        radius,
        num_sides,
        num_lor_endpoints_per_side,
        lor_spacing,
        ring_positions,
        radial_trim=65,
        max_ring_difference=None,
        symmetry_axis=2,
    ):
        self._radius = float(radius)
        self._num_sides = int(num_sides)
        self._num_lor_endpoints_per_side = int(num_lor_endpoints_per_side)
        self._lor_spacing = float(lor_spacing)
        self._ring_positions = np.asarray(ring_positions, dtype=np.float32)
        self._radial_trim = int(radial_trim)
        self._max_ring_difference = max_ring_difference
        self._symmetry_axis = int(symmetry_axis)
        
    @property
    def radius(self):
        """Scanner radius"""
        return self._radius
    
    @property
    def num_sides(self):
        """Number of polygon sides."""
        return self._num_sides
    
    @property
    def num_lor_endpoints_per_side(self):
        """Number of LOR endpoints per side."""
        return self._num_lor_endpoints_per_side
    
    @property
    def lor_spacing(self):
        """Spacing between LOR endpoints"""
        return self._lor_spacing
    
    @property
    def ring_positions(self):
        """Axial positions of detector rings."""
        return self._ring_positions
    
    @property
    def radial_trim(self):
        """Number of radial elements to trim."""
        return self._radial_trim
    
    @property
    def max_ring_difference(self):
        """Maximum ring difference for oblique LORs."""
        return self._max_ring_difference
    
    @property
    def symmetry_axis(self):
        """Axis of symmetry (0, 1, or 2)."""
        return self._symmetry_axis
    
    @property
    def num_rings(self):
        """Number of rings."""
        return len(self.ring_positions)
    
    def __repr__(self):
        return (
            f"{self.__class__.__name__}(\n"
            f"    radius={self.radius},\n"
            f"    num_sides={self.num_sides},\n"
            f"    num_lor_endpoints_per_side={self.num_lor_endpoints_per_side},\n"
            f"    lor_spacing={self.lor_spacing},\n"
            f"    num_rings={self.num_rings},\n"
            f"    radial_trim={self.radial_trim},\n"
            f"    max_ring_difference={self.max_ring_difference}\n"
            f")"
        )
    
