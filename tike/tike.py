#!/usr/bin/env python
# -*- coding: utf-8 -*-

# #########################################################################
# Copyright (c) 2017-2018, UChicago Argonne, LLC. All rights reserved.    #
#                                                                         #
# Copyright 2018. UChicago Argonne, LLC. This software was produced       #
# under U.S. Government contract DE-AC02-06CH11357 for Argonne National   #
# Laboratory (ANL), which is operated by UChicago Argonne, LLC for the    #
# U.S. Department of Energy. The U.S. Government has rights to use,       #
# reproduce, and distribute this software.  NEITHER THE GOVERNMENT NOR    #
# UChicago Argonne, LLC MAKES ANY WARRANTY, EXPRESS OR IMPLIED, OR        #
# ASSUMES ANY LIABILITY FOR THE USE OF THIS SOFTWARE.  If software is     #
# modified to produce derivative works, such modified software should     #
# be clearly marked, so as not to confuse it with the version available   #
# from ANL.                                                               #
#                                                                         #
# Additionally, redistribution and use in source and binary forms, with   #
# or without modification, are permitted provided that the following      #
# conditions are met:                                                     #
#                                                                         #
#     * Redistributions of source code must retain the above copyright    #
#       notice, this list of conditions and the following disclaimer.     #
#                                                                         #
#     * Redistributions in binary form must reproduce the above copyright #
#       notice, this list of conditions and the following disclaimer in   #
#       the documentation and/or other materials provided with the        #
#       distribution.                                                     #
#                                                                         #
#     * Neither the name of UChicago Argonne, LLC, Argonne National       #
#       Laboratory, ANL, the U.S. Government, nor the names of its        #
#       contributors may be used to endorse or promote products derived   #
#       from this software without specific prior written permission.     #
#                                                                         #
# THIS SOFTWARE IS PROVIDED BY UChicago Argonne, LLC AND CONTRIBUTORS     #
# "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT       #
# LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS       #
# FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL UChicago     #
# Argonne, LLC OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT,        #
# INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING,    #
# BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;        #
# LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER        #
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT      #
# LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN       #
# ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE         #
# POSSIBILITY OF SUCH DAMAGE.                                             #
# #########################################################################

"""Define the highest level functions for solving ptycho-tomography problem."""

from __future__ import (absolute_import, division, print_function,
                        unicode_literals)

import numpy as np
import logging
import tike.tomo
import tike.ptycho
from tike.constants import *
from mpi4py import MPI

__author__ = "Doga Gursoy, Daniel Ching"
__copyright__ = "Copyright (c) 2018, UChicago Argonne, LLC."
__docformat__ = 'restructuredtext en'
__all__ = ['admm',
           ]


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

comm = MPI.COMM_WORLD
mpi_rank = comm.Get_rank()
mpi_size = comm.Get_size()


def _combined_interface(
    obj,
    data,
    probe, theta, v, h,
    **kwargs
):
    """Define an interface that all functions in this module match."""
    assert np.all(obj_size > 0), "Detector dimensions must be > 0."
    assert np.all(probe_size > 0), "Probe dimensions must be > 0."
    assert np.all(detector_size > 0), "Detector dimensions must be > 0."
    assert theta.size == h.size == v.size == \
        detector_grid.shape[0] == probe_grid.shape[0], \
        "The size of theta, h, v must be the same as the number of probes."
    logger.info(" _ptycho_interface says {}".format("Hello, World!"))
    return None


def admm(
    obj=None, voxelsize=1.0,
    data=None,
    probe=None, theta=None, h=None, v=None, energy=None,
    niter=1, rho=0.5, gamma=0.25,
    **kwargs
):
    """Solve using the Alternating Direction Method of Multipliers (ADMM).

    Parameters
    ----------
    obj : (Z, X, Y, P) :py:class:`numpy.array` float
        The initial guess for the reconstruction.
    voxelsize : float [cm]
        The side length of an `obj` voxel.
    data : (M, H, V) :py:class:`numpy.array` float
        An array of detector intensities for each of the `M` probes. The
        grid of each detector is `H` pixels wide (the horizontal
        direction) and `V` pixels tall (the vertical direction).
    probe : (H, V) :py:class:`numpy.array` complex
        A single illumination function for the all probes.
    energy : float [keV]
        The energy of the probe
    algorithms : (2, ) string
        The names of the pytchography and tomography reconstruction algorithms.
    niter : int
        The number of ADMM interations.
    kwargs :
        Any keyword arguments for the pytchography and tomography
        reconstruction algorithms.

    """
    if mpi_rank == 0:
        Z, X, Y = obj.shape[0:3]
        T = theta.size
        x = obj
        psi = np.ones([T, Z, Y], dtype=obj.dtype)
        split_psi = np.array_split(psi, mpi_size)
        split_data = np.array_split(data, mpi_size)
        split_theta = np.array_split(theta, mpi_size)
        split_v = np.array_split(v, mpi_size)
        split_h = np.array_split(h, mpi_size)
    else:
        split_psi = None
        split_data = None
        split_theta = None
        split_v = None
        split_h = None
        x = None
    # Scatter psi, data, v, h, hobj, lamda
    psi = comm.scatter(split_psi, root=0)
    data = comm.scatter(split_data, root=0)
    theta = comm.scatter(split_theta, root=0)
    v = comm.scatter(split_v, root=0)
    h = comm.scatter(split_h, root=0)
    lamda = np.zeros_like(psi)
    hobj = np.ones_like(psi)
    # bcast probe, rho, gamma, voxelsize, energy
    probe = comm.bcast(probe, root=0)
    rho = comm.bcast(rho, root=0)
    gamma = comm.bcast(rho, root=0)
    voxelsize = comm.bcast(voxelsize, root=0)
    energy = comm.bcast(energy, root=0)

    logger.info("Rank {} gets {} views.".format(mpi_rank, len(theta)))

    for i in range(niter):
        logger.info("ADMM iteration {}".format(i))
        # Ptychography parallel
        for view in range(len(psi)):
            psi[view] = tike.ptycho.reconstruct(data=data[view],
                                                probe=probe,
                                                v=v[view], h=h[view],
                                                psi=psi[view],
                                                algorithm='grad',
                                                niter=1, rho=rho, gamma=gamma,
                                                reg=hobj[view],
                                                lamda=lamda[view], **kwargs)
        phi = -1j / wavenumber(energy) * np.log(psi + lamda / rho) / voxelsize
        # Recombine phi
        phi = comm.gather(phi, root=0)
        # Tomography single
        if mpi_rank == 0:
            phi = np.concatenate(phi, axis=0)
            x = tike.tomo.reconstruct(obj=x,
                                      theta=theta,
                                      line_integrals=phi,
                                      algorithm='grad', reg_par=-1,
                                      niter=1, **kwargs)
        # bcast x
        x = comm.bcast(x, root=0)
        # Lambda update.
        line_integrals = tike.tomo.forward(obj=x, theta=theta) * voxelsize
        hobj = np.exp(1j * wavenumber(energy) * line_integrals)
        lamda = lamda + rho * (psi - hobj)

        # # Update residuals.
        # r = 1 / M * np.sqrt(np.sum(np.square(np.abs(psi - hobj)),
        #                            axis=(-1, -2)))
        # s = rho * np.sqrt(np.sum(np.square(np.abs(x1 - x0)),
        #                          axis=(-1, -2)))
        # if r < epsilon_prime and s < epsilon_dual:
        #     pass
    return x
