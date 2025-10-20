from __future__ import absolute_import

import numpy as np
import logging

from six.moves import xrange

from dtcwt.coeffs import biort as _biort, qshift as _qshift
from dtcwt.defaults import DEFAULT_BIORT, DEFAULT_QSHIFT
from dtcwt.utils import appropriate_complex_type_for, asfarray
import dtcwt.numpy.lowlevel as c_dtcwt
from dtcwt.numpy.common import Pyramid
from dtcwt.numpy.lowlevel import *

def _zero_pad(l):
    ret = []

    for elem in l:
        ret.append(elem)
        ret.append([0.0])
    return ret

def _mix_list(l1, l2):
    ret = []
    for i in range(len(l1)):
        if i%2 ==1:
            ret.append(l1[i])
        else:
            ret.append(l2[i])
    if len(ret)%2 == 0:
        ret.append([0.0])
    return ret

class Transform2d(object):
    """
    An implementation of the 2D DT-CWT via NumPy. *biort* and *qshift* are the
    wavelets which parameterise the transform.

    If *biort* or *qshift* are strings, they are used as an argument to the
    :py:func:`dtcwt.coeffs.biort` or :py:func:`dtcwt.coeffs.qshift` functions.
    Otherwise, they are interpreted as tuples of vectors giving filter
    coefficients. In the *biort* case, this should be (h0o, g0o, h1o, g1o). In
    the *qshift* case, this should be (h0a, h0b, g0a, g0b, h1a, h1b, g1a, g1b).

    """
    def __init__(self, biort=DEFAULT_BIORT, qshift=DEFAULT_QSHIFT):
        # Load bi-orthogonal wavelets
        try:
            self.biort = _biort(biort)
        except TypeError:
            self.biort = biort

        # Load quarter sample shift wavelets
        try:
            self.qshift = _qshift(qshift)
        except TypeError:
            self.qshift = qshift



    def forward_undec(self, X, nlevels=3, include_scale=False):
        """Perform a *n*-level undecimated DTCWT-2D decompostion on a 2D matrix *X*.

        :param X: 2D real array
        :param nlevels: Number of levels of wavelet decomposition

        :returns: A :py:class:`dtcwt.Pyramid` compatible object representing the transform-domain signal

        .. codeauthor:: Rich Wareham <rjw57@cantab.net>, Aug 2013
        .. codeauthor:: Nick Kingsbury, Cambridge University, Sept 2001
        .. codeauthor:: Cian Shaffrey, Cambridge University, Sept 2001

        """
        # If biort has 6 elements instead of 4, then it's a modified
        # rotationally symmetric wavelet
        # FIXME: there's probably a nicer way to do this
        if len(self.biort) == 4:
            h0o, g0o, h1o, g1o = self.biort
        elif len(self.biort) == 6:
            h0o, g0o, h1o, g1o, h2o, g2o = self.biort
        else:
            raise ValueError('Biort wavelet must have 6 or 4 components.')

        # If qshift has 12 elements instead of 8, then it's a modified
        # rotationally symmetric wavelet
        # FIXME: there's probably a nicer way to do this
        if len(self.qshift) == 8:
            h0a, h0b, g0a, g0b, h1a, h1b, g1a, g1b = self.qshift
        elif len(self.qshift) == 12:
            h0a, h0b, g0a, g0b, h1a, h1b, g1a, g1b, h2a, h2b = self.qshift[:10]
        else:
            raise ValueError('Qshift wavelet must have 12 or 8 components.')

        #X = np.atleast_2d(asfarray(X))
        X = np.atleast_2d(np.asarray(X))
        original_size = X.shape

        if len(X.shape) >= 3:
            raise ValueError('The entered image is {0}, which is invalid '.
                             format('x'.join(list(str(s) for s in X.shape))) +
                             'for the 2D transform in a numpy backend. ' +
                             'Please enter each image slice separately.')

        # The next few lines of code check to see if the image is odd in size, if so an extra ...
        # row/column will be added to the bottom/right of the image
        initial_row_extend = 0  #initialise
        initial_col_extend = 0
        if original_size[0] % 2 != 0:
            # if X.shape[0] is not divisable by 2 then we need to extend X by adding a row at the bottom
            X = np.vstack((X, X[[-1],:]))  # Any further extension will be done in due course.
            initial_row_extend = 1

        if original_size[1] % 2 != 0:
            # if X.shape[1] is not divisable by 2 then we need to extend X by adding a col to the left
            X = np.hstack((X, X[:,[-1]]))
            initial_col_extend = 1

        extended_size = X.shape

        if nlevels == 0:
            if include_scale:
                return Pyramid(X, (), ())
            else:
                return Pyramid(X, ())

        # initialise
        Yh = [None,] * nlevels
        Yh_std = [None,] * nlevels
        if include_scale:
            # this is only required if the user specifies a third output component.
            Yscale = [None,] * nlevels

        complex_dtype = appropriate_complex_type_for(X)

        if nlevels >= 1:
            # Do odd top-level filters on cols.
            Lo0 = colfilter(X,h0o).T
            Hi0 = colfilter(X,h1o).T
            if len(self.biort) >= 6:
                Ba0 = colfilter(X,h2o).T

            HiLo0 = colfilter(Hi0,h0o).T
            LoHi0 = colfilter(Lo0,h1o).T
            if len(self.biort) >= 6:
                BaBa0 = colfilter(Ba0,h2o).T     # Diagonal pair
                diag0 = BaBa0
            else:
                HiHi0 = colfilter(Hi0,h1o).T     # Diagonal pair    
                diag0 = HiHi0

            # Do odd top-level filters on rows.
            LoLo = colfilter(Lo0,h0o).T
            LoLo_std = LoLo
            Yh_std[0] = np.zeros((LoLo_std.shape[0] >> 1, LoLo_std.shape[1] >> 1, 6), dtype=complex_dtype)
            Yh_std[0][:,:,0:6:5] = q2c(HiLo0)     # Horizontal pair
            Yh_std[0][:,:,2:4:1] = q2c(LoHi0)     # Vertical pair
            Yh_std[0][:,:,1:5:3] = q2c(diag0)     # Diagonal pair

            Yh[0] = np.zeros((LoLo.shape[0], LoLo.shape[1], 6), dtype=complex_dtype)
            Yh[0][:,:,0:6:5] = q2c_rec(HiLo0, 1)     # Horizontal pair (dirs 0 & 5)
            Yh[0][:,:,2:4:1] = q2c_rec(LoHi0, 1)     # Vertical pair (dirs 2 & 3)
            Yh[0][:,:,1:5:3] = q2c_rec(diag0, 1)     # Diagonal pair (dirs 1 & 4)

            print(f"Error between Yh and Yh_std at lvl {0} : {np.sum(np.abs(Yh[0][::2, ::2, :] - Yh_std[0]))}")

            if include_scale:
                Yscale[0] = LoLo

        for level in xrange(1, nlevels):
            print(f"Current level of analysis is {level}")
            row_size, col_size = LoLo.shape
            if row_size % 4 != 0:
                # Extend by 2 rows if no. of rows of LoLo are not divisable by 4
                LoLo = np.vstack((LoLo[:1,:], LoLo, LoLo[-1:,:]))

            if col_size % 4 != 0:
                # Extend by 2 cols if no. of cols of LoLo are not divisable by 4
                LoLo = np.hstack((LoLo[:,:1], LoLo, LoLo[:,-1:]))

            # Do even Qshift filters on rows.
            Lo_std = coldfilt(LoLo_std, h0b, h0a).T
            Hi_std = coldfilt(LoLo_std, h1b, h1a).T
            if len(self.qshift) >=   12:
                Ba_std = coldfilt(LoLo_std, h2b, h2a).T

            HiLo_std = coldfilt(Hi_std, h0b, h0a).T
            LoHi_std = coldfilt(Lo_std, h1b, h1a).T
            LoLo_std = coldfilt(Lo_std, h0b, h0a).T
            if len(self.qshift) >=   12:
                BaBa_std = coldfilt(Ba_std, h2b,h2a).T
                diag_std =  BaBa_std
            else:
                HiHi_std =  coldfilt(Hi_std, h1b, h1a).T
                diag_std =  HiHi_std

            Yh_std[level] = np.zeros((LoLo_std.shape[0] >> 1, LoLo_std.shape[1] >> 1, 6), dtype=complex_dtype)
            Yh_std[level][:,:,0:6:5] = q2c(HiLo_std)     # Horizontal pair
            Yh_std[level][:,:,2:4:1] = q2c(LoHi_std)     # Vertical pair
            Yh_std[level][:,:,1:5:3] = q2c(diag_std)     # Diagonal pair
            
            LPFilter = lambda dat : coldfilt(dat, h0b, h0a)
            HPFilter = lambda dat : coldfilt(dat, h1b, h1a)
            if len(self.qshift) >= 12:
                DiagFilter = lambda dat : coldfilt(dat, h2b, h2a)
            else :
                DiagFilter = HPFilter

            Lo = np.zeros(original_size)
            Hi = np.zeros(original_size)
            if len(self.qshift) >= 12:
                Dg = np.zeros(original_size)

            c_dtcwt.undecimate(Lo, LPFilter, LoLo, level)
            #print(f"Lo : {Lo[:10,:10]}")
            #print(f"Lo_std : {Lo_std[:10,:10]}")
            c_dtcwt.undecimate(Hi, HPFilter, LoLo, level)
            if len(self.qshift) >= 12:
                c_dtcwt.undecimate(Dg, DiagFilter, LoLo, level)
            else :
                Dg = Hi
            
            LoLo = np.zeros(original_size)
            LoHi = np.zeros(original_size)
            HiLo = np.zeros(original_size)
            DgDg = np.zeros(original_size)

            c_dtcwt.undecimate(LoLo, LPFilter, Lo.T, level)
            c_dtcwt.undecimate(LoHi, HPFilter, Lo.T, level)
            c_dtcwt.undecimate(HiLo, LPFilter, Hi.T, level)
            c_dtcwt.undecimate(DgDg, DiagFilter, Dg.T, level)
            LoLo = LoLo.T
            LoHi = LoHi.T
            HiLo = HiLo.T
            DgDg = DgDg.T

            #print(f"Lo : {LoLo[:10,:10]}")
            #print(f"Lo_std : {LoLo_std[:10,:10]}")
            
            Yh[level] = np.zeros((LoLo.shape[0], LoLo.shape[1], 6), dtype=complex_dtype)
            c_dtcwt.undecimate2D(Yh[level][:,:,0:6:5], q2c, HiLo, level+1)  # Horizontal   (HiLo)
            c_dtcwt.undecimate2D(Yh[level][:,:,2:4:1], q2c, LoHi, level+1)  # Horizontal   (HiLo)
            c_dtcwt.undecimate2D(Yh[level][:,:,1:5:3], q2c, DgDg, level+1)  # Horizontal   (HiLo)

            if include_scale:
                Yscale[level] = LoLo

        Yl = LoLo

        if initial_row_extend == 1 and initial_col_extend == 1:
            logging.warn('The image entered is now a {0} NOT a {1}.'.format(
                'x'.join(list(str(s) for s in extended_size)),
                'x'.join(list(str(s) for s in original_size))))
            logging.warn(
                'The bottom row and rightmost column have been duplicated, prior to decomposition.')

        if initial_row_extend == 1 and initial_col_extend == 0:
            logging.warn('The image entered is now a {0} NOT a {1}.'.format(
                'x'.join(list(str(s) for s in extended_size)),
                'x'.join(list(str(s) for s in original_size))))
            logging.warn(
                'The bottom row has been duplicated, prior to decomposition.')

        if initial_row_extend == 0 and initial_col_extend == 1:
            logging.warn('The image entered is now a {0} NOT a {1}.'.format(
                'x'.join(list(str(s) for s in extended_size)),
                'x'.join(list(str(s) for s in original_size))))
            logging.warn(
                'The rightmost column has been duplicated, prior to decomposition.')

        if include_scale:
            return Pyramid(Yl, tuple(Yh), tuple(Yscale))
        else:
            return Pyramid(Yl, tuple(Yh))






    def forward(self, X, nlevels=3, include_scale=False):
        """Perform a *n*-level DTCWT-2D decompostion on a 2D matrix *X*.

        :param X: 2D real array
        :param nlevels: Number of levels of wavelet decomposition

        :returns: A :py:class:`dtcwt.Pyramid` compatible object representing the transform-domain signal

        .. codeauthor:: Rich Wareham <rjw57@cantab.net>, Aug 2013
        .. codeauthor:: Nick Kingsbury, Cambridge University, Sept 2001
        .. codeauthor:: Cian Shaffrey, Cambridge University, Sept 2001

        """
        # If biort has 6 elements instead of 4, then it's a modified
        # rotationally symmetric wavelet
        # FIXME: there's probably a nicer way to do this
        if len(self.biort) == 4:
            h0o, g0o, h1o, g1o = self.biort
        elif len(self.biort) == 6:
            h0o, g0o, h1o, g1o, h2o, g2o = self.biort
        else:
            raise ValueError('Biort wavelet must have 6 or 4 components.')

        # If qshift has 12 elements instead of 8, then it's a modified
        # rotationally symmetric wavelet
        # FIXME: there's probably a nicer way to do this
        if len(self.qshift) == 8:
            h0a, h0b, g0a, g0b, h1a, h1b, g1a, g1b = self.qshift
        elif len(self.qshift) == 12:
            h0a, h0b, g0a, g0b, h1a, h1b, g1a, g1b, h2a, h2b = self.qshift[:10]
        else:
            raise ValueError('Qshift wavelet must have 12 or 8 components.')

        X = np.atleast_2d(asfarray(X))
        original_size = X.shape

        if len(X.shape) >= 3:
            raise ValueError('The entered image is {0}, which is invalid '.
                             format('x'.join(list(str(s) for s in X.shape))) +
                             'for the 2D transform in a numpy backend. ' +
                             'Please enter each image slice separately.')

        # The next few lines of code check to see if the image is odd in size, if so an extra ...
        # row/column will be added to the bottom/right of the image
        initial_row_extend = 0  #initialise
        initial_col_extend = 0
        if original_size[0] % 2 != 0:
            # if X.shape[0] is not divisable by 2 then we need to extend X by adding a row at the bottom
            X = np.vstack((X, X[[-1],:]))  # Any further extension will be done in due course.
            initial_row_extend = 1

        if original_size[1] % 2 != 0:
            # if X.shape[1] is not divisable by 2 then we need to extend X by adding a col to the left
            X = np.hstack((X, X[:,[-1]]))
            initial_col_extend = 1

        extended_size = X.shape

        if nlevels == 0:
            if include_scale:
                return Pyramid(X, (), ())
            else:
                return Pyramid(X, ())

        # initialise
        Yh = [None,] * nlevels
        if include_scale:
            # this is only required if the user specifies a third output component.
            Yscale = [None,] * nlevels

        complex_dtype = appropriate_complex_type_for(X)

        if nlevels >= 1:
            # Do odd top-level filters on cols.
            Lo = colfilter(X,h0o).T
            Hi = colfilter(X,h1o).T
            if len(self.biort) >= 6:
                Ba = colfilter(X,h2o).T

            # Do odd top-level filters on rows.
            LoLo = colfilter(Lo,h0o).T
            Yh[0] = np.zeros((LoLo.shape[0] >> 1, LoLo.shape[1] >> 1, 6), dtype=complex_dtype)
            Yh[0][:,:,0:6:5] = q2c(colfilter(Hi,h0o).T)     # Horizontal pair
            Yh[0][:,:,2:4:1] = q2c(colfilter(Lo,h1o).T)     # Vertical pair
            if len(self.biort) >= 6:
                Yh[0][:,:,1:5:3] = q2c(colfilter(Ba,h2o).T)     # Diagonal pair
            else:
                Yh[0][:,:,1:5:3] = q2c(colfilter(Hi,h1o).T)     # Diagonal pair

            if include_scale:
                Yscale[0] = LoLo

        for level in xrange(1, nlevels):
            row_size, col_size = LoLo.shape
            if row_size % 4 != 0:
                # Extend by 2 rows if no. of rows of LoLo are not divisable by 4
                LoLo = np.vstack((LoLo[:1,:], LoLo, LoLo[-1:,:]))

            if col_size % 4 != 0:
                # Extend by 2 cols if no. of cols of LoLo are not divisable by 4
                LoLo = np.hstack((LoLo[:,:1], LoLo, LoLo[:,-1:]))

            # Do even Qshift filters on rows.
            Lo = coldfilt(LoLo,h0b,h0a).T
            Hi = coldfilt(LoLo,h1b,h1a).T
            if len(self.qshift) >= 12:
                Ba = coldfilt(LoLo,h2b,h2a).T

            # Do even Qshift filters on columns.
            LoLo = coldfilt(Lo,h0b,h0a).T

            Yh[level] = np.zeros((LoLo.shape[0]>>1, LoLo.shape[1]>>1, 6), dtype=complex_dtype)
            Yh[level][:,:,0:6:5] = q2c(coldfilt(Hi,h0b,h0a).T)  # Horizontal
            Yh[level][:,:,2:4:1] = q2c(coldfilt(Lo,h1b,h1a).T)  # Vertical
            if len(self.qshift) >= 12:
                Yh[level][:,:,1:5:3] = q2c(coldfilt(Ba,h2b,h2a).T)  # Diagonal
            else:
                Yh[level][:,:,1:5:3] = q2c(coldfilt(Hi,h1b,h1a).T)  # Diagonal

            if include_scale:
                Yscale[level] = LoLo

        Yl = LoLo

        if initial_row_extend == 1 and initial_col_extend == 1:
            logging.warn('The image entered is now a {0} NOT a {1}.'.format(
                'x'.join(list(str(s) for s in extended_size)),
                'x'.join(list(str(s) for s in original_size))))
            logging.warn(
                'The bottom row and rightmost column have been duplicated, prior to decomposition.')

        if initial_row_extend == 1 and initial_col_extend == 0:
            logging.warn('The image entered is now a {0} NOT a {1}.'.format(
                'x'.join(list(str(s) for s in extended_size)),
                'x'.join(list(str(s) for s in original_size))))
            logging.warn(
                'The bottom row has been duplicated, prior to decomposition.')

        if initial_row_extend == 0 and initial_col_extend == 1:
            logging.warn('The image entered is now a {0} NOT a {1}.'.format(
                'x'.join(list(str(s) for s in extended_size)),
                'x'.join(list(str(s) for s in original_size))))
            logging.warn(
                'The rightmost column has been duplicated, prior to decomposition.')

        if include_scale:
            return Pyramid(Yl, tuple(Yh), tuple(Yscale))
        else:
            return Pyramid(Yl, tuple(Yh))

    def inverse(self, pyramid, gain_mask=None):
        """Perform an *n*-level dual-tree complex wavelet (DTCWT) 2D
        reconstruction.

        :param pyramid: A :py:class:`dtcwt.Pyramid`-like class holding the transform domain representation to invert.
        :param gain_mask: Gain to be applied to each subband.

        :returns: A numpy-array compatible instance with the reconstruction.

        The (*d*, *l*)-th element of *gain_mask* is gain for subband with direction
        *d* at level *l*. If gain_mask[d,l] == 0, no computation is performed for
        band (d,l). Default *gain_mask* is all ones. Note that both *d* and *l* are
        zero-indexed.

        .. codeauthor:: Rich Wareham <rjw57@cantab.net>, Aug 2013
        .. codeauthor:: Nick Kingsbury, Cambridge University, May 2002
        .. codeauthor:: Cian Shaffrey, Cambridge University, May 2002

        """
        Yl = pyramid.lowpass
        print(Yl.shape)
        Yh = pyramid.highpasses

        a = len(Yh) # No of levels.

        if gain_mask is None:
            gain_mask = np.ones((6,a)) # Default gain_mask.

        gain_mask = np.array(gain_mask)

        # If biort has 6 elements instead of 4, then it's a modified
        # rotationally symmetric wavelet
        # FIXME: there's probably a nicer way to do this
        if len(self.biort) == 4:
            h0o, g0o, h1o, g1o = self.biort
        elif len(self.biort) == 6:
            h0o, g0o, h1o, g1o, h2o, g2o = self.biort
        else:
            raise ValueError('Biort wavelet must have 6 or 4 components.')

        # If qshift has 12 elements instead of 8, then it's a modified
        # rotationally symmetric wavelet
        # FIXME: there's probably a nicer way to do this
        if len(self.qshift) == 8:
            h0a, h0b, g0a, g0b, h1a, h1b, g1a, g1b = self.qshift
        elif len(self.qshift) == 12:
            h0a, h0b, g0a, g0b, h1a, h1b, g1a, g1b, h2a, h2b, g2a, g2b = self.qshift
        else:
            raise ValueError('Qshift wavelet must have 12 or 8 components.')

        current_level = a
        Z = Yl

        while current_level >= 2: # this ensures that for level 1 we never do the following
            lh = c2q(Yh[current_level-1][:,:,[0, 5]], gain_mask[[0, 5], current_level-1])
            hl = c2q(Yh[current_level-1][:,:,[2, 3]], gain_mask[[2, 3], current_level-1])
            hh = c2q(Yh[current_level-1][:,:,[1, 4]], gain_mask[[1, 4], current_level-1])

            print(f'Shapes : Z {Z.shape} \t,g0b : {g0b.shape}\t,g0a : {g0a.shape}\t  lh : {lh.shape} \t,g1b : {g1b.shape}\t ,g1a : {g1a.shape}')

            # Do even Qshift filters on columns.
            y1 = colifilt(Z,g0b,g0a) + colifilt(lh,g1b,g1a)

            if len(self.qshift) >= 12:
                y2 = colifilt(hl,g0b,g0a)
                y2bp = colifilt(hh,g2b,g2a)

                # Do even Qshift filters on rows.
                Z = (colifilt(y1.T,g0b,g0a) + colifilt(y2.T,g1b,g1a) + colifilt(y2bp.T, g2b, g2a)).T
            else:
                y2 = colifilt(hl,g0b,g0a) + colifilt(hh,g1b,g1a)

                # Do even Qshift filters on rows.
                Z = (colifilt(y1.T,g0b,g0a) + colifilt(y2.T,g1b,g1a)).T

            # Check size of Z and crop as required
            [row_size, col_size] = Z.shape
            S = 2*np.array(Yh[current_level-2].shape)
            if row_size != S[0]:    # check to see if this result needs to be cropped for the rows
                Z = Z[1:-1,:]
            if col_size != S[1]:    # check to see if this result needs to be cropped for the cols
                Z = Z[:,1:-1]

            if np.any(np.array(Z.shape) != S[:2]):
                raise ValueError('Sizes of highpasses are not valid for DTWAVEIFM2')

            current_level = current_level - 1

        if current_level == 1:
            lh = c2q(Yh[current_level-1][:,:,[0, 5]],gain_mask[[0, 5],current_level-1])
            hl = c2q(Yh[current_level-1][:,:,[2, 3]],gain_mask[[2, 3],current_level-1])
            hh = c2q(Yh[current_level-1][:,:,[1, 4]],gain_mask[[1, 4],current_level-1])

            # Do odd top-level filters on columns.
            y1 = colfilter(Z,g0o) + colfilter(lh,g1o)

            if len(self.biort) >= 6:
                y2 = colfilter(hl,g0o)
                y2bp = colfilter(hh,g2o)

                # Do odd top-level filters on rows.
                Z = (colfilter(y1.T,g0o) + colfilter(y2.T,g1o) + colfilter(y2bp.T, g2o)).T
            else:
                y2 = colfilter(hl,g0o) + colfilter(hh,g1o)

                # Do odd top-level filters on rows.
                Z = (colfilter(y1.T,g0o) + colfilter(y2.T,g1o)).T

        return Z

#==========================================================================================
#                       **********    INTERNAL FUNCTIONS    **********
#==========================================================================================

def q2c(y):
    """
    Convert from quads in y to complex numbers in z.
    """

    j2 = (np.sqrt(0.5) * np.array([1, 1j])).astype(appropriate_complex_type_for(y))

    # Arrange pixels from the corners of the quads into
    # 2 subimages of alternate real and imag pixels.
    #  a----b
    #  |    |
    #  |    |
    #  c----d

    # Combine (a,b) and (d,c) to form two complex subimages.
    p = y[0::2, 0::2]*j2[0] + y[0::2, 1::2]*j2[1] # p = (a + jb) / sqrt(2)
    q = y[1::2, 1::2]*j2[0] - y[1::2, 0::2]*j2[1] # q = (d - jc) / sqrt(2)

    # Form the 2 highpasses in z.
    z = np.dstack((p-q,p+q))

    return z

def q2c_rec(y, step):
    
    y = asfarray(y)
    complex_dtype = appropriate_complex_type_for(y)

    new_shape = list(y.shape)
    new_shape.append(2)
    ret = np.zeros(shape=new_shape, dtype=complex_dtype)
    slice_even = slice(0,y.shape[0], 2)
    slice_odd = slice(1,y.shape[0], 2)
    if step !=1:
        ret[slice_even,slice_even] = q2c_rec(y[slice_even,slice_even], step >> 1)
        ret[slice_even,slice_odd] = q2c_rec(y[slice_even,slice_odd], step >> 1)
        ret[slice_odd, slice_even] = q2c_rec(y[slice_odd,slice_even], step >> 1)
        ret[slice_odd, slice_odd] = q2c_rec(y[slice_odd,slice_odd], step >> 1)

        return ret
    else:
        #ret[slice_even,slice_even] = q2c(np.roll(y, axis = (0,1), shift = (0,0)))
        #ret[slice_even,slice_odd] = q2c(np.roll(y, axis = (0,1), shift = (0,-1)))
        #ret[slice_odd, slice_even] = q2c(np.roll(y,  axis = (0,1), shift = (-1,0)))
        #ret[slice_odd, slice_odd] = q2c(np.roll(y,  axis = (0,1), shift = (-1,-1)))
        ret = undec_q2c(y)
        return ret

def undec_q2c(y):
    """
    Convert from quads in y to complex numbers in z.
    """

    j2 = (np.sqrt(0.5) * np.array([1, 1j])).astype(appropriate_complex_type_for(y))

    # Arrange pixels from the corners of the quads into
    # 2 subimages of alternate real and imag pixels.
    #  a----b
    #  |    |
    #  |    |
    #  c----d

    # Combine (a,b) and (d,c) to form two complex subimages.
    p = y[0::2, 0::2]*j2[0] + y[0::2, 1::2]*j2[1] # p = (a + jb) / sqrt(2)
    q = y[1::2, 1::2]*j2[0] - y[1::2, 0::2]*j2[1] # q = (d - jc) / sqrt(2)

    #print(f" shifted q : {(y[1::2, 1::2]*j2[0])[:10, 0]}")
    #print(f" rolled q : {(np.roll(y, axis = (0,1), shift = (-1,-1))*j2[0])[:10,0]}")

    undec_p = np.roll(y, axis = 1, shift = 0)*j2[0] + np.roll(y, axis = 1, shift = -1)*j2[1] 
    undec_q = np.roll(y, axis = (0,1), shift = (-1,-1))*j2[0] - np.roll(y, axis = (0,1), shift = (-1,0))*j2[1] 
    
    #undec_q = np.roll(y, axis = 1, shift = -1)*j2[0] - np.roll(y, axis = 1, shift = 0)*j2[1] 

    #print(f"In undec_q2c() : diff between p and undec_p : {np.sum(np.abs(undec_p[::2,::2] - p))} ")
    #print(f"In undec_q2c() : diff between q and undec_q : {np.sum(np.abs(undec_q[::2,::2] - q))} ")

    # Form the 2 highpasses in z.
    z = np.dstack((p-q,p+q))
    undec_z = np.dstack((undec_p-undec_q,undec_p+undec_q))
    

    return undec_z
def q2c_bis(y, level):
    
    j2 = (np.sqrt(0.5) * np.array([1, 1j])).astype(appropriate_complex_type_for(y))

        # Combine (a,b) and (d,c) to form two complex subimages.
    p = np.roll(y, shift = 0, axis = 0)*j2[0] + 1j*np.roll(y, shift = -2*2**(level-1), axis=1)*j2[1] # p = (a + jb) / sqrt(2)
    p_std = y[0::2, 0::2]*j2[0] + y[0::2, 1::2]*j2[1] # p = (a + jb) / sqrt(2)

    print(f"In q2c_bis, error between std and post decimation for p : {np.sum(np.abs(p_std - p[::2**level, ::2**level]))}")
    q = np.roll(y, shift = (-2*2**(level-1),-2*2**(level-1)), axis = (0,1))*j2[0] - np.roll(y, shift = -2*2**(level-1), axis = 0)*j2[1] # q = (d - jc) / sqrt(2)

    # Form the 2 highpasses in z.
    z = np.dstack((p-q,p+q))

    return z

def c2q(w,gain):
    """
    Scale by gain and convert from complex w(:,:,1:2) to real quad-numbers
    in z.

    Arrange pixels from the real and imag parts of the 2 highpasses
    into 4 separate subimages .
     A----B     Re   Im of w(:,:,1)
     |    |
     |    |
     C----D     Re   Im of w(:,:,2)

    """

    x = np.zeros((w.shape[0] << 1, w.shape[1] << 1), dtype=w.real.dtype)

    sc = np.sqrt(0.5) * gain
    P = w[:,:,0]*sc[0] + w[:,:,1]*sc[1]
    Q = w[:,:,0]*sc[0] - w[:,:,1]*sc[1]

    # Recover each of the 4 corners of the quads.
    x[0::2, 0::2] = P.real  # a = (A+C)*sc
    x[0::2, 1::2] = P.imag  # b = (B+D)*sc
    x[1::2, 0::2] = Q.imag  # c = (B-D)*sc
    x[1::2, 1::2] = -Q.real # d = (C-A)*sc

    return x

