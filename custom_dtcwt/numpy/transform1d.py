from __future__ import absolute_import

import numpy as np
import logging

from six.moves import xrange

from dtcwt.coeffs import biort as _biort, qshift as _qshift
from dtcwt.defaults import DEFAULT_BIORT, DEFAULT_QSHIFT
from dtcwt.numpy.common import Pyramid
import numpy.lowlevel as c_dtcwt
from dtcwt.numpy.lowlevel_orig import colfilter, coldfilt, colifilt
from dtcwt.utils import as_column_vector, asfarray

def _zero_pad(l):
    return _zero_pad_0(l)

def _zero_pad_1(l):
    ret = []

    for elem in l:
        ret.append(elem)
        ret.append([0.0])
    return ret

def _zero_pad_0(l):
    ret = []

    for elem in l:
        ret.append([0.0])
        ret.append(elem)
    return ret

def _offset_and_pad(l, offset, step):
    ret = []
    for i in range(offset):
        ret.append([0.0])
    for elem in l:
        ret.append(elem)
        for i in range(step):
            ret.append([0.0])
    return ret


def print_lists(L):
    len_list = len(L[0])
    print('index', end='\t')
    for n in range(len(L)):
        print('list'+str(n), end='\t')
    print('\n')
    for k in range(len_list):
        print(k,end='\t')
        for l in L:
            print(l[k],end='\t')
        print('\n')




def _mix_list(l1, l2):
    ret = []
    for i in range(len(l1)):
        if i%2 ==1:
            ret.append(l1[i])
        else:
            ret.append(l2[i])
    if len(ret)%2 == 0:         ## Working with odd length filters avoids some complications
        ret.append([0.0])
    return ret

class Transform1d(object):
    """
    An implementation of the 1D DT-CWT in NumPy.

    :param biort: Level 1 wavelets to use. See :py:func:`dtcwt.coeffs.biort`.
    :param qshift: Level >= 2 wavelets to use. See :py:func:`dtcwt.coeffs.qshift`.

    """
    def __init__(self, biort=DEFAULT_BIORT, qshift=DEFAULT_QSHIFT):
        self.biort = biort
        self.qshift = qshift

    def forward_rec(self, X, nlevels=3, include_scale=False):
        """Perform a *n*-level DTCWT decompostion on a 1D column vector *X* (or on
        the columns of a matrix *X*).

        :param X: 1D real array or 2D real array whose columns are to be transformed
        :param nlevels: Number of levels of wavelet decomposition

        :returns: A :py:class:`dtcwt.Pyramid`-like object representing the transform result.

        If *biort* or *qshift* are strings, they are used as an argument to the
        :py:func:`biort` or :py:func:`qshift` functions. Otherwise, they are
        interpreted as tuples of vectors giving filter coefficients. In the *biort*
        case, this should be (h0o, g0o, h1o, g1o). In the *qshift* case, this should
        be (h0a, h0b, g0a, g0b, h1a, h1b, g1a, g1b).

        .. codeauthor:: Rich Wareham <rjw57@cantab.net>, Aug 2013
        .. codeauthor:: Nick Kingsbury, Cambridge University, May 2002
        .. codeauthor:: Cian Shaffrey, Cambridge University, May 2002

        """
        # Which wavelets are to be used?
        biort = self.biort
        qshift = self.qshift

        # Need this because colfilter and friends assumes input is 2d
        X = asfarray(X)
        if len(X.shape) == 1:
           X = np.atleast_2d(X).T

        # Try to load coefficients if biort is a string parameter
        try:
            h0o, g0o, h1o, g1o = _biort(biort)
        except TypeError:
            h0o, g0o, h1o, g1o = biort

        # Try to load coefficients if qshift is a string parameter
        try:
            h0a, h0b, g0a, g0b, h1a, h1b, g1a, g1b = _qshift(qshift)
        except TypeError:
            h0a, h0b, g0a, g0b, h1a, h1b, g1a, g1b = qshift

        L = np.asanyarray(X.shape)


        # ensure that X is an even length, thus enabling it to be extended if needs be.
        if X.shape[0] % 2 != 0:
            raise ValueError('Size of input X must be a multiple of 2')

        if nlevels == 0:
            if include_scale:
                return Pyramid(X, (), ())
            else:
                return Pyramid(X, ())

        # initialise
        Yh = [None,] * nlevels
        dec_Yh = [None,] * nlevels

        if include_scale:
            # This is only required if the user specifies scales are to be outputted
            Yscale = [None,] * nlevels
            dec_Yscale = [None,] * nlevels

        # Level 1.
        Hi = colfilter(X, h1o)
        Lo = colfilter(X, h0o)
        
        dec_Lo = Lo
        Lo_rec = Lo

        dec_Yh[0] = Hi[::2,:] + 1j*Hi[1::2,:] # Convert Hi to complex form.
        Yh[0] = np.roll(Hi, shift = 0, axis = 0) + 1j*np.roll(Hi, shift = -1, axis=0)

        print(f"Error at scale for Hi 0 : {np.sum(np.abs(Yh[0][::2,:] - dec_Yh[0]))}")

        if include_scale:
            Yscale[0] = Lo
            dec_Yscale[0] = Lo

        # Levels 2 and above.

        for level in xrange(1, nlevels):
            # Check to see if height of Lo is divisable by 4, if not extend.
            #if Lo.shape[0] % 4 != 0:
            #    Lo = np.vstack((Lo[0,:], Lo, Lo[-1,:]))

            dec_Hi = coldfilt(dec_Lo,h1b,h1a)
            dec_Lo = coldfilt(dec_Lo,h0b,h0a)

            Hi_rec = c_dtcwt.rec_coldfilt(Lo_rec, 2**(level-1), h1b, h1a)
            print(f"error between ground truth dec_Hi & Hi_rec : {np.sum(np.abs(Hi_rec[::2**level] - dec_Hi))}")
            Lo_rec = c_dtcwt.rec_coldfilt(Lo_rec, 2**(level-1), h0b, h0a)

            Lo = Lo_rec

            print(f"Error between decimated coefs and post decimated Hi at lvl {level} : {np.sum(np.abs(Hi_rec[::2**level] - dec_Hi))}")
            print(f"Error between decimated coefs and post decimated Lo at lvl {level} : {np.sum(np.abs(Lo[::2**level] - dec_Lo))}")

            Yh[level] = np.roll(Hi_rec, shift = 0, axis = 0) + 1j*np.roll(Hi_rec, shift = -2*2**(level-1), axis=0)
            Yh_dec = dec_Hi[::2,:] + 1j*dec_Hi[1::2,:] # Convert Hi to complex form.
            print(f"Error between true HP coefs & post decimated : {np.sum(np.abs(Yh_dec - Yh[level][::2**(1+level)]))} ")

            if include_scale:
                Yscale[level] = Lo

        Yl = Lo

        if include_scale:
            return Pyramid(Yl, Yh, Yscale)
        else:
            return Pyramid(Yl, Yh)
    

    def forward(self, X, nlevels=3, include_scale=False):
        """Perform a *n*-level DTCWT decompostion on a 1D column vector *X* (or on
        the columns of a matrix *X*).

        :param X: 1D real array or 2D real array whose columns are to be transformed
        :param nlevels: Number of levels of wavelet decomposition

        :returns: A :py:class:`dtcwt.Pyramid`-like object representing the transform result.

        If *biort* or *qshift* are strings, they are used as an argument to the
        :py:func:`biort` or :py:func:`qshift` functions. Otherwise, they are
        interpreted as tuples of vectors giving filter coefficients. In the *biort*
        case, this should be (h0o, g0o, h1o, g1o). In the *qshift* case, this should
        be (h0a, h0b, g0a, g0b, h1a, h1b, g1a, g1b).

        .. codeauthor:: Rich Wareham <rjw57@cantab.net>, Aug 2013
        .. codeauthor:: Nick Kingsbury, Cambridge University, May 2002
        .. codeauthor:: Cian Shaffrey, Cambridge University, May 2002

        """
        # Which wavelets are to be used?
        biort = self.biort
        qshift = self.qshift

        # Need this because colfilter and friends assumes input is 2d
        X = asfarray(X)
        if len(X.shape) == 1:
           X = np.atleast_2d(X).T

        # Try to load coefficients if biort is a string parameter
        try:
            h0o, g0o, h1o, g1o = _biort(biort)
        except TypeError:
            h0o, g0o, h1o, g1o = biort

        # Try to load coefficients if qshift is a string parameter
        try:
            h0a, h0b, g0a, g0b, h1a, h1b, g1a, g1b = _qshift(qshift)
        except TypeError:
            h0a, h0b, g0a, g0b, h1a, h1b, g1a, g1b = qshift

        print(f"h1a {h1a}")
        h1a_padded =    h1a
        h1b_padded =    h1b
        h0a_padded =    h0a
        h0b_padded =    h0b
        L = np.asanyarray(X.shape)


        # ensure that X is an even length, thus enabling it to be extended if needs be.
        if X.shape[0] % 2 != 0:
            raise ValueError('Size of input X must be a multiple of 2')

        if nlevels == 0:
            if include_scale:
                return Pyramid(X, (), ())
            else:
                return Pyramid(X, ())

        # initialise
        Yh = [None,] * nlevels
        dec_Yh = [None,] * nlevels

        if include_scale:
            # This is only required if the user specifies scales are to be outputted
            Yscale = [None,] * nlevels
            dec_Yscale = [None,] * nlevels

        # Level 1.
        Hi = colfilter(X, h1o)
        Lo = colfilter(X, h0o)
        dec_Lo = Lo
        Lo_dfilt = Lo
        Lo_cfilt = Lo
        Lo_rec = Lo
        dec_Yh[0] = Hi[::2,:] + 1j*Hi[1::2,:] # Convert Hi to complex form.
        
        Yh[0] = np.roll(Hi, shift = 0, axis = 0) + 1j*np.roll(Hi, shift = -1, axis=0)

        print(f"Error at scale for Hi 0 : {np.sum(np.abs(Yh[0][::2,:] - dec_Yh[0]))}")

        if include_scale:
            Yscale[0] = Lo
            dec_Yscale[0] = Lo

        n_rows = Lo.shape[0]
        print(f"n rows : {n_rows}")
        # Levels 2 and above.
        h0a_padded_0    =   h0a_padded
        h0b_padded_0    =   h0b_padded
        h0a_padded_1    =   h0a_padded
        h0b_padded_1    =   h0b_padded
        print(h0a*h0b)
        h0a_o   = h0a[0::2]
        h0a_e   = h0a[1::2]
        for level in xrange(1, nlevels):
            # Check to see if height of Lo is divisable by 4, if not extend.
            #if Lo.shape[0] % 4 != 0:
            #    Lo = np.vstack((Lo[0,:], Lo, Lo[-1,:]))

            h1a_padded = _zero_pad_1(h1a_padded)
            h1b_padded = _zero_pad_1(h1b_padded)
            h0a_padded_0 = _zero_pad_0(h0a_padded_0)
            h0b_padded_0 = _zero_pad_0(h0b_padded_0)

            h0a_o_padded = _offset_and_pad(h0a_o, offset=0, step = 2**level)
            h0a_e_padded = _offset_and_pad(h0a_e, offset=2, step = 2**level)

            if level%2 != 0:
                #h0a_padded_1 = _zero_pad_1(h0a_padded_1)
                #h0b_padded_1 = _zero_pad_0(h0b_padded_1)
                h0a_padded_1 = _offset_and_pad(h0a, offset=0, step=1)
                h0b_padded_1 = _offset_and_pad(h0b, offset=1, step=1)
            else:
                h0a_padded_1 = _offset_and_pad(h0a, offset=1, step=3)
                h0b_padded_1 = _offset_and_pad(h0b, offset=2, step=3)
                #h0a_padded_1 = _zero_pad_0(h0a_padded_1)
                #h0b_padded_1 = _zero_pad_1(h0b_padded_1)
            #print(f"h0a_padded_1 : {h0a_padded_1}\n h0b_padded_1 : {h0b_padded_1}")
            
            #h1a_padded = _zero_pad_0(h1a_padded)
            #h1b_padded = _zero_pad_1(h1b_padded)
            #h0a_padded = _zero_pad_1(h0a_padded)
            #h0b_padded = _zero_pad_0(h0b_padded)

            dec_Hi = coldfilt(dec_Lo,h1b,h1a)
            #print(f"len dec_Hi {len(dec_Hi)} \t len Lo {len(Lo)}")
            dec_Lo = coldfilt(dec_Lo,h0b,h0a)

            Hi_rec = c_dtcwt.rec_coldfilt(Lo_rec, 2**(level-1), h1b, h1a)
            print(f"error between ground truth dec_Hi & Hi_rec : {np.sum(np.abs(Hi_rec[::2**level] - dec_Hi))}")

            #print_lists([dec_Hi[:32], Hi_rec[:32]])
            

            Lo_rec = c_dtcwt.rec_coldfilt(Lo_rec, 2**(level-1), h0b, h0a)

            #print(f"error between ground truth dec_Lo & Lo_rec : {np.sum(np.abs(Lo_rec[::2**level] - dec_Lo))}")
            #print_lists([dec_Lo[:32], Lo_rec[:32]])

            #Lo_cfilt = c_dtcwt.coldfilt_c(Lo_cfilt, h0b_padded, h0a_padded)
            #print(f"error between ground truth dec_Lo & Lo_cfilt : {np.sum(np.abs(Lo_cfilt[::2**level] - dec_Lo))}")
            #print_lists([dec_Lo[:32], Lo_cfilt[:32]])

            #Lo_dfilt = c_dtcwt.coldfilt_d(Lo_dfilt, h0b, h0a, level-1, dec_Lo)
            #print(f"Lo_dfilt.shape {Lo_dfilt.shape}")
            #print_lists([dec_Lo[:32], Lo_dfilt[:32]])
            #merged_h1 = _mix_list(h1a, h1b)
            #merged_h0 = _mix_list(h0a, h0b)

            #merged_h1 = [h1b[i//2] + (1-i%2)*h1a[i//2] for i in range(2*len(h1a))]
            #merged_h0 = [h0b[i//2] + (1-i%2)*h0a[i//2] for i in range(2*len(h0a))]
            #print(f'Filters : \n h1a : {h1a} \n h1b : {h1b}\n merged h1 : {merged_h1}' )

            #Hi_merged = colfilter(Lo, merged_h1)
            #Lo_merged = colfilter(Lo, merged_h0)

            #Hia = colfilter(Lo, h1a)
            #Hib = colfilter(Lo, h1b)            
            #Hic = np.roll(Hia, shift = 0, axis = 0) + np.roll(Hib, shift = -1, axis=0)

            Hia_padded = colfilter(Lo, h1a_padded)
            Hib_padded = colfilter(Lo, h1b_padded)
            #print(f"len(Hia_padded) : {len(Hia_padded)} \nlen(Lo) : {len(Lo)} \n")
            #print(f"dec_Hi : {dec_Hi[:10]} \n Hia_padded : {Hia_padded[:10]} \n Hib_padded : {Hib_padded[:10]}")
            
            Hia_padded_rolled = np.roll(Hia_padded, shift =-2, axis = 0)
            Hib_padded_rolled = np.roll(Hib_padded, shift = 0, axis = 0)

            Loa_o = colfilter(Lo, h0a_o_padded)
            Loa_e = colfilter(Lo, h0a_e_padded)
            #print_lists([dec_Hi[:10], Hia_padded_rolled[:10], Hib_padded_rolled[:10]])

            #Hia_pr_subsampled = Hia_padded_rolled[::2]
            #Hib_pr_subsampled = Hib_padded_rolled[::2]
            Hid = [None,]*(len(Lo))
            idx = 0
            #for i in range(len(Lo)>>1):
            #    dec_Hid[2*i]    = Hia_padded_rolled[4*i] 
            #    dec_Hid[2*i+1]  = Hib_padded_rolled[4*i+1]
            for i in range(len(Lo)>>2):
                Hid[4*i]    = Hia_padded_rolled[4*i] 
                Hid[4*i+1]  = Hib_padded_rolled[4*i+2] 
                Hid[4*i+2]  = Hib_padded_rolled[4*i+1] 
                Hid[4*i+3]  = Hia_padded_rolled[4*i+3]

                
                
                #Hid[4*i]    = Hia_pr_subsampled[2*i] 
                #Hid[4*i+1]  = Hib_pr_subsampled[2*i] 
                #Hid[4*i+2]  = Hib_pr_subsampled[2*i+1] 
                #Hid[4*i+3]  = Hia_pr_subsampled[2*i+1]
                #if idx ==0:
                #    Hid[2*i] = Hia_pr_subsampled[i] 
                #    Hid[2*i+1] = Hia_pr_subsampled[i+1] 
                #    idx = 1
                #else :
                #    Hid[2*i] = Hib_pr_subsampled[i] 
                #    Hid[2*i+1] = Hib_pr_subsampled[i+1]
                #    idx = 0

            #Loa = colfilter(Lo, h0a)
            #Lob = colfilter(Lo, h0b) 
            #Loc = np.roll(Loa, shift = 0, axis = 0) + np.roll(Lob, shift = -1, axis=0)
           


            #Loa_padded_0 = colfilter(Lo, h0a_padded_0)
            #Lob_padded_0 = colfilter(Lo, h0b_padded_0)
            #Loa_padded_rolled_0 = np.roll(Loa_padded_0, shift =-2, axis = 0)
            #Lob_padded_rolled_0 = np.roll(Lob_padded_0, shift = -2, axis = 0)
            #
            #Loa_padded_1 = colfilter(Lo, h0a_padded_1)
            #Lob_padded_1 = colfilter(Lo, h0b_padded_1)
            #Loa_padded_rolled_1 = np.roll(Loa_padded_1, shift =-1, axis = 0)
            #Lob_padded_rolled_1 = np.roll(Lob_padded_1, shift = -1, axis = 0)
            ##Loa_pr_subsampled = Loa_padded_rolled[::2]
            ##Lob_pr_subsampled = Lob_padded_rolled[::2]
            #
            ##print_lists([dec_Lo[:16], Loa_padded_0[:16], Lob_padded_0[:16], Loa_padded_rolled_0[:16], Lob_padded_rolled_0[:16]])
#
            #Lod = [None,]*(len(Lo))
            #idx = 1
            ##for i in range(len(Lo)>>1):
            ##    dec_Lod[2*i]    = Lob_padded_rolled[4*i]
            ##    dec_Lod[2*i+1]  = Loa_padded_rolled[4*i+1]
            #Loa_padded_rolled = Loa_padded_rolled_1
            #Lob_padded_rolled = Lob_padded_rolled_1
            #for i in range(len(Lo)>>2):
            #    Lod[4*i]    = Lob_padded_rolled[4*i] 
            #    Lod[4*i+1]  = np.zeros_like(Lob_padded_rolled[4*i+2])#Lob_padded_rolled[4*i+2] 
            #    Lod[4*i+2]  = Loa_padded_rolled[4*i+1] 
            #    Lod[4*i+3]  = np.zeros_like(Loa_padded_rolled[4*i+3])               
                
                #if idx ==0:
                #    Lod[2*i]    = Loa_pr_subsampled[i] 
                #    Lod[2*i+1]  = Loa_pr_subsampled[i+1] 
                #    idx = 1
                #else :
                #    Lod[2*i]    = Lob_pr_subsampled[i] 
                #    Lod[2*i+1]  = Lob_pr_subsampled[i+1]
                #    idx = 0
            #print_lists([dec_Lo[0:16], Lod[0:16], Loa_padded_1[0:16], Lob_padded_1[0:16], Loa_padded_rolled_1[0:16], Lob_padded_rolled_1[0:16]])
            #print_lists([Loa_padded_rolled[0:16], Loa_e[0:16], Loa_o[0:16] ])
            Lo = Lo_rec
            #print(f"Hi_merged : {Hi_merged[:10]}\n dec_Hi : {dec_Hi[:10]}\n Hia : {Hia[:10]} \n Hib : {Hib[:10]} \n Hic : {Hic[:10]}\n Hia_padded : {Hia_padded[:10]} \n Hib_padded : {Hib_padded[:10]} \n\n Hia_padded_rolled : {Hia_padded_rolled[:10]} \n Hib_padded_rolled : {Hib_padded_rolled[:10]}\n Hia_pr_subsample : {Hia_pr_subsampled[:10]} \n Hib_pr_subsample : {Hib_pr_subsampled[:10]}\n Hid: {Hid[:10]}")
            #print("\n\nHi : ")
            #print(f"Hid : {Hid[:10]} \n dec_Hi {dec_Hi[:10]}")
            #print(f"Hid shape : {len(Hid)} \t dec_Hi shape : {dec_Hi.shape}")
            print(f"Error between decimated coefs and post decimated Hi at lvl {level} : {np.sum(np.abs(Hi_rec[::2**level] - dec_Hi))}")
            #print(f"dec_Hi : {str(dec_Hi[:10])}\n Hi : {str(Hid[:10])}")
#
            ##print(f"Lo_merged : {Lo_merged[:10]}\n dec_Lo : {dec_Lo[:10]}\n Loa : {Loa[:10]} \n Lob : {Lob[:10]} \n Loc : {Loc[:10]}\n Loa_padded : {Loa_padded[:10]} \n Lob_padded : {Lob_padded[:10]} \n\n Loa_padded_rolled : {Loa_padded_rolled[:10]} \n Lob_padded_rolled : {Lob_padded_rolled[:10]}\n Loa_pr_subsample : {Loa_pr_subsampled[:10]} \n Lob_pr_subsample : {Lob_pr_subsampled[:10]}\n Lod: {Lod[:10]}")
            ##print("\n\nLo")
            ##print(f"Lod : {Lod[:10]} \n dec_Lo {dec_Lo[:10]}")
            #print(f"Lod shape : {len(Lod)} \t dec_Lo shape : {dec_Lo.shape}")
            print(f"Error between decimated coefs and post decimated Lo at lvl {level} : {np.sum(np.abs(Lo[::2**level] - dec_Lo))}")
            #print(f"dec_Lo : {str(dec_Lo[:10])}\n Lo : {str(Lo[:10])}")
            #Loa = colfilter(Lo, h0a)
            #Lob = colfilter(Lo, h0b)
            #
            #Lo = Loa + 1j*Lob

            #Hi = colfilter(Lo,merged_h1)
            #Lo = colfilter(Lo,merged_h0)
            #print(f"Lo.shape : {Lo.shape}")
            #Yh[level] = np.roll(Hia, shift = 0, axis = 0) + 1j*np.roll(Hib, shift = -1, axis=0)
            #Yh[level] = np.roll(Hid, shift = 0, axis = 0) + 1j*np.roll(Hid, shift = -1, axis=0)
            Yh[level] = np.roll(Hi_rec, shift = 0, axis = 0) + 1j*np.roll(Hi_rec, shift = -2*2**(level-1), axis=0)
            Yh_dec = dec_Hi[::2,:] + 1j*dec_Hi[1::2,:] # Convert Hi to complex form.
            print_lists([Yh_dec[:32], Yh[level][:32]])
            print(f"Error between true HP coefs & post decimated : {np.sum(np.abs(Yh_dec - Yh[level][::2**(1+level)]))} ")
            #Yh[level] = (Hi[:-1:2,:] + 1j*Hi[1::2,:]).append(Hi[-1,:] + 1j*Hi[0,:]) # Convert Hi to complex form.
            #print(f'diff between roll and no roll {np.sum(np.abs(Yh[level][::2,:] - Hi[::2,:] - 1j*Hi[1::2,:]))}')
            if include_scale:
                Yscale[level] = Lo

        Yl = Lo

        if include_scale:
            return Pyramid(Yl, Yh, Yscale)
        else:
            return Pyramid(Yl, Yh)

    def inverse(self, pyramid, gain_mask=None):
        """Perform an *n*-level dual-tree complex wavelet (DTCWT) 1D
        reconstruction.

        :param pyramid: A :py:class:`dtcwt.Pyramid`-like object containing the transformed signal.
        :param gain_mask: Gain to be applied to each subband.

        :returns: Reconstructed real array.

        The *l*-th element of *gain_mask* is gain for wavelet subband at level l.
        If gain_mask[l] == 0, no computation is performed for band *l*. Default
        *gain_mask* is all ones. Note that *l* is 0-indexed.

        .. codeauthor:: Rich Wareham <rjw57@cantab.net>, Aug 2013
        .. codeauthor:: Nick Kingsbury, Cambridge University, May 2002
        .. codeauthor:: Cian Shaffrey, Cambridge University, May 2002

        """
        # Which wavelets are to be used?
        biort = self.biort
        qshift = self.qshift

        Yl = pyramid.lowpass
        Yh = pyramid.highpasses

        a = len(Yh) # No of levels.

        if gain_mask is None:
            gain_mask = np.ones(a) # Default gain_mask.

        # Try to load coefficients if biort is a string parameter
        try:
            h0o, g0o, h1o, g1o = _biort(biort)
        except TypeError:
            h0o, g0o, h1o, g1o = biort

        # Try to load coefficients if qshift is a string parameter
        try:
            h0a, h0b, g0a, g0b, h1a, h1b, g1a, g1b = _qshift(qshift)
        except TypeError:
            h0a, h0b, g0a, g0b, h1a, h1b, g1a, g1b = qshift

        level = a-1   # No of levels = no of rows in L.
        if level < 0:
            # if there are no levels in the input, just return the Yl value
            return Yl

        Lo = Yl
        while level >= 1:  # Reconstruct levels 2 and above in reverse order.
           Hi = c2q1d(Yh[level]*gain_mask[level])
           Lo = colifilt(Lo, g0b, g0a) + colifilt(Hi, g1b, g1a)

           if Lo.shape[0] != 2*Yh[level-1].shape[0]:  # If Lo is not the same length as the next Yh => t1 was extended.
              Lo = Lo[1:-1,...]                       # Therefore we have to clip Lo so it is the same height as the next Yh.

           if np.any(np.asanyarray(Lo.shape) != np.asanyarray(Yh[level-1].shape * np.array((2,1)))):
              raise ValueError('Yh sizes are not valid for DTWAVEIFM')

           level -= 1

        if level == 0:  # Reconstruct level 1.
           Hi = c2q1d(Yh[level]*gain_mask[level])
           Z = colfilter(Lo,g0o) + colfilter(Hi,g1o)

        # Return a 1d vector or a column vector
        if Z.shape[1] == 1:
            return Z.flatten()
        else:
            return Z

#==========================================================================================
#                  **********      INTERNAL FUNCTION    **********
#==========================================================================================

def c2q1d(x):
    """An internal function to convert a 1D Complex vector back to a real
    array,  which is twice the height of x.

    """
    a, b = x.shape
    z = np.zeros((a*2, b), dtype=x.real.dtype)
    z[::2, :] = np.real(x)
    z[1::2, :] = np.imag(x)

    return z

# vim:sw=4:sts=4:et

