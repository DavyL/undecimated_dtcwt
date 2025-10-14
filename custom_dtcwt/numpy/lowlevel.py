from __future__ import absolute_import, division

__all__ = [ 'colfilter', 'colifilt', 'coldfilt', ]

import numpy as np
from six.moves import xrange
from dtcwt.utils import as_column_vector, asfarray, appropriate_complex_type_for, reflect



def _centered(arr, newsize):
    # Return the center newsize portion of the array.
    # (Shamelessly cribbed from scipy.)
    newsize = np.asanyarray(newsize)
    currsize = np.array(arr.shape)
    startind = (currsize - newsize) // 2
    endind = startind + newsize
    myslice = [slice(startind[k], endind[k]) for k in range(len(endind))]
    return arr[tuple(myslice)]

# This is to allow easy replacement of these later with, possibly, GPU versions
_rfft = np.fft.rfft
_irfft = np.fft.irfft

def _column_convolve(X, h):
    """Convolve the columns of *X* with *h* returning only the 'valid' section,
    i.e. those values unaffected by zero padding. Irrespective of the ftype of
    *h*, the output will have the dtype of *X* appropriately expanded to a
    floating point type if necessary.

    We assume that h is small and so direct convolution is the most efficient.

    """
    Xshape = np.asanyarray(X.shape)
    h = h.flatten().astype(X.dtype)
    h_size = h.shape[0]

    full_size = X.shape[0] + h_size - 1
    Xshape[0] = full_size

    out = np.zeros(Xshape, dtype=X.dtype)
    for idx in xrange(h_size):
        out[idx:(idx+X.shape[0]),...] += X * h[idx]

    outShape = Xshape.copy()
    outShape[0] = abs(X.shape[0] - h_size) + 1
    return _centered(out, outShape)

def colfilter(X, h):
    """Filter the columns of image *X* using filter vector *h*, without decimation.
    If len(h) is odd, each output sample is aligned with each input sample
    and *Y* is the same size as *X*.  If len(h) is even, each output sample is
    aligned with the mid point of each pair of input samples, and Y.shape =
    X.shape + [1 0].

    :param X: an image whose columns are to be filtered
    :param h: the filter coefficients.
    :returns Y: the filtered image.

    .. codeauthor:: Rich Wareham <rjw57@cantab.net>, August 2013
    .. codeauthor:: Cian Shaffrey, Cambridge University, August 2000
    .. codeauthor:: Nick Kingsbury, Cambridge University, August 2000

    """

    # Interpret all inputs as arrays
    X = asfarray(X)
    h = as_column_vector(h)

    r, c = X.shape
    m = h.shape[0]
    m2 = np.fix(m*0.5)

    # Symmetrically extend with repeat of end samples.
    # Use 'reflect' so r < m2 works OK.
    xe = reflect(np.arange(-m2, r+m2, dtype=np.int32), -0.5, r-0.5)

    # Perform filtering on the columns of the extended matrix X(xe,:), keeping
    # only the 'valid' output samples, so Y is the same size as X if m is odd.
    Y = _column_convolve(X[xe,:], h)

    return Y

def coldfilt(X, ha, hb):
    """Filter the columns of image X using the two filters ha and hb =
    reverse(ha).  ha operates on the odd samples of X and hb on the even
    samples.  Both filters should be even length, and h should be approx linear
    phase with a quarter sample advance from its mid pt (i.e. :math:`|h(m/2)| >
    |h(m/2 + 1)|`).

    .. code-block:: text

                          ext        top edge                     bottom edge       ext
        Level 1:        !               |               !               |               !
        odd filt on .    b   b   b   b   a   a   a   a   a   a   a   a   b   b   b   b
        odd filt on .      a   a   a   a   b   b   b   b   b   b   b   b   a   a   a   a
        Level 2:        !               |               !               |               !
        +q filt on x      b       b       a       a       a       a       b       b
        -q filt on o          a       a       b       b       b       b       a       a

    The output is decimated by two from the input sample rate and the results
    from the two filters, Ya and Yb, are interleaved to give Y.  Symmetric
    extension with repeated end samples is used on the composite X columns
    before each filter is applied.

    Raises ValueError if the number of rows in X is not a multiple of 4, the
    length of ha does not match hb or the lengths of ha or hb are non-even.

    .. codeauthor:: Rich Wareham <rjw57@cantab.net>, August 2013
    .. codeauthor:: Cian Shaffrey, Cambridge University, August 2000
    .. codeauthor:: Nick Kingsbury, Cambridge University, August 2000

    """
    #print(f'Input given to coldfilt has shape : {X.shape}')
    # Make sure all inputs are arrays
    X = asfarray(X)
    ha = asfarray(ha)
    hb = asfarray(hb)

    r, c = X.shape
    if r % 4 != 0:
        raise ValueError('No. of rows in X must be a multiple of 4')

    if ha.shape != hb.shape:
        raise ValueError('Shapes of ha and hb must be the same')

    if ha.shape[0] % 2 != 0:
        raise ValueError('Lengths of ha and hb must be even')

    m = ha.shape[0]
    m2 = np.fix(m*0.5)

    # Set up vector for symmetric extension of X with repeated end samples.
    xe = reflect(np.arange(-m, r+m), -0.5, r-0.5)
    #print(f"xe is {xe}")

    # Select odd and even samples from ha and hb. Note that due to 0-indexing
    # 'odd' and 'even' are not perhaps what you might expect them to be.
    hao = as_column_vector(ha[0:m:2])
    hae = as_column_vector(ha[1:m:2])
    hbo = as_column_vector(hb[0:m:2])
    hbe = as_column_vector(hb[1:m:2])
    t = np.arange(5, r+2*m-2, 4)
    #print(f"t is {t}")

    r2 = r//2
    Y = np.zeros((r2,c), dtype=X.dtype)

    if np.sum(ha*hb) > 0:
        #print(f"In coldfilt() : Did not enter weird condition !")
        s1 = slice(0, r2, 2)
        s2 = slice(1, r2, 2)
    else:
        #Sprint(f"In coldfilt() : Entered weird condition !")
        s2 = slice(0, r2, 2)
        s1 = slice(1, r2, 2)

    #print(f"t is {t} and s1 is {str(s1)}")
    # Perform filtering on columns of extended matrix X(xe,:) in 4 ways.
    Y[s1,:] = _column_convolve(X[xe[t-1],:],hao) + _column_convolve(X[xe[t-3],:],hae)
    Y[s2,:] = _column_convolve(X[xe[t],:],hbo) + _column_convolve(X[xe[t-2],:],hbe)
    
    #print(f'Output of coldfilt has shape : {Y.shape}')

    return Y

    
def coldfilt_d(X, ha, hb, depth, dec_Lo):
    
    # Make sure all inputs are arrays
    X = asfarray(X)
    ha = asfarray(ha)
    hb = asfarray(hb)
    Y = np.zeros_like(X)


    step = 2**depth
    slices = [slice(offset, X.shape[0], 2*step) for offset in range(2*step)]

    ret = []
    for offset in range(step):
        Y[slices[2*offset],:] = coldfilt(X[2*offset::step], ha, hb)
        Y[slices[2*offset+1],:] = coldfilt(X[2*offset::step], hb, ha)


        #print(f"At step {offset} : values of coldfilt are {coldfilt(X[offset::step], ha, hb)[:16]}")
    print(f"In coldfilt_d() : error between ground truth dec_Lo & output[0] : {np.sum(np.abs(Y[slices[0],:] - dec_Lo))}")
    return Y



def coldfilt_c(X, ha, hb):
    """Filter the columns of image X using the two filters ha and hb =
    reverse(ha).  ha operates on the odd samples of X and hb on the even
    samples.  Both filters should be even length, and h should be approx linear
    phase with a quarter sample advance from its mid pt (i.e. :math:`|h(m/2)| >
    |h(m/2 + 1)|`).

    .. code-block:: text

                          ext        top edge                     bottom edge       ext
        Level 1:        !               |               !               |               !
        odd filt on .    b   b   b   b   a   a   a   a   a   a   a   a   b   b   b   b
        odd filt on .      a   a   a   a   b   b   b   b   b   b   b   b   a   a   a   a
        Level 2:        !               |               !               |               !
        +q filt on x      b       b       a       a       a       a       b       b
        -q filt on o          a       a       b       b       b       b       a       a

    The output is decimated by two from the input sample rate and the results
    from the two filters, Ya and Yb, are interleaved to give Y.  Symmetric
    extension with repeated end samples is used on the composite X columns
    before each filter is applied.

    Raises ValueError if the number of rows in X is not a multiple of 4, the
    length of ha does not match hb or the lengths of ha or hb are non-even.

    .. codeauthor:: Rich Wareham <rjw57@cantab.net>, August 2013
    .. codeauthor:: Cian Shaffrey, Cambridge University, August 2000
    .. codeauthor:: Nick Kingsbury, Cambridge University, August 2000

    """
    #print(f'Input given to coldfilt has shape : {X.shape}')
    # Make sure all inputs are arrays
    X = asfarray(X)
    ha = asfarray(ha)
    hb = asfarray(hb)

    r, c = X.shape
    if r % 4 != 0:
        raise ValueError('No. of rows in X must be a multiple of 4')

    if ha.shape != hb.shape:
        raise ValueError('Shapes of ha and hb must be the same')

    if ha.shape[0] % 2 != 0:
        raise ValueError('Lengths of ha and hb must be even')

    m = ha.shape[0]
    m2 = np.fix(m*0.5)

    # Set up vector for symmetric extension of X with repeated end samples.
    xe = reflect(np.arange(-m, r+m), -0.5, r-0.5)
    #print(f"xe is {xe}")

    # Select odd and even samples from ha and hb. Note that due to 0-indexing
    # 'odd' and 'even' are not perhaps what you might expect them to be.
    hao = as_column_vector(ha[0:m:2])
    hae = as_column_vector(ha[1:m:2])
    hbo = as_column_vector(hb[0:m:2])
    hbe = as_column_vector(hb[1:m:2])
    t = np.arange(5, r+2*m-2, 4)
    #print(f"t is {t}")

    r2 = r//2
    Y = np.zeros((r,c), dtype=X.dtype)

    if np.sum(ha*hb) > 0:
        s1 = slice(0, r, 4)
        s2 = slice(2, r, 4)
        s3 = slice(1, r, 4)
        s4 = slice(3, r, 4)
    else:
        print(f"In coldfilt_c() : Entered weird condition !")
        s2 = slice(0, r2, 2)
        s1 = slice(1, r2, 2)

    #print(f"t is {t} and s1 is {str(s1)}")
    # Perform filtering on columns of extended matrix X(xe,:) in 4 ways.
    Y[s1,:] = _column_convolve(X[xe[t-1],:],hao) + _column_convolve(X[xe[t-3],:],hae)
    Y[s2,:] = _column_convolve(X[xe[t],:],hbo) + _column_convolve(X[xe[t-2],:],hbe)
    Y[s3,:] = _column_convolve(X[xe[t-1],:],hae) + _column_convolve(X[xe[t-3],:],hao)
    Y[s4,:] = _column_convolve(X[xe[t],:],hbe) + _column_convolve(X[xe[t-2],:],hbo)
    
    #print(f'Output of coldfilt has shape : {Y.shape}')

    return Y

def undecimate(output, decimating_function, data, N):
    if output.shape != data.shape:
        print("ERROR : In undecimate() : output shape different from data shape.")
    if N < 1:
        N=1
        print("Warning : In undecimate() : N was negative")
    if N == 1:
        s0 = slice(0,data.shape[0], 2)
        s1 = slice(1,data.shape[0], 2)
        output[s0] = decimating_function(data)
        output[s1] = decimating_function(np.roll(data, -1, axis=0))
    else:
        s0 = slice(0,data.shape[0], 2)
        s1 = slice(1,data.shape[0], 2)
        undecimate(output[s0],decimating_function, data[s0],N-1)
        undecimate(output[s1],decimating_function, data[s1],N-1)

def undecimate2D(output, decimating_function, data, N):
    #if output.shape != data.shape:
    #    print("ERROR : In undecimate() : output shape different from data shape.")
    if N < 1:
        N=1
        print("Warning : In undecimate() : N was negative")
    if N == 1:
        s0 = slice(0,data.shape[0], 2)
        s1 = slice(1,data.shape[0], 2)
        output[s0,s0] = decimating_function(data)
        output[s1,s0] = decimating_function(np.roll(data, (-1,0), axis=(0,1)))
        output[s0,s1] = decimating_function(np.roll(data, (0,-1), axis=(0,1)))
        output[s1,s1] = decimating_function(np.roll(data, (-1,-1), axis=(0,1)))
    else:
        s0 = slice(0,data.shape[0], 2)
        s1 = slice(1,data.shape[0], 2)
        undecimate2D(output[s0,s0,:],decimating_function, data[s0,s0],N-1)
        undecimate2D(output[s1,s0,:],decimating_function, data[s1,s0],N-1)
        undecimate2D(output[s0,s1,:],decimating_function, data[s0,s1],N-1)
        undecimate2D(output[s1,s1,:],decimating_function, data[s1,s1],N-1)


def rec_coldfilt(data, step, ha, hb):
    
    #print(f"In rec_coldfilt() with step {step}")

    data = asfarray(data)

    ret = np.zeros_like(data)
    slice_even = slice(0,ret.shape[0], 2)
    slice_odd = slice(1,ret.shape[0], 2)
    if step !=1:
        ret[slice_even,:] = rec_coldfilt(data[0::2,:], step >> 1, ha, hb)
        ret[slice_odd, :] = rec_coldfilt(data[1::2,:], step >> 1, ha, hb)

        return ret
    else:
        ret[slice_even,:] = coldfilt(data, ha, hb)
        ret[slice_odd, :] = coldfilt(data, hb, ha)
        return ret

def rec_coldfilt_2d(data, step, ha, hb):
    
    #print(f"In rec_coldfilt_2d() with step {step}")

    data = asfarray(data)

    ret = np.zeros_like(data)
    slice_even = slice(0,ret.shape[0], 2)
    slice_odd = slice(1,ret.shape[0], 2)
    if step !=1:
        ret[slice_even,] = rec_coldfilt_2d(data[slice_even,], step >> 1, ha, hb)
        ret[slice_odd, ] = rec_coldfilt_2d(data[slice_odd,], step >> 1, ha, hb)
        #ret[slice_even, slice_even] =   rec_coldfilt_2d(data[slice_even,slice_even], step >> 1, ha, hb)
        #ret[slice_even,slice_odd] =     rec_coldfilt_2d(data[slice_even,slice_odd], step >> 1, ha, hb)        
        #ret[slice_odd, slice_even] =    rec_coldfilt_2d(data[slice_odd,slice_even], step >> 1, ha, hb)
        #ret[slice_odd,slice_odd] =      rec_coldfilt_2d(data[slice_odd,slice_odd], step >> 1, ha, hb)
        return ret
    else:
        #ret[slice_even, slice_odd]  = coldfilt(np.roll(data, shift=(0,0), axis = (0,1)), ha, hb)
        #ret[slice_even, slice_odd]  = coldfilt(np.roll(data, shift=(0,-1), axis = (0,1)), ha, hb)        
        #ret[slice_odd, slice_even]  = coldfilt(np.roll(data, shift=(-1,0), axis = (0,1)), ha, hb)
        #ret[slice_odd, slice_odd]   = coldfilt(np.roll(data, shift=(-1,-1), axis = (0,1)), ha, hb)
        ret[slice_even, ]  = coldfilt(np.roll(data, shift=(0), axis = (0)), ha, hb)
        ret[slice_odd, ]  = coldfilt(np.roll(data, shift=(-1), axis = (0)), ha, hb)        
        
        return ret


    """Filter the columns of image X using the two filters ha and hb =
    reverse(ha).  ha operates on the odd samples of X and hb on the even
    samples.  Both filters should be even length, and h should be approx linear
    phase with a quarter sample advance from its mid pt (i.e. :math:`|h(m/2)| >
    |h(m/2 + 1)|`).

    .. code-block:: text

                          ext        top edge                     bottom edge       ext
        Level 1:        !               |               !               |               !
        odd filt on .    b   b   b   b   a   a   a   a   a   a   a   a   b   b   b   b
        odd filt on .      a   a   a   a   b   b   b   b   b   b   b   b   a   a   a   a
        Level 2:        !               |               !               |               !
        +q filt on x      b       b       a       a       a       a       b       b
        -q filt on o          a       a       b       b       b       b       a       a

    The output is decimated by two from the input sample rate and the results
    from the two filters, Ya and Yb, are interleaved to give Y.  Symmetric
    extension with repeated end samples is used on the composite X columns
    before each filter is applied.

    Raises ValueError if the number of rows in X is not a multiple of 4, the
    length of ha does not match hb or the lengths of ha or hb are non-even.

    .. codeauthor:: Rich Wareham <rjw57@cantab.net>, August 2013
    .. codeauthor:: Cian Shaffrey, Cambridge University, August 2000
    .. codeauthor:: Nick Kingsbury, Cambridge University, August 2000

    """
    print(f'Input given to coldfilt has shape : {X.shape}')
    # Make sure all inputs are arrays
    X = asfarray(X)
    ha = asfarray(ha)
    hb = asfarray(hb)

    r, c = X.shape
    if r % 4 != 0:
        raise ValueError('No. of rows in X must be a multiple of 4')

    if ha.shape != hb.shape:
        raise ValueError('Shapes of ha and hb must be the same')

    if ha.shape[0] % 2 != 0:
        raise ValueError('Lengths of ha and hb must be even')

    m = ha.shape[0]
    m2 = np.fix(m*0.5)

    # Set up vector for symmetric extension of X with repeated end samples.
    xe = reflect(np.arange(-m, r+m), -0.5, r-0.5)
    print(f"xe is {xe}")

    # Select odd and even samples from ha and hb. Note that due to 0-indexing
    # 'odd' and 'even' are not perhaps what you might expect them to be.
    hao = as_column_vector(ha[0:m:2])
    hae = as_column_vector(ha[1:m:2])
    hbo = as_column_vector(hb[0:m:2])
    hbe = as_column_vector(hb[1:m:2])
    t = np.arange(5, r+2*m-2, 4)
    print(f"t is {t}")

    r2 = r//2
    Y = np.zeros((r2,c), dtype=X.dtype)

    if np.sum(ha*hb) > 0:
       s1 = slice(0, r2, 2)
       s2 = slice(1, r2, 2)
    else:
       s2 = slice(0, r2, 2)
       s1 = slice(1, r2, 2)

    #print(f"t is {t} and s1 is {str(s1)}")
    # Perform filtering on columns of extended matrix X(xe,:) in 4 ways.
    Y[s1,:] = _column_convolve(X[xe[t-1],:],hao) + _column_convolve(X[xe[t-3],:],hae)
    Y[s2,:] = _column_convolve(X[xe[t],:],hbo) + _column_convolve(X[xe[t-2],:],hbe)
    
    #print(f'Output of coldfilt has shape : {Y.shape}')

    return Y

def colifilt(X, ha, hb):
    """ Filter the columns of image X using the two filters ha and hb =
    reverse(ha).  ha operates on the odd samples of X and hb on the even
    samples.  Both filters should be even length, and h should be approx linear
    phase with a quarter sample advance from its mid pt (i.e `:math:`|h(m/2)| >
    |h(m/2 + 1)|`).

    .. code-block:: text

                          ext       left edge                      right edge       ext
        Level 2:        !               |               !               |               !
        +q filt on x      b       b       a       a       a       a       b       b
        -q filt on o          a       a       b       b       b       b       a       a
        Level 1:        !               |               !               |               !
        odd filt on .    b   b   b   b   a   a   a   a   a   a   a   a   b   b   b   b
        odd filt on .      a   a   a   a   b   b   b   b   b   b   b   b   a   a   a   a

    The output is interpolated by two from the input sample rate and the
    results from the two filters, Ya and Yb, are interleaved to give Y.
    Symmetric extension with repeated end samples is used on the composite X
    columns before each filter is applied.

    .. codeauthor:: Rich Wareham <rjw57@cantab.net>, August 2013
    .. codeauthor:: Cian Shaffrey, Cambridge University, August 2000
    .. codeauthor:: Nick Kingsbury, Cambridge University, August 2000

    """
    # Make sure all inputs are arrays
    X = asfarray(X)
    ha = asfarray(ha)
    hb = asfarray(hb)

    r, c = X.shape
    if r % 2 != 0:
        raise ValueError('No. of rows in X must be a multiple of 2')

    if ha.shape != hb.shape:
        raise ValueError('Shapes of ha and hb must be the same')

    if ha.shape[0] % 2 != 0:
        raise ValueError('Lengths of ha and hb must be even')

    m = ha.shape[0]
    m2 = np.fix(m*0.5)

    Y = np.zeros((r*2,c), dtype=X.dtype)
    if not np.any(np.nonzero(X[:])[0]):
        return Y

    if m2 % 2 == 0:
        # m/2 is even, so set up t to start on d samples.
        # Set up vector for symmetric extension of X with repeated end samples.
        # Use 'reflect' so r < m2 works OK.
        xe = reflect(np.arange(-m2, r+m2, dtype=np.int32), -0.5, r-0.5)

        t = np.arange(3, r+m, 2)
        if np.sum(ha*hb) > 0:
            ta = t
            tb = t - 1
        else:
            ta = t - 1
            tb = t

        # Select odd and even samples from ha and hb. Note that due to 0-indexing
        # 'odd' and 'even' are not perhaps what you might expect them to be.
        hao = as_column_vector(ha[0:m:2])
        hae = as_column_vector(ha[1:m:2])
        hbo = as_column_vector(hb[0:m:2])
        hbe = as_column_vector(hb[1:m:2])

        s = np.arange(0,r*2,4)

        Y[s,:]   = _column_convolve(X[xe[tb-2],:],hae)
        Y[s+1,:] = _column_convolve(X[xe[ta-2],:],hbe)
        Y[s+2,:] = _column_convolve(X[xe[tb  ],:],hao)
        Y[s+3,:] = _column_convolve(X[xe[ta  ],:],hbo)
    else:
        # m/2 is odd, so set up t to start on b samples.
        # Set up vector for symmetric extension of X with repeated end samples.
        # Use 'reflect' so r < m2 works OK.
        xe = reflect(np.arange(-m2, r+m2, dtype=np.int32), -0.5, r-0.5)

        t = np.arange(2, r+m-1, 2)
        if np.sum(ha*hb) > 0:
            ta = t
            tb = t - 1
        else:
            ta = t - 1
            tb = t

        # Select odd and even samples from ha and hb. Note that due to 0-indexing
        # 'odd' and 'even' are not perhaps what you might expect them to be.
        hao = as_column_vector(ha[0:m:2])
        hae = as_column_vector(ha[1:m:2])
        hbo = as_column_vector(hb[0:m:2])
        hbe = as_column_vector(hb[1:m:2])

        s = np.arange(0,r*2,4)

        Y[s,:]   = _column_convolve(X[xe[tb],:],hao)
        Y[s+1,:] = _column_convolve(X[xe[ta],:],hbo)
        Y[s+2,:] = _column_convolve(X[xe[tb],:],hae)
        Y[s+3,:] = _column_convolve(X[xe[ta],:],hbe)

    return Y

# vim:sw=4:sts=4:et

