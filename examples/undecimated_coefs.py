# %%
import numpy as np
import matplotlib.pylab as plt
import dtcwt as dt

def pad_to_power_of_two(image):
    M, N = image.shape
    new_size = 2 ** int(np.ceil(np.log2(max(M, N))))
    padded_image = np.zeros((new_size, new_size), dtype=image.dtype)
    padded_image[:M, :N] = image
    return padded_image

filename = 'aniso'
extension = '.npy'

matlab_key = 'key'
if extension == '.npy':
    input_im = np.load(filename+extension, allow_pickle=True)[()]

if extension == '.png' or extension == '.pgm' or extension =='.tiff' or extension == '.jpg':
    from PIL import Image ## To load png images
    im_frame = Image.open(filename+extension)
    input_im = np.array(im_frame)
if extension == '.mat':
    dict = loadmat(filename + extension,squeeze_me=True)
    input_im = dict[matlab_key]

if len(input_im.shape) > 2:
    input_im = input_im[:,:,0]

print(f'input im has shape {input_im.shape}')
resized_im = pad_to_power_of_two(input_im)

plt.figure()
plt.imshow(input_im, cmap= 'Greys')
plt.axis("off")

# %%

    
J_SCALES = 4
wavelet_coefs = dt.utils.compute_wavelet_coefs(image = resized_im, nlevels=J_SCALES)
print(wavelet_coefs)
# %%
for j in range(J_SCALES):
    fig, axs = plt.subplots(1,6, squeeze = False)
    for b in range(6):
        axs[0,b].imshow(np.abs(wavelet_coefs[b, ..., j]))
        axs[0,b].axis('off')
plt.figure()
for b in range(6):
    plt.plot(np.average(np.log(np.abs(wavelet_coefs[b])), axis = (0,1)))
# %%
