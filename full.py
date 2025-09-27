'''FOURIER'''
import cv2
import numpy as np
import matplotlib.pyplot as plt

# ---------------- Fourier Transform ----------------
def fourier_transform(img):
    f = np.fft.fft2(img)
    fshift = np.fft.fftshift(f)   # Shift zero frequency to center
    magnitude = 20*np.log(np.abs(fshift) + 1)
    phase = np.angle(fshift)
    amplitude = np.abs(fshift)
    return f, fshift, magnitude, phase, amplitude

# Distance function
def distance(u, v, M, N):
    return np.sqrt((u - M/2)**2 + (v - N/2)**2)

# ---------------- Frequency Domain Filters ----------------
def ideal_low_pass(img_shape, D0):
    M, N = img_shape
    H = np.zeros((M, N))
    for u in range(M):
        for v in range(N):
            if distance(u, v, M, N) <= D0:
                H[u, v] = 1
    return H

def ideal_high_pass(img_shape, D0):
    return 1 - ideal_low_pass(img_shape, D0)

def gaussian_low_pass(img_shape, D0):
    M, N = img_shape
    H = np.zeros((M, N))
    for u in range(M):
        for v in range(N):
            D = distance(u, v, M, N)
            H[u, v] = np.exp(-(D**2) / (2*(D0**2)))
    return H

def gaussian_high_pass(img_shape, D0):
    return 1 - gaussian_low_pass(img_shape, D0)

def butterworth_low_pass(img_shape, D0, n=2):
    M, N = img_shape
    H = np.zeros((M, N))
    for u in range(M):
        for v in range(N):
            D = distance(u, v, M, N)
            H[u, v] = 1 / (1 + (D / D0)**(2*n))
    return H

def butterworth_high_pass(img_shape, D0, n=2):
    return 1 - butterworth_low_pass(img_shape, D0, n)

# Apply filter in frequency domain
def apply_filter(img, H):
    f = np.fft.fft2(img)
    fshift = np.fft.fftshift(f)
    G = fshift * H
    magnitude = 20*np.log(np.abs(G) + 1)   # Spectrum after filtering
    img_back = np.fft.ifft2(np.fft.ifftshift(G))
    img_back = np.abs(img_back)
    return img_back, magnitude

#----------- Spatial Domain Filters ----------------
def spatial_filters(img):
    results = {}

    # Low-pass filters
    results["Average Blur"] = cv2.blur(img, (5, 5))
    results["Gaussian Blur"] = cv2.GaussianBlur(img, (5, 5), 1)

    # High-pass filters
    laplacian = cv2.Laplacian(img, cv2.CV_64F)
    results["Laplacian (HPF)"] = cv2.convertScaleAbs(laplacian)

    sobelx = cv2.Sobel(img, cv2.CV_64F, 1, 0, ksize=3)
    sobely = cv2.Sobel(img, cv2.CV_64F, 0, 1, ksize=3)
    results["Sobel (HPF)"] = cv2.convertScaleAbs(sobelx + sobely)

    # Sharpening filter (High-boost like)
    kernel_sharpen = np.array([[0, -1, 0],
                               [-1, 5, -1],
                               [0, -1, 0]])
    results["Sharpen"] = cv2.filter2D(img, -1, kernel_sharpen)

    return results

# ---------------- MAIN ----------------
# Load two grayscale images
img1 = cv2.imread("/content/drive/MyDrive/Semester_9/Computer_Vision/CA 2 portions/Sun.jpg", 0)
img2 = cv2.imread("/content/drive/MyDrive/Semester_9/Computer_Vision/CA 2 portions/sun2.jpg", 0)

# Parameters
D0 = 30
n = 2

# Process both images
for idx, img in enumerate([img1, img2], start=1):
    f, fshift, magnitude, phase, amplitude = fourier_transform(img)

    # Plot Fourier components
    plt.figure(figsize=(12, 8))
    plt.subplot(2, 2, 1), plt.imshow(img, cmap='gray')
    plt.title(f"Original Image {idx}"), plt.axis('off')

    plt.subplot(2, 2, 2), plt.imshow(magnitude, cmap='gray')
    plt.title("Magnitude Spectrum"), plt.axis('off')

    plt.subplot(2, 2, 3), plt.imshow(phase, cmap='gray')
    plt.title("Phase Spectrum"), plt.axis('off')

    plt.subplot(2, 2, 4), plt.imshow(amplitude, cmap='gray')
    plt.title("Amplitude Spectrum"), plt.axis('off')
    plt.suptitle(f"Fourier Analysis for Image {idx}")
    plt.show()

    # ---------- Frequency Domain Filters ----------
    filters = {
        "Ideal LPF": ideal_low_pass(img.shape, D0),
        "Ideal HPF": ideal_high_pass(img.shape, D0),
        "Gaussian LPF": gaussian_low_pass(img.shape, D0),
        "Gaussian HPF": gaussian_high_pass(img.shape, D0),
        "Butterworth LPF": butterworth_low_pass(img.shape, D0, n),
        "Butterworth HPF": butterworth_high_pass(img.shape, D0, n)
    }

    plt.figure(figsize=(14, 10))
    plt.subplot(3, 3, 1), plt.imshow(img, cmap='gray')
    plt.title("Original"), plt.axis('off')

    plt.subplot(3, 3, 2), plt.imshow(magnitude, cmap='gray')
    plt.title("Original Spectrum"), plt.axis('off')

    i = 3
    for name, H in filters.items():
        filtered_img, filtered_mag = apply_filter(img, H)
        plt.subplot(3, 3, i), plt.imshow(filtered_img, cmap='gray')
        plt.title(name), plt.axis('off')
        i += 1

    plt.suptitle(f"Frequency Domain Filtering - Image {idx}")
    plt.tight_layout()
    plt.show()

    # ---------- Spatial Domain Filters ----------
    spatial_results = spatial_filters(img)
    plt.figure(figsize=(14, 10))
    plt.subplot(2, 3, 1), plt.imshow(img, cmap='gray')
    plt.title("Original"), plt.axis('off')

    i = 2
    for name, res in spatial_results.items():
        plt.subplot(2, 3, i), plt.imshow(res, cmap='gray')
        plt.title(name), plt.axis('off')
        i += 1

    plt.suptitle(f"Spatial Domain Filtering - Image {idx}")
    plt.tight_layout()
    plt.show()


'''Histogram'''

import cv2
import numpy as np
import matplotlib.pyplot as plt

def histogram_matching(source, reference):
    """
    Perform histogram specification (matching) of source image to reference image
    """
    # Compute histograms
    src_hist, bins = np.histogram(source.flatten(), 256, [0,256])
    ref_hist, bins = np.histogram(reference.flatten(), 256, [0,256])

    # Compute cumulative distribution function (CDF)
    src_cdf = np.cumsum(src_hist).astype(np.float32)
    src_cdf /= src_cdf[-1]  # normalize to [0,1]

    ref_cdf = np.cumsum(ref_hist).astype(np.float32)
    ref_cdf /= ref_cdf[-1]  # normalize to [0,1]

    # Create mapping
    mapping = np.zeros(256, dtype=np.uint8)
    for i in range(256):
        idx = np.argmin(np.abs(src_cdf[i] - ref_cdf))
        mapping[i] = idx

    # Apply mapping
    matched = cv2.LUT(source, mapping)
    return matched

# ---------------- Main ----------------

# Load grayscale images
source = cv2.imread("/content/drive/MyDrive/Semester_9/Computer_Vision/CA 2 portions/Sun.jpg", 0)
reference =cv2.imread("/content/drive/MyDrive/Semester_9/Computer_Vision/CA 2 portions/sun2.jpg", 0)

# Histogram Matching
matched = histogram_matching(source, reference)

# Plot results
plt.figure(figsize=(12,8))

plt.subplot(2,3,1), plt.imshow(source, cmap='gray'), plt.title("Source Image"), plt.axis('off')
plt.subplot(2,3,2), plt.imshow(reference, cmap='gray'), plt.title("Reference Image"), plt.axis('off')
plt.subplot(2,3,3), plt.imshow(matched, cmap='gray'), plt.title("Matched Image"), plt.axis('off')

# Histograms
plt.subplot(2,3,4), plt.hist(source.ravel(), bins=256, range=(0,256)), plt.title("Source Histogram")
plt.subplot(2,3,5), plt.hist(reference.ravel(), bins=256, range=(0,256)), plt.title("Reference Histogram")
plt.subplot(2,3,6), plt.hist(matched.ravel(), bins=256, range=(0,256)), plt.title("Matched Histogram")

plt.tight_layout()
plt.show()



'''Noise'''

import cv2
import numpy as np
import matplotlib.pyplot as plt
import math

# ---------------- Noise Models ----------------
def add_uniform_noise(img, low=-50, high=50):
    r, c = img.shape
    noise = np.random.uniform(low, high, (r, c))
    noisy = img.astype(np.float32) + noise
    return np.clip(noisy, 0, 255).astype(np.uint8)

def add_gaussian_noise(img, mean=0, sigma=25):
    r, c = img.shape
    noise = np.random.normal(mean, sigma, (r, c))
    noisy = img.astype(np.float32) + noise
    return np.clip(noisy, 0, 255).astype(np.uint8)

def add_rayleigh_noise(img, scale=30):
    r, c = img.shape
    noise = np.random.rayleigh(scale, (r, c))
    noisy = img.astype(np.float32) + noise
    return np.clip(noisy, 0, 255).astype(np.uint8)

def add_exponential_noise(img, scale=0.05):
    r, c = img.shape
    noise = np.random.exponential(scale, (r, c))
    noisy = img.astype(np.float32) + noise * 255.0
    return np.clip(noisy, 0, 255).astype(np.uint8)

def add_laplacian_noise(img, mean=0, scale=20):
    r, c = img.shape
    noise = np.random.laplace(mean, scale, (r, c))
    noisy = img.astype(np.float32) + noise
    return np.clip(noisy, 0, 255).astype(np.uint8)

def add_salt_pepper_noise(img, salt_prob=0.02, pepper_prob=0.02):
    noisy = img.copy()
    r, c = img.shape
    num_salt = int(np.ceil(salt_prob * img.size))
    num_pepper = int(np.ceil(pepper_prob * img.size))
    # Salt
    coords = [np.random.randint(0, i, num_salt) for i in img.shape]
    noisy[coords[0], coords[1]] = 255
    # Pepper
    coords = [np.random.randint(0, i, num_pepper) for i in img.shape]
    noisy[coords[0], coords[1]] = 0
    return noisy

def add_gamma_noise(img, shape=2.0, scale=1.0):
    r, c = img.shape
    noise = np.random.gamma(shape, scale, (r, c))
    noisy = img.astype(np.float32) + noise * 10.0
    return np.clip(noisy, 0, 255).astype(np.uint8)

def add_poisson_noise(img):
    # Simple Poisson model: scale intensities, draw Poisson, rescale
    img_float = img.astype(np.float32) / 255.0
    peak = 30.0  # photon scale (adjustable)
    lam = img_float * peak
    noisy = np.random.poisson(lam).astype(np.float32) / peak
    noisy = np.clip(noisy * 255.0, 0, 255).astype(np.uint8)
    return noisy

def add_additive_noise(img, mean=0, sigma=20):
    r, c = img.shape
    noise = np.random.normal(mean, sigma, (r, c))
    noisy = img.astype(np.float32) + noise
    return np.clip(noisy, 0, 255).astype(np.uint8)

def add_multiplicative_noise(img, mean=1.0, sigma=0.05):
    r, c = img.shape
    noise = np.random.normal(mean, sigma, (r, c))
    noisy = img.astype(np.float32) * noise
    return np.clip(noisy, 0, 255).astype(np.uint8)

# ---------------- Hardcoded Filters (spatial domain) ----------------

def pad_reflect(img, pad):
    return np.pad(img, pad, mode='reflect')

def convolve2d(img, kernel):
    k = kernel.shape[0]
    pad = k // 2
    padded = pad_reflect(img, pad)
    out = np.zeros_like(img, dtype=np.float32)
    for i in range(img.shape[0]):
        for j in range(img.shape[1]):
            region = padded[i:i+k, j:j+k]
            out[i, j] = np.sum(region * kernel)
    return np.clip(out, 0, 255)

def apply_mean_filter(img, kernel_size=3):
    kernel = np.ones((kernel_size, kernel_size), dtype=np.float32) / (kernel_size*kernel_size)
    out = convolve2d(img, kernel)
    return out.astype(np.uint8)

def apply_box_filter(img, kernel_size=3):
    return apply_mean_filter(img, kernel_size)  # same as mean

def apply_median_filter(img, kernel_size=3):
    pad = kernel_size // 2
    padded = pad_reflect(img, pad)
    out = np.zeros_like(img, dtype=np.uint8)
    for i in range(img.shape[0]):
        for j in range(img.shape[1]):
            region = padded[i:i+kernel_size, j:j+kernel_size].ravel()
            out[i, j] = np.median(region)
    return out

def gaussian_kernel(kernel_size=3, sigma=1.0):
    pad = kernel_size // 2
    ax = np.arange(-pad, pad+1, dtype=np.float32)
    xx, yy = np.meshgrid(ax, ax)
    kernel = np.exp(-(xx**2 + yy**2) / (2.0 * sigma**2))
    kernel /= np.sum(kernel)
    return kernel

def apply_gaussian_filter(img, kernel_size=3, sigma=1.0):
    kernel = gaussian_kernel(kernel_size, sigma)
    out = convolve2d(img, kernel)
    return out.astype(np.uint8)

def apply_min_filter(img, kernel_size=3):
    pad = kernel_size // 2
    padded = pad_reflect(img, pad)
    out = np.zeros_like(img, dtype=np.uint8)
    for i in range(img.shape[0]):
        for j in range(img.shape[1]):
            region = padded[i:i+kernel_size, j:j+kernel_size]
            out[i, j] = np.min(region)
    return out

def apply_max_filter(img, kernel_size=3):
    pad = kernel_size // 2
    padded = pad_reflect(img, pad)
    out = np.zeros_like(img, dtype=np.uint8)
    for i in range(img.shape[0]):
        for j in range(img.shape[1]):
            region = padded[i:i+kernel_size, j:j+kernel_size]
            out[i, j] = np.max(region)
    return out

# ---------------- Utility: histogram plotting ----------------
def plot_histogram(img, title=None):
    plt.hist(img.ravel(), bins=256, range=(0, 255), density=True, color='gray')
    if title: plt.title(title)

# ---------------- Main / Demo ----------------

# Load image (use a moderate-size image for performance)
img = cv2.imread("/content/drive/MyDrive/Semester_9/Computer_Vision/CA 2 portions/Sun.jpg", cv2.IMREAD_GRAYSCALE)
if img is None:
    raise FileNotFoundError("Image not found at the path.")

# Resize a bit for speed if large
max_size = 512
h, w = img.shape
if max(h, w) > max_size:
    scale = max_size / max(h, w)
    img = cv2.resize(img, (int(w*scale), int(h*scale)), interpolation=cv2.INTER_AREA)

# Generate noisy images (all requested types)
noises = {
    "Uniform": add_uniform_noise(img),
    "Gaussian": add_gaussian_noise(img),
    "Rayleigh": add_rayleigh_noise(img),
    "Exponential": add_exponential_noise(img),
    "Laplacian": add_laplacian_noise(img),
    "Salt & Pepper": add_salt_pepper_noise(img, salt_prob=0.02, pepper_prob=0.02),
    "Gamma": add_gamma_noise(img),
    "Poisson": add_poisson_noise(img),
    "Additive": add_additive_noise(img),
    "Multiplicative": add_multiplicative_noise(img)
}

# Filters to apply (hardcoded implementations)
filter_funcs = [
    ("Mean", apply_mean_filter),
    ("Median", apply_median_filter),
    ("Gaussian", apply_gaussian_filter),
    ("Min", apply_min_filter),
    ("Max", apply_max_filter)
]

# Show original + all noisy images
n = len(noises) + 1
cols = 4
rows = math.ceil(n / cols)
plt.figure(figsize=(4*cols, 3*rows))
plt.subplot(rows, cols, 1)
plt.imshow(img, cmap='gray'); plt.title("Original"); plt.axis('off')
for i, (name, noisy) in enumerate(noises.items(), start=2):
    plt.subplot(rows, cols, i)
    plt.imshow(noisy, cmap='gray'); plt.title(name); plt.axis('off')
plt.tight_layout()
plt.show()

# Plot histograms for original + noisy images (optional)
plt.figure(figsize=(4*cols, 3*rows))
plt.subplot(rows, cols, 1); plot_histogram(img, "Original Histogram")
for i, (name, noisy) in enumerate(noises.items(), start=2):
    plt.subplot(rows, cols, i); plot_histogram(noisy, f"{name} Hist")
plt.tight_layout()
plt.show()

# Apply filters (hard-coded) to every noisy image and display
for noise_name, noisy_img in noises.items():
    # pick kernel sizes: larger median for salt & pepper
    k_med = 5 if noise_name == "Salt & Pepper" else 3
    k = 5 if noise_name in ["Poisson", "Multiplicative"] else 3

    # apply
    mean_f = apply_mean_filter(noisy_img, kernel_size=k)
    median_f = apply_median_filter(noisy_img, kernel_size=k_med)
    gauss_f = apply_gaussian_filter(noisy_img, kernel_size=k, sigma=1.0)
    min_f = apply_min_filter(noisy_img, kernel_size=3)
    max_f = apply_max_filter(noisy_img, kernel_size=3)

    # show
    plt.figure(figsize=(15,6))
    plt.subplot(1,6,1), plt.imshow(noisy_img, cmap='gray'), plt.title(f"Noisy: {noise_name}"), plt.axis('off')
    plt.subplot(1,6,2), plt.imshow(mean_f, cmap='gray'), plt.title("Mean"), plt.axis('off')
    plt.subplot(1,6,3), plt.imshow(median_f, cmap='gray'), plt.title("Median"), plt.axis('off')
    plt.subplot(1,6,4), plt.imshow(gauss_f, cmap='gray'), plt.title("Gaussian"), plt.axis('off')
    plt.subplot(1,6,5), plt.imshow(min_f, cmap='gray'), plt.title("Min"), plt.axis('off')
    plt.subplot(1,6,6), plt.imshow(max_f, cmap='gray'), plt.title("Max"), plt.axis('off')
    plt.suptitle(f"Filtering results for: {noise_name}")
    plt.tight_layout()
    plt.show()


'''Unsharp'''

# Unsharp Masking & High-Boost Filtering for Grayscale (OpenCV)
# Uses only: cv2, numpy

import cv2
import numpy as np
import matplotlib.pyplot as plt

def unsharp_mask(gray, sigma=1.5, amount=1.5, threshold=0):
    """
    gray: uint8 grayscale image
    sigma: Gaussian blur sigma
    amount: strength of sharpening (typical 0.5–2.0)
    threshold: only sharpen edges with |mask| >= threshold (0–255)
    """
    gray_f = gray.astype(np.float32)
    # Gaussian blur
    blurred = cv2.GaussianBlur(gray_f, ksize=(0,0), sigmaX=sigma, sigmaY=sigma, borderType=cv2.BORDER_REPLICATE)
    # High-frequency mask
    mask = gray_f - blurred

    if threshold > 0:
        # Apply threshold to mask (soft gating)
        mabs = np.abs(mask)
        gate = (mabs >= threshold).astype(np.float32)
        mask = mask * gate

    # Add scaled mask back
    sharp = gray_f + amount * mask
    sharp = np.clip(sharp, 0, 255).astype(np.uint8)
    return sharp

def high_boost(gray, sigma=1.5, k=2.0):
    """
    gray: uint8 grayscale image
    sigma: Gaussian blur sigma
    k: high-boost factor (>1). k=1 reduces to original; larger = stronger boost.
    Formula: output = original + k * (original - blurred)
    """
    gray_f = gray.astype(np.float32)
    blurred = cv2.GaussianBlur(gray_f, ksize=(0,0), sigmaX=sigma, sigmaY=sigma, borderType=cv2.BORDER_REPLICATE)
    hb = gray_f + k * (gray_f - blurred)
    hb = np.clip(hb, 0, 255).astype(np.uint8)
    return hb

if _name_ == "_main_":
    img = cv2.imread("house.jpeg", cv2.IMREAD_GRAYSCALE)

    # Apply filters
    usm_soft  = unsharp_mask(img, sigma=1.0, amount=1.0, threshold=0)
    usm_crisp = unsharp_mask(img, sigma=1.5, amount=1.8, threshold=3)
    hb_mild   = high_boost(img, sigma=1.0, k=1.2)
    hb_strong = high_boost(img, sigma=1.5, k=2.5)

    # Plot results with matplotlib
    plt.figure(figsize=(12,8))

    plt.subplot(2,3,1)
    plt.imshow(img, cmap='gray')
    plt.title("Original")
    plt.axis('off')

    plt.subplot(2,3,2)
    plt.imshow(usm_soft, cmap='gray')
    plt.title("Unsharp Soft")
    plt.axis('off')

    plt.subplot(2,3,3)
    plt.imshow(usm_crisp, cmap='gray')
    plt.title("Unsharp Crisp")
    plt.axis('off')

    plt.subplot(2,3,4)
    plt.imshow(hb_mild, cmap='gray')
    plt.title("High-Boost Mild")
    plt.axis('off')

    plt.subplot(2,3,5)
    plt.imshow(hb_strong, cmap='gray')
    plt.title("High-Boost Strong")
    plt.axis('off')

    plt.tight_layout()
    plt.show()
