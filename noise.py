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