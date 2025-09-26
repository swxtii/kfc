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
