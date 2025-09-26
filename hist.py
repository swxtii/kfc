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
