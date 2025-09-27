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
