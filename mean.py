from scipy.ndimage import uniform_filter, median_filter, maximum_filter, minimum_filter, gaussian_filter
import numpy as np
import matplotlib.pyplot as plt
import cv2
def add_gaussian(img):
  h, w = img.shape
  noise = np.random.normal(0, 25, (h, w)).astype(np.float32)
  n_img = img + noise
  return np.clip(n_img, 0, 255)

def add_gamma(img):
  shape = 2
  scale = 1
  noise = np.random.gamma(shape, scale, img.shape).astype(np.float32)
  n_img = img + noise*10
  return np.clip(n_img, 0, 255)

def add_uniform(img):
  noise = np.random.uniform(-50, 50, img.shape).astype(np.float32)
  n_img = img+noise
  return np.clip(n_img, 0, 255)

def add_exp(img):
  noise = np.random.exponential(0.05, img.shape)
  n_img = img + noise*255
  return np.clip(n_img, 0, 255)

def add_ray(img):
  noise = np.random.rayleigh(30, img.shape)
  n_img = img*noise
  return np.clip(n_img, 0, 255)

def add_lap(img):
  noise = np.random.laplace(0, 20, img.shape)
  n_img = img + noise
  return np.clip(n_img, 0, 255)

def add_poisson(img):
  vals = len(np.unique(img))
  vals = 2**np.ceil(np.log2(vals))
  noisy = np.random.poisson(img*vals)/float(vals)
  return np.clip(noisy, 0, 255)

def add_s_p(img):
  noisy = img.copy()
  num_salt = np.ceil(0.4 * img.size)
  coords = [np.random.randint(0, i-1, int(num_salt)) for i in img.shape]
  noisy[coords[0], coords[1]]= 255
  num_salt = np.ceil(0.4 * img.size)
  coords = [np.random.randint(0, i-1, int(num_salt)) for i in img.shape]
  noisy[coords[0], coords[1]]= 0
  return np.clip(noisy, 0, 255)

def apply_filters(img):
  uf = uniform_filter(img,3, mode= 'reflect')
  gf = gaussian_filter(img,1, mode= 'reflect')
  med_f = median_filter(img,3, mode= 'reflect')
  min_f = minimum_filter(img,3, mode= 'reflect')
  max_f = maximum_filter(img,3, mode= 'reflect')
  bl = cv2.bilateralFilter(img.astype(np.uint8), d=9, sigmaColor=75, sigmaSpace=75)
  #bl = maximum_filter(img,3, mode= 'reflect')
  return uf, gf, med_f, min_f, max_f, bl

img = cv2.imread('sunflower2.jpg', cv2.IMREAD_GRAYSCALE)
noises= {
    'gauss' : add_gaussian(img),
    'gamma' : add_gamma(img),
    'uni' : add_uniform(img),
    'exp' : add_exp(img),
    'ray' : add_ray(img),
    'lap' : add_lap(img),
    's_p' : add_s_p(img),
    'poisson' : add_poisson(img)
}
for noise in noises:
  plt.figure(figsize = (5, 10))
  plt.subplot(1,2,1)
  plt.imshow(img, cmap='gray')
  plt.subplot(1,2,2)
  plt.imshow(noises[noise], cmap='gray')
  plt.title(noise)

  uf, gf, med_f, min_f, max_f, bl = apply_filters(noises[noise])
  plt.figure(figsize = (12, 10))
  plt.subplot(2,3,1)
  plt.imshow(uf, cmap='gray')
  plt.subplot(2,3,2)
  plt.imshow(gf, cmap='gray')
  plt.subplot(2,3,3)
  plt.imshow(med_f, cmap='gray')
  plt.subplot(2,3,4)
  plt.imshow(min_f, cmap='gray')
  plt.subplot(2,3,5)
  plt.imshow(max_f, cmap='gray')
  plt.subplot(2,3,6)
  plt.imshow(bl, cmap='gray')

  hist1 = find_histogram(noises[noise].astype(np.uint8))
  plt.figure(figsize=(8,4))
  print_histogram(hist1)
  plt.title(f"Histogram - {noise}")
  plt.show()