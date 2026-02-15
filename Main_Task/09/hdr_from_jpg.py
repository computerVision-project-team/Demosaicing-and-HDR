import os
import numpy as np
import cv2
from PIL import Image


THIS_DIR = os.path.dirname(__file__)
JPG_DIR = os.path.abspath(os.path.join(THIS_DIR, "..", "..", "hdr-jpg"))


def _read_exposure_time(path: str) -> float:
    img = Image.open(path)
    exif = img._getexif() or {}
    exp = exif.get(33434)  # ExposureTime
    if exp is None:
        raise ValueError(f"Missing ExposureTime EXIF for: {path}")
    if isinstance(exp, tuple) and len(exp) == 2:
        exp = exp[0] / exp[1]
    return float(exp)


def _load_jpg_stack(folder: str):
    paths = sorted(
        os.path.join(folder, f)
        for f in os.listdir(folder)
        if f.lower().endswith(".jpg")
    )
    if not paths:
        raise FileNotFoundError(f"No JPG files found in {folder}")

    exposure_times = []
    images = []
    for p in paths:
        t = _read_exposure_time(p)
        img = cv2.imread(p, cv2.IMREAD_COLOR)
        if img is None:
            raise ValueError(f"Failed to read image: {p}")
        exposure_times.append(t)
        images.append(img)

    # sort by exposure (longest first)
    order = np.argsort(exposure_times)[::-1]
    exposure_times = [exposure_times[i] for i in order]
    images = [images[i] for i in order]
    paths = [paths[i] for i in order]
    return paths, images, np.array(exposure_times, dtype=np.float32)


def _weight(z):
    z_min, z_max = 0.0, 255.0
    z_mid = 0.5 * (z_min + z_max)
    return np.where(z <= z_mid, z - z_min, z_max - z)


def _solve_response_curve(z_samples, log_t, l=50.0):
    """
    Debevec-style response curve estimation without OpenCV.
    z_samples: (P, N) uint8 pixel values for one channel
    log_t: (N,) log exposure times
    returns g: (256,) mapping pixel -> log exposure
    """
    P, N = z_samples.shape
    n = 256
    # unknowns: g(0..255) and lnE(0..P-1)
    rows = P * N + (n - 2) + 1
    cols = n + P
    A = np.zeros((rows, cols), dtype=np.float64)
    b = np.zeros((rows, 1), dtype=np.float64)

    w = _weight(np.arange(n)).astype(np.float64)
    k = 0
    for i in range(P):
        for j in range(N):
            z = int(z_samples[i, j])
            wij = w[z]
            A[k, z] = wij
            A[k, n + i] = -wij
            b[k, 0] = wij * log_t[j]
            k += 1

    # fix the curve by setting g(128) = 0
    A[k, 128] = 1.0
    b[k, 0] = 0.0
    k += 1

    # smoothness constraints
    for z in range(1, n - 1):
        A[k, z - 1] = l * w[z]
        A[k, z] = -2 * l * w[z]
        A[k, z + 1] = l * w[z]
        k += 1

    x, _, _, _ = np.linalg.lstsq(A, b, rcond=None)
    g = x[:n, 0]
    return g


def estimate_response_curve(images, exposure_times, num_samples=2000):
    # sample pixels uniformly for curve estimation
    h, w, _ = images[0].shape
    rng = np.random.default_rng(42)
    ys = rng.integers(0, h, size=num_samples)
    xs = rng.integers(0, w, size=num_samples)

    Z = np.zeros((num_samples, len(images), 3), dtype=np.uint8)
    for j, img in enumerate(images):
        Z[:, j, :] = img[ys, xs, :]

    log_t = np.log(exposure_times)
    response = np.zeros((256, 1, 3), dtype=np.float32)
    for c in range(3):
        g = _solve_response_curve(Z[:, :, c], log_t)
        response[:, 0, c] = g
    return response  # shape (256, 1, 3), BGR order


def linearize_images(images, exposure_times, response):
    g = response[:, 0, :]  # (256, 3), BGR order
    linear_images = []
    for img, t in zip(images, exposure_times):
        img_u8 = img.astype(np.uint8)
        out = np.empty_like(img_u8, dtype=np.float32)
        for c in range(3):
            out[..., c] = np.exp(g[:, c][img_u8[..., c]]) / t
        linear_images.append(out)
    return linear_images


def merge_hdr_weighted(images, exposure_times, response):
    """
    Proper Debevec-style HDR merge:
    lnE = sum_j w(z_ij) * (g(z_ij) - ln t_j) / sum_j w(z_ij)
    images: list of uint8 BGR
    response: (256,1,3) where response[z,0,c] = g(z) = ln exposure
    returns hdr radiance map in float32 (linear), BGR
    """
    g = response[:, 0, :]  # (256,3)
    log_t = np.log(exposure_times).astype(np.float32)

    h, w, _ = images[0].shape
    hdr_ln = np.zeros((h, w, 3), dtype=np.float32)
    wsum   = np.zeros((h, w, 3), dtype=np.float32)

    w_lut = _weight(np.arange(256)).astype(np.float32)  # (256,)

    for j, img in enumerate(images):
        z = img.astype(np.uint8)
        for c in range(3):
            w_ij = w_lut[z[..., c]]
            hdr_ln[..., c] += w_ij * (g[z[..., c], c].astype(np.float32) - log_t[j])
            wsum[..., c]   += w_ij

    hdr_ln /= (wsum + 1e-8)
    hdr = np.exp(hdr_ln)
    return hdr



def tonemap_log(hdr, out_path):
    hdr_log = np.log1p(hdr)
    a = np.percentile(hdr_log, 0.1)
    b = np.percentile(hdr_log, 99.9)
    ldr = np.clip((hdr_log - a) / (b - a + 1e-12), 0, 1)
    out = (ldr * 255).astype(np.uint8)
    cv2.imwrite(out_path, out)


def main():
    paths, images, exposure_times = _load_jpg_stack(JPG_DIR)
    print("Loaded images:")
    for p, t in zip(paths, exposure_times):
        print(f"  {os.path.basename(p)}  t={t:.6f}s")

    response = estimate_response_curve(images, exposure_times)
    np.save(os.path.join(THIS_DIR, "response_curve.npy"), response)

    linear_images = linearize_images(images, exposure_times, response)
    hdr = merge_hdr_weighted(images, exposure_times, response)
    np.save(os.path.join(THIS_DIR, "HDR_linear_from_jpg.npy"), hdr)

    tonemap_log(hdr, os.path.join(THIS_DIR, "HDR_from_jpg.png"))
    print("Saved: response_curve.npy, HDR_linear_from_jpg.npy, HDR_from_jpg.png")


if __name__ == "__main__":
    main()
