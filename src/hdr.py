import subprocess
import numpy as np
from pathlib import Path
import cv2
import argparse as ap
import time
import matplotlib.pyplot as plt
from tqdm import tqdm
from cp_hw2 import writeHDR, readHDR, read_colorchecker_gm, lRGB2XYZ, XYZ2lRGB, xyY_to_XYZ
from utils import gamma_encoding
from scipy import stats


def process_raw(src_dir: Path):
    """Uses dcraw to process raw files to linear, 16-bit TIFFs.
    
    Args:
        src_dir: Path to directory of raw files.

    Returns:
        Writes linear, 16-bit TIFFs to src_dir
    """
    raw_imgs = src_dir.glob("*.nef")
    for raw_img in raw_imgs:
        print(f"Processing {raw_img}...")
        cmd = [
            "sudo", "dcraw", "-w", "-o", "1", "-q", "3", "-T", "-4", "-H", "0",
            str(raw_img)
        ]
        subprocess.run(cmd)


def read_image(img_path: Path, n: int = 1) -> np.ndarray:
    """Reads image in provided path and naively downsamples by n.
    
    Args:
        img_path: Path to image to be read.
        n: Downsampling factor.

    Returns:
        Downsampled image.
    """
    img = cv2.imread(str(img_path), -1)
    return img[::n, ::n, ::-1]


def get_exposures(num_exposures: int = 16) -> np.ndarray:
    """Computes exposures given number of exposures.

    Args:
        num_exposures: Number of exposures.

    Returns:
    
    """
    return np.array([(1 / 2048) * np.power(2, k - 1)
                     for k in range(1, num_exposures + 1)])


def linearize(input_images: Path,
              regularization_weight: float = 0.01,
              weighting_scheme="uniform",
              image_type="jpg"):
    """Linearizes rendered images.
    
    Args:
        input_images: List of images to linearize.
        regularization_weight: How much to weight the regularization term in optimization.
        weighting_scheme: One of ["uniform", "tent", "gaussian", "photon"] indicating the weighting scheme to use.
    Returns:
        Returns linear images.
    """
    h_orig, w_orig, c_orig = read_image(input_images[0], n=1).shape
    num_exposures = len(input_images)
    exposures = get_exposures(num_exposures=num_exposures)
    exposure_stack = [read_image(im, n=200) for im in input_images]
    flattened_exposure_stack = np.hstack(
        [im.reshape(-1, 1) for im in exposure_stack])
    h, w, c = exposure_stack[0].shape
    A = np.zeros(((h * w * c * num_exposures) + 256 + 1, (h * w * c) + 256))
    if weighting_scheme == "photon":
        weights = get_weights(mode=weighting_scheme,
                              exposures=exposures,
                              image_type=image_type).reshape(
                                  (num_exposures, -1))
    elif weighting_scheme == "optimal":
        weights = get_weights(mode=weighting_scheme,
                              exposures=exposures,
                              image_type=image_type).reshape(
                                  (num_exposures, -1, 3))
    else:
        weights = get_weights(mode=weighting_scheme,
                              image_type=image_type).reshape((-1, 1))

    b = np.zeros((A.shape[0], 1))

    k = 0
    n = 256
    for i in range(flattened_exposure_stack.shape[0]):
        for j in range(num_exposures):
            if weighting_scheme == "photon":
                w_ij = weights[j, flattened_exposure_stack[i, j]]
            elif weighting_scheme == "optimal":
                if i == 0 or i % 3 == 0:
                    channel = 0
                elif i == 1 or i % 4 == 0:
                    channel = 1
                else:
                    channel = 2
                w_ij = weights[j, flattened_exposure_stack[i, j], channel]
            else:
                w_ij = weights[flattened_exposure_stack[i, j]]
            A[k, flattened_exposure_stack[i, j]] = w_ij
            A[k, n + i] = -w_ij
            b[k] = w_ij * np.log(exposures[j])
            k += 1

    A[k, 128] = 1
    k += 1

    for z in range(0, 254):
        if weighting_scheme == "photon" or weighting_scheme == "optimal":
            w1, w2, w3 = 1, 1, 1
        else:
            w1, w2, w3 = weights[z:z + 3]
        A[k, z] = regularization_weight * w1
        A[k, z + 1] = -2 * regularization_weight * w2
        A[k, z + 2] = regularization_weight * w3
        k += 1
    t0 = time.time()
    print("Solving least squares problem....")
    v, residuals, rank, singular_values = np.linalg.lstsq(A, b, rcond=None)
    print(f"Took {(time.time() - t0)/60.0} minutes.")
    g = v[:256]
    full_flattened_exposure_stack = np.hstack(
        [read_image(im, n=1).reshape(-1, 1) for im in input_images])
    linearized_full_flattened_exposure_stack = np.zeros_like(
        full_flattened_exposure_stack, dtype=np.float32)
    for j in tqdm(range(num_exposures)):
        linearized_full_flattened_exposure_stack[:, j] = np.exp(
            g[full_flattened_exposure_stack[:, j]].reshape(-1))

    linearized_full_exposure_stack = np.dstack([
        linearized_full_flattened_exposure_stack[:, i].reshape(
            (h_orig, w_orig, c_orig)) for i in range(num_exposures)
    ])

    return linearized_full_exposure_stack


def merge(input_images,
          merging_scheme: str = "linear",
          weighting_scheme: str = "uniform",
          image_type: str = "jpg") -> np.ndarray:
    """Linearly or logarithmically merges linear LDR images.
    
    Args:
        input_images: List of LDR images to merge.
        merging_scheme: One of ["linear", "log"]

    Returns:
        H x W x 3 HDR image.
    """
    print(
        f"Image Type: {image_type}, Weighting Scheme: {weighting_scheme}, Merging Scheme: {merging_scheme}"
    )
    dark_frame = read_image("dark_frame.tiff")
    if image_type == "jpg":
        dark_frame = dark_frame[8:-8, 8:-8, :]

    original_ldr_images = np.dstack([read_image(im) for im in input_images])
    h, w, c = read_image(input_images[0]).shape
    weighted_original_ldr_images = np.zeros_like(original_ldr_images,
                                                 dtype=np.float32)
    num_exposures = len(input_images)
    exposures = get_exposures(num_exposures=num_exposures)
    placeholder = []
    if weighting_scheme == "optimal":
        for i in tqdm(range(original_ldr_images.shape[-1])):
            orig_img = original_ldr_images[:, :, i]
            expos = exposures[i // 3]
            if i == 0 or i % 3 == 0:
                x = orig_img - (dark_frame[:, :, 0] * 5 * expos)
                placeholder.append(x)
            elif i == 1 or i % 4 == 0:
                x = orig_img - (dark_frame[:, :, 1] * 5 * expos)
                placeholder.append(x)
            else:
                x = orig_img - (dark_frame[:, :, 2] * 5 * expos)
                placeholder.append(x)

        original_ldr_images = np.dstack(placeholder)
        original_ldr_images = np.clip(original_ldr_images, 0.0,
                                      original_ldr_images.max()).astype(
                                          np.uint16)
    if weighting_scheme == "photon":
        weights = get_weights(mode=weighting_scheme,
                              exposures=exposures,
                              image_type=image_type).reshape(
                                  num_exposures, -1)
    elif weighting_scheme == "optimal":
        weights = get_weights(mode=weighting_scheme,
                              exposures=exposures,
                              image_type=image_type).reshape(
                                  num_exposures, -1, 3)
    else:
        weights = get_weights(mode=weighting_scheme,
                              image_type=image_type).reshape(-1, 1)

    for i in tqdm(range(num_exposures)):
        if weighting_scheme == "photon":
            weighted_flattened_img = weights[i, original_ldr_images[:, :, (
                i * 3):(i + 1) * 3].reshape(-1)]
        elif weighting_scheme == "optimal":
            weighted_flattened_placeholder = np.zeros_like(
                original_ldr_images[:, :, 0:3].reshape(-1), dtype=np.float32)
            for k in range(3):
                weighted_flattened_placeholder[k::3] = weights[
                    i, original_ldr_images[:, :, i * 3 + k].reshape(-1), k]
            weighted_flattened_img = weighted_flattened_placeholder
        else:
            weighted_flattened_img = weights[original_ldr_images[:, :, (
                i * 3):(i + 1) * 3].reshape(-1)]
        weighted_original_ldr_images[:, :, (i * 3):(i + 1) *
                                     3] = weighted_flattened_img.reshape(
                                         (h, w, c))

    if image_type == "tiff":
        linearized_ldr_images = original_ldr_images
    else:
        linearized_ldr_images = linearize(input_images,
                                          weighting_scheme=weighting_scheme)

    exposures = exposures.reshape((1, 1, -1))
    if merging_scheme == "linear":
        hdr_image = np.dstack([
            (weighted_original_ldr_images[:, :, i::3] *
             (linearized_ldr_images[:, :, i::3] / exposures)).sum(axis=2) /
            ((weighted_original_ldr_images[:, :, i::3]).sum(axis=2))
            for i in range(3)
        ])
    elif merging_scheme == "log":
        hdr_image = np.dstack([
            (weighted_original_ldr_images[:, :, i::3] *
             (np.log(linearized_ldr_images[:, :, i::3] + 1e-8) -
              np.log(exposures))).sum(axis=2) /
            (weighted_original_ldr_images[:, :, i::3].sum(axis=2))
            for i in range(3)
        ])
        hdr_image = np.exp(hdr_image)

    image_type = "raw" if image_type == "tiff" else "jpg"
    out_dir = Path("./out_hdrs")
    if not out_dir.exists():
        out_dir.mkdir(parents=True, exist_ok=False)
    writeHDR(out_dir / f"{image_type}_{merging_scheme}_{weighting_scheme}.hdr",
             hdr_image)


def get_weights(mode: str = "uniform",
                z_min: float = 0.0,
                z_max: float = 1.0,
                exposures: float = None,
                image_type: str = "jpg"):
    """Applies desired weighting scheme to image.

    Args:
        input_image: Image to apply weighting scheme to.
        mode: Weighting scheme to apply. One of ["uniform", "tent", "gaussian", "photon"]
        z_min: Minimum value to threshold on.
        z_max: Maximum value to threshold on.
        exposures: Required only when mode == "photon".
        image_type: One of ["jpg", "tiff"]

    Returns:
        1 x 256 or 1 x 65536 array corresponding to weighting scheme.
    
    """
    if image_type == "jpg":
        values = np.arange(256, dtype=np.float32) / 255.0
    elif image_type == "tiff":
        values = np.arange(np.power(2, 16),
                           dtype=np.float32) / (np.power(2, 16) - 1)

    values = values.reshape(1, -1)
    weights = np.ones_like(values, dtype=np.float32)
    weights[np.where(values < z_min)] = 0.0
    weights[np.where(values > z_max)] = 0.0
    weights = weights.reshape(1, -1)
    if mode == "uniform":
        return weights
    elif mode == "tent":
        weights = weights * (np.minimum(values, 1 - values))
        return weights
    elif mode == "gaussian":
        weights = weights * np.exp(-4 *
                                   (np.square(values - 0.5) / np.square(0.5)))
        return weights
    elif mode == "photon":
        if exposures is None:
            raise RuntimeError("Photon weighting requires the image exposure.")
        weights = np.vstack([weights * exposure for exposure in exposures])
        return weights
    elif mode == "optimal":
        if exposures is None:
            raise RuntimeError("Photon weighting requires the image exposure.")
        gains = [35.91713454375651, 6.352994321756792, 23.52849247392018]
        noises = [518.5012691093725, 142.0744964823043, 2649.807828383473]
        weights = np.dstack([
            np.vstack([
                weights * np.square(exposure) / ((gain * values) + noise)
                for exposure in exposures
            ]) for gain, noise in zip(gains, noises)
        ])
        return weights
    else:
        raise RuntimeError(
            "Please select a valid weighting scheme. One of [uniform, tent, gaussian, photon]."
        )


def color_correct(hdr_image, colorchecker, crops=None):
    if crops is None:
        _, ax = plt.subplots()
        ax.imshow(
            np.clip(gamma_encoding(hdr_image[::, ::] * 16 * (2**-13.2)), 0.0,
                    1.0))
        crops = plt.ginput(n=(24 * 2) + 1, timeout=0)
        plt.close('all')
        final_crops = []
        for i in range(24):  # 24 Squares
            x = []
            y = []
            for j in range(2):  # 2 points per square
                x.append(crops[(i * 2) + j + 1][0])
                y.append(crops[(i * 2) + j + 1][1])
            final_crops.append(
                np.vstack(
                    [np.array(x).reshape(1, -1),
                     np.array(y).reshape(1, -1)]))
        crops = np.dstack(final_crops)
        np.save("crops.npy", crops)

    r_average = []
    g_average = []
    b_average = []
    for i in range(crops.shape[-1]):
        x_coords, y_coords = crops[0, :, i], crops[1, :, i]
        start_y, end_y = int(y_coords.min()), int(y_coords.max())
        start_x, end_x = int(x_coords.min()), int(x_coords.max())
        patch = hdr_image[start_y:end_y, start_x:end_x]

        patch_averages = patch.mean(axis=(0, 1)).reshape(-1)
        r_average.append(patch_averages[0])
        g_average.append(patch_averages[1])
        b_average.append(patch_averages[2])

    r_average = np.array(r_average).reshape((-1, 1))
    g_average = np.array(g_average).reshape((-1, 1))
    b_average = np.array(b_average).reshape((-1, 1))
    averages = np.hstack([r_average, g_average, b_average])

    A = np.zeros((24 * 3, 12), dtype=np.float32)
    b_lstq = np.zeros((24 * 3, 1), dtype=np.float32)

    k = 0
    for i in range(24):
        r, g, b = averages[i]
        for j in range(3):
            A[k, (j * 4):(j + 1) * 4] = np.array([r, g, b, 1])
            b_lstq[k] = colorchecker[i, j]
            k += 1

    v, residuals, rank, singular_values = np.linalg.lstsq(A,
                                                          b_lstq,
                                                          rcond=None)
    affine_matrix = v.reshape((3, 4))
    affine_matrix = affine_matrix.reshape((3, 1, 4))
    homogeneous_hdr = np.dstack(
        [hdr_image,
         np.ones((hdr_image.shape[0], hdr_image.shape[1]))])
    transformed_hdr = np.dstack([
        (homogeneous_hdr * affine_matrix[i]).sum(axis=2)
        for i in range(len(affine_matrix))
    ])
    transformed_hdr = np.clip(transformed_hdr, 0.0, transformed_hdr.max())

    x_coords, y_coords = crops[0, :, 18], crops[1, :, 18]
    start_y, end_y = int(y_coords.min()), int(y_coords.max())
    start_x, end_x = int(x_coords.min()), int(x_coords.max())
    transformed_average = transformed_hdr[start_y:end_y,
                                          start_x:end_x].mean(axis=(0, 1))
    transformed_hdr[:, :, 0] *= transformed_average[1] / transformed_average[0]
    transformed_hdr[:, :, 2] *= transformed_average[1] / transformed_average[2]

    out_dir = Path("./out_hdrs")
    if not out_dir.exists():
        out_dir.mkdir(parents=True, exist_ok=False)
    writeHDR(out_dir / "after_cc.hdr", transformed_hdr)


def tonemap(hdr_image, mode="RGB"):
    """Tonemaps HDR image.
    
    Args:
        mode: One of ["RGB", "xyY"]
    """
    if mode == "xyY":
        XYZ = lRGB2XYZ(hdr_image)
        X, Y, Z = XYZ[:, :, 0], XYZ[:, :, 1], XYZ[:, :, 2]
        x = X / (X + Y + Z)
        y = Y / (X + Y + Z)
        I_hdr = Y
    elif mode == "RGB":
        I_hdr = hdr_image
    else:
        raise RuntimeError(
            "Unrecognized mode. Please provide as input one of [RGB, xyY]")

    K = 0.05  #key
    B = 1.25  #burn
    N = np.prod(I_hdr.shape)
    eps = 1e-8
    I_m_hdr = np.exp(np.log(I_hdr + eps).sum() / N)
    I_tilde_hdr = I_hdr * K / I_m_hdr
    I_tilde_white = B * I_tilde_hdr.max()
    if mode == "xyY":
        Y = (I_tilde_hdr *
             (1 + I_tilde_hdr / np.square(I_tilde_white))) / (1 + I_tilde_hdr)
        X, Y, Z = xyY_to_XYZ(x, y, Y)
        XYZ = np.dstack([X, Y, Z])
        I_tm = XYZ2lRGB(XYZ)
    elif mode == "RGB":
        I_tm = (I_tilde_hdr *
                (1 + I_tilde_hdr / np.square(I_tilde_white))) / (1 +
                                                                 I_tilde_hdr)
    I_tm = np.clip(I_tm, 0.0, 1.0)
    out_dir = Path("./out_hdrs")
    if not out_dir.exists():
        out_dir.mkdir(parents=True, exist_ok=False)

    plt.imsave(out_dir / f"tonemapped_{mode}_K_{K}_B_{B}.png",
               gamma_encoding(I_tm))


def capture_exposure_stack(num_exposures):
    for i in range(1, num_exposures + 1):
        print(f"Capturing Image {i}...")
        cmd = [
            "sudo", "gphoto2", "--set-config-value",
            f"/main/capturesettings/shutterspeed={2**(i - 1)}/2048"
        ]
        subprocess.run(cmd)
        cmd = [
            "sudo", "gphoto2", "--capture-image-and-download", "--filename",
            f"./myexposurestack2/exposure{i}.%C"
        ]
        subprocess.run(cmd)


def generate_ramp_image():
    im = np.tile(np.linspace(0, 1, 4400), (3400, 1))
    im = np.dstack([im, im, im])
    plt.imsave("ramp.png", im)


def capture_noise_calibration_images(n=50, with_lens_cap=True):
    out_dir = "./noise_calib/"
    if not out_dir.exists():
        out_dir.mkdir(parents=True, exist_ok=False)
    if not (out_dir / "lens_cap_on").exists():
        (out_dir / "lens_cap_on").mkdir(parents=True, exist_ok=False)
    if not (out_dir / "lens_cap_off").exists():
        (out_dir / "lens_cap_off").mkdir(parents=True, exist_ok=False)

    for i in range(n):
        print(f"Capturing Image {i}...")
        if with_lens_cap:
            fname = f"./noise_calib/lens_cap_on/{i}.%C"
        else:
            fname = f"./noise_calib/lens_cap_off/{i}.%C"
        cmd = [
            "sudo", "gphoto2", "--set-config-value",
            f"/main/capturesettings/shutterspeed=1/3"
        ]
        subprocess.run(cmd)
        cmd = [
            "sudo", "gphoto2", "--capture-image-and-download", "--filename",
            fname
        ]
        subprocess.run(cmd)


def compute_dark_frame(dark_frame_paths):
    print("Computing Dark Frame...")
    breakpoint()
    dark_frames = np.dstack([read_image(im) for im in dark_frame_paths])
    breakpoint()
    dark_frame = np.dstack(
        [dark_frames[:, :, i::3].mean(axis=2) for i in range(3)])
    cv2.imwrite("dark_frame.tiff", dark_frame)
    print("Done!")


def noise_calibration(ramp_image_paths, dark_frame_path):
    ramp_images = [read_image(im, n=1) for im in tqdm(ramp_image_paths)]
    dark_frame = read_image(dark_frame_path, n=1)
    ramp_images_subtracted = np.dstack(
        [im - dark_frame for im in tqdm(ramp_images)])
    ramp_images_subtracted = np.clip(ramp_images_subtracted, 0,
                                     ramp_images_subtracted.max())
    red = ramp_images_subtracted[:, :, ::3]
    green = ramp_images_subtracted[:, :, 1::3]
    blue = ramp_images_subtracted[:, :, 2::3]

    red_mean = red.mean(axis=2)
    green_mean = green.mean(axis=2)
    blue_mean = blue.mean(axis=2)

    red_var = np.var(red, axis=2, ddof=1)
    blue_var = np.var(blue, axis=2, ddof=1)
    green_var = np.var(green, axis=2, ddof=1)

    red_rounded_means = np.rint(red_mean)
    green_rounded_means = np.rint(green_mean)
    blue_rounded_means = np.rint(blue_mean)
    red_rounded_unique = np.unique(red_rounded_means)
    green_rounded_unique = np.unique(green_rounded_means)
    blue_rounded_unique = np.unique(blue_rounded_means)

    red_averages = []
    green_averages = []
    blue_averages = []
    for r_un in tqdm(red_rounded_unique):
        red_averages.append(
            red_var[np.where(red_rounded_means == r_un)].mean())

    r_slope, r_intercept, _, _, _ = stats.linregress(red_rounded_unique,
                                                     red_averages)
    rx = np.arange(min(red_rounded_unique), max(red_rounded_unique))
    ry = r_slope * rx + r_intercept
    print("R gain: ", r_slope)
    print("R noise: ", r_intercept)
    plt.plot(rx, ry, c='k')
    plt.scatter(red_rounded_unique, red_averages, c='r')
    plt.show()

    for g_un in tqdm(green_rounded_unique):
        green_averages.append(
            green_var[np.where(green_rounded_means == g_un)].mean())
    green_averages = np.array(green_averages)
    g_slope, g_intercept, _, _, _ = stats.linregress(green_rounded_unique,
                                                     green_averages)
    gx = np.arange(min(green_rounded_unique), max(green_rounded_unique))
    gy = g_slope * gx + g_intercept
    print("G gain: ", g_slope)
    print("G noise: ", g_intercept)
    plt.plot(gx, gy, c='k')
    plt.scatter(green_rounded_unique, green_averages, c='g')
    plt.show()

    for b_un in tqdm(blue_rounded_unique):
        blue_averages.append(
            blue_var[np.where(blue_rounded_means == b_un)].mean())
    blue_averages = np.array(blue_averages)
    b_slope, b_intercept, _, _, _ = stats.linregress(blue_rounded_unique,
                                                     blue_averages)
    bx = np.arange(min(blue_rounded_unique), max(blue_rounded_unique))
    by = b_slope * bx + b_intercept
    print("B gain: ", b_slope)
    print("B noise: ", b_intercept)
    plt.plot(bx, by, c='k')
    plt.scatter(blue_rounded_unique, blue_averages, c='b')
    plt.show()


def main(args):
    src_dir = Path(args.src_dir)
    weighting_scheme = args.weight
    regularization_weight = float(args.lam)
    if args.process:
        process_raw(src_dir)
    if args.merge_all:
        for image_type in ["jpg", "tiff"]:
            im_list = sorted(list(src_dir.glob(f"*.{image_type}")),
                             key=lambda x: "%06d" % int(
                                 str(Path(x).stem).split("exposure")[1]))
            for merging_scheme in ["log", "linear"]:
                for weighting_scheme in [
                        "gaussian", "tent", "uniform", "photon"
                ]:
                    merge(im_list,
                          image_type=image_type,
                          weighting_scheme=weighting_scheme,
                          merging_scheme=merging_scheme)

    if args.color_correct:
        hdr_im = readHDR("before_cc.hdr")
        r, g, b = read_colorchecker_gm()
        colorchecker = np.hstack(
            [r.reshape((-1, 1)),
             g.reshape((-1, 1)),
             b.reshape((-1, 1))])
        if Path("crops.npy").exists():
            crops = np.load("crops.npy")
        else:
            crops = None
        color_correct(hdr_image=hdr_im, colorchecker=colorchecker, crops=crops)
    if args.tonemap:
        color_corrected_hdr_im = readHDR("./out_hdrs/after_cc.hdr")
        tonemap(color_corrected_hdr_im)
    if args.capture_dark_frames:
        capture_noise_calibration_images(n=50, with_lens_cap=True)
    if args.capture_ramp_frames:
        capture_noise_calibration_images(n=50, with_lens_cap=False)
    if args.get_dark_frame:
        dark_frame_paths = sorted(
            list(Path("./noise_calib/lens_cap_on/").glob("*.tiff")))
        compute_dark_frame(dark_frame_paths)
    if args.noise_calibrate:
        ramp_image_paths = sorted(list(
            Path("./noise_calib/lens_cap_off/").glob("*.tiff")),
                                  key=lambda x: int(str(Path(x).stem)))
        noise_calibration(ramp_image_paths, compute_dark_frame)


if __name__ == "__main__":
    parser = ap.ArgumentParser()
    parser.add_argument("--src_dir", type=str, required=True)
    parser.add_argument("--process",
                        action=ap.BooleanOptionalAction,
                        default=False)
    parser.add_argument("--merge-all",
                        action=ap.BooleanOptionalAction,
                        default=False)
    parser.add_argument("--colorcorrect",
                        action=ap.BooleanOptionalAction,
                        default=False)
    parser.add_argument("--tonemap",
                        action=ap.BooleanOptionalAction,
                        default=False)
    parser.add_argument("--capture_dark_frames",
                        action=ap.BooleanOptionalAction,
                        default=False)
    parser.add_argument("--capture_ramp_frames",
                        action=ap.BooleanOptionalAction,
                        default=False)
    parser.add_argument("--get_dark_frame",
                        action=ap.BooleanOptionalAction,
                        default=False)
    args = parser.parse_args()
    main(args)