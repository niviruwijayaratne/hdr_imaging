import numpy as np


def gamma_encoding(brightened_image: np.ndarray) -> np.ndarray:
    """Gamma encodes image.
    
    Applies following non-linearity to image:

        12.92 * C_linear if C_linear <= 0.0031308
        (1 + 0.055) * C_linear**(1/2.4) - 0.055 if C_linear > 0.0031308

        where C_linear = {R, G, B} of brightened_image

    Args:
        brightened_image: Brightened image.

    Returns:
        Gamma encoded image.

    """
    # Indices where C_linear <= 0.0031308
    cond1_coords = np.where(brightened_image <= 0.0031308)
    # Indices where C_linear > 0.0031308
    cond2_coords = np.where(brightened_image > 0.0031308)

    brightened_image[cond1_coords] *= 12.92
    brightened_image[cond2_coords] = (
        (1 + 0.055) *
        np.power(brightened_image[cond2_coords], 1 / 2.4)) - 0.055

    return brightened_image