import os
import logging
from pathlib import Path

try:
    import ants
except ImportError:
    ants = None

logger = logging.getLogger("rich")

def register_image(
    moving: str, 
    fixed: str, 
    output: str, 
    transform_type: str = 'syn', 
    num_threads: int = None
) -> None:
    """
    Register a moving image to a fixed template using ANTs.

    Args:
        moving (str): Path to moving image (e.g. subject)
        fixed (str): Path to fixed image (e.g. template)
        output (str): Path to save registered image
        transform_type (str): Type of transform ('rigid', 'affine', 'syn')
        num_threads (int): Number of threads to use (default: None = auto)
    """
    if ants is None:
        raise ImportError("ANTsPy is not installed. Please install it with `pip install antspyx`.")

    if num_threads:
        os.environ["ITK_GLOBAL_DEFAULT_NUMBER_OF_THREADS"] = str(num_threads)

    logger.info(f"Loading moving image: {moving}")
    moving_img = ants.image_read(str(moving))

    logger.info(f"Loading fixed image: {fixed}")
    fixed_img = ants.image_read(str(fixed))

    # Map CLI transform names to ANTs transform types
    transform_map = {
        'rigid': 'Rigid',
        'affine': 'Affine',
        'syn': 'SyN'
    }

    ants_transform = transform_map.get(transform_type.lower(), 'SyN')
    logger.info(f"Starting registration using {ants_transform} transform...")

    try:
        mytx = ants.registration(fixed=fixed_img, moving=moving_img, type_of_transform=ants_transform)
        warped_img = mytx['warpedmovout']

        output_path = Path(output)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        logger.info(f"Saving registered image to {output}")
        ants.image_write(warped_img, str(output))

        # Save transforms option
        # for i, transform in enumerate(mytx['fwdtransforms']):
        #     shutil.copy(transform, output_path.parent / f"transform_{i}.mat")

    except Exception as e:
        logger.error(f"Registration failed: {e}")
        raise
