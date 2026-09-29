from PIL import Image
import pytest

from visionbench.transforms import CoordinateTransform, preprocess_image


def test_scale_and_multiple_crop_have_correct_coordinates():
    image = Image.new("RGB", (101, 61), "white")
    processed, transform, params = preprocess_image(image, (80, 80), "scale", 16)

    assert processed.size == (80, 48)
    assert params["final_crop"] == (0, 0, 80, 48)
    assert transform.forward((0, 0)) == (0, 0)
    assert transform.forward((101, 61)) == (80, 48)
    assert transform.inverse(transform.forward((23.5, 19.25))) == pytest.approx((23.5, 19.25))
    assert transform.visible((99, 59))
    assert not transform.visible((101, 61))


def test_crop_transform_and_annotation_helpers():
    image = Image.new("RGB", (100, 80), "white")
    processed, transform, _ = preprocess_image(image, (60, 50), "crop", 10)

    assert processed.size == (60, 50)
    assert transform.source_crop == (20, 15, 80, 65)
    assert transform.forward((20, 15)) == (0, 0)
    assert transform.inverse((0, 0)) == (20, 15)
    assert transform.crop_box_to_visible((0, 0, 30, 30)) == (0.0, 0.0, 10.0, 15.0)

    mask = Image.new("L", image.size, 0)
    mask.putpixel((20, 15), 255)
    transformed_mask = transform.transform_mask(mask)
    assert transformed_mask.getpixel((0, 0)) == 255
    assert CoordinateTransform.from_dict(transform.to_dict()) == transform
