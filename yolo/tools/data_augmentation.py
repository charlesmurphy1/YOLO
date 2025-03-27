import math
import random
from typing import List

import cv2
import numpy as np
import torch
from PIL import Image
from torchvision.transforms import functional as TF


class AugmentationComposer:
    """Composes several transforms together."""

    def __init__(self, transforms, image_size: int = [640, 640], base_size: int = 640):
        self.transforms = transforms
        # TODO: handle List of image_size [640, 640]
        self.pad_resize = PadAndResize(image_size)
        self.base_size = base_size

        for transform in self.transforms:
            if hasattr(transform, "set_parent"):
                transform.set_parent(self)

    def __call__(self, image, boxes=torch.zeros(0, 5)):
        for transform in self.transforms:
            image, boxes = transform(image, boxes)
        image, boxes, rev_tensor = self.pad_resize(image, boxes)
        image = TF.to_tensor(image)
        return image, boxes, rev_tensor


class RemoveOutliers:
    """Removes outlier bounding boxes that are too small or have invalid dimensions."""

    def __init__(self, min_box_area=1e-8):
        """
        Args:
            min_box_area (float): Minimum area for a box to be kept, as a fraction of the image area.
        """
        self.min_box_area = min_box_area

    def __call__(self, image, boxes):
        """
        Args:
            image (PIL.Image): The cropped image.
            boxes (torch.Tensor): Bounding boxes in normalized coordinates (x_min, y_min, x_max, y_max).
        Returns:
            PIL.Image: The input image (unchanged).
            torch.Tensor: Filtered bounding boxes.
        """
        box_areas = (boxes[:, 3] - boxes[:, 1]) * (boxes[:, 4] - boxes[:, 2])

        valid_boxes = (
            (box_areas > self.min_box_area)
            & (boxes[:, 3] > boxes[:, 1])
            & (boxes[:, 4] > boxes[:, 2])
        )

        return image, boxes[valid_boxes]


class PadAndResize:
    def __init__(self, image_size, background_color=(114, 114, 114)):
        """Initialize the object with the target image size."""
        self.target_width, self.target_height = image_size
        self.background_color = background_color

    def set_size(self, image_size: List[int]):
        self.target_width, self.target_height = image_size

    def __call__(self, image: Image, boxes):
        img_width, img_height = image.size
        scale = min(self.target_width / img_width, self.target_height / img_height)
        new_width, new_height = int(img_width * scale), int(img_height * scale)

        resized_image = image.resize((new_width, new_height), Image.Resampling.LANCZOS)

        pad_left = (self.target_width - new_width) // 2
        pad_top = (self.target_height - new_height) // 2
        padded_image = Image.new(
            "RGB", (self.target_width, self.target_height), self.background_color
        )
        padded_image.paste(resized_image, (pad_left, pad_top))

        boxes[:, [1, 3]] = (boxes[:, [1, 3]] * new_width + pad_left) / self.target_width
        boxes[:, [2, 4]] = (
            boxes[:, [2, 4]] * new_height + pad_top
        ) / self.target_height

        transform_info = torch.tensor([scale, pad_left, pad_top, pad_left, pad_top])
        return padded_image, boxes, transform_info


class HorizontalFlip:
    """Randomly horizontally flips the image along with the bounding boxes."""

    def __init__(self, prob=0.5):
        self.prob = prob

    def __call__(self, image, boxes):
        if torch.rand(1) < self.prob:
            image = TF.hflip(image)
            boxes[:, [1, 3]] = 1 - boxes[:, [3, 1]]
        return image, boxes


class VerticalFlip:
    """Randomly vertically flips the image along with the bounding boxes."""

    def __init__(self, prob=0.5):
        self.prob = prob

    def __call__(self, image, boxes):
        if torch.rand(1) < self.prob:
            image = TF.vflip(image)
            boxes[:, [2, 4]] = 1 - boxes[:, [4, 2]]
        return image, boxes


class Mosaic:
    """Applies the Mosaic augmentation to a batch of images and their corresponding boxes."""

    def __init__(self, prob=0.5):
        self.prob = prob
        self.parent = None

    def set_parent(self, parent):
        self.parent = parent

    def __call__(self, image, boxes):
        if torch.rand(1) >= self.prob:
            return image, boxes

        assert self.parent is not None, (
            "Parent is not set. Mosaic cannot retrieve image size."
        )

        img_sz = self.parent.base_size  # Assuming `image_size` is defined in parent
        more_data = self.parent.get_more_data(3)  # get 3 more images randomly

        data = [(image, boxes)] + more_data
        mosaic_image = Image.new("RGB", (2 * img_sz, 2 * img_sz), (114, 114, 114))
        vectors = np.array([(-1, -1), (0, -1), (-1, 0), (0, 0)])
        center = np.array([img_sz, img_sz])
        all_labels = []

        for (image, boxes), vector in zip(data, vectors):
            this_w, this_h = image.size
            coord = tuple(center + vector * np.array([this_w, this_h]))

            mosaic_image.paste(image, coord)
            xmin, ymin, xmax, ymax = boxes[:, 1], boxes[:, 2], boxes[:, 3], boxes[:, 4]
            xmin = (xmin * this_w + coord[0]) / (2 * img_sz)
            xmax = (xmax * this_w + coord[0]) / (2 * img_sz)
            ymin = (ymin * this_h + coord[1]) / (2 * img_sz)
            ymax = (ymax * this_h + coord[1]) / (2 * img_sz)

            adjusted_boxes = torch.stack([boxes[:, 0], xmin, ymin, xmax, ymax], dim=1)
            all_labels.append(adjusted_boxes)

        all_labels = torch.cat(all_labels, dim=0)
        mosaic_image = mosaic_image.resize((img_sz, img_sz))
        return mosaic_image, all_labels


class MixUp:
    """Applies the MixUp augmentation to a pair of images and their corresponding boxes."""

    def __init__(self, prob=0.5, alpha=1.0):
        self.alpha = alpha
        self.prob = prob
        self.parent = None

    def set_parent(self, parent):
        """Set the parent dataset object for accessing dataset methods."""
        self.parent = parent

    def __call__(self, image, boxes):
        if torch.rand(1) >= self.prob:
            return image, boxes

        assert self.parent is not None, (
            "Parent is not set. MixUp cannot retrieve additional data."
        )

        # Retrieve another image and its boxes randomly from the dataset
        image2, boxes2 = self.parent.get_more_data()[0]

        # Calculate the mixup lambda parameter
        lam = np.random.beta(self.alpha, self.alpha) if self.alpha > 0 else 0.5

        # Mix images
        image1, image2 = TF.to_tensor(image), TF.to_tensor(image2)
        mixed_image = lam * image1 + (1 - lam) * image2

        # Merge bounding boxes
        merged_boxes = torch.cat((boxes, boxes2))

        return TF.to_pil_image(mixed_image), merged_boxes


class RandomCrop:
    """Randomly crops the image to half its size along with adjusting the bounding boxes."""

    def __init__(self, prob=0.5):
        """
        Args:
            prob (float): Probability of applying the crop.
        """
        self.prob = prob

    def __call__(self, image, boxes):
        if torch.rand(1) < self.prob:
            original_width, original_height = image.size
            crop_height, crop_width = original_height // 2, original_width // 2
            top = torch.randint(0, original_height - crop_height + 1, (1,)).item()
            left = torch.randint(0, original_width - crop_width + 1, (1,)).item()

            image = TF.crop(image, top, left, crop_height, crop_width)

            boxes[:, [1, 3]] = boxes[:, [1, 3]] * original_width - left
            boxes[:, [2, 4]] = boxes[:, [2, 4]] * original_height - top

            boxes[:, [1, 3]] = boxes[:, [1, 3]].clamp(0, crop_width)
            boxes[:, [2, 4]] = boxes[:, [2, 4]].clamp(0, crop_height)

            boxes[:, [1, 3]] /= crop_width
            boxes[:, [2, 4]] /= crop_height

        return image, boxes


class RandomHSV:
    """Randomly adjusts the Hue, Saturation, and Value (HSV) channels of an image."""

    def __init__(self, prob=0.5, hgain=0.015, sgain=0.7, vgain=0.4):
        """
        Args:
            prob (float): Probability of applying the HSV augmentation.
            hgain (float): Maximum variation for hue. Range is typically [0, 1].
            sgain (float): Maximum variation for saturation. Range is typically [0, 1].
            vgain (float): Maximum variation for value. Range is typically [0, 1].
        """
        self.prob = prob
        self.hgain = hgain
        self.sgain = sgain
        self.vgain = vgain

    def __call__(self, image, boxes):
        """
        Args:
            image (PIL.Image): Input image.
            boxes (torch.Tensor): Bounding boxes in normalized coordinates.
        Returns:
            PIL.Image: HSV-adjusted image.
            torch.Tensor: Bounding boxes (unchanged by HSV adjustment).
        """
        if torch.rand(1) < self.prob:
            # Convert PIL to numpy array
            img = np.array(image)

            # Generate random gains
            r = np.random.uniform(-1, 1, 3) * [self.hgain, self.sgain, self.vgain] + 1

            # Convert RGB to HSV
            img_hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
            h, s, v = cv2.split(img_hsv)

            # Apply adjustments using lookup tables
            dtype = img.dtype
            x = np.arange(0, 256, dtype=np.float32)

            lut_h = ((x * r[0]) % 180).astype(dtype)
            lut_s = np.clip(x * r[1], 0, 255).astype(dtype)
            lut_v = np.clip(x * r[2], 0, 255).astype(dtype)

            # Apply lookup tables
            h = cv2.LUT(h, lut_h)
            s = cv2.LUT(s, lut_s)
            v = cv2.LUT(v, lut_v)

            # Merge channels and convert back to RGB
            img_hsv = cv2.merge((h, s, v))
            img_rgb = cv2.cvtColor(img_hsv, cv2.COLOR_HSV2RGB)

            # Convert back to PIL
            image = Image.fromarray(img_rgb)

        return image, boxes


class RandomPerspective:
    """Applies random perspective and affine transformations to an image and its bounding boxes."""

    def __init__(
        self,
        prob=0.5,
        degrees=0.0,
        translate=0.1,
        scale=0.5,
        shear=0.0,
        perspective=0.0,
    ):
        """
        Args:
            prob (float): Probability of applying the transformation.
            degrees (float): Maximum degree of rotation.
            translate (float): Maximum translation as a fraction of image size.
            scale (float): Maximum scaling factor.
            shear (float): Maximum shear angle in degrees.
            perspective (float): Perspective distortion factor.
        """
        self.prob = prob
        self.degrees = degrees
        self.translate = translate
        self.scale = scale
        self.shear = shear
        self.perspective = perspective
        self.border = (0, 0)  # Border for mosaic (not used directly in this version)

    def __call__(self, image, boxes):
        """
        Args:
            image (PIL.Image): Input image.
            boxes (torch.Tensor): Bounding boxes in normalized coordinates (cls, x1, y1, x2, y2).
        Returns:
            PIL.Image: Transformed image.
            torch.Tensor: Transformed bounding boxes.
        """
        if torch.rand(1) < self.prob:
            width, height = image.size

            # Convert PIL to numpy for OpenCV operations
            img = np.array(image)

            # Center matrix
            C = np.eye(3, dtype=np.float32)
            C[0, 2] = -width / 2  # x translation (pixels)
            C[1, 2] = -height / 2  # y translation (pixels)

            # Perspective matrix
            P = np.eye(3, dtype=np.float32)
            P[2, 0] = random.uniform(-self.perspective, self.perspective)
            P[2, 1] = random.uniform(-self.perspective, self.perspective)

            # Rotation and Scale matrix
            R = np.eye(3, dtype=np.float32)
            a = random.uniform(-self.degrees, self.degrees)
            s = random.uniform(1 - self.scale, 1 + self.scale)
            R[:2] = cv2.getRotationMatrix2D(angle=a, center=(0, 0), scale=s)

            # Shear matrix
            S = np.eye(3, dtype=np.float32)
            S[0, 1] = math.tan(random.uniform(-self.shear, self.shear) * math.pi / 180)
            S[1, 0] = math.tan(random.uniform(-self.shear, self.shear) * math.pi / 180)

            # Translation matrix
            T = np.eye(3, dtype=np.float32)
            T[0, 2] = random.uniform(0.5 - self.translate, 0.5 + self.translate) * width
            T[1, 2] = (
                random.uniform(0.5 - self.translate, 0.5 + self.translate) * height
            )

            # Combined transformation matrix (order matters)
            M = T @ S @ R @ P @ C

            # Apply transformation to the image
            if self.perspective:
                img = cv2.warpPerspective(
                    img, M, dsize=(width, height), borderValue=(114, 114, 114)
                )
            else:
                img = cv2.warpAffine(
                    img, M[:2], dsize=(width, height), borderValue=(114, 114, 114)
                )

            # Convert back to PIL
            image = Image.fromarray(img)

            # Apply transformations to boxes if there are any
            if len(boxes) > 0:
                # Get box corners (x1y1, x2y2, x1y2, x2y1)
                n = len(boxes)
                points = torch.ones((n * 4, 3), dtype=torch.float32)
                # Extract normalized coordinates
                xy = boxes[:, [1, 2, 3, 4, 1, 4, 3, 2]].reshape(n * 4, 2)
                # Denormalize to pixel coordinates
                xy[:, 0] *= width
                xy[:, 1] *= height

                points[:, :2] = xy

                # Apply transformation matrix
                points = points @ torch.from_numpy(M).T.float()

                if self.perspective:
                    # Apply perspective divide
                    points = points[:, :2] / points[:, 2:3]
                else:
                    points = points[:, :2]

                # Reshape back to box format
                points = points.reshape(n, 8)

                # Get min/max coordinates for new boxes
                x_min = torch.min(points[:, [0, 2, 4, 6]], dim=1)[0]
                y_min = torch.min(points[:, [1, 3, 5, 7]], dim=1)[0]
                x_max = torch.max(points[:, [0, 2, 4, 6]], dim=1)[0]
                y_max = torch.max(points[:, [1, 3, 5, 7]], dim=1)[0]

                # Clip to image boundaries
                x_min = x_min.clamp(0, width)
                y_min = y_min.clamp(0, height)
                x_max = x_max.clamp(0, width)
                y_max = y_max.clamp(0, height)

                # Filter boxes with insufficient area
                new_boxes = torch.zeros_like(boxes)
                new_boxes[:, 0] = boxes[:, 0]  # Keep class unchanged

                # Normalize back to [0,1]
                new_boxes[:, 1] = x_min / width
                new_boxes[:, 2] = y_min / height
                new_boxes[:, 3] = x_max / width
                new_boxes[:, 4] = y_max / height

                # Filter out boxes that got too small or have incorrect aspect ratio
                box_width = (new_boxes[:, 3] - new_boxes[:, 1]) * width
                box_height = (new_boxes[:, 4] - new_boxes[:, 2]) * height
                box_area = box_width * box_height
                orig_box_width = (boxes[:, 3] - boxes[:, 1]) * width
                orig_box_height = (boxes[:, 4] - boxes[:, 2]) * height
                orig_area = orig_box_width * orig_box_height

                # Define candidates: boxes that maintain minimum area and reasonable aspect ratio
                min_area = 2  # Minimum pixel area
                area_ratio = 0.1  # Minimum ratio of new area to old area
                aspect_ratio_max = 20  # Maximum aspect ratio

                ar = torch.max(
                    box_width / (box_height + 1e-6), box_height / (box_width + 1e-6)
                )
                valid = (
                    (box_width > min_area)
                    & (box_height > min_area)
                    & (box_area / (orig_area + 1e-6) > area_ratio)
                    & (ar < aspect_ratio_max)
                )

                boxes = new_boxes[valid]

        return image, boxes
