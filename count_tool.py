import cv2
import numpy as np
import random


class ColorAnalyzer:
    @staticmethod
    def hex_to_bgr(hex_str):
        hex_str = hex_str.lstrip('#')
        return np.array([int(hex_str[4:6], 16), int(hex_str[2:4], 16), int(hex_str[0:2], 16)], dtype=np.uint8)

    @staticmethod
    def bgr_to_hex(bgr):
        return "#{:02x}{:02x}{:02x}".format(bgr[2], bgr[1], bgr[0])

    @staticmethod
    def get_labels_and_palette(img_bgr, k=10):
        """Perform K-Means clustering in LAB space."""
        lab = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2LAB)
        pixels = lab.reshape(-1, 3).astype(np.float32)

        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 20, 1.0)
        _, labels, centers = cv2.kmeans(pixels, k, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)

        centers = centers.astype(np.uint8)
        clustered_lab = centers[labels.flatten()].reshape(lab.shape)
        clustered_bgr = cv2.cvtColor(clustered_lab, cv2.COLOR_LAB2BGR)

        # Convert centers back to BGR for the UI
        palette_bgr = [cv2.cvtColor(np.uint8([[c]]), cv2.COLOR_LAB2BGR)[0][0] for c in centers]
        return clustered_bgr, palette_bgr, labels.flatten().reshape(img_bgr.shape[:2])

    @staticmethod
    def apply_manual_palette(img_bgr, palette_hex_list):
        """Map image to the nearest colors in a provided HEX list."""
        palette_bgr = np.array([ColorAnalyzer.hex_to_bgr(h) for h in palette_hex_list])
        pixels = img_bgr.reshape(-1, 3).astype(np.float32)

        # Use Euclidean distance in BGR for simplicity, or LAB for accuracy
        # Converting to LAB for better perceptual matching
        lab_pixels = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2LAB).reshape(-1, 3).astype(np.float32)
        lab_palette = cv2.cvtColor(palette_bgr.reshape(-1, 1, 3), cv2.COLOR_BGR2LAB).reshape(-1, 3).astype(np.float32)

        dists = np.linalg.norm(lab_pixels[:, None, :] - lab_palette[None, :, :], axis=2)
        labels = np.argmin(dists, axis=1)

        clustered_bgr = palette_bgr[labels].reshape(img_bgr.shape).astype(np.uint8)
        return clustered_bgr, palette_bgr, labels.reshape(img_bgr.shape[:2])

    @staticmethod
    def process_mask(mask, morph_size=3, iterations=1):
        if morph_size > 0:
            kernel = np.ones((morph_size, morph_size), np.uint8)
            mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=iterations)
        return mask

    @staticmethod
    def filter_and_count(mask, min_area, max_aspect, area_ratio):
        # 1. Find components
        num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=4)

        if num_labels <= 1:  # Only background found
            return 0, np.zeros_like(mask), np.zeros_like(mask)

        # 2. Extract stats (skip index 0 which is background)
        areas = stats[1:, cv2.CC_STAT_AREA]
        widths = stats[1:, cv2.CC_STAT_WIDTH]
        heights = stats[1:, cv2.CC_STAT_HEIGHT]

        # 3. Vectorized Filtering (All calculations happen at once)
        # Avoid division by zero
        widths[widths == 0] = 1
        heights[heights == 1] = 1

        current_area_ratios = areas / (widths * heights)
        # Handle aspect ratio for both horizontal and vertical objects
        aspect_ratios = np.maximum(widths / heights, heights / widths)

        # Create a boolean array of component indices that meet all criteria
        valid_mask = (areas >= min_area) & \
                     (current_area_ratios >= area_ratio) & \
                     (aspect_ratios <= max_aspect)

        # 4. Final results
        # valid_mask is offset by 1 because we sliced stats[1:]
        # We add a False at the beginning for the background (label 0)
        valid_labels_map = np.concatenate([[False], valid_mask])

        # Use the map to create the final mask:
        # keep pixels where the label's entry in valid_labels_map is True
        display_mask = np.where(valid_labels_map[labels], 255, 0).astype(np.uint8)

        count = int(np.sum(valid_mask))
        removed_mask = cv2.bitwise_and(mask, cv2.bitwise_not(display_mask))

        return count, display_mask, removed_mask