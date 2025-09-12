"""
HFO Spectral Analyzer Module

This module provides spectral analysis for High Frequency Oscillations (HFO)
detection in EEG spectrograms using computer vision techniques.
Author: Daniel Lachner-Pizarro
Date: 2024-06-20
"""

import cv2
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans, MiniBatchKMeans
from collections import defaultdict
from typing import Tuple, Optional, List, Dict, Any
import logging
import time
from contextlib import contextmanager

# Configure logging
logger = logging.getLogger(__name__)


@contextmanager
def timing_context(operation_name: str, logger: logging.Logger):
    """Context manager for timing operations."""
    start_time = time.time()
    try:
        yield
    finally:
        elapsed_time = time.time() - start_time
        logger.info(f"{operation_name} completed in {elapsed_time:.2f} seconds")
        print(f"{operation_name} completed in {elapsed_time:.2f} seconds")


def _resize_spectrogram(
    spectrogram: np.ndarray, 
    target_width: int, 
    target_height: int
) -> np.ndarray:
    """
    Resize spectrogram to standardized dimensions for consistent feature extraction.
    
    Args:
        spectrogram: Input spectrogram image
        target_width: Target width in pixels (milliseconds)
        target_height: Target height in pixels (frequency range)
    
    Returns:
        Resized spectrogram image
    """
    return cv2.resize(
        spectrogram, 
        (target_width, target_height), 
        interpolation=cv2.INTER_LINEAR
    )


def _extract_rgb_channels(image: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Extract BGR channels from image.
    
    Args:
        image: Input BGR image
    
    Returns:
        Tuple of (blue, green, red) channel arrays
    """
    return cv2.split(image)


def _perform_kmeans_clustering(
    image: np.ndarray, 
    n_clusters: int = 3,
    random_state: int = 42
) -> Tuple[MiniBatchKMeans, np.ndarray]:
    """
    Perform K-means clustering on image pixels for segmentation using MiniBatchKMeans.
    
    Args:
        image: Input BGR image
        n_clusters: Number of clusters for K-means
        random_state: Random state for reproducibility
    
    Returns:
        Tuple of (fitted MiniBatchKMeans model, cluster labels)
    """
    b, g, r = _extract_rgb_channels(image)
    
    # Efficiently stack BGR values for clustering
    bgr_data = np.column_stack((b.ravel(), g.ravel(), r.ravel()))
    
    # Use MiniBatchKMeans for significantly better performance on large images
    kmeans = MiniBatchKMeans(
        n_clusters=n_clusters, 
        random_state=random_state,
        batch_size=1000,      # Process 1000 samples at a time
        max_iter=100,         # Reduced from 300 for faster convergence
        n_init=3,             # Reduced from 10 for speed
        init_size=None,       # Auto-select based on n_clusters
        reassignment_ratio=0.01  # Lower ratio for stability
    )
    
    cluster_labels = kmeans.fit_predict(bgr_data)
    
    return kmeans, cluster_labels


def _calculate_optimal_threshold(red_channel: np.ndarray, cluster_labels: np.ndarray) -> int:
    """
    Calculate optimal threshold value based on cluster centroids.
    
    Args:
        red_channel: Red channel of the image
        cluster_labels: Cluster labels from K-means
    
    Returns:
        Optimal threshold value for binary segmentation
    """
    red_flat = red_channel.ravel()
    centroids = []
    
    for label in np.unique(cluster_labels):
        cluster_mask = cluster_labels == label
        centroid_value = np.mean(red_flat[cluster_mask])
        centroids.append(centroid_value)
    
    return int(np.max(centroids))


def _find_and_filter_contours(binary_image: np.ndarray) -> List[np.ndarray]:
    """
    Find contours in binary image and filter by minimum area.
    
    Args:
        binary_image: Binary thresholded image
    
    Returns:
        List of valid contours
    """
    contours, _ = cv2.findContours(
        binary_image, 
        cv2.RETR_EXTERNAL, 
        cv2.CHAIN_APPROX_SIMPLE
    )
    
    # Filter contours by minimum area
    return [contour for contour in contours if cv2.contourArea(contour) >= 1]


def _calculate_contour_features(
    contour: np.ndarray, 
    cwt_range_hz: Tuple[int, int],
    img_width: int
) -> Dict[str, Any]:
    """
    Calculate comprehensive features for a single contour.
    
    Args:
        contour: OpenCV contour array
        cwt_range_hz: Frequency range tuple (min_freq, max_freq)
        img_width: Image width for boundary checks
    
    Returns:
        Dictionary containing all contour features
    """
    # Basic contour properties
    area = cv2.contourArea(contour)
    perimeter = cv2.arcLength(contour, closed=True)
    
    # Avoid division by zero
    circularity = 100 * (4 * np.pi * area / (perimeter**2)) if perimeter > 0 else 0
    
    # Contour moments for centroid calculation
    moments = cv2.moments(contour)
    if moments['m00'] > 0:
        cx = int(moments['m10'] / moments['m00'])
        cy = int(moments['m01'] / moments['m00'])
    else:
        cx, cy = 0, 0
    
    # Frequency and spatial calculations
    freq_centroid_hz = cwt_range_hz[1] - cy
    
    # Vectorized min/max calculations
    contour_points = contour.reshape(-1, 2)
    x_coords = contour_points[:, 0]
    y_coords = contour_points[:, 1]
    
    freq_min_hz = cwt_range_hz[1] - np.max(y_coords)
    freq_max_hz = cwt_range_hz[1] - np.min(y_coords)
    vspread = freq_max_hz - freq_min_hz
    hspread = np.max(x_coords) - np.min(x_coords)
    hvr = 100 * hspread / vspread if vspread > 0 else 0
    
    # Time domain calculations
    contour_start_ms = np.min(x_coords)
    contour_end_ms = np.max(x_coords)
    contour_center_ms = cx
    
    # Frequency-based duration correction
    contour_period_ms = 1000 / freq_centroid_hz if freq_centroid_hz > 0 else 0
    
    # Extend contour boundaries
    if contour_period_ms > 0:
        contour_start_ms = max(0, contour_start_ms - contour_period_ms)
        contour_end_ms = min(img_width, contour_end_ms + contour_period_ms)
    
    dur_ms = contour_end_ms - contour_start_ms
    nr_oscillations = np.round(dur_ms / contour_period_ms) if contour_period_ms > 0 else 0
    
    return {
        'area': area,
        'perimeter': perimeter,
        'circularity': circularity,
        'hspread': hspread,
        'vspread': vspread,
        'hvr': hvr,
        'center_ms': contour_center_ms,
        'start_ms': contour_start_ms,
        'end_ms': contour_end_ms,
        'dur_ms': dur_ms,
        'freq_centroid_Hz': freq_centroid_hz,
        'freq_min_Hz': freq_min_hz,
        'freq_max_Hz': freq_max_hz,
        'nr_oscillations': nr_oscillations,
        'spect_ok': True,
        'centroid_x': cx,
        'centroid_y': cy
    }


def _visualize_contours(
    original_image: np.ndarray,
    contours: List[np.ndarray],
    contour_features: List[Dict[str, Any]]
) -> np.ndarray:
    """
    Create visualization of detected contours with annotations.
    
    Args:
        original_image: Original spectrogram image
        contours: List of detected contours
        contour_features: List of feature dictionaries for each contour
    
    Returns:
        Annotated image with contours and metrics
    """
    output_image = original_image.copy()
    cont_thickness = 2
    
    for idx, (contour, features) in enumerate(zip(contours, contour_features)):
        # Extract features for display
        area = features['area']
        freq_centroid_hz = features['freq_centroid_Hz']
        nr_oscillations = features['nr_oscillations']
        circularity = features['circularity']
        hvr = features['hvr']
        cx, cy = features['centroid_x'], features['centroid_y']
        
        # Create annotation string
        metrics_str = (
            f"Blob:{idx}, Area:{area:.0f}, Freq:{freq_centroid_hz:.0f}, "
            f"NrOsc:{nr_oscillations:.0f}, Circ:{circularity:.0f}, HVR:{hvr:.0f}"
        )
        
        # Draw contour and annotations
        cont_color = (255, 0, 255)  # Magenta for valid contours
        metrics_color = (255, 0, 255)
        
        cv2.putText(
            output_image, metrics_str, (0, 30 + 15 * idx), 
            cv2.FONT_HERSHEY_SIMPLEX, 0.5, metrics_color, 1, cv2.LINE_AA
        )
        
        cv2.drawContours(
            output_image, [contour], -1, cont_color, cont_thickness
        )
        
        cv2.circle(output_image, (cx, cy), 4, (0, 0, 0), -1)
        
        # Add contour index
        cv2.putText(
            output_image, str(idx), (cx, cy), 
            cv2.FONT_HERSHEY_PLAIN, 1, (0, 0, 0), 1, cv2.LINE_AA
        )
    
    return output_image


def hfo_spectral_analysis(
    hfo_spect_img: np.ndarray,
    fs: Optional[int] = None,
    wdw_duration_ms: int = 1000,
    cwt_range_Hz: Tuple[int, int] = (60, 500),
    plot_ok: bool = False,
    fig_title: str = 'HFO Analysis',
    out_path: Optional[str] = None
) -> Tuple[np.ndarray, pd.DataFrame]:
    """
    Analyze HFO spectrograms to detect and characterize high-frequency oscillations.
    
    This function performs computer vision-based analysis on spectrogram images to identify
    and extract features from potential HFO events. The analysis includes image preprocessing,
    clustering-based segmentation, contour detection, and feature extraction.
    
    Args:
        hfo_spect_img: Input spectrogram image as numpy array
        fs: Sampling frequency (currently unused, kept for compatibility)
        wdw_duration_ms: Window duration in milliseconds for resizing
        cwt_range_Hz: Frequency range tuple (min_freq, max_freq) in Hz
        plot_ok: Whether to generate visualization output
        fig_title: Title for plots and saved images
        out_path: Output path for saved images (if applicable)
    
    Returns:
        Tuple containing:
        - objects: Annotated image with detected contours (if plot_ok=True, else empty array)
        - all_contours_df: DataFrame with extracted features for each detected contour
    
    Raises:
        ValueError: If input image is invalid or parameters are out of range
    """        
    # Input validation
    if hfo_spect_img is None or hfo_spect_img.size == 0:
        raise ValueError("Input spectrogram image cannot be None or empty")
    
    if len(cwt_range_Hz) != 2 or cwt_range_Hz[0] >= cwt_range_Hz[1]:
        raise ValueError("cwt_range_Hz must be a tuple of (min_freq, max_freq) with min < max")

    if hfo_spect_img.shape[1] != fs:
        raise ValueError("Input spectrogram width must match the sampling frequency, i.e. 1 second wdw length")

    # Calculate target dimensions
    img_resize_width = wdw_duration_ms
    img_resize_height = cwt_range_Hz[1] - cwt_range_Hz[0]
    
    logger.debug(f"Resizing spectrogram from {hfo_spect_img.shape} to ({img_resize_width}, {img_resize_height})")

    # Step 1: Resize spectrogram for standardized analysis
    resized_img = _resize_spectrogram(hfo_spect_img, img_resize_width, img_resize_height)
    
    # Step 2: Perform K-means clustering for segmentation
    #with timing_context("K-means clustering", logger):
    kmeans, cluster_labels = _perform_kmeans_clustering(resized_img)
    
    # Step 3: Extract red channel and calculate threshold
    _, _, red_channel = _extract_rgb_channels(resized_img)
    threshold_value = _calculate_optimal_threshold(red_channel, cluster_labels)
    logger.debug(f"Calculated threshold value: {threshold_value}")
    
    # Step 4: Apply binary thresholding
    _, binary_img = cv2.threshold(red_channel, threshold_value, 255, cv2.THRESH_BINARY)
    
    # Step 5: Find and filter contours
    valid_contours = _find_and_filter_contours(binary_img)        
    #logger.info(f"Found {len(valid_contours)} valid contours")
    
    # Step 6: Extract features from each contour
    all_contour_features = []
    for idx, contour in enumerate(valid_contours):
        features = _calculate_contour_features(contour, cwt_range_Hz, img_resize_width)
        features['idx'] = idx
        all_contour_features.append(features)
    
    # Step 7: Create visualization if requested
    visualization_img = None
    if plot_ok and len(valid_contours) > 0:
        with timing_context("Visualization generation", logger):
            visualization_img = np.array([])  # Empty array by default
            visualization_img = _visualize_contours(resized_img, valid_contours, all_contour_features)
    
    # Step 8: Convert results to DataFrame
    contours_df = pd.DataFrame(all_contour_features)        
    #logger.info(f"Analysis complete. Extracted features for {len(contours_df)} contours")
    
    return visualization_img, contours_df


def create_summary_visualization(
    original_img: np.ndarray,
    clustered_img: np.ndarray,
    red_channel: np.ndarray,
    binary_img: np.ndarray,
    annotated_img: np.ndarray,
    fig_title: str
) -> None:
    """
    Create comprehensive visualization showing all analysis steps.
    
    Args:
        original_img: Original spectrogram
        clustered_img: Clustered visualization
        red_channel: Red channel extraction
        binary_img: Binary thresholded image
        annotated_img: Final annotated result
        fig_title: Title for the visualization windows
    """
    new_size = (860, 400)
    
    # Create comparison panels
    panel_a = np.concatenate((
        cv2.resize(original_img, new_size, interpolation=cv2.INTER_LINEAR),
        cv2.resize(clustered_img, new_size, interpolation=cv2.INTER_LINEAR)
    ), axis=1)
    cv2.line(panel_a, (new_size[0], 0), (new_size[0], new_size[1]), (255, 255, 255), 1)
    
    panel_b = np.concatenate((
        cv2.resize(red_channel, new_size, interpolation=cv2.INTER_LINEAR),
        cv2.resize(binary_img, new_size, interpolation=cv2.INTER_LINEAR)
    ), axis=1)
    cv2.line(panel_b, (new_size[0], 0), (new_size[0], new_size[1]), (255, 255, 255), 1)
    
    panel_c = np.concatenate((
        cv2.resize(original_img, new_size, interpolation=cv2.INTER_LINEAR),
        cv2.resize(annotated_img, new_size, interpolation=cv2.INTER_LINEAR)
    ), axis=1)
    cv2.line(panel_c, (new_size[0], 0), (new_size[0], new_size[1]), (255, 255, 255), 1)
    
    # Display windows
    cv2.imshow(f"HFO Spectro vs Clustered Spectro - {fig_title}", panel_a)
    cv2.imshow(f"Red Channel vs Thresholded Red Channel - {fig_title}", panel_b)
    cv2.imshow(f"HFO Spectro vs Contours - {fig_title}", panel_c)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


if __name__ == "__main__":
    # Example usage and testing
    pass
