import numpy as np
import random
import cv2 as cv
from mtcnn import MTCNN
from math import ceil
from sklearn.preprocessing import MinMaxScaler
from typing import Tuple, List, Union, Optional, Dict, Any, cast

import torch
from torch.utils import data
import logging
from functools import lru_cache

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def resize(frames: Union[np.ndarray, List[np.ndarray]], 
           dynamic_det: bool, 
           det_length: int,
           w: int, 
           h: int, 
           larger_box: bool, 
           crop_face: bool, 
           larger_box_size: float) -> Union[np.ndarray, List[np.ndarray]]:
    """
    Resize frames and optionally crop to detected face regions.
    
    Args:
        frames: Input video frames
        dynamic_det: Whether to perform dynamic face detection
        det_length: Interval for dynamic detection
        w: Target width
        h: Target height
        larger_box: Whether to enlarge the detected face region
        crop_face: Whether to crop the frames to face regions
        larger_box_size: Size multiplier for enlarging bounding box
        
    Returns:
        Resized (and optionally cropped) frames
    """
    # Input validation
    if len(frames) == 0:
        logger.warning("Empty frames list provided to resize function")
        return np.array([]) if isinstance(frames, np.ndarray) else []
    
    if det_length <= 0:
        logger.warning(f"Invalid detection length: {det_length}, using default value of 30")
        det_length = 30
    
    # Calculate number of detections needed
    if dynamic_det:
        det_num = ceil(len(frames) / det_length)
    else:
        det_num = 1
        
    face_regions = []
    detector = MTCNN()
    
    # Detect face regions
    for idx in range(det_num):
        frame_idx = min(det_length * idx, len(frames) - 1)
        if crop_face:
            try:
                face_regions.append(
                    facial_detection(
                        detector, 
                        frames[frame_idx],
                        larger_box, 
                        larger_box_size
                    )
                )
            except Exception as e:
                logger.error(f"Face detection failed: {e}")
                # Fallback to full frame
                if isinstance(frames, np.ndarray):
                    face_regions.append([0, 0, frames.shape[2], frames.shape[1]])
                else:
                    face_regions.append([0, 0, frames[0].shape[1], frames[0].shape[0]])
        else:  # No cropping
            if isinstance(frames, np.ndarray):
                face_regions.append([0, 0, frames.shape[2], frames.shape[1]])
            else:
                face_regions.append([0, 0, frames[0].shape[1], frames[0].shape[0]])
    
    face_regions_array = np.asarray(face_regions, dtype=np.int32)
    resized_frames = []

    # Process each frame
    for i in range(len(frames)):
        frame = frames[i]
        
        # Select appropriate face region
        if dynamic_det:
            reference_index = min(i // det_length, len(face_regions_array) - 1)
        else:
            reference_index = 0
            
        # Crop if needed
        if crop_face:
            face_region = face_regions_array[reference_index]
            y1 = max(face_region[1], 0)
            y2 = min(face_region[3], frame.shape[0])
            x1 = max(face_region[0], 0)
            x2 = min(face_region[2], frame.shape[1])
            
            # Check if valid region
            if x2 <= x1 or y2 <= y1:
                logger.warning(f"Invalid crop region: [{x1}, {y1}, {x2}, {y2}], using full frame")
                frame_cropped = frame
            else:
                frame_cropped = frame[y1:y2, x1:x2]
        else:
            frame_cropped = frame
            
        # Resize if dimensions are specified
        if w > 0 and h > 0:
            try:
                # Add padding and then crop to handle border effects
                resized = cv.resize(
                    frame_cropped, 
                    (w + 4, h + 4),
                    interpolation=cv.INTER_CUBIC
                )[2: w + 2, 2: h + 2, :]
                resized_frames.append(resized)
            except Exception as e:
                logger.error(f"Resize failed for frame {i}: {e}")
                # Add a black frame as fallback
                resized_frames.append(np.zeros((h, w, 3), dtype=frame.dtype))
        else:
            resized_frames.append(frame_cropped)
            
    # Return as array or list based on input type
    if isinstance(frames, np.ndarray) and w > 0 and h > 0:
        return np.asarray(resized_frames, dtype=frames.dtype)
    else:
        return resized_frames


def facial_detection(detector: MTCNN, 
                     frame: np.ndarray, 
                     larger_box: bool = False, 
                     larger_box_size: float = 1.0) -> List[int]:
    """
    Detect face region using MTCNN.
    
    Args:
        detector: MTCNN detector instance
        frame: Input image frame
        larger_box: Whether to enlarge the bounding box (for handling movement)
        larger_box_size: Size multiplier for the bounding box
        
    Returns:
        Face region coordinates [x1, y1, x2, y2]
    """
    # Input validation
    if frame is None or frame.size == 0:
        raise ValueError("Empty frame provided to facial_detection")
        
    if larger_box_size <= 0:
        logger.warning(f"Invalid larger_box_size: {larger_box_size}, using default value of 1.0")
        larger_box_size = 1.0
        
    if frame.ndim != 3 or frame.shape[2] != 3:
        raise ValueError(f"Expected RGB image with 3 channels, got shape {frame.shape}")
    
    # Convert to RGB if needed (MTCNN expects RGB)
    if frame.dtype != np.uint8:
        logger.warning("Converting frame to uint8 for face detection")
        frame = (frame * 255).astype(np.uint8) if frame.max() <= 1.0 else frame.astype(np.uint8)
    
    # Detect faces
    try:
        face_zone = detector.detect_faces(frame)
    except Exception as e:
        logger.error(f"MTCNN detection failed: {e}")
        return [0, 0, frame.shape[1], frame.shape[0]]
    
    # Handle no face detected
    if len(face_zone) < 1:
        logger.warning("No face detected! Using full frame.")
        return [0, 0, frame.shape[1], frame.shape[0]]
        
    # Handle multiple faces
    if len(face_zone) >= 2:
        logger.warning(f"Multiple faces detected ({len(face_zone)}). Using the largest one.")
        # Find the face with the largest area
        largest_area = 0
        largest_idx = 0
        for i, face in enumerate(face_zone):
            area = face['box'][2] * face['box'][3]
            if area > largest_area:
                largest_area = area
                largest_idx = i
        result = face_zone[largest_idx]['box'].copy()
    else:
        result = face_zone[0]['box'].copy()
    
    # Get the bounding box dimensions
    h = result[3]
    w = result[2]
    
    # Convert from [x, y, w, h] to [x1, y1, x2, y2]
    result[2] += result[0]  # x2 = x1 + w
    result[3] += result[1]  # y2 = y1 + h
    
    # Enlarge the bounding box if requested
    if larger_box:
        logger.debug("Using enlarged bounding box")
        result[0] = round(max(0, result[0] - (larger_box_size - 1.0) / 2 * w))
        result[1] = round(max(0, result[1] - (larger_box_size - 1.0) / 2 * h))
        result[2] = round(min(frame.shape[1], result[2] + (larger_box_size - 1.0) / 2 * w))
        result[3] = round(min(frame.shape[0], result[3] + (larger_box_size - 1.0) / 2 * h))
        
    return result


def chunk(frames: np.ndarray, 
          gts: np.ndarray, 
          chunk_length: int, 
          chunk_stride: int = -1) -> Tuple[np.ndarray, np.ndarray]:
    """
    Chunk data into clips of specified length.
    
    Args:
        frames: Input video frames
        gts: Ground truth signals
        chunk_length: Length of each chunk
        chunk_stride: Stride between chunks (defaults to chunk_length)
        
    Returns:
        Tuple of chunked frames and ground truths
    """
    # Input validation
    if chunk_length <= 0:
        raise ValueError(f"Invalid chunk_length: {chunk_length}")
    
    if len(frames) < chunk_length or len(gts) < chunk_length:
        raise ValueError(f"Input data too short for chunk_length: {chunk_length}")
        
    if chunk_stride < 0:
        chunk_stride = chunk_length
    
    # Efficient chunking using list comprehension
    frames_clips = [
        frames[i: i + chunk_length]
        for i in range(0, len(frames) - chunk_length + 1, chunk_stride)
    ]
    
    bvps_clips = [
        gts[i: i + chunk_length]
        for i in range(0, len(gts) - chunk_length + 1, chunk_stride)
    ]
    
    return np.array(frames_clips), np.array(bvps_clips)


@lru_cache(maxsize=32)
def _get_block_indices(h: int, w: int, h_num: int, w_num: int) -> List[Tuple[slice, slice]]:
    """
    Generate block indices for efficient frame division.
    Results are cached for performance.
    
    Args:
        h: Frame height
        w: Frame width
        h_num: Number of blocks in height
        w_num: Number of blocks in width
        
    Returns:
        List of (row_slice, col_slice) for each block
    """
    h_len = h // h_num
    w_len = w // w_num
    
    indices = []
    for i in range(h_num):
        for j in range(w_num):
            row_start = i * h_len
            row_end = row_start + h_len
            col_start = j * w_len
            col_end = col_start + w_len
            indices.append((slice(row_start, row_end), slice(col_start, col_end)))
            
    return indices


def get_blocks(frame: np.ndarray, h_num: int = 5, w_num: int = 5) -> List[np.ndarray]:
    """
    Divide frame into h_num × w_num blocks.
    Uses cached block indices for better performance.
    
    Args:
        frame: Input frame
        h_num: Number of blocks in height
        w_num: Number of blocks in width
        
    Returns:
        List of block arrays
    """
    h, w, _ = frame.shape
    
    # Get cached indices
    block_indices = _get_block_indices(h, w, h_num, w_num)
    
    # Extract blocks using the indices
    blocks = [frame[row_slice, col_slice] for row_slice, col_slice in block_indices]
    
    return blocks


def get_STMap(frames: Union[np.ndarray, List[np.ndarray]], 
              hrs: np.ndarray, 
              Fs: float, 
              chunk_length: int = 300, 
              roi_num: int = 25) -> Tuple[np.ndarray, np.ndarray]:
    """
    Generate Spatial-Temporal Maps from video frames.
    
    Args:
        frames: Input video frames (T x H x W x C) or list[H x W x C]
        hrs: Heart rate values
        Fs: Sampling frequency
        chunk_length: Length of each chunk
        roi_num: Number of ROIs (regions of interest)
        
    Returns:
        Tuple of (STMaps, average_hrs)
        STMaps shape: clip_num x chunk_length x roi_num x C
    """
    # Input validation
    if len(frames) < chunk_length:
        logger.warning(f"Input frames length ({len(frames)}) < chunk_length ({chunk_length})")
        if len(frames) == 0:
            return np.array([]), np.array([])
        chunk_length = min(len(frames), chunk_length)
    
    # Half-second stride
    chunk_stride = max(1, round(Fs / 2))
    clip_num = (len(frames) - chunk_length + chunk_stride) // chunk_stride
    
    if clip_num <= 0:
        logger.warning("No clips can be generated with current parameters")
        return np.array([]), np.array([])
    
    # Pre-allocate arrays for better memory efficiency
    STMaps = []
    average_hrs = np.zeros(clip_num, dtype=np.float32)
    scaler = MinMaxScaler()
    
    # Determine if we need h_num and w_num based on roi_num
    h_num = w_num = int(np.sqrt(roi_num))
    if h_num * w_num != roi_num:
        logger.warning(f"roi_num {roi_num} is not a perfect square, using {h_num*w_num} blocks")
    
    for idx, i in enumerate(range(0, len(frames) - chunk_length + 1, chunk_stride)):
        if idx >= clip_num:
            break
            
        # Create temporary storage for this chunk
        temp = np.zeros((chunk_length, roi_num, 3), dtype=np.float32)
        
        # Process each frame in the chunk
        for j, frame in enumerate(frames[i: i + chunk_length]):
            # Get blocks for this frame
            blocks = get_blocks(frame, h_num, w_num)
            
            # Calculate mean for each block and channel - more efficiently
            block_means = np.array([np.mean(block, axis=(0, 1)) for block in blocks[:roi_num]])
            temp[j, :len(block_means)] = block_means
        
        # Apply min-max normalization to each temporal signal
        # Scale values to [0, 255] range - more efficiently with vectorization
        for j in range(roi_num):
            for c in range(3):  # For each color channel
                channel_data = temp[:, j, c].reshape(-1, 1)
                # Skip normalization if all values are the same
                if np.min(channel_data) != np.max(channel_data):
                    try:
                        scaled = scaler.fit_transform(channel_data)
                        temp[:, j, c] = (scaled * 255.).reshape(-1)
                    except Exception as e:
                        logger.warning(f"Normalization failed: {e}, using original values")
                        # Keep original values if normalization fails
                
        # Convert to uint8 after all processing
        temp = temp.astype(np.uint8)
        STMaps.append(temp)
        
        # Calculate average heart rate for this time segment
        hr_start = int(i // Fs)
        hr_end = min(len(hrs) - 1, int((i + chunk_length) // Fs))
        if hr_start < hr_end:
            average_hrs[idx] = np.mean(hrs[hr_start:hr_end])
    
    # Verify clip count
    if len(STMaps) != clip_num:
        logger.warning(f"Expected {clip_num} clips but got {len(STMaps)}!")
    
    return np.asarray(STMaps), average_hrs[:len(STMaps)]


def randomMask(x: np.ndarray) -> np.ndarray:
    """
    Randomly mask portions of spatial-temporal maps.
    
    During training, half of the generated maps are randomly masked,
    with mask length varying from 10 to 30 frames.
    
    Args:
        x: Spatial-temporal maps, shape: clip_num x chunk_length x roi_num x C
        
    Returns:
        Masked spatial-temporal maps
    """
    # Input validation
    if x.size == 0:
        return x
        
    # Create a copy to avoid modifying the original
    x_masked = x.copy()
    
    for i, stmap in enumerate(x_masked):
        # 50% chance of masking
        if random.random() < 0.5:
            continue
            
        # Random mask length between 10-30 frames
        mask_len = random.randint(10, 30)
        
        # Safety check for short sequences
        if mask_len >= len(stmap):
            mask_len = max(1, len(stmap) // 3)
            
        # Random starting position (ensuring it fits within the map)
        max_start = max(0, len(stmap) - mask_len - 1)
        idx = random.randint(0, max_start)
        
        # Apply the mask
        stmap[idx: idx + mask_len, :, :] = 0
        
    return x_masked


def normalize_frame(frame: np.ndarray) -> np.ndarray:
    """
    Normalize pixel values from [0, 255] to [-1, 1].
    
    Args:
        frame: Input frame with values in range [0, 255]
        
    Returns:
        Normalized frame with values in range [-1, 1]
    """
    if frame.size == 0:
        return frame
        
    # More numerically stable normalization
    if frame.dtype == np.uint8:
        return (frame.astype(np.float32) - 127.5) / 128.0
    else:
        # If not uint8, assume it's already in a floating point format
        max_val = np.max(np.abs(frame))
        if max_val > 0:
            return frame / max_val
        return frame


def standardize(data: np.ndarray) -> np.ndarray:
    """
    Standardize data to zero mean and unit variance.
    
    Args:
        data: Input data array
        
    Returns:
        Standardized data (x - μ) / σ
    """
    # Handle empty arrays
    if data.size == 0:
        return data
        
    # Handle constant arrays
    std_val = np.std(data)
    if std_val < 1e-10:
        return np.zeros_like(data)
        
    # Z-score normalization
    mean_val = np.mean(data)
    result = (data - mean_val) / std_val
    
    # Handle any NaNs resulting from division by zero
    return np.nan_to_num(result, nan=0.0)


class VideoProcessor:
    """
    A class for processing video frames for heart rate estimation.
    Encapsulates functionality for face detection, frame resizing, and STMap generation.
    """
    
    def __init__(self, use_mtcnn: bool = True, cache_size: int = 10):
        """
        Initialize the video processor.
        
        Args:
            use_mtcnn: Whether to use MTCNN for face detection
            cache_size: Size of the internal cache for processed results
        """
        self.detector = MTCNN() if use_mtcnn else None
        self.cache_size = cache_size
        self._cache: Dict[str, Any] = {}
        
    def _get_cache_key(self, **kwargs) -> str:
        """Generate a cache key from function parameters"""
        return str(hash(frozenset(kwargs.items())))
        
    def clear_cache(self) -> None:
        """Clear the internal cache"""
        self._cache.clear()
        
    def detect_faces(self, frame: np.ndarray) -> List[Dict[str, Any]]:
        """
        Detect faces in a single frame.
        
        Args:
            frame: Input frame
            
        Returns:
            List of detected faces with bounding boxes and landmarks
        """
        if self.detector is None:
            raise ValueError("MTCNN detector not initialized")
            
        return self.detector.detect_faces(frame)
    
    def process_video(self, 
                      frames: Union[np.ndarray, List[np.ndarray]],
                      hrs: Optional[np.ndarray] = None,
                      Fs: float = 30.0,
                      resize_dims: Tuple[int, int] = (0, 0),
                      crop_face: bool = True,
                      dynamic_det: bool = False,
                      det_length: int = 30,
                      generate_stmaps: bool = True,
                      use_cache: bool = True) -> Dict[str, Any]:
        """
        Process video frames through the entire pipeline.
        
        Args:
            frames: Input video frames
            hrs: Heart rate ground truth values (optional)
            Fs: Sampling frequency
            resize_dims: (width, height) for resizing, (0,0) for no resizing
            crop_face: Whether to crop to face regions
            dynamic_det: Whether to perform dynamic face detection
            det_length: Interval for dynamic detection
            generate_stmaps: Whether to generate STMaps
            use_cache: Whether to use caching for results
            
        Returns:
            Dictionary with processed data
        """
        # Input validation
        if len(frames) == 0:
            logger.warning("Empty frames list provided to process_video")
            return {"frames": frames}
            
        # Check if result is in cache
        if use_cache:
            cache_key = self._get_cache_key(
                frame_count=len(frames),
                resize_dims=resize_dims,
                crop_face=crop_face,
                dynamic_det=dynamic_det,
                generate_stmaps=generate_stmaps
            )
            if cache_key in self._cache:
                logger.info("Using cached result")
                return self._cache[cache_key]
        
        results: Dict[str, Any] = {}
        
        # Resize and crop if requested
        if resize_dims != (0, 0) or crop_face:
            w, h = resize_dims
            try:
                resized_frames = resize(
                    frames, 
                    dynamic_det, 
                    det_length,
                    w, 
                    h, 
                    larger_box=True, 
                    crop_face=crop_face, 
                    larger_box_size=1.2
                )
                results['frames'] = resized_frames
            except Exception as e:
                logger.error(f"Frame resizing failed: {e}")
                results['frames'] = frames
                results['error'] = str(e)
        else:
            results['frames'] = frames
            
        # Generate STMaps if requested
        if generate_stmaps and hrs is not None:
            try:
                stmaps, avg_hrs = get_STMap(
                    results['frames'],
                    hrs,
                    Fs,
                    chunk_length=300
                )
                results['stmaps'] = stmaps
                results['average_hrs'] = avg_hrs
            except Exception as e:
                logger.error(f"STMap generation failed: {e}")
                results['error'] = str(e)
        
        # Cache the result
        if use_cache:
            # Maintain cache size
            if len(self._cache) >= self.cache_size:
                # Remove a random key
                self._cache.pop(next(iter(self._cache)))
            self._cache[cache_key] = results
            
        return results
        
    def extract_features(self, 
                         stmaps: np.ndarray, 
                         method: str = 'mean') -> np.ndarray:
        """
        Extract features from spatial-temporal maps.
        
        Args:
            stmaps: Spatial-temporal maps
            method: Feature extraction method ('mean', 'std', 'max', 'min')
            
        Returns:
            Extracted features
        """
        if stmaps.size == 0:
            return np.array([])
            
        if method == 'mean':
            # Average over all ROIs for each time step
            return np.mean(stmaps, axis=2)
        elif method == 'std':
            # Standard deviation over ROIs
            return np.std(stmaps, axis=2)
        elif method == 'max':
            # Max value over ROIs
            return np.max(stmaps, axis=2)
        elif method == 'min':
            # Min value over ROIs
            return np.min(stmaps, axis=2)
        else:
            raise ValueError(f"Unknown feature extraction method: {method}")
