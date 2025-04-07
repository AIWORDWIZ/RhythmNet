import numpy as np
import random
import cv2 as cv
from mtcnn import MTCNN
from math import ceil
from sklearn.preprocessing import MinMaxScaler
from typing import Tuple, List, Union, Optional

import torch
from torch.utils import data
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def resize(frames: Union[np.ndarray, List], 
           dynamic_det: bool, 
           det_length: int,
           w: int, 
           h: int, 
           larger_box: bool, 
           crop_face: bool, 
           larger_box_size: float) -> Union[np.ndarray, List]:
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
    # Calculate number of detections needed
    if dynamic_det:
        det_num = ceil(len(frames) / det_length)
    else:
        det_num = 1
        
    face_regions = []
    detector = MTCNN()
    
    # Detect face regions
    for idx in range(det_num):
        if crop_face:
            face_regions.append(
                facial_detection(
                    detector, 
                    frames[min(det_length * idx, len(frames) - 1)],
                    larger_box, 
                    larger_box_size
                )
            )
        else:  # No cropping
            face_regions.append([0, 0, frames.shape[1], frames.shape[2]])
    
    face_regions_array = np.asarray(face_regions, dtype='int')
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
            frame = frame[
                max(face_region[1], 0):min(face_region[3], frame.shape[0]),
                max(face_region[0], 0):min(face_region[2], frame.shape[1])
            ]
            
        # Resize if dimensions are specified
        if w > 0 and h > 0:
            # Add padding and then crop to handle border effects
            resized = cv.resize(
                frame, 
                (w + 4, h + 4),
                interpolation=cv.INTER_CUBIC
            )[2: w + 2, 2: h + 2, :]
            resized_frames.append(resized)
        else:
            resized_frames.append(frame)
            
    # Return as array or list based on input dimensions
    if w > 0 and h > 0:
        return np.asarray(resized_frames)
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
    face_zone = detector.detect_faces(frame)
    
    # Handle no face detected
    if len(face_zone) < 1:
        logger.warning("No face detected! Using full frame.")
        return [0, 0, frame.shape[1], frame.shape[0]]
        
    # Handle multiple faces
    if len(face_zone) >= 2:
        logger.warning("Multiple faces detected. Using the largest one.")
    
    # Get the bounding box
    result = face_zone[0]['box']
    h = result[3]
    w = result[2]
    
    # Convert from [x, y, w, h] to [x1, y1, x2, y2]
    result[2] += result[0]  # x2 = x1 + w
    result[3] += result[1]  # y2 = y1 + h
    
    # Enlarge the bounding box if requested
    if larger_box:
        logger.info("Using enlarged bounding box")
        result[0] = round(max(0, result[0] + (1. - larger_box_size) / 2 * w))
        result[1] = round(max(0, result[1] + (1. - larger_box_size) / 2 * h))
        result[2] = round(min(frame.shape[1], result[0] + larger_box_size * w))
        result[3] = round(min(frame.shape[0], result[1] + larger_box_size * h))
        
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
    if chunk_stride < 0:
        chunk_stride = chunk_length
        
    frames_clips = [
        frames[i: i + chunk_length]
        for i in range(0, len(frames) - chunk_length + 1, chunk_stride)
    ]
    
    bvps_clips = [
        gts[i: i + chunk_length]
        for i in range(0, len(gts) - chunk_length + 1, chunk_stride)
    ]
    
    return np.array(frames_clips), np.array(bvps_clips)


def get_blocks(frame: np.ndarray, h_num: int = 5, w_num: int = 5) -> List[np.ndarray]:
    """
    Divide frame into h_num × w_num blocks.
    
    Args:
        frame: Input frame
        h_num: Number of blocks in height
        w_num: Number of blocks in width
        
    Returns:
        List of block arrays
    """
    h, w, _ = frame.shape
    h_len = h // h_num
    w_len = w // w_num
    
    blocks = []
    h_idx = [i * h_len for i in range(h_num)]
    w_idx = [i * w_len for i in range(w_num)]
    
    for i in h_idx:
        for j in w_idx:
            blocks.append(frame[i: i + h_len, j: j + w_len, :])
            
    return blocks


def get_STMap(frames: Union[np.ndarray, List], 
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
    # Half-second stride
    chunk_stride = round(Fs / 2)
    clip_num = (len(frames) - chunk_length + chunk_stride) // chunk_stride
    
    # Pre-allocate arrays for better performance
    STMaps = []
    average_hrs = []
    scaler = MinMaxScaler()
    
    for i in range(0, len(frames) - chunk_length + 1, chunk_stride):
        # Create temporary storage for this chunk
        temp = np.zeros((chunk_length, roi_num, 3), dtype=np.float32)
        
        # Process each frame in the chunk
        for j, frame in enumerate(frames[i: i + chunk_length]):
            blocks = get_blocks(frame)
            
            # Calculate mean for each block and channel
            for k, block in enumerate(blocks):
                if k < roi_num:  # Safety check
                    # More efficient to use numpy's mean directly
                    temp[j, k, :] = np.mean(block, axis=(0, 1))
        
        # Apply min-max normalization to each temporal signal
        # Scale values to [0, 255] range
        for j in range(roi_num):
            for c in range(3):  # For each color channel
                scaled = scaler.fit_transform(temp[:, j, c].reshape(-1, 1))
                temp[:, j, c] = (scaled * 255.).reshape(-1).astype(np.uint8)
        
        STMaps.append(temp)
        
        # Calculate average heart rate for this time segment
        hr_start = int(i // Fs)
        hr_end = min(len(hrs) - 1, int((i + chunk_length) // Fs))
        average_hrs.append(np.mean(hrs[hr_start:hr_end]))
    
    # Verify clip count
    if len(STMaps) != clip_num:
        logger.error(f"Expected {clip_num} clips but got {len(STMaps)}!")
    
    return np.asarray(STMaps), np.asarray(average_hrs)


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
    # Create a copy to avoid modifying the original
    x_masked = x.copy()
    
    for i, stmap in enumerate(x_masked):
        # 50% chance of masking
        if random.random() < 0.5:
            continue
            
        # Random mask length between 10-30 frames
        mask_len = random.randint(10, 30)
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
    return (frame - 127.5) / 128


def standardize(data: np.ndarray) -> np.ndarray:
    """
    Standardize data to zero mean and unit variance.
    
    Args:
        data: Input data array
        
    Returns:
        Standardized data (x - μ) / σ
    """
    # Handle empty or constant arrays
    if len(data) == 0 or np.std(data) == 0:
        return np.zeros_like(data)
        
    # Z-score normalization
    data = data - np.mean(data)
    data = data / np.std(data)
    
    # Handle any NaNs resulting from division by zero
    data = np.nan_to_num(data, nan=0.0)
    
    return data


class VideoProcessor:
    """
    A class for processing video frames for heart rate estimation.
    Encapsulates functionality for face detection, frame resizing, and STMap generation.
    """
    
    def __init__(self, use_mtcnn: bool = True):
        """
        Initialize the video processor.
        
        Args:
            use_mtcnn: Whether to use MTCNN for face detection
        """
        self.detector = MTCNN() if use_mtcnn else None
        
    def process_video(self, 
                      frames: Union[np.ndarray, List],
                      hrs: Optional[np.ndarray] = None,
                      Fs: float = 30.0,
                      resize_dims: Tuple[int, int] = (0, 0),
                      crop_face: bool = True,
                      dynamic_det: bool = False,
                      det_length: int = 30,
                      generate_stmaps: bool = True) -> dict:
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
            
        Returns:
            Dictionary with processed data
        """
        results = {}
        
        # Resize and crop if requested
        if resize_dims != (0, 0) or crop_face:
            w, h = resize_dims
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
        else:
            results['frames'] = frames
            
        # Generate STMaps if requested
        if generate_stmaps and hrs is not None:
            stmaps, avg_hrs = get_STMap(
                results['frames'],
                hrs,
                Fs,
                chunk_length=300
            )
            results['stmaps'] = stmaps
            results['average_hrs'] = avg_hrs
            
        return results
