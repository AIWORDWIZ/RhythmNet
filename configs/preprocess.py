from typing import Optional, Dict, Any, Union, TextIO
from dataclasses import dataclass, field, asdict
import json
import os
import logging


logger = logging.getLogger(__name__)


@dataclass
class FrameConfig:
    """
    Base configuration for RhythmNet frame processing pipeline.
    
    Attributes:
        input_path: Source directory for input frames
        cache_path: Directory to store processed frames
        record_path: CSV file to log processing results
        modify: Whether to allow modifications to the frames
        width: Target width (-1 means no resizing)
        height: Target height (-1 means no resizing)
        dynamic_detection: Whether to perform face detection at intervals
        detection_frequency: How often to perform detection (in frames)
        crop_face: Whether to crop frames to detected face regions
        large_face_box: Whether to use enlarged bounding boxes
        face_box_scale: Size multiplier for face bounding boxes
        do_chunk: Whether to chunk the frames sequence
        chunk_length: Length of each chunk
        chunk_stride: Stride between chunks (-1 means equal to length)
    """
    # Path configurations
    input_path: str = ""
    cache_path: str = "./cache"
    record_path: str = "./record.csv"
    
    # Processing flags
    modify: bool = True
    dynamic_detection: bool = False
    crop_face: bool = True
    large_face_box: bool = True
    do_chunk: bool = True
    
    # Dimensional parameters
    width: int = -1
    height: int = -1
    detection_frequency: int = 1
    face_box_scale: float = 1.2
    chunk_length: int = 300
    chunk_stride: int = -1
    
    # Internal state
    _frozen: bool = field(default=False, repr=False)
    
    def __post_init__(self):
        """Initialize derived values and perform validation."""
        # Handle backward compatibility with uppercase attribute names
        self._initialize_aliases()
        self.validate()
    
    def _initialize_aliases(self):
        """Set up property getters/setters for backward compatibility."""
        # This allows old code to still access parameters with uppercase names
        self.__dict__['W'] = self.width
        self.__dict__['H'] = self.height
        self.__dict__['MODIFY'] = self.modify
        self.__dict__['DYNAMIC_DETECTION'] = self.dynamic_detection
        self.__dict__['DYNAMIC_DETECTION_FREQUENCY'] = self.detection_frequency
        self.__dict__['CROP_FACE'] = self.crop_face
        self.__dict__['LARGE_FACE_BOX'] = self.large_face_box
        self.__dict__['LARGE_BOX_COEF'] = self.face_box_scale
        self.__dict__['DO_CHUNK'] = self.do_chunk
        self.__dict__['CHUNK_LENGTH'] = self.chunk_length
        self.__dict__['CHUNK_STRIDE'] = self.chunk_stride
    
    def __setattr__(self, name, value):
        """Override to support configuration freezing and maintain aliases."""
        if getattr(self, '_frozen', False) and name != '_frozen':
            raise AttributeError(f"Cannot modify frozen configuration: {name}")
            
        # For backward compatibility, update both the new and old attribute names
        if name == 'W':
            super().__setattr__('width', value)
        elif name == 'H':
            super().__setattr__('height', value)
        elif name == 'MODIFY':
            super().__setattr__('modify', value)
        elif name == 'DYNAMIC_DETECTION':
            super().__setattr__('dynamic_detection', value)
        elif name == 'DYNAMIC_DETECTION_FREQUENCY':
            super().__setattr__('detection_frequency', value)
        elif name == 'CROP_FACE':
            super().__setattr__('crop_face', value)
        elif name == 'LARGE_FACE_BOX':
            super().__setattr__('large_face_box', value)
        elif name == 'LARGE_BOX_COEF':
            super().__setattr__('face_box_scale', value)
        elif name == 'DO_CHUNK':
            super().__setattr__('do_chunk', value)
        elif name == 'CHUNK_LENGTH':
            super().__setattr__('chunk_length', value)
        elif name == 'CHUNK_STRIDE':
            super().__setattr__('chunk_stride', value)
            
        # Update the main attribute
        super().__setattr__(name, value)
        
        # Keep aliases in sync
        if name == 'width':
            self.__dict__['W'] = value
        elif name == 'height':
            self.__dict__['H'] = value
        elif name == 'modify':
            self.__dict__['MODIFY'] = value
        elif name == 'dynamic_detection':
            self.__dict__['DYNAMIC_DETECTION'] = value
        elif name == 'detection_frequency':
            self.__dict__['DYNAMIC_DETECTION_FREQUENCY'] = value
        elif name == 'crop_face':
            self.__dict__['CROP_FACE'] = value
        elif name == 'large_face_box':
            self.__dict__['LARGE_FACE_BOX'] = value
        elif name == 'face_box_scale':
            self.__dict__['LARGE_BOX_COEF'] = value
        elif name == 'do_chunk':
            self.__dict__['DO_CHUNK'] = value
        elif name == 'chunk_length':
            self.__dict__['CHUNK_LENGTH'] = value
        elif name == 'chunk_stride':
            self.__dict__['CHUNK_STRIDE'] = value
    
    def validate(self) -> None:
        """
        Validate the configuration parameters.
        
        Raises:
            ValueError: If any parameter is invalid
        """
        if self.chunk_length <= 0:
            raise ValueError("chunk_length must be positive")
            
        if self.width < -1 or self.height < -1:
            raise ValueError("width and height must be -1 or positive")
            
        if self.detection_frequency <= 0:
            raise ValueError("detection_frequency must be positive")
            
        if self.face_box_scale <= 0:
            raise ValueError("face_box_scale must be positive")
            
        # Set chunk_stride to chunk_length if it's -1
        if self.chunk_stride == -1:
            self.chunk_stride = self.chunk_length
    
    def freeze(self) -> 'FrameConfig':
        """
        Freeze the configuration to prevent further modifications.
        
        Returns:
            Self for method chaining
        """
        self._frozen = True
        return self
    
    def unfreeze(self) -> 'FrameConfig':
        """
        Unfreeze the configuration to allow modifications.
        
        Returns:
            Self for method chaining
        """
        self._frozen = False
        return self
    
    def is_frozen(self) -> bool:
        """
        Check if the configuration is frozen.
        
        Returns:
            True if frozen, False otherwise
        """
        return self._frozen
    
    def to_dict(self) -> Dict[str, Any]:
        """
        Convert configuration to a dictionary.
        
        Returns:
            Dictionary representation of the configuration
        """
        return {
            "input_path": self.input_path,
            "cache_path": self.cache_path,
            "record_path": self.record_path,
            "modify": self.modify,
            "width": self.width,
            "height": self.height,
            "dynamic_detection": self.dynamic_detection,
            "detection_frequency": self.detection_frequency,
            "crop_face": self.crop_face,
            "large_face_box": self.large_face_box,
            "face_box_scale": self.face_box_scale,
            "do_chunk": self.do_chunk,
            "chunk_length": self.chunk_length,
            "chunk_stride": self.chunk_stride
        }
    
    def save(self, file_path: str) -> None:
        """
        Save configuration to a JSON file.
        
        Args:
            file_path: Path to the output file
            
        Raises:
            IOError: If the file cannot be written
        """
        try:
            os.makedirs(os.path.dirname(file_path), exist_ok=True)
            with open(file_path, 'w') as f:
                json.dump(self.to_dict(), f, indent=2)
            logger.info(f"Configuration saved to {file_path}")
        except IOError as e:
            logger.error(f"Failed to save configuration: {e}")
            raise
    
    @classmethod
    def load(cls, file_path: str) -> 'FrameConfig':
        """
        Load configuration from a JSON file.
        
        Args:
            file_path: Path to the input file
            
        Returns:
            Loaded configuration object
            
        Raises:
            IOError: If the file cannot be read
            ValueError: If the file contains invalid configuration
        """
        try:
            with open(file_path, 'r') as f:
                config_dict = json.load(f)
            
            # Create a new instance
            config = cls()
            
            # Update with loaded values
            for key, value in config_dict.items():
                if hasattr(config, key):
                    setattr(config, key, value)
            
            # Validate the loaded configuration
            config.validate()
            logger.info(f"Configuration loaded from {file_path}")
            return config
        except IOError as e:
            logger.error(f"Failed to load configuration: {e}")
            raise
        except ValueError as e:
            logger.error(f"Invalid configuration in {file_path}: {e}")
            raise


@dataclass
class FrameTrain(FrameConfig):
    """
    Configuration for RhythmNet training pipeline.
    Uses dynamic detection and no resizing by default.
    """
    modify: bool = True
    width: int = -1
    height: int = -1
    record_path: str = "./record.csv"


@dataclass
class FrameTest(FrameConfig):
    """
    Configuration for RhythmNet testing pipeline.
    Uses fixed frame size and no modifications by default.
    """
    modify: bool = False
    width: int = 120
    height: int = 120
    record_path: str = "./test_record.csv"


def create_frame_config(config_type: str = "train", **kwargs) -> FrameConfig:
    """
    Create a frame configuration object of the specified type.
    
    Args:
        config_type: Type of configuration ('train' or 'test')
        **kwargs: Override default configuration parameters
        
    Returns:
        FrameConfig: Configured object
        
    Raises:
        ValueError: If the configuration type is unknown
    """
    if config_type.lower() == "train":
        config = FrameTrain()
    elif config_type.lower() == "test":
        config = FrameTest()
    else:
        raise ValueError(f"Unknown configuration type: {config_type}")
    
    # Override with any provided kwargs
    for key, value in kwargs.items():
        if hasattr(config, key):
            setattr(config, key, value)
        else:
            logger.warning(f"Unknown configuration parameter: {key}")
    
    # Validate the configuration
    config.validate()
    return config


# Example usage:
if __name__ == "__main__":
    # Configure logging
    logging.basicConfig(level=logging.INFO, 
                        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    
    # Using class directly
    train_config = FrameTrain()
    train_config.input_path = "./data/train"
    
    # Using factory function
    test_config = create_frame_config("test", input_path="./data/test")
    
    # Demonstrate both attribute naming styles
    print(f"Training dimensions: {train_config.width}x{train_config.height}")
    print(f"Training dimensions (legacy): {train_config.W}x{train_config.H}")
    
    # Demonstrate saving and loading
    train_config.save("./train_config.json")
    loaded_config = FrameConfig.load("./train_config.json")
    print(f"Loaded config matches original: {loaded_config.to_dict() == train_config.to_dict()}")
    
    # Demonstrate freezing
    test_config.freeze()
    try:
        test_config.width = 200  # This will raise an AttributeError
    except AttributeError as e:
        print(f"Expected error: {e}")
        
    # Unfreeze to allow modifications
    test_config.unfreeze()
    test_config.width = 200  # Now it works
    print(f"New test dimensions: {test_config.width}x{test_config.height}")
