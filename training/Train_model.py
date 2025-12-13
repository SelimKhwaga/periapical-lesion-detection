"""
Periapical Lesion Detection - Training Script
==============================================
YOLOv8-based detection system for Type3 and Type4 periapical lesions
Author: [Selim Rezk Abdelmawly Khwaga]
Date: 2024-2025
"""

from ultralytics import YOLO
import torch
import yaml
from pathlib import Path
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def train_model(
    data_yaml: str = 'data.yaml',
    model_size: str = 'yolov8s.pt',
    epochs: int = 150,
    imgsz: int = 832,
    batch_size: int = 16,
    project_name: str = 'periapical_detection',
    device: str = '0'
):
    """
    Train YOLOv8 model for periapical lesion detection.
    
    Args:
        data_yaml: Path to data configuration file
        model_size: YOLOv8 model variant (yolov8s.pt recommended)
        epochs: Number of training epochs
        imgsz: Input image size
        batch_size: Batch size for training
        project_name: Name for saving results
        device: GPU device ('0', 'cpu', etc.)
    
    Returns:
        model: Trained YOLO model
    """
    
    logger.info("="*80)
    logger.info("PERIAPICAL LESION DETECTION - TRAINING")
    logger.info("="*80)
    
    # Check CUDA availability
    if device != 'cpu':
        if torch.cuda.is_available():
            logger.info(f"✓ CUDA available: {torch.cuda.get_device_name(0)}")
        else:
            logger.warning("CUDA not available, using CPU")
            device = 'cpu'
    
    # Initialize model
    logger.info(f"\n[1/4] Loading base model: {model_size}")
    model = YOLO(model_size)
    logger.info(f"✓ Model loaded with {sum(p.numel() for p in model.parameters()):,} parameters")
    
    # Training configuration
    logger.info("\n[2/4] Training Configuration:")
    config = {
        'data': data_yaml,
        'epochs': epochs,
        'imgsz': imgsz,
        'batch': batch_size,
        'device': device,
        'project': project_name,
        'name': 'train',
        
        # Optimizer settings
        'optimizer': 'AdamW',
        'lr0': 0.01,
        'lrf': 0.01,
        'momentum': 0.937,
        'weight_decay': 0.0005,
        'warmup_epochs': 3.0,
        
        # Data augmentation (medical-safe)
        'hsv_h': 0.015,  # Hue augmentation (minimal for medical images)
        'hsv_s': 0.7,    # Saturation
        'hsv_v': 0.4,    # Brightness
        'degrees': 10.0,  # Rotation (preserves anatomy)
        'translate': 0.1,
        'scale': 0.5,
        'fliplr': 0.0,   # No horizontal flip (maintains anatomy)
        'flipud': 0.0,   # No vertical flip
        
        # Loss weights
        'box': 7.5,
        'cls': 0.5,
        'dfl': 1.5,
        
        # Validation
        'val': True,
        'save': True,
        'save_period': 10,
        'patience': 50,  # Early stopping patience
        
        # Misc
        'workers': 8,
        'verbose': True,
        'plots': True,
    }
    
    for key, value in config.items():
        logger.info(f"  {key}: {value}")
    
    # Start training
    logger.info("\n[3/4] Starting training...")
    logger.info("="*80)
    
    results = model.train(**config)
    
    logger.info("="*80)
    logger.info("[4/4] Training Complete!")
    logger.info(f"✓ Best weights saved to: {project_name}/train/weights/best.pt")
    logger.info(f"✓ Results saved to: {project_name}/train/")
    
    # Print final metrics
    if hasattr(results, 'results_dict'):
        metrics = results.results_dict
        logger.info("\nFinal Metrics:")
        logger.info(f"  mAP@0.5: {metrics.get('metrics/mAP50(B)', 0):.3f}")
        logger.info(f"  Precision: {metrics.get('metrics/precision(B)', 0):.3f}")
        logger.info(f"  Recall: {metrics.get('metrics/recall(B)', 0):.3f}")
    
    return model


def validate_model(model_path: str, data_yaml: str):
    """
    Validate trained model on test set.
    
    Args:
        model_path: Path to trained model weights
        data_yaml: Path to data configuration
    """
    logger.info("\n" + "="*80)
    logger.info("MODEL VALIDATION")
    logger.info("="*80)
    
    model = YOLO(model_path)
    
    # Run validation
    metrics = model.val(data=data_yaml)
    
    logger.info("\nValidation Results:")
    logger.info(f"  mAP@0.5: {metrics.box.map50:.3f}")
    logger.info(f"  mAP@0.5:0.95: {metrics.box.map:.3f}")
    logger.info(f"  Precision: {metrics.box.mp:.3f}")
    logger.info(f"  Recall: {metrics.box.mr:.3f}")
    logger.info("="*80)
    
    return metrics


if __name__ == "__main__":
    """
    Main training script execution.
    
    Usage:
        python train_model.py
    
    For custom configuration, modify the parameters below or use command-line arguments.
    """
    
    # Configuration
    DATA_YAML = "data.yaml"  # Update with your data.yaml path
    MODEL_SIZE = "yolov8s.pt"  # Small model (11M params)
    EPOCHS = 150
    IMAGE_SIZE = 832
    BATCH_SIZE = 16
    PROJECT_NAME = "periapical_detection"
    
    # Train model
    trained_model = train_model(
        data_yaml=DATA_YAML,
        model_size=MODEL_SIZE,
        epochs=EPOCHS,
        imgsz=IMAGE_SIZE,
        batch_size=BATCH_SIZE,
        project_name=PROJECT_NAME
    )
    
    # Validate model (optional)
    # validate_model(
    #     model_path=f"{PROJECT_NAME}/train/weights/best.pt",
    #     data_yaml=DATA_YAML
    # )
    

    logger.info("\n✅ Training pipeline complete!")
