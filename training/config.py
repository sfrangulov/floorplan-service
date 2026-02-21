"""Segmentation model configuration."""

# Class definitions - must match design doc
CLASSES = [
    "background",      # 0
    "wall",            # 1
    "door",            # 2
    "window",          # 3
    "balcony",         # 4
    "balcony_window",  # 5
    "bedroom",         # 6
    "living_room",     # 7
    "kitchen",         # 8
    "bathroom",        # 9
]

NUM_CLASSES = len(CLASSES)

# CubiCasa5K category mapping -> our 10 classes
CUBICASA_MAPPING = {
    "Wall": 1,
    "Door": 2,
    "Window": 3,
    "Balcony": 4,
    "Bedroom": 6,
    "LivingRoom": 7,
    "Kitchen": 8,
    "Bathroom": 9,
    "Toilet": 9,
    "Bath": 9,
}

# Model
MODEL_NAME = "nvidia/segformer-b2-finetuned-ade-512-512"
INPUT_SIZE = 512

# Training hyperparameters
LEARNING_RATE = 6e-5
WEIGHT_DECAY = 0.01
BATCH_SIZE = 8
NUM_EPOCHS = 100
EARLY_STOPPING_PATIENCE = 10

# Tversky loss parameters (emphasize recall for thin walls)
TVERSKY_ALPHA = 0.3
TVERSKY_BETA = 0.7

# Output
COORD_RANGE = 1000  # normalize coordinates to 0-1000
