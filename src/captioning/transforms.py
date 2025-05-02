# Path: Vision.ai/src/captioning/transforms.py

from torchvision import transforms
# import torch # Not strictly needed for the transforms themselves

def get_transform():
    """
    Returns a transform pipeline for image preprocessing before feeding into CNN.
    Steps:
    1. Resize image to a standard size (e.g., 224x224 for ResNet).
    2. Convert image to PyTorch tensor (Channels x Height x Width).
    3. Normalize pixel values using ImageNet mean and std for pre-trained models.
    """
    return transforms.Compose([
        # Resize the image to the expected input size of the CNN encoder
        # (224x224 is standard for ResNet)
        transforms.Resize((224, 224)),

        # Convert a PIL Image or numpy.ndarray to a PyTorch Tensor.
        # It also automatically scales pixel values from [0, 255] to [0.0, 1.0].
        transforms.ToTensor(),

        # Normalize the tensor's pixel values. This is essential when using
        # pre-trained models that were trained on datasets like ImageNet,
        # as they expect inputs normalized with the same parameters.
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406], # Mean values for R, G, B channels from ImageNet
            std=[0.229, 0.224, 0.225]   # Standard deviation values for R, G, B channels from ImageNet
        )
    ])

# Optional: Add separate transforms for training (with augmentation) and testing/inference
# def get_train_transform():
#     return transforms.Compose([
#         transforms.Resize(256),
#         transforms.RandomCrop(224), # Data augmentation
#         transforms.RandomHorizontalFlip(), # Data augmentation
#         transforms.ToTensor(),
#         transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
#     ])

# def get_test_transform():
#      return transforms.Compose([
#          transforms.Resize((224, 224)), # Or Resize(256), CenterCrop(224)
#          transforms.ToTensor(),
#          transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
#      ])