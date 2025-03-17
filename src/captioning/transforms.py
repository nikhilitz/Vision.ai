from torchvision import transforms

def get_transform():
    """
    Returns a transform pipeline for image preprocessing before feeding into CNN.
    Steps:
    1. Resize image to 224x224 (ResNet standard input)
    2. Convert image to PyTorch tensor (C*H*W)
    3. Normalize pixel values using ImageNet mean and std
    """
    return transforms.Compose([
        transforms.Resize((224,224)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
        
    ])
    