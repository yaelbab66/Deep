import torch

def preprocess_image_tensor(image_tensor, patch_size):
    """
    Function to preprocess an image tensor for MLP-Mixer
    :param image_tensor: Image tensor in format [channels, height, width]
    :param patch_size: Patch size in pixels
    :return: Tensor of patches with shape [num_patches, channels × patch_size × patch_size]
    """
    # 1. Validate the input tensor shape
    assert len(image_tensor.shape) == 3, "Image tensor must be in the format [channels, height, width]"
    channels, height, width = image_tensor.shape
    
    # 2. Check if the image dimensions are divisible by the patch size
    assert height % patch_size == 0 and width % patch_size == 0, \
        "Image height and width must be divisible by the patch size."
    
    # 3. Divide the image into patches
    num_patches_h = height // patch_size  # Number of patches along the height
    num_patches_w = width // patch_size  # Number of patches along the width
    num_patches = num_patches_h * num_patches_w  # Total number of patches
    
    patches = []
    for i in range(0, height, patch_size):
        for j in range(0, width, patch_size):
            # Extract a patch and flatten it (preserve channels)
            patch = image_tensor[:, i:i+patch_size, j:j+patch_size]
            patches.append(patch.flatten())
    
    # Stack all patches into a tensor of shape [num_patches, channels × patch_size × patch_size]
    patches = torch.stack(patches)
    
    return patches

# Example usage:
# Example tensor with size [3, 224, 224] (ImageNet-like dimensions with 3 channels)
image_tensor = torch.randn(3, 224, 224)  # Random values for demonstration
patch_size = 16  # Patch size of 16x16
processed_data = preprocess_image_tensor(image_tensor, patch_size)

print("Processed data shape:", processed_data.shape)  # Output: [num_patches, channels × patch_size × patch_size]
