import os
import numpy as np
from PIL import Image

# Create output directory
output_dir = os.path.join(os.path.dirname(__file__), 'data')
os.makedirs(output_dir, exist_ok=True)

# Generate and save 10 random images
for i in range(10):
    # Generate random pixel values (uint8: 0-255)
    random_image = np.random.randint(0, 256, size=(224, 224, 3), dtype=np.uint8)
    
    # Convert to PIL image
    img = Image.fromarray(random_image)
    
    # Save as PNG
    img.save(os.path.join(output_dir, f'image_{i+1}.png'))

print("10 random images saved in 'data/' directory.")
