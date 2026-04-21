import os
from pathlib import Path
from torchvision import datasets
from PIL import Image

def main():
    out_dir = Path("artifacts/clean_ui_test_images")
    out_dir.mkdir(parents=True, exist_ok=True)
    
    # Load CIFAR-10 test dataset
    dataset = datasets.CIFAR10(root="./data", train=False, download=True)
    
    classes = dataset.classes
    count = 10
    
    # Pick a few images to save
    for i in range(count):
        image, label = dataset[i]
        class_name = classes[label]
        
        filename = out_dir / f"clean_{i:02d}_{class_name}.png"
        image.save(filename)
        print(f"Saved: {filename}")

if __name__ == "__main__":
    main()
