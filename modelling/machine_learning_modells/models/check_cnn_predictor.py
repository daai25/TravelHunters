import os

def check_cnn_predictor():
    """Check what the CNN predictor is looking for"""
    
    cnn_predictor_path = "/Users/leonakryeziu/PycharmProjects/SummerSchool/TravelHunters/modelling/cnn/predictor.py"
    
    if os.path.exists(cnn_predictor_path):
        print(f"Reading CNN predictor: {cnn_predictor_path}")
        print("=" * 60)
        
        with open(cnn_predictor_path, 'r') as f:
            content = f.read()
        
        print("CNN Predictor Content:")
        print("-" * 40)
        print(content)
        
        # Look for model path references
        lines = content.split('\n')
        print("\nLines that might reference model paths:")
        print("-" * 40)
        for i, line in enumerate(lines, 1):
            if any(keyword in line.lower() for keyword in ['model', 'path', 'load', '.pth', '.h5', '.pkl']):
                print(f"Line {i}: {line.strip()}")
    else:
        print(f"CNN predictor not found at: {cnn_predictor_path}")

if __name__ == "__main__":
    check_cnn_predictor()