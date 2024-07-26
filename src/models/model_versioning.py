import os
import json
from datetime import datetime
import torch

def save_model_version(symbol, model, performance_metrics):
    versions_dir = f"models/versions/{symbol}"
    os.makedirs(versions_dir, exist_ok=True)
    
    # Get the current version number
    version = len([f for f in os.listdir(versions_dir) if f.endswith('_model.pth')]) + 1
    
    # Save the model
    model_path = f"{versions_dir}/v{version}_model.pth"
    model_scripted = torch.jit.script(model)
    model_scripted.save(model_path)

    #torch.save(model.state_dict(), model_path)
    
    # Save metadata
    metadata = {
        "version": version,
        "timestamp": datetime.now().isoformat(),
        "performance_metrics": performance_metrics
    }
    
    metadata_path = f"{versions_dir}/v{version}_metadata.json"
    with open(metadata_path, 'w') as f:
        json.dump(metadata, f, indent=2)
    
    print(f"Model version {version} saved for {symbol}")
    return version

def get_latest_model_version(symbol):
    versions_dir = f"models/versions/{symbol}"
    if not os.path.exists(versions_dir):
        return None
    
    versions = [int(f.split('_')[0][1:]) for f in os.listdir(versions_dir) if f.endswith('_model.pth')]
    if not versions:
        return None
    
    latest_version = max(versions)
    return latest_version

def load_model_version(symbol, version=None):
    if version is None:
        version = get_latest_model_version(symbol)
        if version is None:
            raise FileNotFoundError(f"No model versions found for {symbol}")
    
    model_path = f"models/versions/{symbol}/v{version}_model.pth"
    metadata_path = f"models/versions/{symbol}/v{version}_metadata.json"
    
    if not os.path.exists(model_path) or not os.path.exists(metadata_path):
        raise FileNotFoundError(f"Model version {version} not found for {symbol}")
    
    
    model = torch.jit.load(model_path)
    model.eval()  # Set the model to evaluation mode
    
    # Load metadata
    with open(metadata_path, 'r') as f:
        metadata = json.load(f)
    
    return model, metadata