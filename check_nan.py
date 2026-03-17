import torch
import sys

def check_checkpoint(path):
    print(f"Checking {path}...")
    try:
        ckpt = torch.load(path, map_location='cpu', weights_only=False)
        
        # In YOLOv8/v11, the model is often in 'model' key
        # If it's a full trainer state, it has 'model', 'optimizer', etc.
        model_data = ckpt.get('model', ckpt)
        
        # Check if it has state_dict
        if hasattr(model_data, 'state_dict'):
            sd = model_data.state_dict()
        elif isinstance(model_data, dict):
            sd = model_data
        else:
            # Maybe it is the model object itself (unlikely for .pt)
            sd = getattr(model_data, '__dict__', {})

        found_nan = False
        found_inf = False
        
        for k, v in sd.items():
            if torch.is_tensor(v):
                if torch.isnan(v).any():
                    print(f"NaN found in {k}")
                    found_nan = True
                if torch.isinf(v).any():
                    print(f"Inf found in {k}")
                    found_inf = True
            elif isinstance(v, dict):
                # Nested dicts (like optimizer state)
                for sk, sv in v.items():
                    if torch.is_tensor(sv):
                        if torch.isnan(sv).any():
                            print(f"NaN found in {k}.{sk}")
                            found_nan = True
                        if torch.isinf(sv).any():
                            print(f"Inf found in {k}.{sk}")
                            found_inf = True
        
        if not found_nan and not found_inf:
            print("No NaNs or Infs found in parameters.")
        else:
            print("!!! Checkpoint contains NaNs/Infs !!!")
            
        # Also check optimizer if it exists
        if 'optimizer' in ckpt:
            print("Checking optimizer state...")
            opt_state = ckpt['optimizer']
            # Optimizer states are usually in 'state'
            for p_id, p_state in opt_state.get('state', {}).items():
                for sk, sv in p_state.items():
                    if torch.is_tensor(sv):
                        if torch.isnan(sv).any():
                            print(f"NaN found in optimizer state for param {p_id}")
                        if torch.isinf(sv).any():
                            print(f"Inf found in optimizer state for param {p_id}")

    except Exception as e:
        print(f"Error reading checkpoint: {e}")

if __name__ == "__main__":
    path = r"d:\work\yolo_training\runs\yolo26n_drones\baseline_wo_crop\weights\best.pt"
    if len(sys.argv) > 1:
        path = sys.argv[1]
    check_checkpoint(path)
