import torch
import pandas as pd
from PIL import Image

from config import CLASS_NAMES, CHECKPOINT_PATH, device
from dataset import eval_transforms
from model import get_model


def load_trained_model(checkpoint_path=CHECKPOINT_PATH):
    model = get_model()
    model.load_state_dict(torch.load(checkpoint_path, map_location=device))
    model.eval()
    return model


def classify_skin_lesions(list_of_img_paths, checkpoint_path=CHECKPOINT_PATH):
    model = load_trained_model(checkpoint_path)

    results = []
    for img_path in list_of_img_paths:
        img    = Image.open(img_path).convert('RGB')
        tensor = eval_transforms(img).unsqueeze(0).to(device)

        with torch.no_grad():
            output = model(tensor)
            probs  = torch.softmax(output, dim=1)[0]
            pred_idx = probs.argmax().item()

        results.append({
            'path': img_path,
            'predicted_class': CLASS_NAMES[pred_idx],
            'confidence': round(probs[pred_idx].item() * 100, 2),
            'probabilities': {
                CLASS_NAMES[i]: round(probs[i].item() * 100, 2)
                for i in range(len(CLASS_NAMES))
            }
        })

    return results


def batch_inference_to_csv(list_of_img_paths, output_csv='predictions.csv', checkpoint_path=CHECKPOINT_PATH):
    results = classify_skin_lesions(list_of_img_paths, checkpoint_path)
    df = pd.DataFrame(results)
    df.to_csv(output_csv, index=False)
    print(f'Saved predictions to {output_csv}')
    return df
