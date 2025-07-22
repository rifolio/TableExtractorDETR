import argparse
import os
from pathlib import Path
from PIL import Image
import torch
from transformers import AutoModelForObjectDetection, TableTransformerForObjectDetection
from huggingface_hub import snapshot_download
from torchvision import transforms
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.patches import Patch
import numpy as np
import easyocr
from tqdm.auto import tqdm

# --- Spanning-cell support functions ---
def intersects_interval(a1, a2, b1, b2):
    """Return True if [a1,a2] overlaps [b1,b2] by any amount."""
    return not (a2 <= b1 or b2 <= a1)


def get_cell_grid(table_data, iou_thresh=0.1):
    """
    Build a grid of cells, accounting for spanning cells.
    Returns a list of rows, each a list of dicts with keys:
      - row_index, col_index, bbox, is_span
    """
    # 1) Sort detected row and column boundaries
    rows = sorted(
        (e for e in table_data if e['label'] == 'table row' and e['score'] >= iou_thresh),
        key=lambda x: x['bbox'][1]
    )
    cols = sorted(
        (e for e in table_data if e['label'] == 'table column' and e['score'] >= iou_thresh),
        key=lambda x: x['bbox'][0]
    )
    spans = [e for e in table_data if e['label'] == 'table spanning cell' and e['score'] >= iou_thresh]

    # 2) Initialize grid slots
    grid = []
    for r_idx, row in enumerate(rows):
        top, bot = row['bbox'][1], row['bbox'][3]
        row_cells = []
        for c_idx, col in enumerate(cols):
            left, right = col['bbox'][0], col['bbox'][2]
            row_cells.append({
                'row_index': r_idx,
                'col_index': c_idx,
                'bbox': [left, top, right, bot],
                'is_span': False,
            })
        grid.append(row_cells)

    # 3) Assign spanning-cell bboxes to overlapped slots
    for span in spans:
        sx1, sy1, sx2, sy2 = span['bbox']
        # rows touched by this span
        touched_rows = [i for i, row in enumerate(rows)
                        if intersects_interval(sy1, sy2, row['bbox'][1], row['bbox'][3])]
        # cols touched by this span
        touched_cols = [j for j, col in enumerate(cols)
                        if intersects_interval(sx1, sx2, col['bbox'][0], col['bbox'][2])]
        for i in touched_rows:
            for j in touched_cols:
                grid[i][j]['bbox'] = span['bbox']
                grid[i][j]['is_span'] = True

    # 4) Sanity check
    R, C = len(rows), len(cols)
    assert sum(len(r) for r in grid) == R * C, "Grid size mismatch"

    return grid


# Initialize the EasyOCR reader just once
txt_reader = easyocr.Reader(['en'])

def apply_ocr(grid, cropped_table):
    """
    OCR each cell in the grid, reusing cached text for spanning cells.
    Returns a dict mapping row_index -> list of text strings.
    """
    span_cache = {}
    data = {}
    max_cols = 0

    for r_idx, row in enumerate(tqdm(grid, desc="OCR rows")):
        texts = []
        for cell in row:
            bbox = cell['bbox']
            key = tuple(bbox)

            if cell['is_span'] and key in span_cache:
                # reuse OCR for this spanning bbox
                text = span_cache[key]
            else:
                # crop and OCR
                crop_arr = np.array(cropped_table.crop(bbox))
                results = txt_reader.readtext(crop_arr)
                text = " ".join([res[1] for res in results]) if results else ""
                if cell['is_span']:
                    span_cache[key] = text

            texts.append(text)

        max_cols = max(max_cols, len(texts))
        data[r_idx] = texts

    # pad rows to equal length
    for r_idx, texts in data.items():
        if len(texts) < max_cols:
            texts += [""] * (max_cols - len(texts))
        data[r_idx] = texts

    return data


def ensure_local_model(repo_id: str, folder_name: str) -> Path:
    project_root = Path(__file__).resolve().parent
    local_dir = project_root / 'models' / folder_name

    if not local_dir.exists():
        print(f"📥 Downloading {repo_id} to {local_dir} …")
        snapshot_download(
            repo_id=repo_id,
            local_dir=str(local_dir),
            local_dir_use_symlinks=False
        )
        print("✅ Download complete\n")
    else:
        print(f"✅ Found local model at {local_dir}\n")

    return local_dir


def box_cxcywh_to_xyxy(x):
    x_c, y_c, w, h = x.unbind(-1)
    b = [x_c - 0.5 * w, y_c - 0.5 * h, x_c + 0.5 * w, y_c + 0.5 * h]
    return torch.stack(b, dim=1)


def rescale_bboxes(out_bbox, size):
    img_w, img_h = size
    b = box_cxcywh_to_xyxy(out_bbox)
    scale = torch.tensor([img_w, img_h, img_w, img_h], dtype=torch.float32)
    return b * scale


def outputs_to_objects(outputs, img_size, id2label):
    # softmax scores
    scores = outputs.logits.softmax(-1)
    max_scores, indices = scores.max(-1)
    labels = indices[0].cpu().numpy()
    confidences = max_scores[0].cpu().numpy()
    raw_boxes = outputs.pred_boxes[0].cpu()
    boxes = [list(map(float, box)) for box in rescale_bboxes(raw_boxes, img_size)]

    objects = []
    for lbl, conf, bbox in zip(labels, confidences, boxes):
        label = id2label.get(int(lbl), 'unknown')
        if label != 'no object':
            objects.append({'label': label, 'score': float(conf), 'bbox': bbox})
    return objects


def objects_to_crops(img, objects, class_thresholds, padding=10):
    crops = []
    for obj in objects:
        if obj['score'] < class_thresholds.get(obj['label'], 0.5):
            continue
        x1, y1, x2, y2 = obj['bbox']
        pad_box = [x1 - padding, y1 - padding, x2 + padding, y2 + padding]
        crops.append(img.crop(pad_box))
    return crops


def visualize_detected_tables(img, det_tables, out_path=None):
    plt.imshow(img, interpolation='lanczos')
    fig = plt.gcf()
    fig.set_size_inches(20, 20)
    ax = plt.gca()

    for det in det_tables:
        bbox = det['bbox']
        if det['label'] == 'table':
            facecolor, edgecolor = (1, 0, 0.45), (1, 0, 0.45)
        else:
            facecolor, edgecolor = (0.95, 0.6, 0.1), (0.95, 0.6, 0.1)

        # filled rectangle
        ax.add_patch(patches.Rectangle(
            (bbox[0], bbox[1]), bbox[2]-bbox[0], bbox[3]-bbox[1],
            linewidth=1, edgecolor='none', facecolor=facecolor, alpha=0.1
        ))
        # outline
        ax.add_patch(patches.Rectangle(
            (bbox[0], bbox[1]), bbox[2]-bbox[0], bbox[3]-bbox[1],
            linewidth=2, edgecolor=edgecolor, facecolor='none', alpha=0.3
        ))
        # hatched
        ax.add_patch(patches.Rectangle(
            (bbox[0], bbox[1]), bbox[2]-bbox[0], bbox[3]-bbox[1],
            linewidth=0, edgecolor=edgecolor, facecolor='none', linestyle='-', hatch='//////', alpha=0.2
        ))

    plt.xticks([])
    plt.yticks([])

    legend_items = [
        Patch(facecolor=(1, 0, 0.45), edgecolor=(1, 0, 0.45), label='Table', hatch='//////', alpha=0.3),
        Patch(facecolor=(0.95, 0.6, 0.1), edgecolor=(0.95, 0.6, 0.1), label='Table (rotated)', hatch='//////', alpha=0.3)
    ]
    plt.legend(handles=legend_items, bbox_to_anchor=(0.5, -0.02), loc='upper center', ncol=2)
    plt.axis('off')

    if out_path:
        plt.savefig(out_path, bbox_inches='tight', dpi=150)
        plt.close()
    return fig


def extract_table_structure_data(cells):
    """Separate row, column, and spanning-cell detections."""
    rows = [c for c in cells if c['label'] == 'table row']
    cols = [c for c in cells if c['label'] == 'table column']
    spans = [c for c in cells if c['label'] == 'table spanning cell']
    return {'table_rows': rows, 'table_columns': cols, 'table_spans': spans}


def extract_tables_from_images(image_paths):
    """Main pipeline: detect, structure, grid, OCR."""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Ensure models
    det_dir = ensure_local_model(
        'microsoft/table-transformer-detection', 'table-transformer-detection')
    struct_dir = ensure_local_model(
        'microsoft/table-structure-recognition-v1.1-all', 'table-structure-recognition')

    # Load detection model
    det_model = AutoModelForObjectDetection.from_pretrained(
        str(det_dir), local_files_only=True, revision='no_timm')
    det_model.to(device)
    det_id2label = det_model.config.id2label.copy()
    det_id2label[len(det_id2label)] = 'no object'

    # Load structure model
    struct_model = TableTransformerForObjectDetection.from_pretrained(
        str(struct_dir), local_files_only=True)
    struct_model.to(device)
    struct_id2label = struct_model.config.id2label.copy()
    struct_id2label[len(struct_id2label)] = 'no object'

    # Transforms
    class MaxResize:
        def __init__(self, max_size): self.max_size = max_size
        def __call__(self, image):
            w,h = image.size
            scale = self.max_size / max(w,h)
            return image.resize((int(w*scale), int(h*scale)))

    det_transform = transforms.Compose([
        MaxResize(800), transforms.ToTensor(),
        transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])
    ])
    struct_transform = transforms.Compose([
        MaxResize(1000), transforms.ToTensor(),
        transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])
    ])
    det_thresholds = {'table': 0.5, 'table rotated': 0.5}

    all_tables = []
    for path in image_paths:
        print(f"\nProcessing {path}")
        img = Image.open(path).convert('RGB')
        img_path = Path(path)
        output_dir = img_path.parent / f"{img_path.stem}_predicted"
        output_dir.mkdir(exist_ok=True)

        # 1) Table detection
        pixels = det_transform(img).unsqueeze(0).to(device)
        with torch.no_grad(): det_out = det_model(pixels)
        tables = outputs_to_objects(det_out, img.size, det_id2label)
        print(f"Detected {len(tables)} table region(s)")

        # Save detection viz
        if tables:
            vis_path = output_dir / f"{img_path.stem}_tables_detected.png"
            visualize_detected_tables(img, tables, str(vis_path))
            print(f"Saved detection visualization to {vis_path}")

        # 2) Crop table regions
        crops = objects_to_crops(img, tables, det_thresholds)
        for i, crop in enumerate(crops):
            print(f"\nStructure for table {i}")
            px = struct_transform(crop).unsqueeze(0).to(device)
            with torch.no_grad(): struct_out = struct_model(px)
            cells = outputs_to_objects(struct_out, crop.size, struct_id2label)

            # 3) Build grid
            grid = get_cell_grid(cells)

            # 4) OCR
            ocr_dict = apply_ocr(grid, crop)

            # Format as list of rows: each cell as dict{text:…}
            ocr_data = []
            for r in sorted(ocr_dict.keys()):
                ocr_data.append([{'text': t} for t in ocr_dict[r]])

            all_tables.append({'table': ocr_data})
            print(f"Extracted table {i}: {ocr_data}")

    print("Done.")
    return all_tables


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Extract tables from images")
    parser.add_argument('images', nargs='+', help="Paths to input images")
    args = parser.parse_args()
    extract_tables_from_images(args.images)
