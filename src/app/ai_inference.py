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
import csv
import easyocr
from tqdm.auto import tqdm


def ensure_local_model(repo_id: str, folder_name: str) -> Path:
    """
    Ensure that `models/<folder_name>/` exists, downloading it if necessary.
    Returns the Path to the local model directory.
    """
    project_root = Path(__file__).resolve().parent.parent.parent
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
    b = [(x_c - 0.5 * w), (y_c - 0.5 * h), (x_c + 0.5 * w), (y_c + 0.5 * h)]
    return torch.stack(b, dim=1)


def rescale_bboxes(out_bbox, size):
    img_w, img_h = size
    b = box_cxcywh_to_xyxy(out_bbox)
    b = b * torch.tensor([img_w, img_h, img_w, img_h], dtype=torch.float32)
    return b


def outputs_to_objects(outputs, img_size, id2label):
    soft = outputs.logits.softmax(-1)
    scores, indices = soft.max(-1)
    pred_labels = indices[0].cpu().numpy()
    pred_scores = scores[0].cpu().numpy()
    raw_bboxes = outputs.pred_boxes[0].cpu()
    bboxes = [list(map(float, box)) for box in rescale_bboxes(raw_bboxes, img_size)]

    objects = []
    for lbl, scr, bbox in zip(pred_labels, pred_scores, bboxes):
        label = id2label.get(int(lbl), 'unknown')
        if label != 'no object':
            objects.append({'label': label, 'score': float(scr), 'bbox': bbox})
    return objects


def objects_to_crops(img, objects, class_thresholds, padding=10):
    crops = []
    for obj in objects:
        if obj['score'] < class_thresholds.get(obj['label'], 0.5):
            continue
        bbox = obj['bbox']
        padded = [bbox[0]-padding, bbox[1]-padding, bbox[2]+padding, bbox[3]+padding]
        crops.append(img.crop(padded))
    return crops


def extract_table_structure_data(cells):
    rows = [c for c in cells if c['label'] == 'table row']
    cols = [c for c in cells if c['label'] == 'table column']
    return {'table_rows': rows, 'table_columns': cols}


def fig2img(fig):
    """Convert a Matplotlib figure to a PIL Image and return it"""
    import io
    buf = io.BytesIO()
    fig.savefig(buf)
    buf.seek(0)
    img = Image.open(buf)
    return img


def visualize_detected_tables(img, det_tables, out_path=None):
    plt.imshow(img, interpolation="lanczos")
    fig = plt.gcf()
    fig.set_size_inches(20, 20)
    ax = plt.gca()

    for det_table in det_tables:
        bbox = det_table['bbox']

        if det_table['label'] == 'table':
            facecolor = (1, 0, 0.45)
            edgecolor = (1, 0, 0.45)
            alpha = 0.3
            linewidth = 2
            hatch='//////'
        elif det_table['label'] == 'table rotated':
            facecolor = (0.95, 0.6, 0.1)
            edgecolor = (0.95, 0.6, 0.1)
            alpha = 0.3
            linewidth = 2
            hatch='//////'
        else:
            continue

        rect = patches.Rectangle(bbox[:2], bbox[2]-bbox[0], bbox[3]-bbox[1], linewidth=linewidth,
                                    edgecolor='none',facecolor=facecolor, alpha=0.1)
        ax.add_patch(rect)
        rect = patches.Rectangle(bbox[:2], bbox[2]-bbox[0], bbox[3]-bbox[1], linewidth=linewidth,
                                    edgecolor=edgecolor,facecolor='none',linestyle='-', alpha=alpha)
        ax.add_patch(rect)
        rect = patches.Rectangle(bbox[:2], bbox[2]-bbox[0], bbox[3]-bbox[1], linewidth=0,
                                    edgecolor=edgecolor,facecolor='none',linestyle='-', hatch=hatch, alpha=0.2)
        ax.add_patch(rect)

    plt.xticks([], [])
    plt.yticks([], [])

    legend_elements = [Patch(facecolor=(1, 0, 0.45), edgecolor=(1, 0, 0.45),
                                label='Table', hatch='//////', alpha=0.3),
                        Patch(facecolor=(0.95, 0.6, 0.1), edgecolor=(0.95, 0.6, 0.1),
                                label='Table (rotated)', hatch='//////', alpha=0.3)]
    plt.legend(handles=legend_elements, bbox_to_anchor=(0.5, -0.02), loc='upper center', borderaxespad=0,
                    fontsize=10, ncol=2)
    plt.gcf().set_size_inches(10, 10)
    plt.axis('off')

    if out_path is not None:
        plt.savefig(out_path, bbox_inches='tight', dpi=150)
        plt.close()

    return fig


def extract_tables_from_images(image_paths):
    """
    Process a list of image paths, extract tables, and return OCR data as a list of tables (list of rows, each row is a list of cell strings).
    Also saves table detection and structure visualization PNGs locally for each image/table.
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    det_dir = ensure_local_model(
        repo_id="microsoft/table-transformer-detection",
        folder_name="table-transformer-detection"
    )
    struct_dir = ensure_local_model(
        repo_id="microsoft/table-structure-recognition-v1.1-all",
        folder_name="table-structure-recognition"
    )
    print("Loading table detection model from disk…")
    det_model = AutoModelForObjectDetection.from_pretrained(
        str(det_dir), local_files_only=True, revision="no_timm"
    )
    det_model.to(device)
    det_id2label = det_model.config.id2label.copy()
    det_id2label[len(det_id2label)] = 'no object'
    class MaxResize:
        def __init__(self, max_size=800): self.max_size = max_size
        def __call__(self, image):
            w, h = image.size
            scale = self.max_size / max(w, h)
            return image.resize((int(w*scale), int(h*scale)))
    det_transform = transforms.Compose([
        MaxResize(800), transforms.ToTensor(),
        transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])
    ])
    print("Loading table structure model from disk…")
    struct_model = TableTransformerForObjectDetection.from_pretrained(
        str(struct_dir), local_files_only=True
    )
    struct_model.to(device)
    struct_id2label = struct_model.config.id2label.copy()
    struct_id2label[len(struct_id2label)] = 'no object'
    struct_transform = transforms.Compose([
        MaxResize(1000), transforms.ToTensor(),
        transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])
    ])
    det_thresholds = {'table': 0.5, 'table rotated': 0.5}
    all_tables = []
    for path in image_paths:
        print(f"\nProcessing {path}")
        img = Image.open(path).convert("RGB")
        img_path = Path(path)
        # Create output directory for predictions
        output_dir = img_path.parent / f"{img_path.stem}_predicted"
        output_dir.mkdir(exist_ok=True)
        # Table detection
        pixels = det_transform(img).unsqueeze(0).to(device)
        with torch.no_grad():
            det_out = det_model(pixels)
        tables = outputs_to_objects(det_out, img.size, det_id2label)
        print(f"Detected {len(tables)} table region(s):", tables)
        # Save table detection visualization
        if tables:
            out_path = output_dir / f"{img_path.stem}_tables_detected.png"
            visualize_detected_tables(img, tables, str(out_path))
            print(f"Saved table detection visualization to {out_path}")
        crops = objects_to_crops(img, tables, det_thresholds)
        for i, crop in enumerate(crops):
            print(f"\nAnalyzing structure for table {i}")
            px = struct_transform(crop).unsqueeze(0).to(device)
            with torch.no_grad():
                struct_out = struct_model(px)
            cells = outputs_to_objects(struct_out, crop.size, struct_id2label)
            struct = extract_table_structure_data(cells)
            print(f"Rows: {len(struct['table_rows'])}, Columns: {len(struct['table_columns'])}")
            # Save the cropped table with structure visualization
            out_path = output_dir / f"{img_path.stem}_table_{i}_structure.png"
            plt.figure(figsize=(10, 10))
            plt.imshow(crop)
            # Draw row and column lines
            for row in struct['table_rows']:
                bbox = row['bbox']
                plt.axhline(y=bbox[1], color='blue', linestyle='-', alpha=0.5)
                plt.axhline(y=bbox[3], color='blue', linestyle='-', alpha=0.5)
            for col in struct['table_columns']:
                bbox = col['bbox']
                plt.axvline(x=bbox[0], color='red', linestyle='-', alpha=0.5)
                plt.axvline(x=bbox[2], color='red', linestyle='-', alpha=0.5)
            plt.axis('off')
            plt.savefig(str(out_path), bbox_inches='tight', dpi=150)
            plt.close()
            print(f"Saved table structure visualization to {out_path}")
            if struct['table_rows'] and struct['table_columns']:
                sorted_rows = sorted(struct['table_rows'], key=lambda r: (r['bbox'][1] + r['bbox'][3]) / 2)
                sorted_cols = sorted(struct['table_columns'], key=lambda c: (c['bbox'][0] + c['bbox'][2]) / 2)
                cell_coordinates = []
                for row in sorted_rows:
                    row_cells = []
                    row_y1, row_y2 = row['bbox'][1], row['bbox'][3]
                    for col in sorted_cols:
                        col_x1, col_x2 = col['bbox'][0], col['bbox'][2]
                        cell_bbox = [col_x1, row_y1, col_x2, row_y2]
                        row_cells.append({'cell': cell_bbox})
                    cell_coordinates.append({'cells': row_cells})
                reader = easyocr.Reader(['en'])
                def apply_ocr(cell_coordinates, cropped_table):
                    data = []
                    max_num_columns = 0
                    for idx, row in enumerate(tqdm(cell_coordinates)):
                        row_cells = []
                        for cell in row["cells"]:
                            cell_image = np.array(cropped_table.crop(cell["cell"]))
                            result = reader.readtext(cell_image)
                            if len(result) > 0:
                                text = " ".join([x[1] for x in result])
                            else:
                                text = ""
                            row_cells.append({"text": text})
                        if len(row_cells) > max_num_columns:
                            max_num_columns = len(row_cells)
                        data.append(row_cells)
                    # Pad rows
                    for row_cells in data:
                        if len(row_cells) != max_num_columns:
                            row_cells += [{"text": ""} for _ in range(max_num_columns - len(row_cells))]
                    return data
                print("Applying EasyOCR to table cells...")
                ocr_data = apply_ocr(cell_coordinates, crop)
                all_tables.append({"table": ocr_data})
                print(f"Extracted table: {ocr_data}")
    print("Done.")
    return all_tables