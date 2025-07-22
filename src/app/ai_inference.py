import argparse
import sys
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
import logging

# ─── Logging setup ─────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(name)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger(__name__)

# --- Spanning-cell support functions ---

def intersects_interval(a1, a2, b1, b2):
    """Return True if [a1,a2] overlaps [b1,b2] by any amount."""
    return not (a2 <= b1 or b2 <= a1)


def get_cell_grid(table_data, iou_thresh=0.1):
    rows = sorted((e for e in table_data if e['label']=='table row' and e['score']>=iou_thresh), key=lambda x: x['bbox'][1])
    cols = sorted((e for e in table_data if e['label']=='table column' and e['score']>=iou_thresh), key=lambda x: x['bbox'][0])
    spans = [e for e in table_data if e['label']=='table spanning cell' and e['score']>=iou_thresh]

    grid=[]
    for r_idx,row in enumerate(rows):
        top,bot=row['bbox'][1],row['bbox'][3]
        row_cells=[]
        for c_idx,col in enumerate(cols):
            left,right=col['bbox'][0],col['bbox'][2]
            row_cells.append({'row_index':r_idx,'col_index':c_idx,'bbox':[left,top,right,bot],'is_span':False})
        grid.append(row_cells)

    for span in spans:
        sx1,sy1,sx2,sy2=span['bbox']
        touched_rows=[i for i,row in enumerate(rows) if intersects_interval(sy1,sy2,row['bbox'][1],row['bbox'][3])]
        touched_cols=[j for j,col in enumerate(cols) if intersects_interval(sx1,sx2,col['bbox'][0],col['bbox'][2])]
        for i in touched_rows:
            for j in touched_cols:
                grid[i][j]['bbox']=span['bbox']
                grid[i][j]['is_span']=True

    R,C=len(rows),len(cols)
    assert sum(len(r) for r in grid)==R*C, "Grid size mismatch"
    return grid

# Initialize OCR reader once
txt_reader=easyocr.Reader(['en'])

def apply_ocr(grid,cropped_table):
    span_cache={}
    data={}
    max_cols=0
    for r_idx,row in enumerate(tqdm(grid,desc="OCR rows")):
        texts=[]
        for cell in row:
            bbox=cell['bbox']; key=tuple(bbox)
            if cell['is_span'] and key in span_cache:
                text=span_cache[key]
            else:
                arr=np.array(cropped_table.crop(bbox))
                res=txt_reader.readtext(arr)
                text=" ".join([r[1] for r in res]) if res else ""
                if cell['is_span']: span_cache[key]=text
            texts.append(text)
        max_cols=max(max_cols,len(texts))
        data[r_idx]=texts
    for r_idx,texts in data.items():
        if len(texts)<max_cols: texts+=[""]*(max_cols-len(texts))
        data[r_idx]=texts
    return data

# Local model download
from pathlib import Path
def ensure_local_model(repo_id:str,folder_name:str)->Path:
    root=Path(__file__).parent
    local=root/'models'/folder_name
    if not local.exists() or not any(local.iterdir()):
        logger.info("Downloading %s -> %s",repo_id,local)
        try:
            snapshot_download(repo_id=repo_id,local_dir=str(local),local_dir_use_symlinks=False)
            logger.info("Downloaded %s",repo_id)
        except Exception:
            logger.exception("Download failed %s",repo_id)
            raise
    else:
        logger.info("Using local model %s",local)
    return local

# Box helpers
def box_cxcywh_to_xyxy(x):
    xc,yc,w,h=x.unbind(-1)
    return torch.stack([xc-0.5*w,yc-0.5*h,xc+0.5*w,yc+0.5*h],dim=1)
def rescale_bboxes(out_bbox,size):
    w,h=size; b=box_cxcywh_to_xyxy(out_bbox)
    scale=torch.tensor([w,h,w,h],dtype=torch.float32)
    return b*scale

def outputs_to_objects(out,img_size,id2label):
    scores=out.logits.softmax(-1); max_s,idx=scores.max(-1)
    labels=idx[0].cpu().numpy(); conf=max_s[0].cpu().numpy()
    boxes=[list(map(float,b)) for b in rescale_bboxes(out.pred_boxes[0].cpu(),img_size)]
    objs=[]
    for lbl,sc,b in zip(labels,conf,boxes):
        l=id2label.get(int(lbl),'unknown')
        if l!='no object': objs.append({'label':l,'score':float(sc),'bbox':b})
    return objs

def objects_to_crops(img,objs,thresh,pad=10):
    crops=[]
    for o in objs:
        if o['score']<thresh.get(o['label'],0.5): continue
        x1,y1,x2,y2=o['bbox']; box=[x1-pad,y1-pad,x2+pad,y2+pad]
        crops.append(img.crop(box))
    return crops

# ─── Visualization ─────────────────────────────────────────────────────────────
def visualize_detected_tables(img, det_tables, out_path=None):
    """
    Draw detected tables in red/orange and save.
    """
    fig, ax = plt.subplots(figsize=(10, 10))
    ax.axis('off')
    arr = np.array(img)
    ax.imshow(arr, interpolation='lanczos')
    for det in det_tables:
        bbox = det['bbox']
        color = (1, 0, 0.45) if det['label'] == 'table' else (0.95, 0.6, 0.1)
        ax.add_patch(patches.Rectangle((bbox[0], bbox[1]), bbox[2]-bbox[0], bbox[3]-bbox[1],
                                       linewidth=0, facecolor=color, alpha=0.1))
        ax.add_patch(patches.Rectangle((bbox[0], bbox[1]), bbox[2]-bbox[0], bbox[3]-bbox[1],
                                       linewidth=2, edgecolor=color, facecolor='none', alpha=0.3))
        ax.add_patch(patches.Rectangle((bbox[0], bbox[1]), bbox[2]-bbox[0], bbox[3]-bbox[1],
                                       linewidth=0, edgecolor=color, facecolor='none', hatch='//////', alpha=0.2))
    legend = [
        Patch(facecolor=(1, 0, 0.45), edgecolor=(1, 0, 0.45), label='Table', hatch='//////', alpha=0.3),
        Patch(facecolor=(0.95, 0.6, 0.1), edgecolor=(0.95, 0.6, 0.1), label='Table (rotated)', hatch='//////', alpha=0.3)
    ]
    ax.legend(handles=legend, bbox_to_anchor=(0.5, -0.02), loc='upper center', ncol=2, fontsize=10)
    if out_path:
        fig.savefig(out_path, bbox_inches='tight', dpi=150)
    plt.close(fig)

def visualize_structure(img, structure_objs, out_path):
    """
    Draw rows (red), columns (orange), and spanning cells (teal).
    """
    fig, ax = plt.subplots(figsize=(10, 10))
    ax.axis('off')
    arr = np.array(img)
    ax.imshow(arr, interpolation='lanczos')
    for obj in structure_objs:
        bbox = obj['bbox']
        if obj['label'] == 'table row':
            face, edge, hatch = (1, 0, 0.45), (1, 0, 0.45), '//////'
        elif obj['label'] == 'table column':
            face, edge, hatch = (0.95, 0.6, 0.1), (0.95, 0.6, 0.1), '//////'
        elif obj['label'] == 'table spanning cell':
            face, edge, hatch = (0.3, 0.74, 0.8), (0.3, 0.7, 0.6), '\\'
        else:
            continue
        ax.add_patch(patches.Rectangle((bbox[0], bbox[1]), bbox[2]-bbox[0], bbox[3]-bbox[1],
                                       linewidth=0, facecolor=face, alpha=0.1))
        ax.add_patch(patches.Rectangle((bbox[0], bbox[1]), bbox[2]-bbox[0], bbox[3]-bbox[1],
                                       linewidth=2, edgecolor=edge, facecolor='none', alpha=0.3))
        ax.add_patch(patches.Rectangle((bbox[0], bbox[1]), bbox[2]-bbox[0], bbox[3]-bbox[1],
                                       linewidth=0, edgecolor=edge, facecolor='none', hatch=hatch, alpha=0.2))
    legend = [
        Patch(facecolor=(1, 0, 0.45), edgecolor=(1, 0, 0.45), label='Rows', hatch='//////', alpha=0.3),
        Patch(facecolor=(0.95, 0.6, 0.1), edgecolor=(0.95, 0.6, 0.1), label='Columns', hatch='//////', alpha=0.3),
        Patch(facecolor=(0.3, 0.74, 0.8), edgecolor=(0.3, 0.7, 0.6), label='Spanning', hatch='\\', alpha=0.3)
    ]
    ax.legend(handles=legend, bbox_to_anchor=(0.5, -0.02), loc='upper center', ncol=3, fontsize=10)
    if out_path:
        fig.savefig(out_path, bbox_inches='tight', dpi=150)
    plt.close(fig)

def extract_tables_from_images(image_paths):
    device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    det_dir=ensure_local_model('microsoft/table-transformer-detection','table-transformer-detection')
    struct_dir=ensure_local_model('microsoft/table-structure-recognition-v1.1-all','table-structure-recognition')
    det_model=AutoModelForObjectDetection.from_pretrained(str(det_dir),local_files_only=True,revision='no_timm').to(device)
    struct_model=TableTransformerForObjectDetection.from_pretrained(str(struct_dir),local_files_only=True).to(device)
    det_id2label=det_model.config.id2label; det_id2label[len(det_id2label)]='no object'
    struct_id2label=struct_model.config.id2label; struct_id2label[len(struct_id2label)]='no object'

    class MaxResize:
        def __init__(self,m): self.m=m
        def __call__(self,img): w,h=img.size; s=self.m/max(w,h); return img.resize((int(w*s),int(h*s)))
    
    det_t=transforms.Compose([MaxResize(800),transforms.ToTensor(),transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])])
    str_t=transforms.Compose([MaxResize(1000),transforms.ToTensor(),transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])])
    thresholds={'table':0.5,'table rotated':0.5}
    results=[]
    
    for path in image_paths:
        logger.info("Processing %s",path)
        img=Image.open(path).convert('RGB')
        base=Path(path).stem; out_dir=Path(path).parent/f"{base}_predicted"; out_dir.mkdir(exist_ok=True)
        # detect
        inp=det_t(img).unsqueeze(0).to(device)
        with torch.no_grad(): dout=det_model(inp)
        tables=outputs_to_objects(dout,img.size,det_id2label)
        if tables:
            fp=out_dir/f"{base}_tables.png"; visualize_detected_tables(img,tables,str(fp)); logger.info("Saved %s",fp)
        # structure
        crops=objects_to_crops(img,tables,thresholds)
        for i,crop in enumerate(crops):
            logger.info("Struct %d",i)
            inp2=str_t(crop).unsqueeze(0).to(device)
            with torch.no_grad(): sout=struct_model(inp2)
            cells=outputs_to_objects(sout,crop.size,struct_id2label)
            fp2=out_dir/f"{base}_structure_{i}.png"; visualize_structure(crop,cells,str(fp2)); logger.info("Saved %s",fp2)
            # OCR
            grid=get_cell_grid(cells)
            ocr=apply_ocr(grid,crop)
            table=[ [{'text':t} for t in ocr[r]] for r in sorted(ocr) ]
            results.append({'table':table})
    return results

if __name__=='__main__':
    p=argparse.ArgumentParser("Extract tables")
    p.add_argument('images',nargs='+')
    extract_tables_from_images(p.parse_args().images)
