import os
import torch
import pandas as pd

from crnn_sed import (
    CFG,
    CRNN,
    build_cache,
    CachedDataset,
    run_inference,
    get_class_names,
)

cfg = CFG()

#Load class names
train_ann = os.path.join(cfg.data_root, "train", "annotations.csv")
classes = get_class_names(train_ann)

#Load best checkpoint
ckpt_path = os.path.join(cfg.out_dir, "best.pt")
print("Loading checkpoint:", ckpt_path)

ckpt = torch.load(ckpt_path, map_location=cfg.device)

model = CRNN(cfg, len(classes)).to(cfg.device)
model.load_state_dict(ckpt["model"])
model.eval()

#Build/reuse validation cache
val_cache_dir = build_cache(
    "validation",
    os.path.join(cfg.data_root, "validation", "audio"),
    os.path.join(cfg.data_root, "validation", "annotations.csv"),
    classes,
    cfg,
)

val_ds = CachedDataset(val_cache_dir, cfg, train=False)
print("Validation files:", len(val_ds))

#Run inference only, no training
val_pred, _ ,_= run_inference(
    model=model,
    ds=val_ds,
    class_names=classes,
    cfg=cfg,
    labeled=False,
)

out_path = os.path.join(cfg.out_dir, "validation_predictions.csv")
val_pred.to_csv(out_path, index=False)

print("Wrote:", out_path)
print("Rows:", len(val_pred))
print(val_pred.head())