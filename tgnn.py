import csv
import random
import re
from collections import Counter, defaultdict
from pathlib import Path
from zipfile import ZipFile
import xml.etree.ElementTree as ET
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

RANDOM_SEED = 42
DATASET_PATH = Path("attackmitre (1).xlsx")
N_FOLDS = 5
TOP_K = 10

NS = {
    "a": "http://schemas.openxmlformats.org/spreadsheetml/2006/main",
    "r": "http://schemas.openxmlformats.org/officeDocument/2006/relationships",
}

def col_to_idx(ref):
    letters = "".join(ch for ch in ref if ch.isalpha())
    idx = 0
    for ch in letters:
        idx = idx * 26 + ord(ch.upper()) - 64
    return idx - 1

def read_xlsx_dicts(path):
    with ZipFile(path) as zf:
        shared = []
        if "xl/sharedStrings.xml" in zf.namelist():
            root = ET.fromstring(zf.read("xl/sharedStrings.xml"))
            for item in root.findall("a:si", NS):
                shared.append("".join(t.text or "" for t in item.findall(".//a:t", NS)))
        workbook = ET.fromstring(zf.read("xl/workbook.xml"))
        rels = ET.fromstring(zf.read("xl/_rels/workbook.xml.rels"))
        rel_target = {rel.attrib["Id"]: rel.attrib["Target"] for rel in rels}
        sheet = workbook.find("a:sheets/a:sheet", NS)
        rel_id = sheet.attrib["{http://schemas.openxmlformats.org/officeDocument/2006/relationships}id"]
        target = rel_target[rel_id]
        sheet_path = "xl/" + target.lstrip("/") if not target.startswith("xl/") else target
        sheet_root = ET.fromstring(zf.read(sheet_path))
        rows = []
        max_col = 0
        for row in sheet_root.findall(".//a:sheetData/a:row", NS):
            values = {}
            for cell in row.findall("a:c", NS):
                idx = col_to_idx(cell.attrib["r"])
                max_col = max(max_col, idx)
                value_node = cell.find("a:v", NS)
                if value_node is None:
                    value = ""
                elif cell.attrib.get("t") == "s":
                    value = shared[int(value_node.text)]
                else:
                    value = value_node.text or ""
                values[idx] = value
            rows.append([values.get(i, "") for i in range(max_col + 1)])
    header = rows[0]
    return [dict(zip(header, row)) for row in rows[1:]]

def normalize_label(value):
    return re.sub(r"\s+", " ", str(value).strip()) if value is not None else ""

def split_techniques(value):
    text = normalize_label(value)
    if not text or text.lower() == "nan":
        return []
    techniques, seen = [], set()
    for item in text.split(";"):
        t = item.strip()
        if t and t.lower() not in seen:
            techniques.append(t)
            seen.add(t.lower())
    return techniques

def load_samples():
    rows = read_xlsx_dicts(DATASET_PATH)
    samples = []
    for row in rows:
        apt = normalize_label(row.get("APT Group Name"))
        software = normalize_label(row.get("Software ID"))
        techniques = split_techniques(row.get("Software Techniques"))
        if apt and software and techniques:
            samples.append({
                "SampleID": len(samples),
                "APT Group Name": apt,
                "Software ID": software,
                "TrueTechniques": techniques,
            })
    return samples

def make_folds(samples, n_folds=N_FOLDS):
    return [[i for i in range(len(samples)) if i % n_folds == fold] for fold in range(n_folds)]

def normalized_set(values):
    return {normalize_label(v).lower() for v in values if normalize_label(v)}

def hit_at_k(true_values, ranked_values, k):
    return float(bool(normalized_set(true_values) & normalized_set(ranked_values[:k])))

def reciprocal_rank(true_values, ranked_values):
    true_set = normalized_set(true_values)
    for idx, pred in enumerate(ranked_values, start=1):
        if normalize_label(pred).lower() in true_set:
            return 1.0 / idx
    return 0.0

def f1_at_k(true_values, ranked_values, k=3):
    true_set, pred_set = normalized_set(true_values), normalized_set(ranked_values[:k])
    if not true_set or not pred_set:
        return 0.0
    tp = len(true_set & pred_set)
    p, r = tp / len(pred_set), tp / len(true_set)
    return (2 * p * r / (p + r)) if (p + r) else 0.0

def rank_from_scores(scores):
    return [item for item, _ in sorted(scores.items(), key=lambda kv: (-kv[1], kv[0]))]

class TemporalGRN(nn.Module):
    def __init__(self, num_apts, num_software, num_techniques, hidden_dim=128):
        super().__init__()
        self.apt_emb = nn.Embedding(num_apts, hidden_dim)
        self.software_emb = nn.Embedding(num_software, hidden_dim)
        self.history_proj = nn.Linear(num_techniques, hidden_dim)
        self.fc1 = nn.Linear(hidden_dim, hidden_dim)
        self.dropout = nn.Dropout(0.30)
        self.fc2 = nn.Linear(hidden_dim, num_techniques)

    def forward(self, apt_ids, software_ids, history):
        h = (
            self.apt_emb(apt_ids)
            + self.software_emb(software_ids)
            + torch.tanh(self.history_proj(history.float()))
        )
        h = self.dropout(F.relu(self.fc1(h)))
        return self.fc2(h)

def train_temporal_gnn(train_samples, techniques, epochs=100):
    apts = sorted({s["APT Group Name"] for s in train_samples})
    software = sorted({s["Software ID"] for s in train_samples})
    apt_idx = {v: i for i, v in enumerate(apts)}
    software_idx = {v: i for i, v in enumerate(software)}
    tech_idx = {v: i for i, v in enumerate(techniques)}

    model = TemporalGRN(max(1, len(apts)), max(1, len(software)), len(techniques))
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-5)
    history = defaultdict(Counter)

    x_apt, x_soft, x_hist, y = [], [], [], []
    for s in sorted(train_samples, key=lambda item: item["SampleID"]):
        apt, soft = s["APT Group Name"], s["Software ID"]
        if apt not in apt_idx or soft not in software_idx:
            continue
        hist_vec = torch.zeros(len(techniques), dtype=torch.float32)
        for tech, count in history[apt].items():
            if tech in tech_idx:
                hist_vec[tech_idx[tech]] = float(count)
        if hist_vec.sum() > 0:
            hist_vec = hist_vec / hist_vec.sum()

        label = torch.zeros(len(techniques), dtype=torch.float32)
        for tech in s["TrueTechniques"]:
            if tech in tech_idx:
                label[tech_idx[tech]] = 1.0

        x_apt.append(apt_idx[apt])
        x_soft.append(software_idx[soft])
        x_hist.append(hist_vec)
        y.append(label)

        for tech in s["TrueTechniques"]:
            history[apt][tech] += 1

    if not x_apt:
        return model, apt_idx, software_idx, history

    x_apt = torch.tensor(x_apt, dtype=torch.long)
    x_soft = torch.tensor(x_soft, dtype=torch.long)
    x_hist = torch.stack(x_hist)
    y = torch.stack(y)

    model.train()
    for _ in range(epochs):
        logits = model(x_apt, x_soft, x_hist)
        loss = F.binary_cross_entropy_with_logits(logits, y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    return model, apt_idx, software_idx, history

def evaluate_temporal_gnn(samples, techniques, folds):
    ranked_by_sample = {}
    global_counts = Counter(t for s in samples for t in s["TrueTechniques"])
    fallback = rank_from_scores(global_counts)
    tech_idx = {t: i for i, t in enumerate(techniques)}

    for test_indices in folds:
        train_samples = [s for i, s in enumerate(samples) if i not in set(test_indices)]
        model, apt_idx, software_idx, history = train_temporal_gnn(train_samples, techniques, epochs=100)
        model.eval()
        with torch.no_grad():
            for idx in test_indices:
                s = samples[idx]
                apt, soft = s["APT Group Name"], s["Software ID"]
                if apt not in apt_idx or soft not in software_idx:
                    ranked_by_sample[s["SampleID"]] = fallback
                    continue
                hist_vec = torch.zeros(len(techniques), dtype=torch.float32)
                for tech, count in history.get(apt, {}).items():
                    if tech in tech_idx:
                        hist_vec[tech_idx[tech]] = float(count)
                if hist_vec.sum() > 0:
                    hist_vec = hist_vec / hist_vec.sum()

                logits = model(
                    torch.tensor([apt_idx[apt]], dtype=torch.long),
                    torch.tensor([software_idx[soft]], dtype=torch.long),
                    hist_vec.unsqueeze(0),
                )[0]
                scores = {tech: float(logits[i]) for i, tech in enumerate(techniques)}
                ranked_by_sample[s["SampleID"]] = rank_from_scores(scores)
    return ranked_by_sample

def main():
    random.seed(RANDOM_SEED)
    np.random.seed(RANDOM_SEED)
    torch.manual_seed(RANDOM_SEED)
    torch.set_num_threads(1)

    samples = load_samples()
    techniques = sorted({t for s in samples for t in s["TrueTechniques"]})
    folds = make_folds(samples)

    ranked_by_sample = evaluate_temporal_gnn(samples, techniques, folds)
    rows = []
    for s in samples:
        ranked = ranked_by_sample.get(s["SampleID"], [])
        rows.append({
            "Model": "Temporal GNN",
            "SampleID": s["SampleID"],
            "APT Group Name": s["APT Group Name"],
            "Software ID": s["Software ID"],
            "TrueTechniques": "; ".join(s["TrueTechniques"]),
            "RankedPredictions": "; ".join(ranked[:TOP_K]),
            "Hits@1": hit_at_k(s["TrueTechniques"], ranked, 1),
            "Hits@3": hit_at_k(s["TrueTechniques"], ranked, 3),
            "MRR": reciprocal_rank(s["TrueTechniques"], ranked),
            "F1": f1_at_k(s["TrueTechniques"], ranked, 3),
        })

    with open("TemporalGNN_Predictions.csv", "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    n = len(rows)
    print({
        "Model": "Temporal GNN",
        "Samples": n,
        "Hits@1": round(sum(r["Hits@1"] for r in rows) / n, 4),
        "Hits@3": round(sum(r["Hits@3"] for r in rows) / n, 4),
        "MRR": round(sum(r["MRR"] for r in rows) / n, 4),
        "F1": round(sum(r["F1"] for r in rows) / n, 4),
    })

if __name__ == "__main__":
    main()