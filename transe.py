import csv
import random
import re
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

class TransEModel(nn.Module):
    def __init__(self, n_entities, n_relations, dim=32):
        super().__init__()
        self.entities = nn.Embedding(n_entities, dim)
        self.relations = nn.Embedding(n_relations, dim)
        nn.init.xavier_uniform_(self.entities.weight)
        nn.init.xavier_uniform_(self.relations.weight)

    def score(self, heads, rels, tails):
        return -torch.linalg.norm(self.entities(heads) + self.relations(rels) - self.entities(tails), dim=1)

def train_transe(train_samples, techniques, epochs=45, dim=32):
    entities = sorted(
        {s["APT Group Name"] for s in train_samples}
        | {s["Software ID"] for s in train_samples}
        | set(techniques)
    )
    rels = ["USES", "USES_TECHNIQUE", "APT_USES_TECHNIQUE"]
    e_idx = {e: i for i, e in enumerate(entities)}
    r_idx = {r: i for i, r in enumerate(rels)}
    triples = []
    for s in train_samples:
        apt, soft = s["APT Group Name"], s["Software ID"]
        if apt in e_idx and soft in e_idx:
            triples.append((e_idx[apt], r_idx["USES"], e_idx[soft]))
        for t in s["TrueTechniques"]:
            if t in e_idx:
                triples.append((e_idx[soft], r_idx["USES_TECHNIQUE"], e_idx[t]))
                triples.append((e_idx[apt], r_idx["APT_USES_TECHNIQUE"], e_idx[t]))

    model = TransEModel(len(entities), len(rels), dim=dim)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.02)
    triples_t = torch.tensor(triples, dtype=torch.long)
    tech_ids = torch.tensor([e_idx[t] for t in techniques if t in e_idx], dtype=torch.long)
    if len(triples_t) == 0 or len(tech_ids) == 0:
        return model, e_idx, r_idx

    for _ in range(epochs):
        perm = torch.randperm(len(triples_t))
        for start in range(0, len(perm), 256):
            batch = triples_t[perm[start : start + 256]]
            neg_tails = tech_ids[torch.randint(0, len(tech_ids), (len(batch),))]
            pos = model.score(batch[:, 0], batch[:, 1], batch[:, 2])
            neg = model.score(batch[:, 0], batch[:, 1], neg_tails)
            loss = F.margin_ranking_loss(pos, neg, torch.ones_like(pos), margin=1.0)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
    return model, e_idx, r_idx

def evaluate_transe(samples, techniques, folds):
    ranked_by_sample = {}
    for test_indices in folds:
        train_samples = [s for i, s in enumerate(samples) if i not in set(test_indices)]
        model, e_idx, r_idx = train_transe(train_samples, techniques)
        model.eval()
        tech_pool = [t for t in techniques if t in e_idx]
        tech_tensor = torch.tensor([e_idx[t] for t in tech_pool], dtype=torch.long)
        with torch.no_grad():
            for idx in test_indices:
                s = samples[idx]
                scores = {}
                for rel_name, source in [("USES_TECHNIQUE", s["Software ID"]), ("APT_USES_TECHNIQUE", s["APT Group Name"])]:
                    if source not in e_idx:
                        continue
                    heads = torch.full((len(tech_pool),), e_idx[source], dtype=torch.long)
                    rels = torch.full((len(tech_pool),), r_idx[rel_name], dtype=torch.long)
                    vals = model.score(heads, rels, tech_tensor).tolist()
                    for tech, val in zip(tech_pool, vals):
                        scores[tech] = scores.get(tech, 0.0) + float(val)
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

    ranked_by_sample = evaluate_transe(samples, techniques, folds)
    rows = []
    for s in samples:
        ranked = ranked_by_sample.get(s["SampleID"], [])
        rows.append({
            "Model": "TransE",
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

    with open("TransE_Predictions.csv", "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    n = len(rows)
    print({
        "Model": "TransE",
        "Samples": n,
        "Hits@1": round(sum(r["Hits@1"] for r in rows) / n, 4),
        "Hits@3": round(sum(r["Hits@3"] for r in rows) / n, 4),
        "MRR": round(sum(r["MRR"] for r in rows) / n, 4),
        "F1": round(sum(r["F1"] for r in rows) / n, 4),
    })

if __name__ == "__main__":
    main()