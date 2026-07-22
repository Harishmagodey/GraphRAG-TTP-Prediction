import csv
import math
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
    if value is None:
        return ""
    return re.sub(r"\s+", " ", str(value).strip())


def split_techniques(value):
    text = normalize_label(value)
    if not text or text.lower() == "nan":
        return []
    techniques = []
    seen = set()
    for item in text.split(";"):
        technique = item.strip()
        key = technique.lower()
        if technique and key not in seen:
            techniques.append(technique)
            seen.add(key)
    return techniques


def load_samples():
    rows = read_xlsx_dicts(DATASET_PATH)
    samples = []
    for row in rows:
        apt = normalize_label(row.get("APT Group Name"))
        software = normalize_label(row.get("Software ID"))
        techniques = split_techniques(row.get("Software Techniques"))
        if apt and software and techniques:
            samples.append(
                {
                    "SampleID": len(samples),
                    "APT Group Name": apt,
                    "Software ID": software,
                    "TrueTechniques": techniques,
                    "TechniqueSequence": techniques,
                }
            )
    return samples


def make_folds(samples, n_folds=N_FOLDS):
    return [[i for i in range(len(samples)) if i % n_folds == fold] for fold in range(n_folds)]


def normalized_set(values):
    return {normalize_label(value).lower() for value in values if normalize_label(value)}


def hit_at_k(true_values, ranked_values, k):
    return float(bool(normalized_set(true_values) & normalized_set(ranked_values[:k])))


def reciprocal_rank(true_values, ranked_values):
    true_set = normalized_set(true_values)
    for index, prediction in enumerate(ranked_values, start=1):
        if normalize_label(prediction).lower() in true_set:
            return 1.0 / index
    return 0.0


def f1_at_k(true_values, ranked_values, k=3):
    true_set = normalized_set(true_values)
    pred_set = normalized_set(ranked_values[:k])
    if not true_set and not pred_set:
        return 1.0
    if not true_set or not pred_set:
        return 0.0
    tp = len(true_set & pred_set)
    precision = tp / len(pred_set) if pred_set else 0.0
    recall = tp / len(true_set) if true_set else 0.0
    return (2 * precision * recall / (precision + recall)) if precision + recall else 0.0


def rank_from_scores(scores):
    return [item for item, _ in sorted(scores.items(), key=lambda kv: (-kv[1], kv[0]))]


def write_predictions(path, rows):
    fieldnames = [
        "Model",
        "SampleID",
        "APT Group Name",
        "Software ID",
        "TrueTechniques",
        "RankedPredictions",
        "TopPrediction",
        "Hits@1",
        "Hits@3",
        "MRR",
        "F1",
    ]
    with open(path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def score_prediction_rows(model_name, samples, ranked_by_sample):
    rows = []
    for sample in samples:
        ranked = ranked_by_sample.get(sample["SampleID"], [])
        rows.append(
            {
                "Model": model_name,
                "SampleID": sample["SampleID"],
                "APT Group Name": sample["APT Group Name"],
                "Software ID": sample["Software ID"],
                "TrueTechniques": "; ".join(sample["TrueTechniques"]),
                "RankedPredictions": "; ".join(ranked[:TOP_K]),
                "TopPrediction": ranked[0] if ranked else "",
                "Hits@1": hit_at_k(sample["TrueTechniques"], ranked, 1),
                "Hits@3": hit_at_k(sample["TrueTechniques"], ranked, 3),
                "MRR": reciprocal_rank(sample["TrueTechniques"], ranked),
                "F1": f1_at_k(sample["TrueTechniques"], ranked, 3),
            }
        )
    return rows


def summarize(rows):
    n = len(rows)
    return {
        "Samples": n,
        "Hits@1": round(sum(float(row["Hits@1"]) for row in rows) / n, 4),
        "Hits@3": round(sum(float(row["Hits@3"]) for row in rows) / n, 4),
        "MRR": round(sum(float(row["MRR"]) for row in rows) / n, 4),
        "F1": round(sum(float(row["F1"]) for row in rows) / n, 4),
    }


def evaluate_markov_chain(samples, techniques, folds):
    ranked_by_sample = {}
    for test_indices in folds:
        train_indices = set(range(len(samples))) - set(test_indices)
        apt_start = defaultdict(Counter)
        apt_transition = defaultdict(Counter)
        software_counts = defaultdict(Counter)
        global_counts = Counter()

        for idx in train_indices:
            sample = samples[idx]
            apt = sample["APT Group Name"]
            software = sample["Software ID"]
            seq = sample["TechniqueSequence"]
            if seq:
                apt_start[apt][seq[0]] += 1
            for left, right in zip(seq, seq[1:]):
                apt_transition[apt][right] += 1
            for technique in seq:
                software_counts[software][technique] += 1
                global_counts[technique] += 1

        for idx in test_indices:
            sample = samples[idx]
            scores = Counter()
            apt = sample["APT Group Name"]
            software = sample["Software ID"]
            for tech in techniques:
                scores[tech] += 3.0 * apt_start[apt][tech]
                scores[tech] += 2.0 * apt_transition[apt][tech]
                scores[tech] += 2.0 * software_counts[software][tech]
                scores[tech] += 0.1 * global_counts[tech]
            ranked_by_sample[sample["SampleID"]] = rank_from_scores(scores)
    return ranked_by_sample


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
    e_idx = {entity: i for i, entity in enumerate(entities)}
    r_idx = {rel: i for i, rel in enumerate(rels)}
    triples = []
    for sample in train_samples:
        apt = sample["APT Group Name"]
        software = sample["Software ID"]
        if apt in e_idx and software in e_idx:
            triples.append((e_idx[apt], r_idx["USES"], e_idx[software]))
        for technique in sample["TrueTechniques"]:
            if technique in e_idx:
                triples.append((e_idx[software], r_idx["USES_TECHNIQUE"], e_idx[technique]))
                triples.append((e_idx[apt], r_idx["APT_USES_TECHNIQUE"], e_idx[technique]))

    model = TransEModel(len(entities), len(rels), dim=dim)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.02)
    triples_t = torch.tensor(triples, dtype=torch.long)
    technique_ids = torch.tensor([e_idx[t] for t in techniques if t in e_idx], dtype=torch.long)
    if len(triples_t) == 0 or len(technique_ids) == 0:
        return model, e_idx, r_idx

    for _ in range(epochs):
        perm = torch.randperm(len(triples_t))
        for start in range(0, len(perm), 256):
            batch = triples_t[perm[start : start + 256]]
            negative_tails = technique_ids[torch.randint(0, len(technique_ids), (len(batch),))]
            pos = model.score(batch[:, 0], batch[:, 1], batch[:, 2])
            neg = model.score(batch[:, 0], batch[:, 1], negative_tails)
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
        technique_pool = [t for t in techniques if t in e_idx]
        tech_tensor = torch.tensor([e_idx[t] for t in technique_pool], dtype=torch.long)
        with torch.no_grad():
            for idx in test_indices:
                sample = samples[idx]
                scores = {}
                for rel_name, source in [
                    ("USES_TECHNIQUE", sample["Software ID"]),
                    ("APT_USES_TECHNIQUE", sample["APT Group Name"]),
                ]:
                    if source not in e_idx:
                        continue
                    heads = torch.full((len(technique_pool),), e_idx[source], dtype=torch.long)
                    rels = torch.full((len(technique_pool),), r_idx[rel_name], dtype=torch.long)
                    vals = model.score(heads, rels, tech_tensor).tolist()
                    for tech, val in zip(technique_pool, vals):
                        scores[tech] = scores.get(tech, 0.0) + float(val)
                ranked_by_sample[sample["SampleID"]] = rank_from_scores(scores)
    return ranked_by_sample


class RelationalGNN(nn.Module):
    def __init__(self, n_apts, n_software, n_techniques, dim=64):
        super().__init__()
        self.apt_emb = nn.Embedding(n_apts, dim)
        self.software_emb = nn.Embedding(n_software, dim)
        self.tech_emb = nn.Embedding(n_techniques, dim)
        self.apt_gate = nn.Linear(dim, dim)
        self.software_gate = nn.Linear(dim, dim)
        self.output = nn.Linear(dim, n_techniques)

    def forward(self, apt_ids, software_ids):
        apt_h = torch.tanh(self.apt_gate(self.apt_emb(apt_ids)))
        software_h = torch.tanh(self.software_gate(self.software_emb(software_ids)))
        query = apt_h + software_h
        return self.output(query) + query @ self.tech_emb.weight.T


def train_relational_gnn(train_samples, techniques, epochs=60):
    apts = sorted({s["APT Group Name"] for s in train_samples})
    software = sorted({s["Software ID"] for s in train_samples})
    apt_idx = {v: i for i, v in enumerate(apts)}
    software_idx = {v: i for i, v in enumerate(software)}
    tech_idx = {v: i for i, v in enumerate(techniques)}

    rows = [s for s in train_samples if s["APT Group Name"] in apt_idx and s["Software ID"] in software_idx]
    x_apt = torch.tensor([apt_idx[s["APT Group Name"]] for s in rows], dtype=torch.long)
    x_soft = torch.tensor([software_idx[s["Software ID"]] for s in rows], dtype=torch.long)
    y = torch.zeros((len(rows), len(techniques)), dtype=torch.float32)
    for row_idx, sample in enumerate(rows):
        for technique in sample["TrueTechniques"]:
            y[row_idx, tech_idx[technique]] = 1.0

    model = RelationalGNN(max(1, len(apts)), max(1, len(software)), len(techniques))
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=1e-4)
    if len(rows) == 0:
        return model, apt_idx, software_idx

    for _ in range(epochs):
        logits = model(x_apt, x_soft)
        loss = F.binary_cross_entropy_with_logits(logits, y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    return model, apt_idx, software_idx


def evaluate_relational_gnn(samples, techniques, folds, model_name="GAT/R-GCN"):
    ranked_by_sample = {}
    global_counts = Counter(t for s in samples for t in s["TrueTechniques"])
    fallback = rank_from_scores(global_counts)
    for test_indices in folds:
        train_samples = [s for i, s in enumerate(samples) if i not in set(test_indices)]
        model, apt_idx, software_idx = train_relational_gnn(train_samples, techniques)
        model.eval()
        with torch.no_grad():
            for idx in test_indices:
                sample = samples[idx]
                if sample["APT Group Name"] not in apt_idx or sample["Software ID"] not in software_idx:
                    ranked_by_sample[sample["SampleID"]] = fallback
                    continue
                logits = model(
                    torch.tensor([apt_idx[sample["APT Group Name"]]], dtype=torch.long),
                    torch.tensor([software_idx[sample["Software ID"]]], dtype=torch.long),
                )[0]
                scores = {tech: float(logits[i]) for i, tech in enumerate(techniques)}
                ranked_by_sample[sample["SampleID"]] = rank_from_scores(scores)
    return ranked_by_sample


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

        history = history.float()

        h = (
            self.apt_emb(apt_ids)
            + self.software_emb(software_ids)
            + torch.tanh(self.history_proj(history))
        )

        h = F.relu(self.fc1(h))
        h = self.dropout(h)

        return self.fc2(h)
def train_temporal_gnn(train_samples, techniques, epochs=100):

    apts = sorted({s["APT Group Name"] for s in train_samples})
    software = sorted({s["Software ID"] for s in train_samples})

    apt_idx = {v: i for i, v in enumerate(apts)}
    software_idx = {v: i for i, v in enumerate(software)}
    tech_idx = {v: i for i, v in enumerate(techniques)}

    model = TemporalGRN(
        max(1, len(apts)),
        max(1, len(software)),
        len(techniques)
    )

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=0.001,
        weight_decay=1e-5
    )

    history = defaultdict(Counter)

    x_apt = []
    x_soft = []
    x_hist = []
    y = []

    for sample in sorted(train_samples, key=lambda s: s["SampleID"]):

        apt = sample["APT Group Name"]
        soft = sample["Software ID"]

        if apt not in apt_idx or soft not in software_idx:
            continue

        hist_vec = torch.zeros(len(techniques), dtype=torch.float32)

        for tech, count in history[apt].items():
            if tech in tech_idx:
                hist_vec[tech_idx[tech]] = float(count)

        if hist_vec.sum() > 0:
            hist_vec = hist_vec / hist_vec.sum()

        label = torch.zeros(len(techniques), dtype=torch.float32)

        for tech in sample["TrueTechniques"]:
            if tech in tech_idx:
                label[tech_idx[tech]] = 1.0

        x_apt.append(apt_idx[apt])
        x_soft.append(software_idx[soft])
        x_hist.append(hist_vec)
        y.append(label)

        for tech in sample["TrueTechniques"]:
            history[apt][tech] += 1

    if len(x_apt) == 0:
        return model, apt_idx, software_idx, history

    x_apt = torch.tensor(x_apt, dtype=torch.long)
    x_soft = torch.tensor(x_soft, dtype=torch.long)
    x_hist = torch.stack(x_hist)
    y = torch.stack(y)

    model.train()

    for _ in range(epochs):

        logits = model(
            x_apt,
            x_soft,
            x_hist
        )

        loss = F.binary_cross_entropy_with_logits(logits, y)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    return model, apt_idx, software_idx, history
def evaluate_temporal_gnn(samples, techniques, folds):

    ranked_by_sample = {}

    global_counts = Counter(
        t
        for s in samples
        for t in s["TrueTechniques"]
    )

    fallback = rank_from_scores(global_counts)

    for test_indices in folds:

        train_samples = [
            s
            for i, s in enumerate(samples)
            if i not in set(test_indices)
        ]

        model, apt_idx, software_idx, history = train_temporal_gnn(
            train_samples,
            techniques,
            epochs=100
        )

        model.eval()

        tech_idx = {
            t: i
            for i, t in enumerate(techniques)
        }

        with torch.no_grad():

            for idx in test_indices:

                sample = samples[idx]

                apt = sample["APT Group Name"]
                soft = sample["Software ID"]

                if apt not in apt_idx or soft not in software_idx:
                    ranked_by_sample[sample["SampleID"]] = fallback
                    continue

                hist_vec = torch.zeros(
                    len(techniques),
                    dtype=torch.float32
                )

                if apt in history:

                    for tech, count in history[apt].items():

                        if tech in tech_idx:
                            hist_vec[tech_idx[tech]] = float(count)

                if hist_vec.sum() > 0:
                    hist_vec = hist_vec / hist_vec.sum()

                logits = model(
                    torch.tensor(
                        [apt_idx[apt]],
                        dtype=torch.long
                    ),
                    torch.tensor(
                        [software_idx[soft]],
                        dtype=torch.long
                    ),
                    hist_vec.unsqueeze(0)
                )[0]

                scores = {
                    tech: float(logits[i])
                    for i, tech in enumerate(techniques)
                }

                ranked_by_sample[sample["SampleID"]] = rank_from_scores(scores)

    return ranked_by_sample    
def write_unavailable_llm(samples):
    path = Path("Experiment3_LLMOnly_Unavailable.csv")
    with open(path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["Model", "Status", "Reason", "Samples"])
        writer.writeheader()
        writer.writerow(
            {
                "Model": "LLM-only",
                "Status": "Not evaluated",
                "Reason": "No local google.generativeai package or configured LLM client is available in this repository/environment.",
                "Samples": len(samples),
            }
        )


def main():
    random.seed(RANDOM_SEED)
    np.random.seed(RANDOM_SEED)
    torch.manual_seed(RANDOM_SEED)
    torch.set_num_threads(1)

    samples = load_samples()
    techniques = sorted({tech for sample in samples for tech in sample["TrueTechniques"]})
    folds = make_folds(samples)
    print(f"Loaded {len(samples)} evaluation samples and {len(techniques)} unique techniques.")

    evaluators = [
        ("Markov Chain", "Experiment3_MarkovChain_Predictions.csv", evaluate_markov_chain),
        ("TransE", "Experiment3_TransE_Predictions.csv", evaluate_transe),
        ("GAT/R-GCN", "Experiment3_GAT_RGCN_Predictions.csv", evaluate_relational_gnn),
        ("Temporal GNN", "Experiment3_TemporalGNN_Predictions.csv", evaluate_temporal_gnn),
    ]

    comparison = []
    for model_name, output_path, evaluator in evaluators:
        print(f"Evaluating {model_name}...")
        ranked = evaluator(samples, techniques, folds)
        rows = score_prediction_rows(model_name, samples, ranked)
        write_predictions(output_path, rows)
        summary = summarize(rows)
        comparison.append({"Model": model_name, **summary})
        print(model_name, summary)

    write_unavailable_llm(samples)
    comparison.append(
        {
            "Model": "LLM-only",
            "Samples": len(samples),
            "Hits@1": "",
            "Hits@3": "",
            "MRR": "",
            "F1": "",
        }
    )

    with open("Experiment3_Comparison.csv", "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["Model", "Samples", "Hits@1", "Hits@3", "MRR", "F1"])
        writer.writeheader()
        writer.writerows(comparison)

    # Keep the project-level results file aligned with the combined comparison.
    with open("Experiment3_Results.csv", "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["Model", "Samples", "Hits@1", "Hits@3", "MRR", "F1"])
        writer.writeheader()
        writer.writerows(comparison)

    print("Saved Experiment3_Comparison.csv and Experiment3_Results.csv.")


if __name__ == "__main__":
    main()
 