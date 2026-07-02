"""Evaluate steering outputs with MobiDB-lite for compositional bias experiments."""
import argparse
import re
import subprocess
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

MOBIDB_CMD = (
    f"PYTHONPATH=/scratch/tmp/hrbanh/MobiDB-lite/src "
    f"python /scratch/tmp/hrbanh/MobiDB-lite/src/mobidb_lite/__main__.py"
)

AA_GROUPS = {
    "acidic": set("DE"),
    "basic": set("KRH"),
    "polar": set("STQNCYW"),
    "hydrophobic": set("ACFGILMPVW"),
    "gly": set("G"),
    "pro": set("P"),
}


def parse_fasta(path: Path):
    seqs, nlls, ppls, ids = [], [], [], []
    with open(path) as f:
        header, seq = None, []
        for line in f:
            line = line.rstrip()
            if line.startswith(">"):
                if header and seq:
                    seqs.append("".join(seq))
                    seq = []
                header = line
                ids.append(line[1:])
                m_nll = re.search(r"nll=([0-9.]+)", line)
                m_ppl = re.search(r"ppl=([0-9.]+)", line)
                nlls.append(float(m_nll.group(1)) if m_nll else float("nan"))
                ppls.append(float(m_ppl.group(1)) if m_ppl else float("nan"))
            else:
                seq.append(line)
        if header and seq:
            seqs.append("".join(seq))
    return ids, seqs, nlls, ppls


def clean_fasta_for_mobidb(input_path: Path, output_path: Path):
    """Write a FASTA with clean accession IDs and only standard AAs."""
    valid_aa = set("ACDEFGHIKLMNPQRSTVWY")
    with open(input_path) as fi, open(output_path, "w") as fo:
        seq_idx = 0
        header, seq = None, []
        for line in fi:
            line = line.rstrip()
            if line.startswith(">"):
                if header and seq:
                    s = "".join(seq)
                    s_clean = "".join(c for c in s if c in valid_aa)
                    if len(s_clean) >= 20:
                        fo.write(f">seq{seq_idx}\n{s_clean}\n")
                        seq_idx += 1
                    seq = []
                header = line
            else:
                seq.append(line)
        if header and seq:
            s = "".join(seq)
            s_clean = "".join(c for c in s if c in valid_aa)
            if len(s_clean) >= 20:
                fo.write(f">seq{seq_idx}\n{s_clean}\n")


def run_mobidb(fasta_path: Path, out_path: Path):
    cmd = f"{MOBIDB_CMD} {fasta_path} {out_path}"
    result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
    if result.returncode != 0:
        print(f"MobiDB-lite error: {result.stderr[:500]}", file=sys.stderr)
    return result.returncode == 0


def parse_mobidb_output(path: Path):
    """Parse MobiDB-lite tab output into per-seq disorder annotations.

    Returns dict: seq_id -> {'-': [(start,end),...], 'Polar': [...], 'Negative Polyelectrolyte': [...], ...}
    """
    annotations = defaultdict(lambda: defaultdict(list))
    if not path.exists():
        return annotations
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split("\t")
            if len(parts) < 4:
                continue
            seq_id, start, end, feat = parts[0], int(parts[1]), int(parts[2]), parts[3]
            annotations[seq_id][feat].append((start, end))
    return annotations


def disorder_fraction(annotations: dict, seq_len: int) -> float:
    """Fraction of residues covered by consensus disorder (-) regions."""
    if "-" not in annotations:
        return 0.0
    covered = set()
    for start, end in annotations["-"]:
        covered.update(range(start, end + 1))
    return len(covered) / seq_len


def aa_composition(seq: str) -> dict:
    n = max(len(seq), 1)
    return {grp: sum(seq.count(aa) for aa in aas) / n for grp, aas in AA_GROUPS.items()}


def diversity_stats(seq: str) -> tuple[int, int]:
    """Return (unique_aa_count, max_homopolymer_run) for a sequence."""
    unique = len(set(seq))
    if not seq:
        return 0, 0
    max_run, run = 1, 1
    for i in range(1, len(seq)):
        if seq[i] == seq[i - 1]:
            run += 1
            max_run = max(max_run, run)
        else:
            run = 1
    return unique, max_run


def summarize(label: str, fasta_path: Path, mobidb_out_path: Path, seqs: list, nlls: list):
    annotations = parse_mobidb_output(mobidb_out_path)
    # Map clean seq ids back to positions
    valid_aa = set("ACDEFGHIKLMNPQRSTVWY")
    clean_seqs = ["".join(c for c in s if c in valid_aa) for s in seqs if len("".join(c for c in s if c in valid_aa)) >= 20]
    n = len(clean_seqs)

    dis_fracs, has_disorder, has_neg_poly, has_polar = [], [], [], []
    comp_acidic, comp_basic, comp_polar_aa, comp_gly, comp_pro = [], [], [], [], []
    unique_counts, max_runs = [], []

    for i, seq in enumerate(clean_seqs):
        sid = f"seq{i}"
        ann = annotations.get(sid, {})
        dis_fracs.append(disorder_fraction(ann, len(seq)))
        has_disorder.append(1 if ann.get("-") else 0)
        has_neg_poly.append(1 if ann.get("Negative Polyelectrolyte") else 0)
        has_polar.append(1 if ann.get("Polar") else 0)
        comp = aa_composition(seq)
        comp_acidic.append(comp["acidic"])
        comp_basic.append(comp["basic"])
        comp_polar_aa.append(comp["polar"])
        comp_gly.append(comp["gly"])
        comp_pro.append(comp["pro"])
        u, mr = diversity_stats(seq)
        unique_counts.append(u)
        max_runs.append(mr)

    nlls_arr = np.array([x for x in nlls if not np.isnan(x)])
    lengths = [len("".join(c for c in s if c in valid_aa)) for s in seqs if len("".join(c for c in s if c in valid_aa)) >= 20]

    print(f"\n{'='*60}")
    print(f"  {label}  (n={n})")
    print(f"{'='*60}")
    print(f"  NLL (mean±std):     {nlls_arr.mean():.3f} ± {nlls_arr.std():.3f}")
    print(f"  Length (mean):      {np.mean(lengths):.1f}")
    print(f"  Unique AAs (mean):  {np.mean(unique_counts):.1f}")
    print(f"  Max run (mean):     {np.mean(max_runs):.1f}")
    print(f"  % any disorder:     {100*np.mean(has_disorder):.1f}%")
    print(f"  Disorder fraction:  {100*np.mean(dis_fracs):.1f}% of residues")
    print(f"  % Neg Polyelectr.:  {100*np.mean(has_neg_poly):.1f}%")
    print(f"  % Polar region:     {100*np.mean(has_polar):.1f}%")
    print(f"  AA comp D+E:        {100*np.mean(comp_acidic):.1f}%")
    print(f"  AA comp K+R+H:      {100*np.mean(comp_basic):.1f}%")
    print(f"  AA comp Polar AA:   {100*np.mean(comp_polar_aa):.1f}%")
    print(f"  AA comp G:          {100*np.mean(comp_gly):.1f}%")
    print(f"  AA comp P:          {100*np.mean(comp_pro):.1f}%")

    return {
        "label": label, "n": n,
        "nll_mean": float(nlls_arr.mean()), "nll_std": float(nlls_arr.std()),
        "mean_length": float(np.mean(lengths)),
        "unique_aa_mean": float(np.mean(unique_counts)),
        "max_run_mean": float(np.mean(max_runs)),
        "pct_disorder": float(100*np.mean(has_disorder)),
        "disorder_frac": float(100*np.mean(dis_fracs)),
        "pct_neg_poly": float(100*np.mean(has_neg_poly)),
        "pct_polar": float(100*np.mean(has_polar)),
        "comp_acidic": float(100*np.mean(comp_acidic)),
        "comp_basic": float(100*np.mean(comp_basic)),
        "comp_gly": float(100*np.mean(comp_gly)),
        "comp_pro": float(100*np.mean(comp_pro)),
    }


def run_condition(label: str, fasta_path: Path, tmpdir: Path) -> dict:
    ids, seqs, nlls, ppls = parse_fasta(fasta_path)
    if not seqs:
        print(f"[{label}] No sequences found in {fasta_path}")
        return {}
    clean_path = tmpdir / f"{label}_clean.fasta"
    mobidb_path = tmpdir / f"{label}_mobidb.tsv"
    clean_fasta_for_mobidb(fasta_path, clean_path)
    print(f"[{label}] Running MobiDB-lite on {sum(1 for _ in open(clean_path)) // 2} seqs...", flush=True)
    run_mobidb(clean_path, mobidb_path)
    return summarize(label, fasta_path, mobidb_path, seqs, nlls)


def run_condition_prefix_split(label: str, fasta_path: Path, tmpdir: Path, prefix_len: int) -> dict:
    """Like run_condition but reports disorder stats separately for prefix and continuation regions."""
    ids, seqs, nlls, ppls = parse_fasta(fasta_path)
    if not seqs:
        return {}
    valid_aa = set("ACDEFGHIKLMNPQRSTVWY")
    clean_path = tmpdir / f"{label}_clean.fasta"
    mobidb_path = tmpdir / f"{label}_mobidb.tsv"
    clean_fasta_for_mobidb(fasta_path, clean_path)
    print(f"[{label}] Running MobiDB-lite (prefix_split, prefix_len={prefix_len})...", flush=True)
    run_mobidb(clean_path, mobidb_path)
    annotations = parse_mobidb_output(mobidb_path)

    clean_seqs = ["".join(c for c in s if c in valid_aa) for s in seqs if len("".join(c for c in s if c in valid_aa)) >= 20]
    n = len(clean_seqs)

    # Disorder stats for continuation only (residues > prefix_len)
    cont_dis_fracs, cont_has_neg_poly = [], []
    cont_unique_counts, cont_max_runs = [], []
    for i, seq in enumerate(clean_seqs):
        sid = f"seq{i}"
        ann = annotations.get(sid, {})
        cont_len = max(len(seq) - prefix_len, 0)
        cont = seq[prefix_len:]
        if cont_len == 0:
            cont_dis_fracs.append(0.0)
            cont_has_neg_poly.append(0)
            cont_unique_counts.append(0)
            cont_max_runs.append(0)
            continue
        # Count disorder residues in continuation
        covered = set()
        for start, end in ann.get("-", []):
            for r in range(start, end + 1):
                if r > prefix_len:
                    covered.add(r)
        cont_dis_fracs.append(len(covered) / cont_len)
        neg_poly_in_cont = any(start > prefix_len for start, end in ann.get("Negative Polyelectrolyte", []))
        cont_has_neg_poly.append(1 if neg_poly_in_cont else 0)
        u, mr = diversity_stats(cont)
        cont_unique_counts.append(u)
        cont_max_runs.append(mr)

    nlls_arr = np.array([x for x in nlls if not np.isnan(x)])
    lengths = [len(s) for s in clean_seqs]
    cont_lengths = [max(l - prefix_len, 0) for l in lengths]

    print(f"\n{'='*60}")
    print(f"  {label}  (n={n}, prefix_len={prefix_len})")
    print(f"{'='*60}")
    print(f"  NLL (mean±std):            {nlls_arr.mean():.3f} ± {nlls_arr.std():.3f}")
    print(f"  Total length (mean):       {np.mean(lengths):.1f}")
    print(f"  Continuation length (mean):{np.mean(cont_lengths):.1f}")
    print(f"  Unique AAs cont (mean):    {np.mean(cont_unique_counts):.1f}")
    print(f"  Max run cont (mean):       {np.mean(cont_max_runs):.1f}")
    print(f"  Continuation disorder frac:{100*np.mean(cont_dis_fracs):.1f}%")
    print(f"  % Neg Polyelectr (cont):   {100*np.mean(cont_has_neg_poly):.1f}%")

    # AA composition of continuation only
    comp_acidic_cont, comp_pro_cont = [], []
    for seq in clean_seqs:
        cont = seq[prefix_len:]
        if cont:
            comp_acidic_cont.append(sum(cont.count(c) for c in "DE") / len(cont))
            comp_pro_cont.append(cont.count("P") / len(cont))
    print(f"  D+E fraction (cont):       {100*np.mean(comp_acidic_cont):.1f}%")
    print(f"  P fraction (cont):         {100*np.mean(comp_pro_cont):.1f}%")

    return {
        "label": label, "n": n, "prefix_len": prefix_len,
        "nll_mean": float(nlls_arr.mean()),
        "cont_length_mean": float(np.mean(cont_lengths)),
        "cont_unique_aa_mean": float(np.mean(cont_unique_counts)),
        "cont_max_run_mean": float(np.mean(cont_max_runs)),
        "cont_disorder_frac": float(100*np.mean(cont_dis_fracs)),
        "pct_neg_poly_cont": float(100*np.mean(cont_has_neg_poly)),
        "comp_acidic_cont": float(100*np.mean(comp_acidic_cont)) if comp_acidic_cont else 0.0,
        "comp_pro_cont": float(100*np.mean(comp_pro_cont)) if comp_pro_cont else 0.0,
    }


def main():
    p = argparse.ArgumentParser(description="Evaluate steering FASTAs with MobiDB-lite")
    p.add_argument("--conditions", nargs="+", required=True,
                   help="label:fasta_path pairs, e.g. 'baseline:out/base.fasta'")
    p.add_argument("--tmpdir", type=Path, default=Path("steering_experiments/compositional_bias/mobidb"))
    p.add_argument("--prefix_len", type=int, default=0,
                   help="If >0, also report disorder stats for the continuation (residues after prefix_len)")
    p.add_argument("--output_json", type=Path, default=None,
                   help="Path to write JSON summary (default: tmpdir/summary.json)")
    args = p.parse_args()

    args.tmpdir.mkdir(parents=True, exist_ok=True)
    results = []
    for cond in args.conditions:
        label, fpath = cond.split(":", 1)
        if args.prefix_len > 0:
            r = run_condition_prefix_split(label, Path(fpath), args.tmpdir, args.prefix_len)
        else:
            r = run_condition(label, Path(fpath), args.tmpdir)
        if r:
            results.append(r)

    if len(results) > 1:
        import json
        out_json = args.output_json if args.output_json else args.tmpdir / "summary.json"
        with open(out_json, "w") as f:
            json.dump(results, f, indent=2)
        print(f"\nSummary saved to {out_json}")


if __name__ == "__main__":
    main()
