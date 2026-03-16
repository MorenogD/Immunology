# predict_CS_BP.py
import sys
import pandas as pd
import itertools
import joblib
from Bio.SeqUtils.ProtParam import ProteinAnalysis

amino_acids = list("ACDEFGHIKLMNPQRSTVWY")

def extract_global_descriptors(seq):
    try:
        seq = str(seq).replace(" ", "").upper()
        analysis = ProteinAnalysis(seq)
        return {
            "length": len(seq),
            "molecular_weight": analysis.molecular_weight(),
            "aromaticity": analysis.aromaticity(),
            "instability_index": analysis.instability_index(),
            "isoelectric_point": analysis.isoelectric_point(),
            "gravy": analysis.gravy()
        }
    except Exception:
        return {k: None for k in [
            "length","molecular_weight","aromaticity",
            "instability_index","isoelectric_point","gravy"
        ]}

def amino_acid_composition(seq):
    seq = seq.upper()
    length = len(seq)
    return {aa: seq.count(aa)/length if length > 0 else 0 for aa in amino_acids}

def dipeptide_composition(seq):
    seq = seq.upper()
    length = len(seq) - 1
    dipeptides = [''.join(p) for p in itertools.product(amino_acids, repeat=2)]
    counts = {dp: 0 for dp in dipeptides}
    for i in range(len(seq) - 1):
        dp = seq[i:i+2]
        if dp in counts:
            counts[dp] += 1
    if length > 0:
        counts = {k: v/length for k, v in counts.items()}
    return counts

def predict_sequence(seq, model_path="modelo_CS_BP.pkl"):
    global_feats = extract_global_descriptors(seq)
    aac_feats = amino_acid_composition(seq)
    dpc_feats = dipeptide_composition(seq)
    features = {**global_feats, **aac_feats, **dpc_feats}
    X = pd.DataFrame([features])

    model = joblib.load(model_path)

    pred = model.predict(X)[0]
    probs = model.predict_proba(X)[0] if hasattr(model, "predict_proba") else None
    return pred, probs

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Uso: python predict_CS_BP.py <SECUENCIA_AA> [ruta_modelo.pkl]")
        sys.exit(1)
    seq_input = sys.argv[1]
    model_path = sys.argv[2] if len(sys.argv) >= 3 else "modelo_CS_BP.pkl"
    pred, probs = predict_sequence(seq_input, model_path)
    print(f"Predicción: {pred}")
    if probs is not None:
        for i, p in enumerate(probs):
            print(f"Clase {i}: {p:.4f}")

