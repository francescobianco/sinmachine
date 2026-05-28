#!/usr/bin/env python3
"""Crea un modello da 1000 armoniche casuali e ascoltalo cercando parole di senso compiuto.

Approccio vettorizzato: per phase_feedback=0 vale
  y(k, φ) = (S[k]·cos(φ) + C[k]·sin(φ)) / amp_total
dove S[k] e C[k] si precalcolano una volta sola, poi si scansionano
N_PHASES fasi contemporaneamente via numpy.
"""

import json, math, sys, time
import numpy as np
from pathlib import Path
from collections import Counter

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))
from sinmachine import (build_default_perm, y_center, char_to_idx,
                        _VOCAB_SIZE, MODELS_DIR, _TOKEN_DISPLAY)

SEP = "─" * 66

# ── parametri ────────────────────────────────────────────────────────
N_HARM   = 1000
STEPS    = 40
DT       = 1.0
N_PHASES = 200_000
SEED     = 1729

# ── 1. Genera il modello caotico ─────────────────────────────────────
rng    = np.random.default_rng(SEED)

# Frequenze: tutte diverse, distribuite su [0.3, 40]
omegas = rng.uniform(0.3, 40.0, N_HARM)

# Ampiezze: log-normale con σ=3 → coda pesante, poche armoniche dominano
amps   = rng.lognormal(mean=0.0, sigma=3.0, size=N_HARM)

PERM        = build_default_perm(multi_end=True)
amp_total   = float(np.sum(amps))
perm_serial = [_TOKEN_DISPLAY.get(c, c) for c in PERM]

model = {
    "name":           "chaos-1000",
    "description":    f"modello caotico: {N_HARM} armoniche, ampiezze log-normali σ=3",
    "harmonics":      [[round(float(o), 6), round(float(a), 6)]
                       for o, a in zip(omegas, amps)],
    "dt":             DT,
    "phase_feedback": 0.0,
    "steps":          STEPS,
    "multi_end":      True,
    "perm":           perm_serial,
}
out_path = MODELS_DIR / "chaos-1000.json"
with open(out_path, "w") as f:
    json.dump(model, f, indent=2)

print(SEP)
print(f"  CHAOS-1000  —  {N_HARM} armoniche casuali")
print(SEP)
print(f"  Modello salvato : {out_path}")
print(f"  amp_total       : {amp_total:.1f}")

top_idx = np.argsort(amps)[-5:][::-1]
print(f"  Top-5 dominanti :")
for i in top_idx:
    pct = amps[i] / amp_total * 100
    print(f"    ω={omegas[i]:.3f}  A={amps[i]:.1f}  ({pct:.2f}% del totale)")
print()

# ── 2. Precalcola S[k] e C[k] ────────────────────────────────────────
# y(k,φ) = (S[k]·cos φ + C[k]·sin φ) / amp_total
t_vals   = np.arange(STEPS, dtype=float) * DT          # (STEPS,)
angles_0 = np.outer(t_vals, omegas)                     # (STEPS, N_HARM)
S = np.dot(np.sin(angles_0), amps)                      # (STEPS,)
C = np.dot(np.cos(angles_0), amps)                      # (STEPS,)

r_k = np.sqrt(S**2 + C**2) / amp_total                 # ampiezza per ogni step
print(f"  Range y per step: max={r_k.max():.4f}  min={r_k.min():.4f}  media={r_k.mean():.4f}")
lo_idx = int(max(0,   ((1 - r_k.max()) / 2) * _VOCAB_SIZE))
hi_idx = int(min(96,  ((1 + r_k.max()) / 2) * _VOCAB_SIZE))
vocab_sample = [_TOKEN_DISPLAY.get(PERM[i], PERM[i]) for i in range(lo_idx, hi_idx+1)]
print(f"  Caratteri raggiungibili: {lo_idx}..{hi_idx}  →  {''.join(vocab_sample)!r}")
print()

# ── 3. Scansione vettorizzata di tutte le fasi ────────────────────────
print(f"  Scansione {N_PHASES:,} fasi (vettorizzata)...")
t0 = time.time()

phi_grid  = np.linspace(0, 2 * math.pi, N_PHASES, endpoint=False)
cos_phi   = np.cos(phi_grid)
sin_phi   = np.sin(phi_grid)

# y_grid[k, p] — (STEPS, N_PHASES)
y_grid   = (np.outer(S, cos_phi) + np.outer(C, sin_phi)) / amp_total
idx_grid = np.clip(((y_grid + 1) / 2 * _VOCAB_SIZE).astype(int), 0, _VOCAB_SIZE - 1)

# Decode: perm_arr[idx_grid] → matrice di caratteri
perm_arr    = np.array(PERM, dtype=object)              # (97,) object
char_matrix = perm_arr[idx_grid]                        # (STEPS, N_PHASES) object

# Sostituisci token speciali con tag leggibili nella matrice di testo
display_arr = np.array([_TOKEN_DISPLAY.get(c, c) for c in PERM], dtype=object)
disp_matrix = display_arr[idx_grid]                     # (STEPS, N_PHASES) object

# Costruisci i testi (grezzi, per word search con char normali)
raw_matrix  = char_matrix.T   # (N_PHASES, STEPS) — ogni riga è un testo

elapsed = time.time() - t0
print(f"  Completata in {elapsed:.2f}s")
print()

# ── 4. Lista parole (senza 'c' — è un token END nel vocab) ───────────
# 'c' (idx 67) è <end3> nel perm multi-END, non compare come lettera
WORDS = sorted({
    # italiano 3 lettere
    "uno","due","tre","per","con","non","del","nel","sul","dal","tra",
    "una","lei","lui","noi","voi","mai","poi","ora","era","ero","sei",
    "qui","via","bar","ore","hai","ahi","amo","ami","usa","fai","vai",
    "sai","dai","fra","fin","sin","ben","men","ten","pan","ran","ban",
    "dan","fan","lan","tan","van","don","ton","son","bon","ron","ion",
    "pia","mia","tua","sua","zia",
    # italiano 4+ lettere
    "mano","moto","pane","vino","luce","mare","sole","luna","aria","vita",
    "idea","onda","dato","bene","tela","filo","naso","vela","lago","riva",
    "rete","nota","alba","sera","gelo","velo","pelo","melo","telo",
    "sale","vale","tale","bale","male","pale","real","anno","mese",
    "solo","fare","dire","malo","polo","ramo","tano","rano","dano",
    "ferro","terra","torre","porta","forte","bello","bella","notte",
    "mondo","tempo","tanto","lungo","altro","primo","gran",
    # inglese 3 lettere
    "the","and","for","are","but","not","you","now","new","old","get",
    "has","him","his","its","may","day","way","run","sun","gun","fun",
    "one","two","six","ten","own","age","ago","any","ask","bad","big",
    "end","fly","few","got","had","hat","job","key","kid","low","map",
    "mix","odd","out","pay","put","raw","red","say","she","sky","top",
    "try","use","war","why","win","yes","yet","zoo","bit","fit","hit",
    "sit","art","far","jar","bar","tar","ear","den","hen","men","pen",
    # inglese 4+ lettere
    "this","that","with","have","from","they","know","time","year","good",
    "some","into","than","then","when","here","were","will","your","what",
    "more","most","move","need","next","only","open","over","read","real",
    "road","same","show","side","sign","song","soon","stop","take","talk",
    "tell","them","also","area","away","base","been","best","body","book",
    "both","find","fine","fire","five","form","four","free","full","gave",
    "give","glad","gone","grew","grow","half","hand","hard","head","help",
    "held","high","hold","home","hope","hour","huge","just","keep","kind",
    "knew","land","last","late","lead","left","less","life","line","live",
    "long","look","loss","lost","loud","made","main","mean","meet","mind",
    "miss","near","news","nine","none","note","name","plan","play","post",
    "pull","push","rain","rate","rest","ride","role","room","rule","rush",
    "safe","sail","sale","self","sell","send","ship","shop","sing","size",
    "slow","snow","sold","sole","star","stay","step","tree","true","turn",
    "type","used","user","vary","vast","view","vote","wait","walk","want",
    "warm","wash","wave","weak","well","wide","wife","wind","wine","wish",
    "word","work","wrap","yard","zero","said","tell","fell","bell","tell",
    "ball","fall","fill","bill","hill","kill","mill","pill","till","will",
    "hall","tall","wall","well","dell","bell","sell","hell","tell","yell",
    "mine","line","pine","vine","fine","wine","dine","nine","sine","tide",
    "side","wide","hide","ride","aide","fire","hire","tire","wire","mire",
    "bone","tone","lone","hone","done","gone","none","zone","bore","more",
    "gore","lore","pore","sore","tore","wore","fore","role","mole","pole",
    "sole","hole","mode","node","rode","lode","mode","ode",
}, key=len)

# Filtra: rimuovi parole con caratteri fuori range raggiungibile
reachable = set(PERM[lo_idx:hi_idx+1]) - {'\x1c', '\x1d', '\x1e', '\x02', '\x03'}
reachable_lower = {c.lower() for c in reachable if c.isalpha()}
WORDS = [w for w in WORDS if all(ch in reachable_lower for ch in w.lower())]
print(f"  Parole nel vocabolario raggiungibile: {len(WORDS)}")

# ── 5. Ricerca parole ────────────────────────────────────────────────
print(f"  Ricerca in corso...")
t0 = time.time()

found = []
for p in range(N_PHASES):
    row = raw_matrix[p]   # (STEPS,) array di char (o tag)
    # Testo grezzo (caratteri normali, non tag)
    text = "".join(c if isinstance(c, str) and len(c) == 1 else "" for c in row)
    text_lower = text.lower()
    for word in WORDS:
        if word in text_lower:
            pos = text_lower.index(word)
            found.append({
                "word":    word,
                "phi":     phi_grid[p],
                "deg":     math.degrees(phi_grid[p]),
                "text":    text,
                "pos":     pos,
                "p":       p,
            })

elapsed = time.time() - t0
print(f"  Completata in {elapsed:.1f}s  —  {len(found)} occorrenze trovate")
print()

# ── 6. Risultati ─────────────────────────────────────────────────────
print(SEP)
if not found:
    # Mostra distribuzione caratteri e parole più simili
    print("  Nessuna parola trovata.")
    print()
    sample_text = "".join(
        "".join(c if isinstance(c, str) and len(c) == 1 else "?" for c in raw_matrix[p])
        for p in range(0, N_PHASES, N_PHASES // 200)
    )
    cnt = Counter(c for c in sample_text if c.isalpha())
    print(f"  Caratteri alfa più frequenti: {cnt.most_common(15)}")
    print()
    print("  Esempi di output (φ=0.0, π/4, π/2, π, 3π/2):")
    for frac in [0, 0.125, 0.25, 0.5, 0.75]:
        p = int(frac * N_PHASES)
        row = raw_matrix[p]
        text = "".join(c if isinstance(c, str) and len(c) == 1 else "·" for c in row)
        print(f"    φ={phi_grid[p]:.4f} rad ({math.degrees(phi_grid[p]):.0f}°)  →  {text!r}")
else:
    # Mostra le prime occorrenze uniche, ordinate per lunghezza parola
    found.sort(key=lambda x: (-len(x["word"]), x["pos"]))
    seen = {}
    for f in found:
        w = f["word"]
        if w not in seen:
            seen[w] = f

    print(f"  PAROLE TROVATE ({len(seen)} uniche):")
    print()
    for w in sorted(seen, key=lambda w: -len(w)):
        f  = seen[w]
        p  = f["pos"]
        t  = f["text"]
        hi = t[:p] + f"[{t[p:p+len(w)]}]" + t[p+len(w):]
        print(f"  '{w}'  φ={f['phi']:.5f} rad ({f['deg']:.2f}°)")
        print(f"    {hi!r}")
        print()
        if len(seen) > 30 and w != list(sorted(seen, key=lambda w: -len(w)))[-1]:
            pass  # continua

    print()
    print(f"  Parola più lunga trovata: '{max(seen, key=len)}'")
    print(f"  Fase corrispondente     : φ={seen[max(seen, key=len)]['phi']:.6f} rad")

print(SEP)