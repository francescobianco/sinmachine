#!/usr/bin/env python3
"""Interroga il modello caotico chaos-1000 come un oracolo.

Per ogni parola trovata nel waveform:
  1. phase_search → trova φ tale che la parola emerga all'inizio
  2. continua la traiettoria → leggi la risposta
  3. cerca parole di senso compiuto nella risposta
  4. cerca co-occorrenze: frasi di due parole nella stessa finestra lunga
"""

import sys, math
from pathlib import Path
from collections import defaultdict
import numpy as np

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

from sinmachine import (load_model, query, search_phase, _simulate,
                        display_tags, _TOKEN_DISPLAY, _VOCAB_SIZE,
                        build_default_perm, y_center, char_to_idx)

SEP  = "─" * 66
SEP2 = "═" * 66

# ── 1. Carica il modello caotico ─────────────────────────────────────
model = load_model("chaos-1000")
PERM  = model["perm"]
print(SEP2)
print(f"  ORACOLO CAOTICO  —  {len(model['harmonics'])} armoniche")
print(SEP2)
print()

# ── 2. Parole da interrogare ──────────────────────────────────────────
# Mix italiano / inglese, diversi range di y-space
QUERIES = [
    # trovate nel scan precedente
    "lost", "hole", "role", "pole", "mole",
    "mai",  "via",  "ero",  "con",  "lei",
    "bar",  "del",  "old",  "one",  "red",
    # nuove (ancora non cercate)
    "sole", "mare", "vita", "mano", "sera",
    "luna", "onda", "vela", "alba", "notte",
    "time", "word", "note", "door", "star",
    "slow", "true", "real", "free", "mind",
]

# Lista parole per cercare nella risposta (senza 'c' → END3)
ANSWER_WORDS = sorted({
    "uno","due","tre","per","non","del","nel","una","lei","lui",
    "mai","poi","ora","era","ero","sei","via","bar","ore","hai",
    "amo","ami","fai","vai","sai","dai","fra","ben","men","pan",
    "mano","pane","mare","sole","luna","vita","idea","dato","bene",
    "tela","vela","lago","riva","nota","alba","sera","gelo","velo",
    "sale","vale","tale","male","real","anno","solo","fare","dire",
    "ferro","terra","porta","forte","bello","notte","mondo","tempo",
    "the","and","for","are","not","you","now","new","old","get",
    "has","him","his","may","day","way","run","sun","fun","one","two",
    "end","few","got","had","job","key","low","map","out","pay","red",
    "say","she","sky","top","try","use","war","why","win","yes","yet",
    "this","that","with","have","from","know","time","year","good",
    "some","into","than","then","when","here","were","will","what",
    "more","most","move","need","next","only","open","over","read",
    "real","road","same","show","side","sign","song","soon","stop",
    "take","talk","tell","them","away","been","best","body","book",
    "both","find","fine","fire","five","form","four","free","full",
    "give","grow","half","hand","hard","head","help","held","high",
    "hold","home","hope","hour","just","keep","kind","land","last",
    "late","lead","left","less","life","line","live","long","look",
    "loss","lost","loud","main","mean","meet","mind","miss","near",
    "news","nine","none","note","name","plan","play","post","pull",
    "rain","rate","rest","ride","role","room","rule","safe","sail",
    "sale","self","send","ship","slow","snow","sold","sole","star",
    "stay","step","tree","true","turn","type","used","vary","vast",
    "view","wait","walk","want","warm","wash","wave","weak","well",
    "wide","wife","wind","wine","wish","word","work","yard","zero",
    "pole","mole","hole","role",
}, key=len)

def words_in(text):
    tl = text.lower()
    return [(w, tl.index(w)) for w in ANSWER_WORDS if w in tl]

# ── 3. Interrogazione ─────────────────────────────────────────────────
print(f"  Interrogo l'oracolo con {len(QUERIES)} parole...")
print(f"  (resolution=4000 per maggiore precisione)")
print()

results = []
for q_word in QUERIES:
    r = query(q_word, model, answer_len=60, resolution=4000)
    answer_clean = r["answer"]
    found_in_answer = words_in(answer_clean)
    results.append({
        "query":    q_word,
        "phi":      r["phi_q"],
        "loss":     r["search_loss"],
        "recon":    r["reconstructed"],
        "match":    r["match"],
        "answer":   answer_clean,
        "ended":    r["ended"],
        "words":    found_in_answer,
    })

# ── 4. Risultati: ordina per match poi per parole trovate ─────────────
results.sort(key=lambda x: (-x["match"], -len(x["words"])))

print(SEP)
print(f"  RISPOSTE DELL'ORACOLO")
print(SEP)
print()

best_pairs = []   # (query_word, answer_word, full_result) per frasi trovate

for r in results:
    match_pct = r["match"] / len(r["query"]) * 100
    bar       = "█" * r["match"] + "░" * (len(r["query"]) - r["match"])
    status    = "✓" if r["match"] == len(r["query"]) else "~"

    print(f"  {status}  domanda: {r['query']!r:<8}  "
          f"φ={r['phi']:.4f} rad ({math.degrees(r['phi']):.1f}°)  "
          f"loss={r['loss']:.4f}")
    print(f"     ricostruita: {r['recon']!r}  [{bar}] {r['match']}/{len(r['query'])}")

    if r["words"]:
        for w, pos in r["words"][:3]:
            snippet = r["answer"][max(0, pos-3):pos+len(w)+3]
            print(f"     ✦ risposta contiene '{w}' → …{snippet!r}…")
            best_pairs.append((r["query"], w, r))
    else:
        # Mostra comunque primi 20 char della risposta
        print(f"     risposta: {r['answer'][:20]!r}…")

    print()

# ── 5. Frasi trovate (domanda + risposta con parola) ──────────────────
if best_pairs:
    print(SEP)
    print(f"  FRASI TROVATE  ({len(best_pairs)} coppie domanda→risposta)")
    print(SEP)
    print()
    seen_pairs = set()
    for q_word, a_word, r in best_pairs:
        key = (q_word, a_word)
        if key in seen_pairs:
            continue
        seen_pairs.add(key)
        print(f"  ❝ {q_word}  →  {a_word} ❞")
        print(f"    φ = {r['phi']:.5f} rad  ({math.degrees(r['phi']):.1f}°)")
        print(f"    risposta completa: {r['answer'][:40]!r}")
        print()

# ── 6. Scan a finestra lunga: cerca frasi di 2 parole adiacenti ───────
print(SEP)
print(f"  SCAN FINESTRA LUNGA (80 passi, 300 000 fasi)")
print(f"  Cerco frasi: due parole entro 10 char l'una dall'altra")
print(SEP)

import time
omegas    = np.array([h[0] for h in model["harmonics"]])
amps      = np.array([h[1] for h in model["harmonics"]])
amp_total = float(np.sum(amps))
DT        = model["dt"]
STEPS2    = 80
N2        = 300_000

t_vals  = np.arange(STEPS2, dtype=float) * DT
ang0    = np.outer(t_vals, omegas)
S       = ang0.__matmul__(amps) if False else np.dot(np.sin(ang0), amps)
C       = np.dot(np.cos(ang0), amps)

phi_grid = np.linspace(0, 2*math.pi, N2, endpoint=False)
y_grid   = (np.outer(S, np.cos(phi_grid)) + np.outer(C, np.sin(phi_grid))) / amp_total
idx_grid = np.clip(((y_grid + 1) / 2 * _VOCAB_SIZE).astype(int), 0, _VOCAB_SIZE - 1)

perm_arr = np.array(PERM, dtype=object)
char_mat = perm_arr[idx_grid]   # (STEPS2, N2)

t0 = time.time()
phrases = []
SHORT = [w for w in ANSWER_WORDS if len(w) <= 4]   # parole corte, scan veloce

for p in range(N2):
    col  = char_mat[:, p]
    text = "".join(c if len(c) == 1 else "" for c in col).lower()
    hits = [(w, text.index(w)) for w in SHORT if w in text]
    if len(hits) >= 2:
        # Ordina per posizione, cerca coppie entro 10 char
        hits.sort(key=lambda x: x[1])
        for i in range(len(hits) - 1):
            w1, p1 = hits[i]
            w2, p2 = hits[i+1]
            if p2 - (p1 + len(w1)) <= 12:
                phrases.append({
                    "phi":   phi_grid[p],
                    "deg":   math.degrees(phi_grid[p]),
                    "w1": w1, "p1": p1,
                    "w2": w2, "p2": p2,
                    "text":  text,
                })

elapsed = time.time() - t0
print(f"  Scansione completata in {elapsed:.1f}s")
print(f"  Frasi a due parole trovate: {len(phrases)}")
print()

if phrases:
    # Ordina per distanza minima tra le parole, poi per lunghezza totale
    phrases.sort(key=lambda x: (x["p2"] - x["p1"] - len(x["w1"]),
                                 -(len(x["w1"]) + len(x["w2"]))))
    seen = set()
    print(f"  Le più compatte:")
    print()
    for ph in phrases[:15]:
        key = (ph["w1"], ph["w2"])
        if key in seen:
            continue
        seen.add(key)
        t  = ph["text"]
        p1, p2 = ph["p1"], ph["p2"]
        gap = p2 - (p1 + len(ph["w1"]))
        ctx = t[max(0, p1-2) : p2 + len(ph["w2"]) + 2]
        print(f"  ❝ {ph['w1']} … {ph['w2']} ❞   (gap={gap})  "
              f"φ={ph['phi']:.4f} ({ph['deg']:.1f}°)")
        print(f"    {ctx!r}")
        print()
else:
    print("  Nessuna frase trovata. Prova con un finestra più lunga.")

print(SEP2)