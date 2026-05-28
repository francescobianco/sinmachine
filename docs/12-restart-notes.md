# 12 — Restart Notes

Questa nota serve per ripartire dopo il riavvio di Codex in modalita
privileged.

## Stato del lavoro

Sono stati creati esperimenti per due linee:

1. `say-hello`: programmini oneline che devono rispondere `hello`;
2. cinque micro-modelli piu piccoli e significativi, da usare prima di tornare
   a `say-hello`.

## File nuovi

Dataset:

```text
datasets/say-hello-noend.jsonl
datasets/identity-bits-noend.jsonl
datasets/not-bits-noend.jsonl
datasets/and-bits-noend.jsonl
datasets/next-digit-noend.jsonl
datasets/lowercase-abc-noend.jsonl
```

Diario:

```text
docs/10-say-hello-oneline-programs.md
docs/11-five-micro-models.md
docs/12-restart-notes.md
```

File modificato:

```text
Makefile
```

Il `Makefile` ora include:

```text
make micro-bench
make micro-identity
make micro-not
make micro-and
make micro-next
make micro-lower
```

## Risultati importanti gia osservati

### `say-hello`

Il primo caso:

```text
echo "hello"; -> hello
```

non e stato trovato con benchmark leggero:

```text
N=1  loss=0.121157  time=6.7s   correct=0/1
N=2  loss=0.190821  time=11.1s  correct=0/2
N=3  loss=0.233150  time=17.4s  correct=0/3
```

La prova profonda sul solo primo caso ha migliorato la loss ma non ha trovato
la sequenza:

```text
N=1  loss=0.048256  time=86.0s  correct=0/1
got q='j]ie40Zjajb15'
got a='`gcm^'
```

Anche `vocab_align.py` su target piatto:

```text
echo "hello";hello
```

non converge in 12 round da circa 10 secondi.

### Micro-modelli

Con benchmark leggero:

```bash
python3 benchmark.py <dataset> --base dense --de-iters 200 --de-pop 6 --seeds 42
```

classifica provvisoria:

```text
identity-bits
lowercase-abc
not-bits
next-digit
and-bits
```

`identity-bits` e `lowercase-abc` sono i piu promettenti.

## Problema tecnico incontrato

`apply_patch` a volte fallisce con:

```text
bwrap: loopback: Failed RTM_NEWADDR: Operation not permitted
```

Non e un errore della patch: e il sandbox helper. Alcune modifiche sono state
fatte con `perl -0pi` come workaround.

Al riavvio privileged conviene verificare:

```bash
git status --short
make -n micro-bench
sh -n Makefile
```

## Prossimi passi consigliati

1. Committare e pushare questo stato.
2. Eseguire `make micro-bench` con budget standard.
3. Allenare prima `identity-bits` o `lowercase-abc`.
4. Tornare a `say-hello` solo dopo aver capito la soglia di lunghezza e la
   geometria del vocabolario.
