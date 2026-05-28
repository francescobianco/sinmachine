# 11 — Five micro-models before Say-Hello

## Perche servono

`say-hello` e risultato troppo ambizioso come primo salto:

```text
echo "hello"; -> hello
```

richiede 18 caratteri vincolati nella stessa traiettoria. Prima di insistere su
programmini oneline conviene costruire una scala di modelli piccoli ma
significativi, nello stesso spirito di `simple-sums`.

Questi micro-modelli testano capacita diverse:

1. memoria / identita;
2. trasformazione simbolica;
3. tabella booleana;
4. progressione ordinata;
5. mapping tra classi di caratteri.

## Dataset implementati

### 1. `identity-bits-noend`

File:

```text
datasets/identity-bits-noend.jsonl
```

Scopo: copiare bit singoli e coppie di bit.

```text
0  -> 0
1  -> 1
00 -> 00
01 -> 01
10 -> 10
11 -> 11
```

Questo e il piu semplice controllo di memoria: la macchina deve continuare con
la stessa sequenza.

### 2. `not-bits-noend`

File:

```text
datasets/not-bits-noend.jsonl
```

Scopo: invertire bit singoli e coppie.

```text
0  -> 1
1  -> 0
00 -> 11
01 -> 10
10 -> 01
11 -> 00
```

E il primo test di trasformazione sistematica.

### 3. `and-bits-noend`

File:

```text
datasets/and-bits-noend.jsonl
```

Scopo: imparare una tabella booleana completa.

```text
0&0= -> 0
0&1= -> 0
1&0= -> 0
1&1= -> 1
```

E piccolo, ma contiene caratteri non contigui nel vocabolario (`&`, `=`, `0`,
`1`) e una risposta sbilanciata: tre zeri e un uno.

### 4. `next-digit-noend`

File:

```text
datasets/next-digit-noend.jsonl
```

Scopo: successore modulo 10.

```text
0> -> 1
1> -> 2
...
9> -> 0
```

E un test di progressione ordinata. Non basta memorizzare una coppia: il dataset
completo richiede una struttura ciclica.

### 5. `lowercase-abc-noend`

File:

```text
datasets/lowercase-abc-noend.jsonl
```

Scopo: normalizzare maiuscole in minuscole per un alfabeto minimo.

```text
A   -> a
B   -> b
C   -> c
AA  -> aa
AB  -> ab
BA  -> ba
ABC -> abc
```

E un test di mapping tra regioni distanti del vocabolario ASCII.

## Primo benchmark leggero

Parametri usati:

```bash
python3 benchmark.py <dataset> --base dense --de-iters 200 --de-pop 6 --seeds 42
```

### `identity-bits-noend`, primi 3 casi

```text
N=1  loss=0.000000  time=0.9s  correct=1/1
N=2  loss=0.000048  time=1.0s  correct=1/2
N=3  loss=0.000005  time=1.4s  correct=3/3
Total time: 3.3s
```

Nota: N=2 fallisce pur avendo loss quasi zero. Questo conferma che piccole
differenze continue possono cambiare bucket discreto.

### `not-bits-noend`, primi 3 casi

```text
N=1  loss=0.000000  time=1.0s  correct=1/1
N=2  loss=0.000002  time=1.1s  correct=2/2
N=3  loss=0.000064  time=1.4s  correct=0/3
Total time: 3.5s
```

Il passaggio da bit singoli a coppie rompe la soluzione con budget basso.

### `and-bits-noend`, 4 casi

```text
N=1  loss=0.000849  time=1.2s  correct=0/1
N=2  loss=0.000710  time=1.5s  correct=0/2
N=3  loss=0.016972  time=1.9s  correct=0/3
N=4  loss=0.000861  time=4.9s  correct=0/4
Total time: 9.5s
```

Sorprendente: anche il singolo caso `0&0= -> 0` non viene centrato. Il problema
sembra il prompt di 4 caratteri con simboli lontani, non la risposta.

### `next-digit-noend`, primi 3 casi

```text
N=1  loss=0.000053  time=1.1s  correct=0/1
N=2  loss=0.000435  time=1.2s  correct=0/2
N=3  loss=0.002836  time=1.5s  correct=0/3
Total time: 3.8s
```

Anche qui la loss bassa non basta. Il carattere `>` nel prompt sposta il vincolo
fuori dalla zona naturale dei digit.

### `lowercase-abc-noend`, primi 3 casi

```text
N=1  loss=0.000000  time=1.0s  correct=1/1
N=2  loss=0.000024  time=1.1s  correct=2/2
N=3  loss=0.000111  time=1.2s  correct=2/3
Total time: 3.3s
```

Questo e un buon candidato per training successivo: i primi due casi funzionano
subito, il terzo e vicino.

## Classifica provvisoria

Dal piu promettente al piu problematico con budget basso:

```text
identity-bits
lowercase-abc
not-bits
next-digit
and-bits
```

## Lezione provvisoria

La difficolta non dipende solo dal numero di coppie. Dipende molto da:

1. lunghezza del prompt;
2. distanza dei caratteri nel vocabolario;
3. presenza di simboli come `&`, `=`, `>`, `"`, `;`;
4. margine tra valore continuo e bucket discreto.

Questo conferma una direzione gia emersa in `say-hello`: prima di aumentare la
semantica del task, bisogna controllare la geometria del vocabolario.
