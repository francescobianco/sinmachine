# 10 — Say-Hello: oneline programs that answer `hello`

## Obiettivo

Creare un modello chiamato `say-hello`.

La relazione da imparare non e piu una coppia linguistica breve come:

```text
hello -> world
```

ma una famiglia di piccoli programmi oneline, in linguaggi diversi, che quando
vengono interrogati devono produrre:

```text
hello
```

Primo caso scelto:

```text
echo "hello"; -> hello
```

Dataset iniziale creato:

```text
datasets/say-hello-noend.jsonl
```

Contiene casi senza token `<end>`, per misurare prima la capacita minima:
dopo aver ricostruito il programmino, i successivi 5 token devono essere
`hello`.

## Casi iniziali

```jsonl
{"q": "echo \"hello\";", "a": "hello"}
{"q": "print(\"hello\")", "a": "hello"}
{"q": "console.log(\"hello\");", "a": "hello"}
{"q": "printf(\"hello\");", "a": "hello"}
{"q": "puts \"hello\"", "a": "hello"}
{"q": "print \"hello\";", "a": "hello"}
{"q": "System.out.print(\"hello\");", "a": "hello"}
{"q": "fmt.Print(\"hello\")", "a": "hello"}
```

## Nota importante emersa prima di partire

Il modello `hello-world` gia presente non risponde correttamente nella pipeline
di inference attuale.

Comando:

```bash
python3 sinmachine.py --model hello-world hello
```

Risultato osservato:

```text
Reconstructed : 'hglkn'
Original      : 'hello'
Match         : 2/5 chars
Answer        : vnrme<start>nnnluqrpa|rnpjssrt_yuoriptqx^twntj
```

Questo significa che non possiamo considerare la chat/inference lunga come
fonte di verita per l'esperimento `say-hello`.

Per ora la misura piu pulita e il benchmark diretto:

1. ricostruzione esatta di `q`;
2. continuazione esatta di `a`;
3. risposta lunga esattamente 5 caratteri, senza END.

## Primo benchmark leggero

Comando:

```bash
python3 benchmark.py datasets/say-hello-noend.jsonl \
  --base dense \
  --max 3 \
  --de-iters 300 \
  --de-pop 8 \
  --seeds 42
```

Risultato:

```text
N=1  loss=0.121157  time=6.7s   correct=0/1
N=2  loss=0.190821  time=11.1s  correct=0/2
N=3  loss=0.233150  time=17.4s  correct=0/3

Total time: 35.1s
Scaling exponent alpha ~= 0.86
Estimated time for N=8: 39s
```

Esempi di fallimento:

```text
want q='echo "hello";'          got q='hacfB?[afkjQ<'    got a='[cYok'
want q='print("hello")'         got q='oa`\\SDAWcgaQ@O'  got a='fihfc'
want q='console.log("hello");'  got q='~iubaP=[lidU:Kksih[FS' got a='jhuln'
```

## Interpretazione provvisoria

Il caso `echo "hello"; -> hello` e molto piu difficile del vecchio
`hello -> world`.

Nel vecchio caso il vincolo era di circa 10 caratteri totali.

Nel primo caso `say-hello` il vincolo e:

```text
len('echo "hello";') + len('hello') = 13 + 5 = 18 caratteri
```

Con una traiettoria armonica discreta, ogni carattere aggiunto restringe molto
lo spazio delle soluzioni. Il primo benchmark leggero quindi non dimostra che
il modello sia impossibile, ma dimostra che il budget basso non basta.

## Processo sbagliato / lezione

All'inizio sembrava naturale riusare direttamente l'esperimento `hello-world`,
ma la verifica ha mostrato che il modello salvato non e allineato con la
pipeline corrente. Questo va annotato perche puo falsare esperimenti futuri:
un training puo stampare una ricostruzione buona nel suo spazio interno, ma il
modello salvato puo poi non essere recuperabile con `search_phase()` nella
pipeline standard.

Possibili cause da verificare:

1. differenza tra fase ottimizzata salvata (`_training_phi`) e fase ritrovata
   da `search_phase()`;
2. perdita di precisione per rounding dei parametri nel JSON;
3. divergenza tra loss di training e mapping/inference corrente;
4. END e `<start>` vicini nella coda del vocabolario.

## Prossimo passo

Eseguire una prova piu profonda solo su `N=1`:

```bash
python3 benchmark.py datasets/say-hello-noend.jsonl \
  --base dense \
  --max 1 \
  --de-iters 2000 \
  --de-pop 12 \
  --seeds 42,7
```

Scopo: capire se il primo caso emerge aumentando il budget o se serve cambiare
strategia, per esempio:

1. partire da prompt piu corti (`echo hello`, `print hello`);
2. aumentare armoniche o usare una base piu espressiva;
3. usare `stream` come memoria di programma-risposta;
4. introdurre un vocabolario/permutazione piu favorevole per i caratteri
   frequenti nei programmini.

