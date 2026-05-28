# SinMachine — ipotesi di modello armonico per generazione intelligente

L’idea di partenza è costruire un modello alternativo al paradigma classico degli LLM autoregressivi basati su predizione del prossimo token.

L’obiettivo non è partire da:

```text
prompt → next token → next token → next token
```

ma da:

```text
domanda → fase iniziale → dinamica armonica → ascolto → testo
```

La macchina viene chiamata provvisoriamente **SinMachine**.

## 1. Ipotesi fondamentale

Si assume che possa esistere una funzione armonica, o una somma di funzioni armoniche, capace di contenere informazione linguistica e conoscenza compressa.

Il testo non sarebbe quindi generato come semplice catena discreta di token, ma come campionamento ricorsivo di una dinamica continua.

La forma generale può essere pensata come:

```text
y(t) = Σ sin(ωᵢ t + φᵢ)
```

oppure, più in generale:

```text
y(t) = F(t, φ, ω)
```

dove:

* `t` è il tempo di ascolto/campionamento;
* `φ` è la fase;
* `ω` è la frequenza;
* `y(t)` produce valori che vengono mappati su token o concetti.

## 2. Fase come coordinata semantica

La fase è il parametro principale per interrogare la SinMachine.

L’idea è che una domanda non debba modificare tutta la funzione, ma debba determinare il punto di fase da cui iniziare ad ascoltare.

Quindi:

```text
domanda → fase iniziale φ₀
```

e poi:

```text
φ₀ → ascolto della funzione → sequenza testuale
```

La fase diventa una coordinata semantica.

In altre parole, porre una domanda significa trovare il punto giusto nello spazio armonico da cui cominciare l’estrazione del testo.

## 3. Ampiezza come supporto di mappatura

L’ampiezza non viene considerata il parametro principale della conoscenza.

Poiché la sinusoide varia naturalmente tra `-1` e `1`, questo intervallo può essere usato come spazio normalizzato per mappare i token.

Quindi:

```text
[-1, 1] → vocabolario/token/concept space
```

L’ampiezza può essere vista più come supporto di lettura o attivazione che come contenuto profondo della conoscenza.

## 4. Frequenza come profondità e densità di conoscenza

La frequenza rappresenta la profondità o densità informativa.

L’intuizione è:

```text
frequenza lenta → struttura ampia, profonda, densa
frequenza veloce → dettaglio locale, variazione fine, superficie
```

Una bassa frequenza può contenere più struttura distribuita lungo l’onda, mentre una frequenza alta rappresenta variazioni più locali e granulari.

Quindi la conoscenza profonda non sarebbe necessariamente associata a oscillazioni rapide, ma a onde lente, ampie, capaci di contenere strutture semantiche più estese.

## 5. Generazione come ascolto

La generazione testuale viene vista come un processo di ascolto.

Dato un punto iniziale di fase:

```text
φ₀
```

si campiona ricorsivamente la funzione.

Il primo token estratto può diventare parte della coordinata successiva, influenzando la fase o il passo successivo di ascolto.

Schema:

```text
φ₀ → token₁
token₁ + φ₀ → φ₁
φ₁ → token₂
token₂ + φ₁ → φ₂
...
```

La generazione diventa quindi un processo autodeterminato, in cui la funzione viene ascoltata e il suo stesso output contribuisce a determinare i campionamenti successivi.

## 6. Problema inverso: fare una domanda alla macchina

Se la SinMachine produce testo quando viene ascoltata da una certa fase, allora il problema fondamentale diventa inverso:

```text
come trasformare una domanda nella fase corretta?
```

Serve quindi un encoder:

```text
E(domanda) = φ₀
```

Il modello completo diventa:

```text
domanda
  ↓
encoder semantico
  ↓
fase iniziale φ₀
  ↓
funzione armonica
  ↓
campionamento ricorsivo
  ↓
testo
```

In questa visione, la domanda non è un prompt da continuare, ma una coordinata di accesso allo spazio armonico.

## 7. Differenza rispetto agli LLM

Un LLM classico fa:

```text
P(token successivo | contesto)
```

La SinMachine invece mira a fare:

```text
fase semantica iniziale → traiettoria armonica → token
```

Quindi il testo non è la causa primaria del processo, ma una proiezione discreta di una dinamica continua.

Il linguaggio sarebbe una superficie osservabile di una struttura più profonda.

## 8. Obiettivo teorico

L’obiettivo non è ottenere solo un’altra implementazione del transformer, ma verificare se esiste una core idea diversa:

```text
intelligenza = risonanza armonica compressa
```

invece di:

```text
intelligenza = predizione autoregressiva del prossimo token
```

La domanda di ricerca è:

```text
può la conoscenza essere rappresentata come struttura armonica compatta?
```

e ancora:

```text
può il testo essere estratto da una dinamica periodica attraverso una mappa fase-token?
```

## 9. Ipotesi computazionale minima

Una possibile implementazione locale minimale potrebbe avere questi componenti:

```text
1. vocabolario di token
2. funzione sinusoidale o somma di sinusoidi
3. mappa valore → token
4. encoder domanda → fase iniziale
5. regola ricorsiva di aggiornamento fase
6. decoder testuale
```

Pseudo-schema:

```text
input domanda q

φ = Encoder(q)

for step in range(N):
    y = sin(ω * t + φ)
    token = map_to_token(y)
    output.append(token)
    φ = update_phase(φ, token)
    t = t + Δt
```

## 10. Possibile forma base

```text
y(t) = sin(ωt + φ)
```

con:

```text
token = Quantize(y)
```

oppure:

```text
y(t) = Σ sin(ωᵢt + φ)
```

con più frequenze che rappresentano più livelli di profondità.

La versione più semplice deve partire da un singolo oscillatore, poi estendersi a più armoniche.

## 11. Interpretazione dei parametri

```text
φ = fase semantica / punto di accesso / domanda
ω = profondità o densità di conoscenza
y = valore osservato
token = discretizzazione dell’osservazione
t = tempo di ascolto
```

## 12. Idea guida

Se oggi sappiamo empiricamente che un artefatto intelligente computazionale può esistere, allora non è necessario assumere che l’unica via sia l’architettura autoregressiva attuale.

Gli LLM dimostrano che l’intelligenza computazionale è possibile.

La SinMachine prova a chiedere:

```text
esiste una rappresentazione più semplice, armonica, compressa e deterministica della conoscenza?
```

Il punto non è imitare gli LLM, ma cercare una funzione generatrice diversa.

## 13. Frase sintetica del progetto

La SinMachine è un’ipotesi di macchina generativa in cui una domanda viene trasformata in una fase iniziale, e la risposta emerge ascoltando ricorsivamente una funzione armonica la cui frequenza rappresenta la profondità della conoscenza e il cui valore viene discretizzato in token.
