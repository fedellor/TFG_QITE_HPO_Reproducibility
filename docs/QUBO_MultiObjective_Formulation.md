# Formulación del Problema HPO como un Modelo QUBO de 30 Cúbits para el Procesador Cuántico Qmio

Este documento detalla la formulación matemática perfecta y el plan metodológico para modelar la Optimización de Hiperparámetros (HPO) de un modelo Transformer como un problema **QUBO (Quadratic Unconstrained Binary Optimization) de exactamente 30 variables**, adaptado al procesador físico **Qmio** del CESGA.

---

## 1. Mapeo Óptimo: 30 Cúbits Simétricos ($6 \text{ parámetros} \times 5 \text{ opciones}$)

Para lograr un equilibrio geométrico perfecto en el espacio de búsqueda clásica y dejar un **margen de seguridad física de 2 cúbits libres en el procesador Qmio de 32 cúbits**, estructuramos un sistema simétrico de exactamente 5 opciones para cada uno de los 6 hiperparámetros:

1. **Batch Size ($BS$)**: 5 opciones (ej. `[16, 32, 64, 128, 256]`) $\implies x_1 \dots x_5$
2. **Learning Rate ($LR$)**: 5 opciones logarítmicas (ej. `[1e-3, 5e-4, 2e-4, 1e-4, 5e-5]`) $\implies x_6 \dots x_{10}$
3. **Embedding Size ($EMB$)**: 5 opciones (ej. `[64, 128, 256, 512, 1024]`) $\implies x_{11} \dots x_{15}$
4. **Attention Heads ($HEADS$)**: 5 opciones (ej. `[1, 2, 4, 8, 16]`) $\implies x_{16} \dots x_{20}$
5. **Encoder Layers ($ENC$)**: 5 opciones (ej. `[1, 2, 3, 4, 6]`) $\implies x_{21} \dots x_{25}$
6. **Decoder Layers ($DEC$)**: 5 opciones (ej. `[1, 2, 3, 4, 6]`) $\implies x_{26} \dots x_{30}$

$$\text{Total del sistema} = 6 \text{ hiperparámetros} \times 5 \text{ opciones cada uno} = 30 \text{ variables binarias} \implies 30 \text{ cúbits}$$

> [!TIP]
> **Por qué 30 Cúbits es Metodológicamente la Decisión Perfecta**:
> 1. **Resolución Equilibrada**: Cada dimensión arquitectónica de tu Transformer se explora con la misma densidad experimental (5 opciones homogéneas).
> 2. **Margen de Resiliencia en Hardware Cuántico Real**: Al no saturar la capacidad límite de Qmio (32 cúbits), dejamos 2 cúbits libres. En caso de que algunos cúbits físicos en Qmio presenten un ruido elevado o baja tasa de coherencia en el momento de la prueba real, el transpilador de Qiskit seleccionará de forma automática los **30 cúbits sanos con mejor rendimiento físico** dentro de la QPU.

---

## 2. Formulación Matemática Rigurosa y Perfecta del QUBO

Siguiendo las pautas del modelo formal para HPO, nuestro objetivo es minimizar una función de coste $f(x)$ que representa el rendimiento del Transformer (combinando precisión y tiempo). Para resolverlo en la QPU **Qmio**, planteamos una aproximación cuadrática de la función de coste más una penalización cuadrática estricta de Lagrange.

### A. La Función de Coste Aproximada (Física del Paisaje HPO)
Definimos la aproximación cuadrática del paisaje de rendimiento del Transformer (equivalente a la ecuación (1) de tu guía pero extendida a 30 variables) como:
$$\hat{f}(x) = c + \sum_{i=1}^{30} h_i x_i + \sum_{i < j}^{30} J_{ij} x_i x_j$$

Donde:
* **$x_i \in \{0, 1\}$**: Es la variable binaria que vale $1$ si seleccionamos la opción $i$ (por ejemplo, Batch Size = 32), y $0$ si no.
* **$c$**: Es la constante o intercepto base de rendimiento del modelo.
* **$h_i$**: Son los efectos o coeficientes lineales individuales de cada opción $i$.
* **$J_{ij}$**: Son las fuerzas de acoplamiento cuadrático real entre dos opciones $i$ y $j$ de hiperparámetros distintos.

### B. La Restricción Metodológica (One-Hot)
Como tenemos $6$ hiperparámetros y cada uno tiene exactamente $5$ opciones exclusivas, dividimos nuestras 30 variables en 6 grupos disjuntos $S_p$ (para $p = 1, \dots, 6$), donde cada grupo contiene las 5 opciones de un hiperparámetro. La restricción física obliga a seleccionar exactamente una opción por hiperparámetro:
$$\sum_{i \in S_p} x_i = 1, \quad \forall p \in \{1, \dots, 6\}$$

### C. Derivación Matemática del Término de Penalización (Lagrange)
Para convertir esta restricción en un problema sin restricciones (QUBO), sumamos una penalización cuadrática con peso $\lambda > 0$ por cada hiperparámetro $p$:
$$H_{\text{penalty}}(x) = \lambda \sum_{p=1}^6 \left( \sum_{i \in S_p} x_i - 1 \right)^2$$

Si expandimos matemáticamente este binomio para un hiperparámetro $p$:
$$\left( \sum_{i \in S_p} x_i - 1 \right)^2 = \left( \sum_{i \in S_p} x_i \right)^2 - 2 \sum_{i \in S_p} x_i + 1$$

Dado que nuestras variables son estrictamente binarias ($x_i \in \{0, 1\}$), se cumple que $x_i^2 = x_i$ (idempotencia binaria). Al abrir el término cuadrático de la sumatoria, obtenemos:
$$\left( \sum_{i \in S_p} x_i \right)^2 = \sum_{i \in S_p} x_i^2 + 2 \sum_{\substack{i, j \in S_p \\ i < j}} x_i x_j = \sum_{i \in S_p} x_i + 2 \sum_{\substack{i, j \in S_p \\ i < j}} x_i x_j$$

Sustituyendo esto de nuevo en la ecuación de la penalización:
$$\left( \sum_{i \in S_p} x_i - 1 \right)^2 = \left( \sum_{i \in S_p} x_i + 2 \sum_{\substack{i, j \in S_p \\ i < j}} x_i x_j \right) - 2 \sum_{i \in S_p} x_i + 1 = -\sum_{i \in S_p} x_i + 2 \sum_{\substack{i, j \in S_p \\ i < j}} x_i x_j + 1$$

Por lo tanto, la penalización total del sistema se reduce a:
$$H_{\text{penalty}}(x) = -\lambda \sum_{i=1}^{30} x_i + 2\lambda \sum_{p=1}^6 \sum_{\substack{i, j \in S_p \\ i < j}} x_i x_j + 6\lambda$$

### D. El Hamiltoniano QUBO Final y la Matriz $Q$
Sumamos la función de coste aproximada y la penalización:
$$H_{\text{QUBO}}(x) = \hat{f}(x) + H_{\text{penalty}}(x)$$
$$H_{\text{QUBO}}(x) = (c + 6\lambda) + \sum_{i=1}^{30} (h_i - \lambda) x_i + \sum_{i < j}^{30} Q_{ij} x_i x_j$$

Omitiendo la constante global de energía $(c + 6\lambda)$ que solo desplaza el espectro sin cambiar el mínimo, la matriz definitiva $Q \in \mathbb{R}^{30 \times 30}$ queda definida matemáticamente de manera exacta:

1. **Elementos de la Diagonal (Términos Lineales):**
   $$Q_{ii} = h_i - \lambda$$
   *El término $-\lambda$ actúa como un incentivo atractivo para activar variables, evitando la solución trivial nula.*

2. **Elementos fuera de la Diagonal (Términos Cuadráticos):**
   * **Si pertenecen al mismo hiperparámetro (mismo $S_p$):**
     $$Q_{ij} = J_{ij} + 2\lambda$$
     *El término $+2\lambda$ actúa como una barrera energética infranqueable (multa) que impide activar dos opciones excluyentes a la vez.*
   * **Si pertenecen a hiperparámetros distintos (diferente $S_p$):**
     $$Q_{ij} = J_{ij}$$
     *No hay penalización, simplemente representa la interacción real o acoplamiento físico obtenido en la regresión.*

---

### E. Nota de Experto: ¿Qué pasa con las Interacciones de Mayor Orden? (Reducción de Rosenberg)
Si durante nuestro análisis detectásemos interacciones de tercer orden (por ejemplo, un acoplamiento cúbico $K_{123} x_1 x_2 x_3$), romperíamos la estructura cuadrática requerida por Qmio. Para solventarlo, aplicaríamos la técnica de **reducción de Rosenberg** introduciendo una variable auxiliar $z = x_1 x_2$, de modo que:
$$x_1 x_2 x_3 \implies z x_3$$
Para garantizar que se cumpla $z = x_1 x_2$ en el mínimo, sumamos al QUBO el término de penalización cuadrático:
$$P(x_1, x_2, z) = M (x_1 x_2 - 2x_1 z - 2x_2 z + 3z)$$
Donde $M > 0$ es una penalización suficientemente grande. Este término vale exactamente $0$ si $z = x_1 x_2$ y suma energía positiva si se viola la consistencia.

---

## 3. Derivación de los Coeficientes del Paisaje ($h_i, J_{ij}$)

La función objetivo para cada muestra clásica $m$ busca balancear exactitud y velocidad mediante Pareto:
$$f^{(m)} = -(1 - \beta) \cdot \text{Acc}^{(m)} + \beta \cdot \text{Time}_{\text{norm}}^{(m)}$$

Los pesos individuales $h_i$ y las fuerzas de acoplamiento real $J_{ij}$ se obtienen entrenando una regresión lineal regularizada mediante **Lasso (L1)** sobre los datos clásicos del pool de 300 experimentos:

$$\min_{h, J} \frac{1}{300} \sum_{m=1}^{300} \left( \sum_{i=1}^{30} h_i x_i^{(m)} + \sum_{i < j}^{30} J_{ij} x_i^{(m)} x_j^{(m)} - f^{(m)} \right)^2 + \gamma \sum_{i < j}^{30} |J_{ij}|$$

**Nota de hardware:** Lasso fuerza a cero todos los acoplamientos débiles. Al esparcir la matriz $Q$, reducimos el número de puertas de acoplamiento físico en Qmio, logrando una simulación limpia y robusta frente al ruido cuántico del procesador real.

---

## 4. Roadmap Experimental del TFG

El plan de trabajo temporal del proyecto consta de 4 hitos secuenciales y claros:

1. **Fase 1: Simulación Clásica en FT3 (En cola - Estimado hasta el domingo):**
   Monitoreo autónomo del Job Array en FinisTerrae III para entrenar 300 combinaciones de Transformers (5 folds, 50 épocas) y registrar exactitud y tiempo físico en `resultados_hpo.csv`.
2. **Fase 2: Regresión y Generación del QUBO (Lunes):**
   Mapeo One-Hot a 30 variables del pool de datos. Ejecución de la regresión Lasso L1 para extraer los pesos $h_i$ e interacciones $J_{ij}$, ensamblando la matriz definitiva $Q \in \mathbb{R}^{30 \times 30}$.
3. **Fase 3: Simulación Cuántica de Qiskit (Martes):**
   Ejecución y evaluación en el simulador de Qiskit en Finisterrae III utilizando los 3 algoritmos de comparación: Optimización Bayesiana clásica, VQE estándar y QITE-VQE.
4. **Fase 4: Ejecución Física en Qmio (Miércoles):**
   Mapeo directo del Hamiltoniano óptimo de 30 cúbits en el chip cuántico real Qmio de CESGA, contrastando los perfiles de convergencia y la robustez al ruido físico frente al simulador.

---

## 5. Análisis de Latencia y Tiempos de Cómputo: CPU clásica vs. QPU física (Qmio)

Una de las observaciones experimentales más relevantes de este estudio es que la **Optimización Bayesiana clásica se ejecuta en menos de 30 segundos**, mientras que el resolvedor **VQE en la QPU real de Qmio requiere aproximadamente 35 minutos por semilla**. Este contraste resulta paradójico bajo el pretexto habitual de la "velocidad de la computación cuántica", pero responde a fundamentos físicos e ingenieriles estrictos de la tecnología de NISQ (Noisy Intermediate-Scale Quantum):

### 5.1. El Límite de la Ventaja Cuántica en Escala (Frontera de Complejidad)
* **Poder Clásico en Dimensiones Bajas:** Una matriz QUBO de $30 \times 30$ es computacionalmente minúscula para una CPU moderna. El cálculo de la energía de un espín clásico $x^T Q x$ involucra operaciones matriciales elementales de coma flotante que una CPU estándar de varios GHz (gigahercios, $10^9$ ciclos por segundo) resuelve en **microsegundos**.
* **Ventaja Cuántica Asintótica:** Los ordenadores cuánticos no poseen procesadores con mayor velocidad de reloj que las CPUs clásicas. De hecho, los procesadores superconductores operan en frecuencias de microondas de unos pocos kHz o MHz (órdenes de magnitud más lentos en ciclos de reloj que un chip de silicio clásico). Su ventaja no radica en la velocidad bruta, sino en la **complejidad algorítmica** (la capacidad de representar superposiciones de $2^N$ estados simultáneamente). La ventaja cuántica real solo se manifiesta a partir de escalas inaccesibles para supercomputadores clásicos (ej. $N \ge 100$ variables, donde evaluar $2^{100}$ combinaciones requeriría millones de años clásicos, pero es tratable cuánticamente). A escala de 30 variables, el rendimiento clásico de CPU siempre dominará en velocidad pura.

### 5.2. Cuello de Botella Híbrido de los Algoritmos Variacionales (VQE/QITE)
VQE y QITE-VQE son **algoritmos híbridos clásico-cuánticos**. No obtienen la solución en un único disparo cuántico, sino que ejecutan un proceso iterativo de optimización en bucle:

```mermaid
graph TD
    subgraph CPU Clásica
        Opt[Optimizador Clásico COBYLA] -- 1. Selecciona Parámetros θ --> CPUExec[CPU Controller]
    end
    subgraph QPU Física Qmio
        QPUPrep[2. Prepara Estado cuántico |ψ(θ)⟩] --> QPUMeas[3. Mide expectation values y shots]
    end
    CPUExec -- Envía Job de Circuitos --> QPUPrep
    QPUMeas -- 4. Retorna Energía E(θ) --> Opt
```

Este ciclo de retroalimentación se repite secuencialmente **entre 80 y 100 veces por ejecución de semilla**. La latencia final es el sumatorio acumulado de cada iteración del bucle.

### 5.3. Latencia Física Inherente del Hardware Cuántico Real
A diferencia de un simulador digital clásico (que calcula la probabilidad matemáticamente de forma teórica e instantánea en RAM), la QPU superconductora física de Qmio debe lidiar con los tiempos reales que exige la física de partículas:

1. **Muestreo Estadístico Obligatorio (Shots):** Dado que la medición cuántica colapsa la función de onda de forma probabilística, para estimar el valor esperado de la energía con alta precisión estadística, cada circuito debe prepararse y medirse **8,192 veces (shots)** en cada paso del optimizador.
2. **Tiempo de Relajación Térmica y Reset Activo (Active Reset):** Después de medir un cúbit físico en un shot, este queda alterado en un estado colapsado. Antes de poder iniciar el siguiente shot, debemos esperar a que el cúbit se enfríe y regrese físicamente a su estado fundamental de reposo $|0\rangle$ (tiempo de relajación $T_1$) o forzarlo activamente mediante pulsos de microondas específicos de desexcitación (*active reset*). Esto añade microsegundos o milisegundos obligatorios por shot.
   $$\text{Tiempo de Muestreo} = 8192 \text{ shots} \times \text{Latencia física de reset} \approx 2 \text{ a } 5 \text{ segundos por paso.}$$
3. **Sobrecarga de Red y Serialización (Network Overhead):** Qmio opera bajo un frontend cuántico remoto dentro de la red del CESGA. En cada una de las 100 iteraciones del optimizador clásico, la lista de instrucciones de circuitos debe serializarse en JSON/ZMQ, enviarse por red, encolarse en el controlador de hardware físico, traducirse a pulsos eléctricos analógicos de microondas a temperaturas criogénicas ($10\text{ mK}$), medirse y enviarse de vuelta. Esta sobrecarga de red remota acumulada 100 veces introduce una penalización temporal significativa que no existe en ejecuciones puramente locales.
4. **Calibraciones de Estabilidad en Caliente:** Durante la ejecución en cola de Slurm, las líneas de microondas físicas realizan micropausas internas automatizadas para ajustar fases y compensar fluctuaciones de temperatura criogénica del chip, garantizando la fidelidad de las puertas frente al ruido físico real a costa de pequeños retrasos temporales.

---

### 5.4. Análisis de la Brecha de Precisión (RMSE) y Viabilidad: CPU vs. QPU Real

Una observación crítica en los resultados del proyecto es la **brecha en la tasa de viabilidad y el error cuadrático medio (RMSE)**:
* **Optimización Bayesiana (CPU clásica):** Consigue una tasa de viabilidad del **100%** (0 violaciones de la restricción One-Hot en todas las semillas) y un RMSE extremadamente bajo (cercano a $0.0$).
* **VQE (QPU Qmio física):** Muestra una tasa de viabilidad del **0%** (entre 1 y 3 violaciones de exclusión en cada semilla) y valores de RMSE elevados (entre $5.2$ y $6.5$).

Desde una perspectiva científica y académica, **este no es un fallo del experimento, sino el resultado de mayor valor e interés científico de este TFG**, ya que ilustra con total honestidad el estado del arte y los límites prácticos del hardware cuántico NISQ real frente a la computación clásica exacta:

#### A. La Fragilidad de los Multiplicadores de Lagrange ante el Ruido Físico
Para obligar al resolvedor a cumplir la exclusión de hiperparámetros (elegir exactamente 1 opción de 5), la formulación QUBO introduce barreras de potencial energético positivas de $+2\lambda \approx +1.318$. 
* En la CPU clásica, la evaluación de la energía es **determinista y exacta**. Si una variable viola la restricción, la CPU suma el coste de penalización de forma matemática precisa y el optimizador descarta ese vector de inmediato.
* En la QPU física real de Qmio, el **ruido cuántico distorsiona por completo las barreras de energía**. Debido a errores de lectura y despolarización, el valor esperado medido de un estado que viola la restricción puede verse reducido artificialmente, haciéndole parecer ante el optimizador clásico (COBYLA) como un estado de baja energía factible. El ruido "difumina" los límites de las penalizaciones de Lagrange, haciendo que el resolvedor se quede atrapado en estados no factibles.

#### B. La Densidad de Puertas CNOT y la Pérdida de Fidelidad
El ansatz variacional cuántico utilizado para VQE es `RealAmplitudes(reps=2, linear)`. Para un sistema de 30 cúbits, este circuito cuántico tiene una profundidad física de **34 niveles** y consta de **148 puertas lógicas en total**, incluyendo **58 puertas CNOT (entrelazadoras de dos cúbits)**.
* Las puertas CNOT son los elementos físicos superconductores más propensos al ruido, con tasas de error físico de puerta que oscilan habitualmente entre el $0.5\%$ y el $1.5\%$ en procesadores NISQ modernos.
* La fidelidad acumulada estimada del circuito completo (la probabilidad de que el estado cuántico final se prepare sin que ocurra ni un solo error físico en ninguna puerta) disminuye exponencialmente con el número de CNOTs:
  $$F_{\text{circuito}} \approx (1 - \epsilon_{\text{CNOT}})^{58} \approx (1 - 0.01)^{58} \approx 0.55 = 55\%$$
* Con una fidelidad de apenas el $55\%$, casi la mitad de los shots lógicos medidos colapsan en estados mezclados térmicos o ruidos aleatorios. Esta pérdida drástica de coherencia aplana el paisaje energético del QUBO (barren plateaus inducidos por ruido), impidiendo que el optimizador COBYLA distinga el gradiente real y forzándolo a converger en mínimos locales no factibles con un RMSE elevado.

#### C. El Impacto del Ruido Estadístico de Disparo (Shot Noise) en COBYLA
COBYLA es un optimizador clásico libre de derivadas que se basa en la interpolación lineal de los puntos evaluados. 
* Debido a que la medición cuántica en Qmio se realiza sobre $8,192$ shots, la energía reportada tiene una **desviación estándar estadística intrínseca** (ruido de disparo).
* Esta pequeña fluctuación o "jitter" estadístico confunde a COBYLA en sistemas multivariables de gran dimensión (30 parámetros lógicos), haciéndole creer que ha convergido o que está subiendo una pendiente inexistente, provocando paradas prematuras en zonas de alta energía y violaciones.

### 5.5. Conclusiones y Vías de Mitigación para el TFG
Presentar y discutir con rigurosidad esta brecha entre CPU y QPU real en la defensa de tu TFG demuestra un **profundo conocimiento de la física del hardware cuántico actual**, elevando exponencialmente la calidad científica de la memoria frente a una simulación puramente idealizada:
1. **Mitigación de Errores (Error Mitigation):** En el futuro de la computación cuántica, técnicas como la Extrapolación de Ruido Cero (ZNE) o la Mitigación por Subespacio Simétrico ayudarán a filtrar este ruido de puertas en Qmio para aproximar el espectro ideal.
2. **Uso de Solvers Cuánticos Nativos de Tiempo Imaginario (QITE-VQE):** Como se observa en simulaciones, el algoritmo VarQITE se beneficia de una estructura geométrica de optimización más robusta que VQE ante las mesetas planas, abriendo una vía de investigación prometedora para mitigar estos fallos lógicos.


