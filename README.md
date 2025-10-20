# 🎙️ So-VITS-SVC Fork Ermis

Biblioteca Python para **conversão de voz (Voice Conversion)** usando o modelo So-VITS-SVC, otimizada para inferência.

Ideal para aplicações de **síntese e conversão de voz**, **clonagem vocal** ou **transformação de áudio**, onde a qualidade e performance de inferência são importantes.

---

## 🧬 Funcionalidades Implementadas

### 1. Conversão de Voz (Voice Conversion)

- Conversão de voz de alta qualidade
- Processamento sequencial de arquivos de áudio
- Processamento em batch para múltiplos arquivos
- Suporte a múltiplos speakers (vozes)
- Sem dependências de treinamento

### 2. Algoritmos de F0 (Pitch Detection)

**Suporte a múltiplos métodos:**

- `crepe` - Alta precisão, mais lento
- `dio` - Rápido, boa qualidade
- `pm` - Muito rápido, qualidade moderada
- `harvest` - Qualidade intermediária

### 3. Processamento de Áudio

**Recursos:**

- Suporte a GPU (CUDA) e CPU
- Ajuste de pitch via transpose (semitons)
- Predição automática de F0
- Processamento em lote para múltiplos arquivos
- Controle de noise_scale para ajuste fino

### 4. Configurações Flexíveis

**Parâmetros ajustáveis:**

- `speaker` - ID (int) ou nome (str) do speaker/voz alvo
- `transpose` - Ajuste de pitch em semitons (-12 a +12)
- `auto_predict_f0` - Predição automática de pitch (True/False)
- `f0_method` - Algoritmo de detecção de pitch
- `device` - Dispositivo de processamento (GPU ou CPU)
- `noise_scale` - Escala de ruído (controle fino da qualidade)

---

## 🧠 Exemplos de Uso

### Exemplo Básico

```python
from so_vits_svc_fork_ermis.inference.core import Svc

# Carregar modelo
model = Svc(
    net_g_path="model.pth",
    config_path="config.json",
    device="cuda"  # ou "cpu"
)

# Converter voz
audio_tensor, audio_length = model.infer(
    speaker=0,
    transpose=2,  # +2 semitons
    audio=input_audio_array,
    auto_predict_f0=True,
    f0_method="crepe"
)
# ➞ Áudio convertido retornado como tensor PyTorch
```

### Exemplo com Processamento em Batch

```python
from so_vits_svc_fork_ermis.inference.core import Svc
import numpy as np

# Carregar modelo
model = Svc(
    net_g_path="model.pth",
    config_path="config.json"
)

# Preparar múltiplos áudios
audios = [audio1, audio2, audio3]  # Lista de arrays numpy
speakers = [0, 0, 1]  # Diferentes speakers
transposes = [0, 2, -2]  # Diferentes ajustes de pitch

# Processar em batch
results = model.infer_batch(
    speakers=speakers,
    transposes=transposes,
    audios=audios,
    auto_predict_f0=True,
    f0_method="dio"
)

# Resultados é uma lista de tuplas (audio_tensor, length)
for i, (audio, length) in enumerate(results):
    print(f"Áudio {i+1}: {length} samples")
```
---

## � Instalação

Instale a biblioteca via pip (Git):

```bash
pip install git+https://github.com/Ermisai/so-vits-svc-fork-ermis.git
```

Para instalar a versão de desenvolvimento a partir do código-fonte local:

```bash
pip install -e .
```

Ou instalar diretamente do repositório Git:

```bash
pip install git+https://github.com/Ermisai/so-vits-svc-fork-ermis.git
```