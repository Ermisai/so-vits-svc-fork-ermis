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

### Exemplo Básico - Conversão de Arquivo de Áudio

```python
from so_vits_svc_fork_ermis.inference.core import Svc
import librosa
import soundfile as sf
import torch
import numpy as np

# Carregar modelo
model = Svc(
    net_g_path="model.pth",
    config_path="config.json",
    device="cuda"  # ou "cpu"
)

# Carregar áudio de entrada
input_audio, sr = librosa.load("input.wav", sr=None)

# Reamostrar para a taxa do modelo se necessário
sovits_sample_rate = model.target_sample
if sr != sovits_sample_rate:
    input_audio = librosa.resample(
        input_audio, 
        orig_sr=sr, 
        target_sr=sovits_sample_rate
    )

# Converter voz
converted_audio, _ = model.infer(
    speaker=0,
    transpose=0,  # semitons de ajuste
    audio=input_audio.astype(np.float32),
    cluster_infer_ratio=0.0,
    auto_predict_f0=False,
    noise_scale=0.4,
    f0_method="dio"
)

# Converter tensor para numpy se necessário
if isinstance(converted_audio, torch.Tensor):
    output_audio = converted_audio.cpu().numpy()
else:
    output_audio = converted_audio

# Salvar resultado
sf.write("output.wav", output_audio, sovits_sample_rate)
print("✅ Conversão concluída! Arquivo salvo em: output.wav")
```

### Exemplo com Processamento em Batch

```python
from so_vits_svc_fork_ermis.inference.core import Svc
import librosa
import soundfile as sf
import torch
import numpy as np

# Carregar modelo
model = Svc(
    net_g_path="model.pth",
    config_path="config.json",
    device="cuda"
)

# Preparar dados para processamento em batch
input_files = ["audio1.wav", "audio2.wav", "audio3.wav"]
audios_for_conversion = []
speakers = []
transposes = []
cluster_ratios = []

sovits_sample_rate = model.target_sample

# Carregar e preparar cada áudio
for file in input_files:
    audio, sr = librosa.load(file, sr=None)
    
    # Reamostrar se necessário
    if sr != sovits_sample_rate:
        audio = librosa.resample(
            audio, 
            orig_sr=sr, 
            target_sr=sovits_sample_rate
        )
    
    audios_for_conversion.append(audio.astype(np.float32))
    speakers.append(0)  # Speaker ID
    transposes.append(0)  # Ajuste de pitch em semitons
    cluster_ratios.append(0.0)  # Cluster infer ratio

# Processar em batch
converted_results = model.infer_batch(
    speakers=speakers,
    transposes=transposes,
    audios=audios_for_conversion,
    cluster_infer_ratios=cluster_ratios,
    auto_predict_f0=False,
    noise_scale=0.4,
    f0_method="dio"
)

# Salvar resultados
for i, (converted_audio_tensor, _) in enumerate(converted_results):
    # Converter tensor para numpy se necessário
    if isinstance(converted_audio_tensor, torch.Tensor):
        output_audio = converted_audio_tensor.cpu().numpy()
    else:
        output_audio = converted_audio_tensor
    
    # Salvar arquivo
    output_file = f"output_{i+1}.wav"
    sf.write(output_file, output_audio, sovits_sample_rate)
    print(f"✅ Áudio {i+1} salvo: {output_file}")
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
