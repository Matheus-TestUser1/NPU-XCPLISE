# Exynos Neural Probe — etapa 1 do Neural Upscaler

Protótipo Android ARM64, em Java + C++ (NNAPI/NDK), para **testar o driver ENN/NPU do Samsung Galaxy S24 FE** sem root.

> **Status:** código-fonte experimental, ainda não compilado/testado num S24 FE. Não é APK pronto nem integração com Winlator.

## Faz agora

- Lista dispositivos NNAPI, nomes, tipos e versões.
- Seleciona exclusivamente um acelerador cujo nome pareça Samsung/ENN/Exynos/NPU.
- Verifica suporte às operações `CONV_2D` e `DEPTH_TO_SPACE`.
- Compila o grafo apenas nesse acelerador, **sem fallback para CPU**.
- Executa 5 warmups + 30 inferências e mede média, mediana e mínimo/máximo em ms.
- Valida uma saída RGB de `8×8` para `16×16` (2×) e exporta `.txt` pelo seletor de arquivos.

### Limites

O grafo é um **ensaio de execução com pesos fixos**, equivalente a replicação de pixels. Ainda não usa uma rede neural treinada, não melhora detalhes como EDSR e não tem ponte com Vulkan/Winlator. Os tempos obtidos **não** representam quadros 540p ou 1080p.

## Como compilar com GitHub Actions (sem Android Studio)

1. Crie um repositório próprio no GitHub.
2. Envie o conteúdo desta pasta para a raiz do repositório, incluindo `.github/workflows/build-apk.yml`.
3. Vá em **Actions → Compilar APK de diagnóstico → Run workflow**.
4. Após a execução, baixe o artefato `ExynosNeuralProbe-debug-apk`, extraia o `app-debug.apk` e instale no S24 FE.
5. No aplicativo, toque em **Identificar aceleradores**, depois em **Executar testes NPU / 2x**, e então em **Exportar relatório .TXT**.
6. Compartilhe o `.txt` na conversa para identificarmos operações suportadas e tempos reais.

O workflow prepara Gradle 8.7, JDK 17, Android SDK 35 e NDK 26.3.11579264. Pode haver ajustes de compilação após o primeiro CI; envie o log se houver falha.

## Como compilar no Android Studio

Abra a pasta como projeto Gradle e instale SDK 35, NDK 26.3.11579264 e CMake 3.22.1. Compile `app` e execute no S24 FE Android 14.

## Próximas etapas

1. Selecionar um modelo 2× treinado e compatível com Exynos 2400e (ENN/NNC ou NNAPI).
2. Confirmar a execução **na NPU**, não em CPU/GPU, e medir latência por imagem 540p → 1080p.
3. Integrar AHardwareBuffer/DMABUF e sincronização para reduzir cópias.
4. Criar uma camada de pós-processamento nativa para o Winlator/Bannerlator; medir latência fim a fim, energia e qualidade.

## Fontes de referência

- Android NNAPI: https://developer.android.com/ndk/guides/neuralnetworks
- Samsung ENN samples: https://github.com/exynos-eco/enn-sdk-samples
- MLPerf Mobile: https://github.com/mlcommons/mobile_app_open

Não altera `/sys`, `/dev` nem modelos Samsung; não precisa de KernelSU.
