# NPU-XCPLISE — Exynos Neural Probe v0.2

Aplicativo experimental Android (Java + C++/NNAPI) para medir um **upscaler neural ESPCN 2×** usando o acelerador Samsung ENN no Galaxy S24 FE / Exynos 2400e.

> **Status:** v0.1 testada no aparelho (dispositivo enn, convolução e DepthToSpace funcionando). A v0.2 inclui uma pequena rede realmente treinada, mas **os novos testes ainda precisam ser executados no S24 FE**. Não é um módulo de upscaling do Winlator pronto.

## Funcionalidades

- Identificação de aceleradores NNAPI (tipo, nome e versão).
- Teste de referência 8×8 → 16×16, com pesos fixos (v0.1).
- ESPCN 2× demonstrativo treinado: 2 convoluções 3×3, canais 3→12→12, ReLU e DepthToSpace.
- Benchmarks de 160×90→320×180, 320×180→640×360 e 480×270→960×540.
- Benchmark separado de 960×540→1920×1080; pode consumir dezenas de MB e falhar se o driver não suportar o grafo.
- Métricas: operações suportadas, tempo de compilação, tempo de NNAPI compute (média/p50/p90), quantidade de saídas finitas, bytes de entrada e saída.
- Exportação do relatório para TXT e seleção explícita do acelerador 'enn', sem fallback NNAPI automático para CPU.

## Compilação e instalação

1. Acesse a aba **Actions → Compilar APK de diagnóstico**.
2. Escolha a execução mais recente da branch main, ou use **Run workflow**.
3. Após sucesso, baixe **Artifacts → ExynosNeuralProbe-debug-apk**, extraia e instale o app-debug.apk.
4. Abra no S24 FE. Execute **Identificar aceleradores**, depois **ESPCN 2× — 90p / 180p / 270p**.
5. Se o teste leve concluir, execute separadamente **ESPCN 2× — 540p → 1080p (pesado)**.
6. Exporte o relatório TXT para analisar a latência.

Build: Gradle 8.7, AGP 8.5.2, JDK 17, Android SDK 35, NDK 26.3.11579264, CMake 3.22.1.

## Modelo neural

O modelo experimental foi treinado usando PyTorch e algumas imagens de exemplo do scikit-image, sem alterar nem extrair os modelos proprietários da Samsung.

- Script: tools/train_espcn.py (execução: python tools/train_espcn.py --steps 700).
- Pesos: app/src/main/assets/espcn2x_fp16.b64.
- Layout de filtros: PyTorch OIHW convertido para NNAPI OHWI.
- Origem dos pesos: treino demonstrativo com reconstrução espacial RGB 2×.
- Em uma única imagem de validação (rocket), o modelo obteve ~30,12 dB de PSNR contra ~29,84 dB da interpolação bicúbica. **Isso não prova qualidade superior em outros conteúdos e não representa jogos.**
- Entradas dos testes Android são sintéticas e os benchmarks não processam uma imagem escolhida pelo usuário.

## Limitações importantes

O tempo de NNAPI compute não corresponde à latência total de um jogo. Não inclui o compositor Winlator, DMA-BUF, cópias GPU/NPU, sincronização Vulkan, nem qualidade temporal. Mesmo restringindo ao driver NNAPI enn, não é possível comprovar como o fornecedor distribui operações entre GNPU/SNPU/UNPU internamente.

Este projeto **não tem integração Winlator/RADV concluída**, não fornece FSR, DLSS nem frame generation. Precisamos medir inferência em resoluções reais antes de planejar a ponte Vulkan.

Referências: https://developer.android.com/ndk/guides/neuralnetworks · https://github.com/exynos-eco/enn-sdk-samples · https://github.com/mlcommons/mobile_app_open
