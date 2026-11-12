# NPU-XCPLISE — Exynos Neural Probe v0.3

Aplicativo experimental Android (Java + C++/NNAPI) para medir um **upscaler neural ESPCN 2×** usando o acelerador Samsung ENN no Galaxy S24 FE / Exynos 2400e.

> **Status:** v0.1 testada no aparelho (dispositivo enn, convolução e DepthToSpace funcionando). A v0.2 foi executada no S24 FE: até 960×540→1920×1080 (média 59,466 ms em FP32). A v0.3 testa **FP16 real em tensores, buffers e pesos**, ainda sem validação no aparelho. Não é um módulo de upscaling do Winlator pronto.

## Funcionalidades

- Identificação de aceleradores NNAPI (tipo, nome e versão).
- Teste de referência 8×8 → 16×16, com pesos fixos (v0.1).
- ESPCN 2× demonstrativo treinado: 2 convoluções 3×3, canais 3→12→12, ReLU e DepthToSpace.
- Comparações FP32 x FP16 de 160×90→320×180, 320×180→640×360 e 480×270→960×540.
- Comparação FP32 x FP16 separada de 960×540→1920×1080; pode consumir dezenas de MB e falhar se o driver não suportar o grafo em FP16.
- Métricas: operações suportadas para cada precisão, tempo de compilação, compute NNAPI (média/p50/p90), quantidade de saídas finitas e bytes reais de entrada/saída. FP16 reduz a quantidade de bytes em ~50% em relação a FP32.
- Exportação do relatório para TXT e seleção explícita do acelerador 'enn', sem fallback NNAPI automático para CPU.

## Compilação e instalação

1. Acesse a aba **Actions → Compilar APK de diagnóstico**.
2. Escolha a execução mais recente da branch main, ou use **Run workflow**.
3. Após sucesso, baixe **Artifacts → ExynosNeuralProbe-debug-apk**, extraia e instale o app-debug.apk.
4. Abra no S24 FE. Execute **Identificar aceleradores**, depois **ESPCN 2× — 90p / 180p / 270p**.
5. Se o teste leve concluir, execute separadamente **Comparar FP32 x FP16 — 540p → 1080p**. O app tentará ambos; se FP16 for incompatível, registre a mensagem SKIP/erro.
6. Exporte o relatório TXT para comparar as duas precisões (FP32 x FP16), incluindo as diferenças de bytes e latência.

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

Este projeto **não tem integração Winlator/RADV concluída**, não fornece FSR, DLSS nem frame generation. INT8, execução reutilizável e compartilhamento Vulkan/NNAPI são experimentos futuros, ainda não implementados. Precisamos medir inferência em resoluções reais antes de planejar a ponte Vulkan.

Referências: https://developer.android.com/ndk/guides/neuralnetworks · https://github.com/exynos-eco/enn-sdk-samples · https://github.com/mlcommons/mobile_app_open


## Histórico medido na v0.2 (FP32, S24 FE)

| Entrada → saída | Média NNAPI compute |
|---|---:|
| 160×90 → 320×180 | 2,508 ms |
| 320×180 → 640×360 | 7,247 ms |
| 480×270 → 960×540 | 14,821 ms |
| 960×540 → 1920×1080 | 59,466 ms |

Todos são tempos de inferência síncrona de um modelo ESPCN demonstrativo e entrada sintética. **Não equivalem a FPS finais do Winlator.** A v0.3 deverá ser comparada usando o mesmo telefone e condições térmicas semelhantes.

## Sobre a aceleração

A v0.3 cria um grafo NNAPI com tipo **ANEURALNETWORKS_TENSOR_FLOAT16** para entrada, camadas, pesos e saída — não é apenas armazenamento FP16. Só compila e executa para o driver selecionado **enn** via ANeuralNetworksCompilation_createForDevices. Se alguma operação for recusada, o app informa e ignora a execução FP16 (sem fallback automático à CPU). O driver proprietário ainda pode ter estratégias internas não observáveis.

A v0.3 **não implementa INT8** nem sincronização Vulkan, pois precisam de quantização calibrada e testes específicos de buffers. Usar FP16 pode acelerar ou não, dependendo do driver, da largura de banda e do overhead. O teste A/B fornece a medição.
