# Arquitetura NPU — Exynos 2400 / Galaxy S24 FE (2400e)

**Pesquisa em 12/11/2026.** Este documento distingue fatos publicados pela Samsung, fatos observados nos nossos testes e hipóteses de otimização. A Samsung confirmou que o Exynos 2400e usa o mesmo silício do Exynos 2400, alterando o clock máximo da CPU de 3,2 para 3,1 GHz (declaração à Android Authority em 27/09/2024). Por isso, a configuração física da NPU 2400 é fortemente aplicável ao 2400e; ainda falta um manual público completo da microarquitetura e de seu firmware.

## Informações oficiais

- Samsung descreve o **Exynos 2400** com *AI Engine with 17K MAC NPU (2 GNPU + 2 SNPU) and DSP*. GNPU = General NPU; SNPU = Shallow NPU.
- O material de apresentação **ISSCC 2025, paper 16.3**, identifica arquitetura heterogênea com motores vetoriais e dois tipos de motores tensoriais (**8K MAC** e **512 MAC**), hierarquia de memória com *Q-caches*, prefetching e tiling de tensores. **Hipótese compatível** com os totais: dois grupos de 8K + dois grupos de 512 = 17.408 MAC. O artigo não deve ser tomado como prova de acesso direto a cada grupo pelo Android.
- A Samsung descreve o **Exynos AI Studio** como compilador específico para SoCs Exynos, com otimização de grafos, quantização e lowering de modelos ONNX/TFLite/PyTorch; isso é potencialmente mais apropriado que construir grafos NNAPI manualmente.
- O **Exynos AI LiteCore** oferece caminho para compilação em NNC e integrações LiteRT/ExecuTorch. Verificar requisitos, acesso e compatibilidade com o S24 FE antes de migrar.

## Evidência do nosso aparelho

Relatórios reais no S24 FE:
- NNAPI detectou dispositivo `enn` (tipo 4, versão E0.0.1). Um segundo dispositivo, `nnapi-reference`, é CPU.
- `ANeuralNetworksCompilation_createForDevices` foi usado com somente `enn`, evitando fallback **do NNAPI**.
- `CONV_2D` e `DEPTH_TO_SPACE` aceitos e executados.
- ESPCN simples com 1.644 pesos foi executado com a saída 960×540 → 1920×1080.
- FP32 59,295 ms versus FP16 30,040 ms na v0.3: velocidade ~1,97×. Esses valores são de `NNAPI compute()` isolado; **não medem FPS Winlator**, qualidade em jogos, nem máxima utilização da NPU.
- Perfis de frequência vistos anteriormente apontam dispositivos devfreq `npu0`, `npu1` e `unpu`; não comprovam vínculo 1:1 com GNPU/SNPU.

## Otimização com verificação empírica

1. **Reuso de execução:** comparar `ANeuralNetworksExecution_compute` tradicional com `ANeuralNetworksExecution_setReusable(true)`. API 31+; o modelo e buffers devem permanecer válidos. Recompilar o modelo por execução não é necessário.
2. **INT8:** calibrar entradas/intermediários, verificar suporte a `TENSOR_QUANT8_ASYMM` e pesos `TENSOR_QUANT8_SYMM_PER_CHANNEL`, medir tempo e degradação de PSNR/SSIM contra FP16. Conv quant exige bias INT32 e escala correta. Não supor suporte sem teste.
3. **Rearquitetar modelo:** experimentar reconstrução **somente de luminância Y**, retornando CbCr ao processamento na GPU, com pesos treinados apropriadamente. Possível diminuição de banda/saída, não garantida.
4. **Distribuição de canais e blocos:** benchmark de 8, 12, 16, 32 canais para identificar preferência do compilador. Evitar supor que 32 seja automaticamente melhor. Inspecionar eficiência de convoluções 3×3/1×1 e *tiling*.
5. **Medição detalhada:** usar `ANeuralNetworksExecution_setMeasureTiming` e `ANeuralNetworksExecution_getDuration` quando o driver aceitar, separando custo host/driver/hardware.
6. **Integração GPU:** experimentar buffers AHardwareBuffer/DMABUF e sincronização. O trabalho no NNAPI não torna o RADV capaz de executar a NPU por conta própria.
7. **Arquitetura de longo prazo:** avaliar Exynos AI LiteCore/Exynos AI Studio e suporte ExecuTorch como caminho de compilação diretamente especializado para a geração Exynos; verificar licenças e disponibilidade.

**Limites de conhecimento:** código da NPU, escalonador, caches exatos, ISA, frequências por motor e mapeamento GNPU↔dispositivos Linux não estão inteiramente públicos. Sem documentação específica, afirmações de "100% de uso dos 17K MAC" ou "44 TOPS reais" seriam especulativas.

## Referências

- Samsung, Exynos 2400: https://semiconductor.samsung.com/processor/mobile-processor/exynos-2400/
- Samsung, Exynos 2400 em Device Farm (17K MAC): https://soc-developer.semiconductor.samsung.com/global/remotelab/AI
- ISSCC 2025, press kit 16.3: https://submissions.mirasmart.com/ISSCC2025/PDF/ISSCC2025PressKit.pdf
- Artigo científico: https://doi.org/10.1109/ISSCC49661.2025.10904722
- Samsung, AI Studio optimization: https://semiconductor.samsung.com/news-events/tech-blog/unpacking-samsungs-comprehensive-on-device-ai-sdk-toolchain-strategy/
- Samsung, Exynos AI LiteCore: https://soc-developer.semiconductor.samsung.com/global/development/ai-litecore/
- Android NDK NNAPI reference: https://developer.android.com/ndk/reference/group/neural-networks

- Declaração Samsung sobre silício 2400e (Android Authority): https://www.androidauthority.com/samsung-galaxy-s4-fe-eyxnos-2400e-chip-difference-3485738/
