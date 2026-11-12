# NPU-XCPLISE — Exynos Neural Probe v0.6

Aplicativo experimental Android (Java + C++/NNAPI) para medir um **upscaler neural ESPCN 2×** usando o acelerador Samsung ENN no Galaxy S24 FE / Exynos 2400e.

> **Status:** v0.1 testada no aparelho (dispositivo enn, convolução e DepthToSpace funcionando). A v0.2 foi executada no S24 FE: até 960×540→1920×1080 (média 59,466 ms em FP32). A v0.3 validou **FP16 real**, alcançando média de 30,040 ms (FP32 59,295 ms) na tarefa de 540p→1080p. **A v0.4 foi executada no S24 FE**: FP16 reutilizável 21,283 ms para 960×540→1920×1080, contra FP16 normal 30,248 ms. A **v0.5** adiciona testes de arquitetura 1×1 e 8/12 canais ainda sem validação no aparelho. Não é um módulo de upscaling do Winlator pronto.

## Funcionalidades

- Identificação de aceleradores NNAPI (tipo, nome e versão).
- Teste de referência 8×8 → 16×16, com pesos fixos (v0.1).
- ESPCN 2× demonstrativo treinado: 2 convoluções 3×3, canais 3→12→12, ReLU e DepthToSpace.
- Comparações FP32 x FP16 x FP16 reutilizável de 160×90→320×180, 320×180→640×360 e 480×270→960×540.
- Comparação FP32 x FP16 x FP16 reutilizável separada de 960×540→1920×1080; pode consumir dezenas de MB e falhar se o driver não suportar o grafo em FP16.
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


## v0.4 — Primeiro teste de escalonamento NNAPI

Agora cada resolução testa **três caminhos** no mesmo modelo ESPCN: (1) FP32 criando uma execução por quadro, (2) FP16 criando uma execução por quadro e (3) FP16 com **ANeuralNetworksExecution_setReusable(true)**, compartilhando a mesma execução entre quadros sequenciais. O caminho reutilizável é API 31+; por segurança o Android mínimo foi elevado a 31, sendo compatível com o S24 FE Android 14.

O teste reutilizável ainda usa buffers em memória comum e chamadas síncronas; **não** é DMA-BUF, não é zero-copy, não demonstra múltiplas unidades GNPU simultâneas e não deve acelerar em 2x só por reutilizar. Se o driver recusar, o relatório apresenta falha e mantém os testes tradicionais.

A pesquisa de arquitetura, separando fatos e hipóteses de otimização, está em [docs/exynos2400-npu-architecture.md](docs/exynos2400-npu-architecture.md). O caminho de modelos potencialmente mais específico do Exynos é Exynos AI LiteCore / Exynos AI Studio, sujeito à disponibilidade de ferramentas e compatibilidade.

**Importante:** NNAPI está depreciada em versões recentes do Android. Nossa implementação é uma prova de conceito funcional no dispositivo existente; para evolução de longo prazo vale pesquisar LiteRT/ExecuTorch + backend Exynos.


## v0.5 — Buscando <16 ms no ESPCN 540p → 1080p

Novo botão **5. LITE 1x1/8 canais — 90p a 270p** e botão **6. LITE 1x1/8 canais — 540p a 1080p**. Ambos fazem comparação A/B/C no mesmo driver **enn**, usando **FP16 e execução reutilizável**:

| Variante | Convolução 1 | Convolução 2 | Características |
|---|---|---|---|
| BASELINE_12_K3 | 3×3, 3→12 canais | 3×3, 12→12 canais | Rede treinada original v0.4 |
| LITE_12_K1 | 3×3, 3→12 canais | 1×1, 12→12 canais | Pesos centrais extraídos, **não retreinada** |
| LITE_8_K1 | 3×3, 3→8 canais | 1×1, 8→12 canais | Pesos podados, **não retreinada** |

As versões LITE_12_K1 e LITE_8_K1 são **apenas experimentos de arquitetura e tempo de inferência**. Seus pesos foram derivados matematicamente do modelo anterior sem treinamento novo; portanto **não representam upscalers visuais utilizáveis**. A próxima etapa, caso alguma configuração seja rápida, é treinar essa topologia no PyTorch e validar PSNR/SSIM e imagens reais antes de considerá-la.

O objetivo de 16 ms em 1080p é uma **meta**, não um resultado. A v0.5 ainda não foi compilada/testada no S24 FE. Recomendação: executar o teste leve primeiro e depois o 1080p, e exportar `.txt`. Comparar médias e p50 entre as três opções, observando aquecimento e variação de clock.

O FPS do Winlator exige GPU → NPU → GPU e sincronização; atingir 16 ms **somente na inferência** não garante 60 FPS finais. A API NNAPI foi depreciada no Android 15, mas o aparelho Android 14 com driver enn continua servindo como base experimental. Futuramente avaliar Exynos AI LiteCore / Exynos AI Studio e arquitetura mais adequada ao hardware Samsung.


## v0.5 resultados reais no S24 FE — AVISO DE CORREÇÃO

Teste 960×540→1920×1080 com FP16 e execução reutilizável:

| Arquitetura | Média de inferência | Resultado de amostragem |
|---|---:|---|
| BASELINE_12_K3 | 21,549 ms | Amostras finitas 2049/2049 |
| LITE_12_K1 | 16,399 ms | Amostras finitas 2049/2049, p90 18,291 ms |
| LITE_8_K1 | 13,653 ms | **Amostras finitas 1534/2049**, valores chegando a 10.000 e negativos extremos — resultado inválido |

**A variante LITE_8_K1 NÃO está aprovada**, pois o benchmark terminou mas a saída apresentou indícios de corrupção/incompatibilidade numérica. Estar abaixo de 16 ms não significa que produz imagem válida. A variante LITE_12_K1 é candidata a treinamento após comprovação de integridade completa e qualidade visual.

### v0.6 — verificação integral do tensor de saída

Depois de rodar cada variante, a v0.6 percorre **todos os valores** do tensor RGB, em vez de amostrar aproximadamente 2.048 componentes, e registra:

- quantidade de valores finitos e não finitos;
- sentinelas NaN FP16 ainda presentes (detectam possíveis componentes não escritos no último quadro);
- quantidade de valores com módulo acima de 2 (anomalia indicativa para entradas normalizadas, não uma prova universal de erro);
- coordenada e canal da primeira saída suspeita;
- `VALIDACAO_NUMERICA=SEM_ANOMALIAS_BASICAS` ou `FALHA`.

A saída é preenchida com sentinelas antes da **última inferência**, fora do cronômetro. A verificação integral ocorre **após** a medição, para não contaminar diretamente a latência de NNAPI compute. Ainda assim, o preenchimento tem custo de memória/energia e pode afetar indiretamente as frequências; repetir várias rodadas sob temperaturas semelhantes é recomendável.

**Validação numérica básica não mede fidelidade visual.** Para aprovar uma rede neural de verdade faltam comparar contra saída CPU e imagens de referência, medir PSNR/SSIM e treinar os pesos da nova topologia. Não há afirmação de que a NPU usa todos os seus motores internos.

### Como testar

Instale a v0.6 quando o Actions compilar com sucesso, execute os botões Lite 5 e 6, exporte o TXT e compare os novos contadores de integridade. Repetir o 1080p em ambiente termicamente estável ajuda a comparar a latência.
