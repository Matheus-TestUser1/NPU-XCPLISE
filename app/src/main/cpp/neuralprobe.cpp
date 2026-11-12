#include <jni.h>
#include <android/NeuralNetworks.h>
#include <algorithm>
#include <cctype>
#include <initializer_list>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <iomanip>
#include <sstream>
#include <string>
#include <vector>

namespace {

std::string errorCode(int status) {
    std::ostringstream ss;
    ss << status;
    if (status == ANEURALNETWORKS_NO_ERROR) ss << " (NO_ERROR)";
    else if (status == ANEURALNETWORKS_BAD_DATA) ss << " (BAD_DATA)";
    else if (status == ANEURALNETWORKS_OP_FAILED) ss << " (OP_FAILED)";
    else if (status == ANEURALNETWORKS_UNAVAILABLE_DEVICE) ss << " (UNAVAILABLE_DEVICE)";
    return ss.str();
}

std::string getName(const ANeuralNetworksDevice* d) {
    const char* name = nullptr;
    if (ANeuralNetworksDevice_getName(d, &name) != ANEURALNETWORKS_NO_ERROR || !name) {
        return "<nome indisponivel>";
    }
    return name;
}

std::string lower(std::string v) {
    std::transform(v.begin(), v.end(), v.begin(), [](unsigned char c) { return static_cast<char>(std::tolower(c)); });
    return v;
}

bool likelySamsungNpu(const ANeuralNetworksDevice* d) {
    int32_t type = ANEURALNETWORKS_DEVICE_TYPE_UNKNOWN;
    if (ANeuralNetworksDevice_getType(d, &type) != ANEURALNETWORKS_NO_ERROR) return false;
    if (type == ANEURALNETWORKS_DEVICE_TYPE_CPU) return false;
    const std::string name = lower(getName(d));
    return name.find("enn") != std::string::npos ||
           name.find("samsung") != std::string::npos ||
           name.find("exynos") != std::string::npos ||
           name.find("npu") != std::string::npos;
}

std::string describeDevices() {
    uint32_t count = 0;
    std::ostringstream out;
    int rc = ANeuralNetworks_getDeviceCount(&count);
    if (rc != ANEURALNETWORKS_NO_ERROR) {
        out << "Falha ao consultar os dispositivos NNAPI: " << errorCode(rc);
        return out.str();
    }
    out << "=== DISPOSITIVOS NNAPI ===\nTotal: " << count << "\n";
    for (uint32_t i = 0; i < count; ++i) {
        ANeuralNetworksDevice* dev = nullptr;
        rc = ANeuralNetworks_getDevice(i, &dev);
        if (rc != ANEURALNETWORKS_NO_ERROR || !dev) {
            out << "[" << i << "] erro de descoberta: " << errorCode(rc) << "\n";
            continue;
        }
        int32_t type = ANEURALNETWORKS_DEVICE_TYPE_UNKNOWN;
        ANeuralNetworksDevice_getType(dev, &type);
        const char* version = nullptr;
        ANeuralNetworksDevice_getVersion(dev, &version);
        out << "[" << i << "] " << getName(dev)
            << " | tipo=" << type
            << " | versao=" << (version ? version : "?")
            << " | candidato_NPU=" << (likelySamsungNpu(dev) ? "SIM" : "nao")
            << "\n";
    }
    out << "Obs: candidato_NPU e uma heuristica por nome, nao uma confirmacao de inferencia.\n";
    return out.str();
}

constexpr uint32_t kW = 8;
constexpr uint32_t kH = 8;
constexpr uint32_t kC = 3;
constexpr uint32_t kConvOut = 12;

struct ModelHolder {
    ANeuralNetworksModel* model = nullptr;
    bool twoX = false;
    int numOps = 1;
    ~ModelHolder() { if (model) ANeuralNetworksModel_free(model); }
};

int addTensor(ANeuralNetworksModel* model, std::initializer_list<uint32_t> dims) {
    std::vector<uint32_t> d(dims);
    const ANeuralNetworksOperandType t = {
        ANEURALNETWORKS_TENSOR_FLOAT32,
        static_cast<uint32_t>(d.size()), d.data(), 0.f, 0
    };
    return ANeuralNetworksModel_addOperand(model, &t);
}

int addScalar(ANeuralNetworksModel* model) {
    ANeuralNetworksOperandType t = { ANEURALNETWORKS_INT32, 0, nullptr, 0.f, 0 };
    int rc = ANeuralNetworksModel_addOperand(model, &t);
    if (rc != ANEURALNETWORKS_NO_ERROR) return rc;
    return ANEURALNETWORKS_NO_ERROR;
}

int makeModel(ModelHolder& target, bool twoX) {
    ANeuralNetworksModel* m = nullptr;
    int rc = ANeuralNetworksModel_create(&m);
    if (rc != ANEURALNETWORKS_NO_ERROR) return rc;
    target.model = m;
    target.twoX = twoX;
    target.numOps = twoX ? 2 : 1;

    rc = addTensor(m, {1, kH, kW, kC}); if (rc) return rc;
    rc = addTensor(m, {kConvOut, 1, 1, kC}); if (rc) return rc;
    rc = addTensor(m, {kConvOut}); if (rc) return rc;
    rc = addScalar(m); if (rc) return rc;
    rc = addScalar(m); if (rc) return rc;
    rc = addScalar(m); if (rc) return rc;
    rc = addScalar(m); if (rc) return rc;
    rc = addTensor(m, {1, kH, kW, kConvOut}); if (rc) return rc;
    if (twoX) {
        rc = addScalar(m); if (rc) return rc;
        rc = addTensor(m, {1, 2*kH, 2*kW, kC}); if (rc) return rc;
    }

    float weights[kConvOut * kC] = {};
    float bias[kConvOut] = {};
    for (uint32_t oc = 0; oc < kConvOut; ++oc) {
        weights[oc * kC + (oc % kC)] = 1.f;
    }
    rc = ANeuralNetworksModel_setOperandValue(m, 1, weights, sizeof(weights)); if (rc) return rc;
    rc = ANeuralNetworksModel_setOperandValue(m, 2, bias, sizeof(bias)); if (rc) return rc;
    int32_t padding = 1, stride = 1, fuse = 0, block = 2;
    rc = ANeuralNetworksModel_setOperandValue(m, 3, &padding, sizeof(padding)); if (rc) return rc;
    rc = ANeuralNetworksModel_setOperandValue(m, 4, &stride, sizeof(stride)); if (rc) return rc;
    rc = ANeuralNetworksModel_setOperandValue(m, 5, &stride, sizeof(stride)); if (rc) return rc;
    rc = ANeuralNetworksModel_setOperandValue(m, 6, &fuse, sizeof(fuse)); if (rc) return rc;
    if (twoX) {
        rc = ANeuralNetworksModel_setOperandValue(m, 8, &block, sizeof(block)); if (rc) return rc;
    }
    const uint32_t convInputs[] = {0, 1, 2, 3, 4, 5, 6};
    const uint32_t convOutputs[] = {7};
    rc = ANeuralNetworksModel_addOperation(m, ANEURALNETWORKS_CONV_2D, 7, convInputs, 1, convOutputs);
    if (rc) return rc;
    if (twoX) {
        const uint32_t dInputs[] = {7, 8};
        const uint32_t dOutputs[] = {9};
        rc = ANeuralNetworksModel_addOperation(m, ANEURALNETWORKS_DEPTH_TO_SPACE, 2, dInputs, 1, dOutputs);
        if (rc) return rc;
    }
    const uint32_t inputIds[] = {0};
    const uint32_t outputIds[] = {twoX ? 9u : 7u};
    rc = ANeuralNetworksModel_identifyInputsAndOutputs(m, 1, inputIds, 1, outputIds);
    if (rc) return rc;
    return ANeuralNetworksModel_finish(m);
}

std::string testOne(const ANeuralNetworksDevice* device, bool twoX) {
    std::ostringstream out;
    out << "\n=== " << (twoX ? "CONV_2D + DEPTH_TO_SPACE (2x)" : "CONV_2D isolado") << " ===\n";
    ModelHolder holder;
    int rc = makeModel(holder, twoX);
    if (rc != ANEURALNETWORKS_NO_ERROR) {
        out << "Criacao do grafo falhou: " << errorCode(rc) << "\n";
        return out.str();
    }

    const ANeuralNetworksDevice* devices[1] = { device };
    bool supported[2] = {false, false};
    rc = ANeuralNetworksModel_getSupportedOperationsForDevices(holder.model, devices, 1, supported);
    if (rc != ANEURALNETWORKS_NO_ERROR) {
        out << "Consulta de suporte falhou: " << errorCode(rc) << "\n";
        return out.str();
    }
    out << "CONV_2D suportado=" << (supported[0] ? "SIM" : "NAO") << "\n";
    if (twoX) out << "DEPTH_TO_SPACE suportado=" << (supported[1] ? "SIM" : "NAO") << "\n";
    if (!supported[0] || (twoX && !supported[1])) {
        out << "SKIP: o acelerador nao suporta todas as operacoes. CPU fallback DESATIVADO.\n";
        return out.str();
    }

    ANeuralNetworksCompilation* compilation = nullptr;
    rc = ANeuralNetworksCompilation_createForDevices(holder.model, devices, 1, &compilation);
    if (rc != ANEURALNETWORKS_NO_ERROR || compilation == nullptr) {
        out << "Compilacao especifica para NPU falhou: " << errorCode(rc) << "\n";
        return out.str();
    }
    ANeuralNetworksCompilation_setPreference(compilation, ANEURALNETWORKS_PREFER_FAST_SINGLE_ANSWER);
    const auto compileStart = std::chrono::steady_clock::now();
    rc = ANeuralNetworksCompilation_finish(compilation);
    const auto compileEnd = std::chrono::steady_clock::now();
    if (rc != ANEURALNETWORKS_NO_ERROR) {
        out << "Finalizacao da compilacao falhou: " << errorCode(rc) << "\n";
        ANeuralNetworksCompilation_free(compilation);
        return out.str();
    }
    const double compileMs = std::chrono::duration<double, std::milli>(compileEnd - compileStart).count();
    out << "Compilacao (ms): " << compileMs << "\n";

    std::vector<float> input(kW*kH*kC);
    for (size_t i = 0; i < input.size(); ++i) input[i] = static_cast<float>(i % 23) / 23.f;
    std::vector<float> output(kW*kH*kConvOut);
    const int runs = 30;
    const int warmups = 5;
    std::vector<double> durations;
    int successful = 0;
    for (int run = 0; run < runs + warmups; ++run) {
        ANeuralNetworksExecution* exec = nullptr;
        rc = ANeuralNetworksExecution_create(compilation, &exec);
        if (rc != ANEURALNETWORKS_NO_ERROR || !exec) {
            out << "Execution_create falhou: " << errorCode(rc) << "\n";
            break;
        }
        rc = ANeuralNetworksExecution_setInput(exec, 0, nullptr, input.data(), input.size()*sizeof(float));
        if (rc == ANEURALNETWORKS_NO_ERROR) {
            rc = ANeuralNetworksExecution_setOutput(exec, 0, nullptr, output.data(), output.size()*sizeof(float));
        }
        const auto t0 = std::chrono::steady_clock::now();
        if (rc == ANEURALNETWORKS_NO_ERROR) rc = ANeuralNetworksExecution_compute(exec);
        const auto t1 = std::chrono::steady_clock::now();
        ANeuralNetworksExecution_free(exec);
        if (rc != ANEURALNETWORKS_NO_ERROR) {
            out << "Inferencia falhou no ciclo " << run << ": " << errorCode(rc) << "\n";
            break;
        }
        if (run >= warmups) {
            durations.push_back(std::chrono::duration<double, std::milli>(t1 - t0).count());
            ++successful;
        }
    }
    ANeuralNetworksCompilation_free(compilation);
    if (successful == 0) {
        out << "Nenhuma inferencia concluida.\n";
        return out.str();
    }
    std::sort(durations.begin(), durations.end());
    double sum = 0;
    for (double ms : durations) sum += ms;
    const double mean = sum / durations.size();
    out << std::fixed << std::setprecision(3);
    out << "Sucesso: " << successful << " execucoes (5 warmups)\n";
    out << "Latencia media (ms): " << mean << "\n";
    out << "p50 (ms): " << durations[durations.size()/2] << "\n";
    out << "min/max (ms): " << durations.front() << " / " << durations.back() << "\n";
    out << "Vazao ilustrativa (1/media): " << (1000.0/mean) << " consultas/s\n";
    if (twoX) {
        float maxError = 0;
        for (uint32_t y = 0; y < kH*2; ++y) {
            for (uint32_t x = 0; x < kW*2; ++x) {
                for (uint32_t c = 0; c < kC; ++c) {
                    const float original = input[((y/2)*kW + (x/2))*kC + c];
                    const float value = output[(y*(kW*2)+x)*kC+c];
                    maxError = std::max(maxError, std::abs(original-value));
                }
            }
        }
        out << "Saida 2x: " << kW*2 << "x" << kH*2 << " RGB; erro max=" << maxError << "\n";
    }
    out << "ATENCAO: entrada 8x8. Isto testa o caminho NPU; NAO mede upscaling 540p->1080p.\n";
    return out.str();
}

std::string runAllTests() {
    std::ostringstream out;
    out << "EXYNOS NEURAL PROBE v0.1 — SEM FALLBACK CPU\n";
    out << "Sem root; nao modifica clocks, kernel ou arquivos de sistema.\n";
    out << describeDevices() << "\n";
    uint32_t count = 0;
    int rc = ANeuralNetworks_getDeviceCount(&count);
    if (rc != ANEURALNETWORKS_NO_ERROR) return out.str();
    ANeuralNetworksDevice* selected = nullptr;
    for (uint32_t i = 0; i < count; ++i) {
        ANeuralNetworksDevice* device = nullptr;
        if (ANeuralNetworks_getDevice(i, &device) == ANEURALNETWORKS_NO_ERROR &&
            device && likelySamsungNpu(device)) {
            selected = device;
            break;
        }
    }
    if (!selected) {
        out << "NPU Samsung nao encontrada pelo NNAPI. Verificar nome real do driver acima.\n";
        return out.str();
    }
    out << "Acelerador selecionado: " << getName(selected) << "\n";
    out << "Compilacao EXCLUSIVA para esse dispositivo.\n";
    out << testOne(selected, false);
    out << testOne(selected, true);
    out << "\nNota: grafo com pesos fixos equivale a ampliacao simples, NAO a rede treinada ESRGAN/EDSR.\n";
    return out.str();
}

} // namespace

extern "C" JNIEXPORT jstring JNICALL
Java_com_exynos_neuralprobe_NativeProbe_listDevices(JNIEnv* env, jclass) {
    return env->NewStringUTF(describeDevices().c_str());
}

extern "C" JNIEXPORT jstring JNICALL
Java_com_exynos_neuralprobe_NativeProbe_runNNAPITest(JNIEnv* env, jclass) {
    return env->NewStringUTF(runAllTests().c_str());
}
