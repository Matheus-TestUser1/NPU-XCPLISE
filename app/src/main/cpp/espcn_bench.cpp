#include <jni.h>
#include <android/asset_manager.h>
#include <android/asset_manager_jni.h>
#include <android/NeuralNetworks.h>
#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <iomanip>
#include <initializer_list>
#include <limits>
#include <numeric>
#include <sstream>
#include <string>
#include <utility>
#include <vector>

namespace {

constexpr uint32_t kFeatures = 12;
constexpr uint32_t kScale = 2;

static std::string statusName(int code) {
    return "NNAPI status=" + std::to_string(code);
}

static std::vector<uint8_t> decodeBase64(const std::string& source) {
    auto digit=[](char c)->int {
        if(c>='A'&&c<='Z')return c-'A';
        if(c>='a'&&c<='z')return c-'a'+26;
        if(c>='0'&&c<='9')return c-'0'+52;
        if(c=='+')return 62;
        if(c=='/')return 63;
        return -1;
    };
    std::vector<uint8_t> bytes;
    uint32_t current=0;
    int bits=0;
    for(char c:source) {
        if(c=='=')break;
        int value=digit(c);
        if(value<0)continue;
        current=(current<<6)|static_cast<uint32_t>(value);
        bits+=6;
        if(bits>=8){
            bits-=8;
            bytes.push_back(static_cast<uint8_t>((current>>bits)&255));
        }
    }
    return bytes;
}

static uint32_t u32le(const uint8_t* p) {
    return uint32_t(p[0])|(uint32_t(p[1])<<8)|(uint32_t(p[2])<<16)|(uint32_t(p[3])<<24);
}

static float fp16toFloat(uint16_t bits) {
    const float sign=(bits&0x8000)?-1.0f:1.0f;
    const int exponent=(bits>>10)&31;
    const int mantissa=bits&1023;
    if(exponent==0)return sign*std::ldexp(float(mantissa),-24);
    if(exponent==31)return mantissa?NAN:sign*INFINITY;
    return sign*std::ldexp(1.0f+float(mantissa)/1024.0f,exponent-15);
}

static uint16_t floatToHalf(float value) {
    const __fp16 h = static_cast<__fp16>(value);
    uint16_t bits=0;
    std::memcpy(&bits,&h,sizeof(bits));
    return bits;
}

struct Weights {
    std::vector<float> w1,b1,w2,b2;
};

// PERFORMANCE-ONLY ablation of trained ESPCN weights: not retrained!
static Weights makePrunedWeights(const Weights& source,uint32_t features,uint32_t kernel2) {
    Weights dst;
    if(features<1 || features>kFeatures || (kernel2!=1 && kernel2!=3))return dst;
    dst.w1.assign(source.w1.begin(),source.w1.begin()+features*27);
    dst.b1.assign(source.b1.begin(),source.b1.begin()+features);
    dst.b2=source.b2;
    for(uint32_t oc=0;oc<12;++oc) {
        const uint32_t offset=kernel2==1?1:0;
        for(uint32_t y=0;y<kernel2;++y)
            for(uint32_t x=0;x<kernel2;++x)
                for(uint32_t ic=0;ic<features;++ic)
                    dst.w2.push_back(source.w2[((oc*3+(y+offset))*3+(x+offset))*kFeatures+ic]);
    }
    return dst;
}

static bool loadWeights(JNIEnv* env,jobject javaAssets,Weights& result,std::string& error) {
    if(javaAssets==nullptr){error="AssetManager ausente";return false;}
    AAssetManager* manager=AAssetManager_fromJava(env,javaAssets);
    if(!manager){error="Falha ao obter AssetManager";return false;}
    AAsset* asset=AAssetManager_open(manager,"espcn2x_fp16.b64",AASSET_MODE_BUFFER);
    if(!asset){error="Modelo espcn2x_fp16.b64 ausente";return false;}
    const off_t size=AAsset_getLength(asset);
    if(size<=0||size>128*1024){
        AAsset_close(asset);error="Tamanho de modelo invalido";return false;
    }
    std::string encoded(static_cast<size_t>(size),'\0');
    off_t received=0;
    while(received<size) {
        int n=AAsset_read(asset,&encoded[static_cast<size_t>(received)],static_cast<size_t>(size-received));
        if(n<=0)break;
        received+=n;
    }
    AAsset_close(asset);
    if(received!=size){error="Erro ao ler modelo";return false;}
    auto bytes=decodeBase64(encoded);
    if(bytes.size()<28||std::memcmp(bytes.data(),"ESPCN2X1",8)!=0){
        error="Assinatura ou versao do modelo incorreta";return false;
    }
    const uint32_t features=u32le(bytes.data()+8);
    const std::array<uint32_t,4> counts={
        u32le(bytes.data()+12),u32le(bytes.data()+16),
        u32le(bytes.data()+20),u32le(bytes.data()+24)
    };
    const std::array<uint32_t,4> expected={
        kFeatures*3*3*3,kFeatures,3*kScale*kScale*3*3*kFeatures,3*kScale*kScale
    };
    if(features!=kFeatures||counts!=expected){
        error="Dimensoes dos pesos inesperadas";return false;
    }
    size_t total=std::accumulate(counts.begin(),counts.end(),size_t(0));
    if(bytes.size()!=28+2*total){error="Comprimento dos pesos inconsistente";return false;}
    size_t offset=28;
    std::array<std::vector<float>*,4> arrays={&result.w1,&result.b1,&result.w2,&result.b2};
    for(size_t a=0;a<4;++a){
        auto& v=*arrays[a];v.reserve(counts[a]);
        for(uint32_t i=0;i<counts[a];++i){
            uint16_t half=uint16_t(bytes[offset]|(bytes[offset+1]<<8));
            offset+=2;
            float value=fp16toFloat(half);
            if(!std::isfinite(value)){error="Peso nao-finito";return false;}
            v.push_back(value);
        }
    }
    return true;
}

struct Model {
    ANeuralNetworksModel* model=nullptr;
    ANeuralNetworksCompilation* compilation=nullptr;
    ~Model(){
        if(compilation)ANeuralNetworksCompilation_free(compilation);
        if(model)ANeuralNetworksModel_free(model);
    }
};

static int tensor(ANeuralNetworksModel* model,std::initializer_list<uint32_t> dims,bool fp16) {
    std::vector<uint32_t> shape(dims);
    ANeuralNetworksOperandType type={
        fp16 ? ANEURALNETWORKS_TENSOR_FLOAT16 : ANEURALNETWORKS_TENSOR_FLOAT32,
        uint32_t(shape.size()),shape.data(),0.0f,0
    };
    return ANeuralNetworksModel_addOperand(model,&type);
}

static int scalar(ANeuralNetworksModel* model,int32_t value,int index) {
    ANeuralNetworksOperandType type={ANEURALNETWORKS_INT32,0,nullptr,0.0f,0};
    int rc=ANeuralNetworksModel_addOperand(model,&type);
    if(rc)return rc;
    return ANeuralNetworksModel_setOperandValue(model,index,&value,sizeof(value));
}

static bool buildModel(Model& holder,const Weights& weights,const ANeuralNetworksDevice* device,
                       uint32_t width,uint32_t height,bool fp16,uint32_t features,
                       uint32_t kernel2,std::ostringstream& out) {
    int rc=ANeuralNetworksModel_create(&holder.model);
    if(rc){out<<"create: "<<statusName(rc)<<"\n";return false;}
    auto* m=holder.model;
    const std::array<int,17> errors={
        tensor(m,{1,height,width,3},fp16),                   // 0 input NHWC
        tensor(m,{features,3,3,3},fp16),                   // 1 conv1 weights OHWI
        tensor(m,{features},fp16),                         // 2 conv1 bias
        scalar(m,1,3),scalar(m,1,4),scalar(m,1,5),scalar(m,1,6), // SAME/stride/RELU
        tensor(m,{1,height,width,features},fp16),         // 7 conv1 output
        tensor(m,{12,kernel2,kernel2,features},fp16),                 // 8 conv2 weights
        tensor(m,{12},fp16),                               // 9 conv2 bias
        scalar(m,1,10),scalar(m,1,11),scalar(m,1,12),scalar(m,0,13), // SAME/stride/NONE
        tensor(m,{1,height,width,12},fp16),                // 14 conv2 output
        scalar(m,2,15),                               // 15 block size
        tensor(m,{1,height*2,width*2,3},fp16)              // 16 RGB upscaled
    };
    for(int e:errors)if(e){out<<"operand: "<<statusName(e)<<"\n";return false;}
    const std::array<std::pair<int,const std::vector<float>*>,4> values={{
        {1,&weights.w1},{2,&weights.b1},{8,&weights.w2},{9,&weights.b2}
    }};
    for(const auto& v:values){
        if(fp16) {
            std::vector<uint16_t> halfWeights;
            halfWeights.reserve(v.second->size());
            for(float value:*v.second)halfWeights.push_back(floatToHalf(value));
            rc=ANeuralNetworksModel_setOperandValue(m,v.first,halfWeights.data(),
                                                   halfWeights.size()*sizeof(uint16_t));
        } else {
            rc=ANeuralNetworksModel_setOperandValue(m,v.first,v.second->data(),
                                                   v.second->size()*sizeof(float));
        }
        if(rc){out<<"pesos: "<<statusName(rc)<<"\n";return false;}
    }
    const uint32_t c1[]={0,1,2,3,4,5,6}, c2[]={7,8,9,10,11,12,13}, d2s[]={14,15};
    const uint32_t o1[]={7},o2[]={14},o3[]={16},input[]={0};
    rc=ANeuralNetworksModel_addOperation(m,ANEURALNETWORKS_CONV_2D,7,c1,1,o1);
    if(!rc)rc=ANeuralNetworksModel_addOperation(m,ANEURALNETWORKS_CONV_2D,7,c2,1,o2);
    if(!rc)rc=ANeuralNetworksModel_addOperation(m,ANEURALNETWORKS_DEPTH_TO_SPACE,2,d2s,1,o3);
    if(!rc)rc=ANeuralNetworksModel_identifyInputsAndOutputs(m,1,input,1,o3);
    if(!rc)rc=ANeuralNetworksModel_finish(m);
    if(rc){out<<"model/ops: "<<statusName(rc)<<"\n";return false;}
    bool supported[3]={false,false,false};
    rc=ANeuralNetworksModel_getSupportedOperationsForDevices(m,&device,1,supported);
    if(rc){out<<"getSupportedOperations: "<<statusName(rc)<<"\n";return false;}
    out<<"Precision="<<(fp16?"FP16":"FP32")<<" | Operacoes suportadas: conv1="<<supported[0]<<" conv2="<<supported[1]
       <<" depth_to_space="<<supported[2]<<"\n";
    if(!supported[0]||!supported[1]||!supported[2]){
        out<<"SKIP: ENN nao suporta este grafo. Sem fallback CPU.\n";return false;
    }
    rc=ANeuralNetworksCompilation_createForDevices(m,&device,1,&holder.compilation);
    if(rc){out<<"createForDevices: "<<statusName(rc)<<"\n";return false;}
    ANeuralNetworksCompilation_setPreference(holder.compilation,ANEURALNETWORKS_PREFER_FAST_SINGLE_ANSWER);
    auto t0=std::chrono::steady_clock::now();
    rc=ANeuralNetworksCompilation_finish(holder.compilation);
    auto t1=std::chrono::steady_clock::now();
    out<<"Compilacao ms="<<std::chrono::duration<double,std::milli>(t1-t0).count()<<"\n";
    if(rc){out<<"compilation_finish: "<<statusName(rc)<<"\n";return false;}
    return true;
}

static const ANeuralNetworksDevice* findEnn(std::ostringstream& out) {
    uint32_t count=0;
    int rc=ANeuralNetworks_getDeviceCount(&count);
    if(rc){out<<"getDeviceCount: "<<statusName(rc)<<"\n";return nullptr;}
    for(uint32_t i=0;i<count;++i) {
        ANeuralNetworksDevice* device=nullptr;
        if(ANeuralNetworks_getDevice(i,&device)!=ANEURALNETWORKS_NO_ERROR||!device)continue;
        const char* name=nullptr;
        int32_t type=ANEURALNETWORKS_DEVICE_UNKNOWN;
        ANeuralNetworksDevice_getName(device,&name);
        ANeuralNetworksDevice_getType(device,&type);
        if(name&&std::string(name)=="enn"&&type!=ANEURALNETWORKS_DEVICE_CPU){
            out<<"Selecionado NNAPI: "<<name<<" tipo="<<type<<"\n";return device;
        }
    }
    out<<"Dispositivo enn ausente. Nao executando na CPU.\n";
    return nullptr;
}

static void benchmarkOne(const ANeuralNetworksDevice* device,const Weights& weights,
                         uint32_t width,uint32_t height,int measured,bool fp16,bool reusable,std::ostringstream& out,
                         uint32_t features=kFeatures,uint32_t kernel2=3,
                         const char* tag="ORIGINAL") {
    out<<"\n=== ESPCN "<<tag<<" "<<(fp16?"FP16 ":"FP32 ")<<(reusable?"REUSABLE ":"FRESH ")<<width<<"x"<<height<<" -> "<<width*2<<"x"<<height*2<<" ===\n";
    Model m;
    if(!buildModel(m,weights,device,width,height,fp16,features,kernel2,out))return;
    const size_t inCount=size_t(width)*height*3;
    const size_t outCount=inCount*4;
    std::vector<float> input32,output32;
    std::vector<uint16_t> input16,output16;
    if(fp16) {
        input16.resize(inCount);
        output16.resize(outCount);
    } else {
        input32.resize(inCount);
        output32.resize(outCount);
    }
    for(uint32_t y=0;y<height;++y)for(uint32_t x=0;x<width;++x){
        size_t i=(size_t(y)*width+x)*3;
        const float v0=float(x)/std::max(1u,width-1);
        const float v1=float(y)/std::max(1u,height-1);
        const float v2=(((x/8)^(y/8))&1)?0.35f:0.75f;
        if(fp16) {
            input16[i]=floatToHalf(v0);
            input16[i+1]=floatToHalf(v1);
            input16[i+2]=floatToHalf(v2);
        } else {
            input32[i]=v0;
            input32[i+1]=v1;
            input32[i+2]=v2;
        }
    }
    const size_t elementBytes=fp16?sizeof(uint16_t):sizeof(float);
    const void* inputPtr=fp16?static_cast<const void*>(input16.data()):
                               static_cast<const void*>(input32.data());
    void* outputPtr=fp16?static_cast<void*>(output16.data()):
                           static_cast<void*>(output32.data());
    std::vector<double> times;
    const int warmups=2;
    ANeuralNetworksExecution* shared=nullptr;
    if(reusable) {
        int rc=ANeuralNetworksExecution_create(m.compilation,&shared);
        if(rc||!shared){out<<"execution_create reusable: "<<statusName(rc)<<"\n";return;}
        rc=ANeuralNetworksExecution_setInput(shared,0,nullptr,inputPtr,inCount*elementBytes);
        if(!rc)rc=ANeuralNetworksExecution_setOutput(shared,0,nullptr,outputPtr,outCount*elementBytes);
        if(!rc)rc=ANeuralNetworksExecution_setReusable(shared,true);
        if(rc){
            out<<"reusable nao suportado/preparacao falhou: "<<statusName(rc)<<"\n";
            ANeuralNetworksExecution_free(shared);
            return;
        }
    }
    for(int n=0;n<warmups+measured;++n) {
        ANeuralNetworksExecution* execution=shared;
        int rc=ANEURALNETWORKS_NO_ERROR;
        if(!reusable) {
            rc=ANeuralNetworksExecution_create(m.compilation,&execution);
            if(rc||!execution){out<<"execution_create: "<<statusName(rc)<<"\n";break;}
            rc=ANeuralNetworksExecution_setInput(execution,0,nullptr,inputPtr,inCount*elementBytes);
            if(!rc)rc=ANeuralNetworksExecution_setOutput(execution,0,nullptr,outputPtr,outCount*elementBytes);
        }
        // Diagnostic only: poison output before the LAST execution, outside timing.
        // An untouched NaN sentinel suggests that the driver did not populate every pixel.
        if(n==warmups+measured-1 && rc==ANEURALNETWORKS_NO_ERROR) {
            if(fp16)std::fill(output16.begin(),output16.end(),static_cast<uint16_t>(0x7e00));
            else std::fill(output32.begin(),output32.end(),std::numeric_limits<float>::quiet_NaN());
        }
        auto start=std::chrono::steady_clock::now();
        if(!rc)rc=ANeuralNetworksExecution_compute(execution);
        auto end=std::chrono::steady_clock::now();
        if(!reusable)ANeuralNetworksExecution_free(execution);
        if(rc){out<<"Inferencia ciclo "<<n<<": "<<statusName(rc)<<"\n";break;}
        if(n>=warmups)times.push_back(std::chrono::duration<double,std::milli>(end-start).count());
    }
    if(shared)ANeuralNetworksExecution_free(shared);
    if(times.empty()){out<<"Nenhuma inferencia medida.\n";return;}
    std::sort(times.begin(),times.end());
    const double mean=std::accumulate(times.begin(),times.end(),0.0)/times.size();
    // Validate EVERY component of output (outside timed inference).
    size_t finite=0, nonfinite=0, untouched=0, outliers=0;
    float minimum=INFINITY, maximum=-INFINITY;
    size_t firstBad=outCount;
    for(size_t i=0;i<outCount;++i){
        if(fp16 && output16[i]==0x7e00u)++untouched;
        const float v=fp16?fp16toFloat(output16[i]):output32[i];
        if(!std::isfinite(v)) {
            ++nonfinite;
            if(firstBad==outCount)firstBad=i;
            continue;
        }
        ++finite;
        minimum=std::min(minimum,v);
        maximum=std::max(maximum,v);
        // Diagnostic threshold for unit-range RGB input, not a general NNAPI constraint.
        if(std::abs(v)>2.0f) {
            ++outliers;
            if(firstBad==outCount)firstBad=i;
        }
    }
    out<<std::fixed<<std::setprecision(3);
    out<<"Inferencias medidas="<<times.size()<<" (warmups=2)\n";
    out<<"Compute medio="<<mean<<" ms | p50="<<times[times.size()/2]
       <<" ms | p90="<<times[(times.size()-1)*9/10]<<" ms\n";
    out<<"Min/Max="<<times.front()<<" / "<<times.back()<<" ms\n";
    out<<"Validacao COMPLETA RGB 2x: componentes="<<outCount
       <<" finitos="<<finite<<" nao_finitos="<<nonfinite
       <<" sentinelas_NaN="<<untouched<<" abs_maior_2="<<outliers<<"\n";
    if(finite>0)out<<"Intervalo de saida FINITA="<<minimum<<" a "<<maximum<<"\n";
    if(firstBad!=outCount) {
        const size_t index=firstBad/3;
        const size_t dstWidth=static_cast<size_t>(width)*2;
        out<<"Primeiro elemento suspeito: x="<<(index%dstWidth)
           <<" y="<<(index/dstWidth)<<" canal="<<(firstBad%3)<<"\n";
    }
    out<<"VALIDACAO_NUMERICA="<<(nonfinite==0&&outliers==0?"SEM_ANOMALIAS_BASICAS":"FALHA")
       <<" (nao avalia fidelidade/PSNR).\n";
    out<<"Entrada bytes="<<inCount*elementBytes<<" saida bytes="<<outCount*elementBytes<<"\n";
    out<<"Nao inclui compilacao nem renderizacao Vulkan/Winlator. Inclui NNAPI compute sincrono.\n";
}

static std::string runLite(JNIEnv* env,jobject assets,bool full) {
    std::ostringstream out;
    out<<"EXYNOS ESPCN v0.5 — ARCHITECTURE ABLATION (NOT RETRAINED)\n";
    out<<"FP16 reusable + Samsung ENN; 2nd convolution 3x3 -> 1x1, channels 12/8.\n";
    out<<"Weights are center-cropped/pruned from v0.4, NOT optimized for visual quality.\n";
    out<<"This measures only inference performance with synthetic input.\n";
    const ANeuralNetworksDevice* device=findEnn(out);
    if(!device)return out.str();
    Weights source;
    std::string error;
    if(!loadWeights(env,assets,source,error)){
        out<<"Failed to load v0.4 source weights: "<<error<<"\n";return out.str();
    }
    const Weights lite12=makePrunedWeights(source,12,1);
    const Weights lite8=makePrunedWeights(source,8,1);
    out<<"Features count: baseline=12 kernel2=3, lite12=12 kernel2=1, lite8=8 kernel2=1\n";
    const auto bench=[&](uint32_t w,uint32_t h,int repeats) {
        benchmarkOne(device,source,w,h,repeats,true,true,out,12,3,"BASELINE_12_K3");
        // The benchmark only reads model weights; buildModel doesn't mutate them.
        benchmarkOne(device,lite12,w,h,repeats,true,true,out,12,1,"LITE_12_K1");
        benchmarkOne(device,lite8,w,h,repeats,true,true,out,8,1,"LITE_8_K1");
    };
    if(full)bench(960,540,7);
    else {
        bench(160,90,8);
        bench(320,180,8);
        bench(480,270,6);
    }
    out<<"\nNO QUALITY CLAIM: pruned model is an experimental throughput probe. Train and validate a new compatible model before using it to enhance images.\n";
    return out.str();
}

static std::string run(JNIEnv* env,jobject assets,bool full) {
    std::ostringstream out;
    out<<"EXYNOS ESPCN 2X v0.4 — FP32, FP16, FP16 REUSABLE (ENN)\n";
    out<<"2x Conv3x3 (3->12->12), ReLU, DepthToSpace. Pesos armazenados FP16.\n";
    out<<"Entradas sinteticas. Nao e FSR temporal nem jogo real.\n";
    out<<"Sem root, sem alteracao de clocks, sem fallback NNAPI CPU.\n";
    auto* device=findEnn(out);
    if(!device)return out.str();
    Weights weights;
    std::string error;
    if(!loadWeights(env,assets,weights,error)){
        out<<"Falha ao carregar pesos: "<<error<<"\n";return out.str();
    }
    out<<"Pesos: "<<(weights.w1.size()+weights.b1.size()+weights.w2.size()+weights.b2.size())
       <<" valores FP16 armazenados no asset.\n";
    if(full) {
        benchmarkOne(device,weights,960,540,5,false,false,out);
        benchmarkOne(device,weights,960,540,5,true,false,out);
        benchmarkOne(device,weights,960,540,5,true,true,out);
    } else {
        for(const auto& item:std::array<std::pair<uint32_t,uint32_t>,3>{{{160,90},{320,180},{480,270}}}) {
            const int measured=item.first==480?5:8;
            benchmarkOne(device,weights,item.first,item.second,measured,false,false,out);
            benchmarkOne(device,weights,item.first,item.second,measured,true,false,out);
            benchmarkOne(device,weights,item.first,item.second,measured,true,true,out);
        }
    }
    out<<"\nAVISO: latencia do modelo demonstrativo, nao desempenho final de jogos.\n";
    return out.str();
}

} // namespace

extern "C" JNIEXPORT jstring JNICALL
Java_com_exynos_neuralprobe_NativeProbe_runESPCNBenchmark(JNIEnv* env,jclass,
                                                           jobject assets,jboolean full) {
    const std::string result=run(env,assets,full==JNI_TRUE);
    return env->NewStringUTF(result.c_str());
}

extern "C" JNIEXPORT jstring JNICALL
Java_com_exynos_neuralprobe_NativeProbe_runLiteBenchmark(JNIEnv* env,jclass,
                                                          jobject assets,jboolean full) {
    const std::string result=runLite(env,assets,full==JNI_TRUE);
    return env->NewStringUTF(result.c_str());
}
