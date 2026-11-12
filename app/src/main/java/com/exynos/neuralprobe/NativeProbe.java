package com.exynos.neuralprobe;

public final class NativeProbe {
    static {
        System.loadLibrary("neuralprobe");
    }

    private NativeProbe() {}

    public static native String listDevices();
    public static native String runNNAPITest();
    public static native String runESPCNBenchmark(android.content.res.AssetManager assets, boolean full);
    public static native String runLiteBenchmark(android.content.res.AssetManager assets, boolean full);
}
