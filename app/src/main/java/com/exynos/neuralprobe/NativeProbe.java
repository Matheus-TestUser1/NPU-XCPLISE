package com.exynos.neuralprobe;

public final class NativeProbe {
    static {
        System.loadLibrary("neuralprobe");
    }

    private NativeProbe() {}

    public static native String listDevices();
    public static native String runNNAPITest();
}
