package com.exynos.neuralprobe;

import android.app.Activity;
import android.content.Intent;
import android.os.Bundle;
import android.graphics.Color;
import android.graphics.Typeface;
import android.view.View;
import android.widget.Button;
import android.widget.LinearLayout;
import android.widget.ScrollView;
import android.widget.TextView;
import android.widget.Toast;
import java.io.OutputStream;
import java.nio.charset.StandardCharsets;
import java.text.SimpleDateFormat;
import java.util.Date;
import java.util.Locale;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;
import java.util.concurrent.Callable;

public class MainActivity extends Activity {
    private static final int SAVE_TXT = 101;
    private final ExecutorService executor = Executors.newSingleThreadExecutor();
    private final StringBuilder report = new StringBuilder();
    private TextView logView;
    private Button listButton;
    private Button testButton;
    private Button saveButton;
    private Button espcnSmallButton;
    private Button espcnFullButton;

    @Override
    public void onCreate(Bundle state) {
        super.onCreate(state);
        LinearLayout root = new LinearLayout(this);
        root.setOrientation(LinearLayout.VERTICAL);
        root.setPadding(24, 24, 24, 16);
        root.setBackgroundColor(Color.rgb(245, 247, 250));

        TextView title = new TextView(this);
        title.setText("Exynos Neural Probe");
        title.setTextSize(24);
        title.setTypeface(null, Typeface.BOLD);
        title.setTextColor(Color.rgb(25, 52, 90));
        root.addView(title);

        TextView intro = new TextView(this);
        intro.setText("Prova de execução na NPU Samsung ENN pelo NNAPI. Sem root e sem modificar drivers ou clocks. O teste antigo usa pesos fixos. Os testes ESPCN v0.2 usam uma pequena rede treinada (experimental) e entradas sintéticas.");
        intro.setTextSize(14);
        intro.setPadding(0, 12, 0, 14);
        intro.setTextColor(Color.DKGRAY);
        root.addView(intro);

        listButton = makeButton("1. Identificar aceleradores", root);
        testButton = makeButton("2. Executar testes NPU / 2x", root);
        espcnSmallButton = makeButton("3. ESPCN 2x — 90p / 180p / 270p", root);
        espcnFullButton = makeButton("4. ESPCN 2x — 540p → 1080p (pesado)", root);
        saveButton = makeButton("5. Exportar relatório .TXT", root);
        listButton.setOnClickListener(v -> runTask(() -> NativeProbe.listDevices()));
        testButton.setOnClickListener(v -> runTask(() -> NativeProbe.runNNAPITest()));
        espcnSmallButton.setOnClickListener(v -> runTask(() -> NativeProbe.runESPCNBenchmark(getAssets(), false)));
        espcnFullButton.setOnClickListener(v -> runTask(() -> NativeProbe.runESPCNBenchmark(getAssets(), true)));
        saveButton.setOnClickListener(v -> saveReport());

        ScrollView scroll = new ScrollView(this);
        logView = new TextView(this);
        logView.setTextSize(12);
        logView.setTypeface(Typeface.MONOSPACE);
        logView.setTextColor(Color.BLACK);
        logView.setPadding(14, 14, 14, 14);
        logView.setTextIsSelectable(true);
        scroll.addView(logView);
        root.addView(scroll, new LinearLayout.LayoutParams(-1, 0, 1));
        setContentView(root);
        addLine("Exynos Neural Probe 0.2 — S24 FE / Android 14");
        addLine("Execute a identificação antes do teste.\n");
    }

    private Button makeButton(String label, LinearLayout root) {
        Button b = new Button(this);
        b.setText(label);
        b.setAllCaps(false);
        root.addView(b, new LinearLayout.LayoutParams(-1, -2));
        return b;
    }

    private void addLine(String message) {
        String time = new SimpleDateFormat("HH:mm:ss", Locale.US).format(new Date());
        report.append("[").append(time).append("] ").append(message).append("\n");
        logView.setText(report.toString());
    }

    private void runTask(Callable<String> work) {
        listButton.setEnabled(false);
        testButton.setEnabled(false);
        espcnSmallButton.setEnabled(false);
        espcnFullButton.setEnabled(false);
        addLine("Executando... ");
        executor.execute(() -> {
            String result;
            try {
                result = work.call();
            } catch (Throwable e) {
                result = "ERRO: " + e.toString();
            }
            final String output = result;
            runOnUiThread(() -> {
                addLine(output + "\n");
                listButton.setEnabled(true);
                testButton.setEnabled(true);
                espcnSmallButton.setEnabled(true);
                espcnFullButton.setEnabled(true);
            });
        });
    }

    private void saveReport() {
        Intent intent = new Intent(Intent.ACTION_CREATE_DOCUMENT);
        intent.addCategory(Intent.CATEGORY_OPENABLE);
        intent.setType("text/plain");
        intent.putExtra(Intent.EXTRA_TITLE, "exynos_neural_probe_" +
            new SimpleDateFormat("yyyyMMdd_HHmmss", Locale.US).format(new Date()) + ".txt");
        startActivityForResult(intent, SAVE_TXT);
    }

    @Override
    protected void onActivityResult(int requestCode, int resultCode, Intent data) {
        super.onActivityResult(requestCode, resultCode, data);
        if (requestCode != SAVE_TXT || resultCode != RESULT_OK || data == null || data.getData() == null) return;
        try (OutputStream os = getContentResolver().openOutputStream(data.getData())) {
            if (os == null) throw new IllegalStateException("Nao foi possivel abrir arquivo");
            os.write(report.toString().getBytes(StandardCharsets.UTF_8));
            Toast.makeText(this, "Relatorio salvo", Toast.LENGTH_LONG).show();
        } catch (Exception e) {
            Toast.makeText(this, "Falha: " + e.getMessage(), Toast.LENGTH_LONG).show();
        }
    }

    @Override
    protected void onDestroy() {
        executor.shutdown();
        super.onDestroy();
    }
}
