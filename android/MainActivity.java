package com.cwdecoder.app;

import android.app.Activity;
import android.Manifest;
import android.content.Intent;
import android.content.pm.PackageManager;
import android.graphics.Color;
import android.net.Uri;
import android.os.Bundle;
import android.webkit.PermissionRequest;
import android.webkit.ValueCallback;
import android.webkit.WebChromeClient;
import android.webkit.WebSettings;
import android.webkit.JavascriptInterface;
import android.webkit.WebView;
import android.webkit.WebViewClient;
import android.webkit.WebResourceRequest;
import android.webkit.WebResourceResponse;
import android.widget.Toast;
import android.os.Build;
import android.os.Environment;
import android.provider.MediaStore;
import android.content.ContentValues;
import android.util.Base64;
import java.io.ByteArrayOutputStream;
import java.io.File;
import java.io.FileOutputStream;
import java.io.InputStream;
import java.io.OutputStream;

public class MainActivity extends Activity {
    private WebView web;
    private ValueCallback<Uri[]> fileCb;

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);

        /* v2.16：权限合并请求——录音 + 通知（Android 13+ 前台服务通知需运行时授权） */
        java.util.List<String> need = new java.util.ArrayList<>();
        if (checkSelfPermission(Manifest.permission.RECORD_AUDIO)
                != PackageManager.PERMISSION_GRANTED) {
            need.add(Manifest.permission.RECORD_AUDIO);
        }
        if (Build.VERSION.SDK_INT >= 33
                && checkSelfPermission(Manifest.permission.POST_NOTIFICATIONS)
                != PackageManager.PERMISSION_GRANTED) {
            need.add(Manifest.permission.POST_NOTIFICATIONS);
        }
        if (!need.isEmpty()) requestPermissions(need.toArray(new String[0]), 1);

        /* v2.16②：App 前台期间屏幕常亮（观察解码/发报），退后台或手动息屏照常 */
        getWindow().addFlags(android.view.WindowManager.LayoutParams.FLAG_KEEP_SCREEN_ON);

        web = new WebView(this);
        web.setBackgroundColor(Color.parseColor("#0b0f14"));
        WebSettings ws = web.getSettings();
        ws.setJavaScriptEnabled(true);
        ws.setDomStorageEnabled(true);
        ws.setMediaPlaybackRequiresUserGesture(false);
        ws.setAllowFileAccess(true);

        web.setWebChromeClient(new WebChromeClient() {
            @Override
            public void onPermissionRequest(PermissionRequest request) {
                final PermissionRequest req = request;
                runOnUiThread(new Runnable() {
                    @Override public void run() { req.grant(req.getResources()); }
                });
            }
            @Override
            public boolean onShowFileChooser(WebView w, ValueCallback<Uri[]> cb,
                                             FileChooserParams params) {
                fileCb = cb;
                Intent i = new Intent(Intent.ACTION_GET_CONTENT);
                i.addCategory(Intent.CATEGORY_OPENABLE);
                i.setType("audio/*");
                startActivityForResult(Intent.createChooser(i, "选择音频文件"), 2);
                return true;
            }
        });
        web.setWebViewClient(new WebViewClient() {
            /* v1.84-P15b：https://cw.local/ 虚拟域资产拦截——loadDataWithBaseURL 没有真实站点，
               页面的 fetch/script 相对路径（deepcw/model.onnx、ort wasm 等）在此桥接到 APK assets。 */
            @Override
            public WebResourceResponse shouldInterceptRequest(WebView view, WebResourceRequest req) {
                Uri u = req.getUrl();
                if (u != null && "cw.local".equals(u.getHost())) {
                    String p = u.getPath();
                    if (p != null && p.startsWith("/")) p = p.substring(1);
                    if (p != null && !p.isEmpty() && !p.equals("index.html") && !p.contains("..")) {
                        try {
                            InputStream in = getAssets().open("www/" + p);
                            String mime = p.endsWith(".wasm") ? "application/wasm"
                                    : p.endsWith(".js")   ? "text/javascript"
                                    : p.endsWith(".mjs")  ? "text/javascript"   /* ort wasm 胶水是 ES module，MIME 必须是 JS */
                                    : p.endsWith(".json") ? "application/json"
                                    : "application/octet-stream";
                            return new WebResourceResponse(mime, null, in);
                        } catch (Exception e) { /* 资产不存在 → 交回默认（404） */ }
                    }
                }
                return super.shouldInterceptRequest(view, req);
            }
        });
        web.addJavascriptInterface(new Object() {
            @JavascriptInterface
            public void save(String b64, String name, String mime) {
                String where;
                try {
                    byte[] data = Base64.decode(b64, Base64.DEFAULT);
                    if (Build.VERSION.SDK_INT >= 29) {
                        ContentValues v = new ContentValues();
                        v.put(MediaStore.Downloads.DISPLAY_NAME, name);
                        v.put(MediaStore.Downloads.MIME_TYPE, mime);
                        v.put(MediaStore.Downloads.RELATIVE_PATH,
                              Environment.DIRECTORY_DOWNLOADS + "/CW解码器");
                        Uri uri = getContentResolver().insert(
                                MediaStore.Downloads.EXTERNAL_CONTENT_URI, v);
                        OutputStream os = getContentResolver().openOutputStream(uri);
                        os.write(data); os.close();
                        where = "下载/CW解码器/" + name;
                    } else {
                        File dir = getExternalFilesDir(Environment.DIRECTORY_DOWNLOADS);
                        File f = new File(dir, name);
                        FileOutputStream os = new FileOutputStream(f);
                        os.write(data); os.close();
                        where = f.getAbsolutePath();
                    }
                    final String msg = "已保存：" + where;
                    runOnUiThread(new Runnable() {
                        @Override public void run() {
                            Toast.makeText(MainActivity.this, msg, Toast.LENGTH_LONG).show();
                        }
                    });
                } catch (Exception e) {
                    final String msg = "保存失败：" + e.getMessage();
                    runOnUiThread(new Runnable() {
                        @Override public void run() {
                            Toast.makeText(MainActivity.this, msg, Toast.LENGTH_LONG).show();
                        }
                    });
                }
            }
        }, "CwSave");

        /* v2.16①：后台持续收听桥——页面 start()/stop() 时经此桥起/撤前台服务，
           服务持麦克风类型前台服务 + 部分唤醒锁，切后台/息屏解码不中断。 */
        web.addJavascriptInterface(new Object() {
            @JavascriptInterface
            public void start() {
                runOnUiThread(new Runnable() {
                    @Override public void run() {
                        Intent i = new Intent(MainActivity.this, CwRxService.class);
                        if (Build.VERSION.SDK_INT >= 26) startForegroundService(i);
                        else startService(i);
                    }
                });
            }
            @JavascriptInterface
            public void stop() {
                runOnUiThread(new Runnable() {
                    @Override public void run() {
                        stopService(new Intent(MainActivity.this, CwRxService.class));
                    }
                });
            }
        }, "CwRx");

        String html = readAsset("www/index.html");
        web.loadDataWithBaseURL("https://cw.local/", html, "text/html", "UTF-8", null);

        setContentView(web);
    }

    private String readAsset(String name) {
        try (InputStream in = getAssets().open(name);
             ByteArrayOutputStream out = new ByteArrayOutputStream()) {
            byte[] buf = new byte[8192];
            int n;
            while ((n = in.read(buf)) != -1) out.write(buf, 0, n);
            return out.toString("UTF-8");
        } catch (Exception e) {
            return "<html><body>resource load failed</body></html>";
        }
    }

    @Override
    protected void onActivityResult(int requestCode, int resultCode, Intent data) {
        super.onActivityResult(requestCode, resultCode, data);
        if (requestCode == 2 && fileCb != null) {
            Uri r = (resultCode == RESULT_OK && data != null) ? data.getData() : null;
            fileCb.onReceiveValue(r == null ? null : new Uri[]{r});
            fileCb = null;
        }
    }

    @Override
    protected void onDestroy() {
        stopService(new Intent(this, CwRxService.class));   /* v2.16：活动销毁同步撤收听服务 */
        if (web != null) web.destroy();
        super.onDestroy();
    }
}
