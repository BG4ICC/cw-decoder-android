package com.cwdecoder.app;

import android.app.Notification;
import android.app.NotificationChannel;
import android.app.NotificationManager;
import android.app.PendingIntent;
import android.app.Service;
import android.content.Intent;
import android.content.pm.ServiceInfo;
import android.os.Build;
import android.os.IBinder;
import android.os.PowerManager;

/* v2.16①：后台持续收听前台服务。
   Android 10+ 后台持麦必须由 microphone 类型前台服务承载（targetSdk 34 强制）；
   服务持部分唤醒锁，息屏后 CPU 不休眠、解码继续。
   常驻通知是系统对后台用麦的硬性要求（隐私提示），不可去除；点通知回主界面。 */
public class CwRxService extends Service {
    private static final String CH = "cw_rx";
    private PowerManager.WakeLock wl;

    @Override
    public void onCreate() {
        super.onCreate();
        PowerManager pm = (PowerManager) getSystemService(POWER_SERVICE);
        wl = pm.newWakeLock(PowerManager.PARTIAL_WAKE_LOCK, "cwdecoder:rx");
        wl.setReferenceCounted(false);
    }

    @Override
    public int onStartCommand(Intent intent, int flags, int startId) {
        NotificationManager nm = (NotificationManager) getSystemService(NOTIFICATION_SERVICE);
        if (Build.VERSION.SDK_INT >= 26) {
            nm.createNotificationChannel(new NotificationChannel(
                    CH, "后台收听", NotificationManager.IMPORTANCE_LOW));
        }
        PendingIntent pi = PendingIntent.getActivity(this, 0,
                new Intent(this, MainActivity.class),
                Build.VERSION.SDK_INT >= 23 ? PendingIntent.FLAG_IMMUTABLE : 0);
        Notification.Builder b = Build.VERSION.SDK_INT >= 26
                ? new Notification.Builder(this, CH) : new Notification.Builder(this);
        b.setContentTitle("CW解码器")
         .setContentText("正在收听，解码保持运行")
         .setSmallIcon(android.R.drawable.stat_notify_more)
         .setContentIntent(pi)
         .setOngoing(true);
        if (Build.VERSION.SDK_INT >= 29) {
            startForeground(1, b.build(), ServiceInfo.FOREGROUND_SERVICE_TYPE_MICROPHONE);
        } else {
            startForeground(1, b.build());
        }
        if (wl != null && !wl.isHeld()) wl.acquire();
        return START_STICKY;
    }

    @Override
    public void onDestroy() {
        if (wl != null && wl.isHeld()) wl.release();
        stopForeground(true);
        super.onDestroy();
    }

    @Override
    public IBinder onBind(Intent intent) { return null; }
}
