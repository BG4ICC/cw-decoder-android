#!/bin/bash
# CW 解码器命令行构建（免 Android Studio）：SDK 自举 + ecj 编译 + aapt2 打包 + 签名
# 用法：bash build.sh [b|n]（默认 b）
#   b=三通道全功能版：VC=version.txt，包名 com.cwdecoder.app
#   n=双通道精简版：VC=version.txt-100，包名 com.cwdecoder.app.n（可与 b 版同机共存）
#   n 版在暂存副本上由 python 断言唯一替换 EDITION 开关（仓库母版恒为 b，严禁手改出包）
set -e
B="$(cd "$(dirname "$0")" && pwd)"   # 仓库根目录
WORK=/tmp/cwapp                      # 构建暂存（每次重建）

echo "=== 1. SDK ==="
if [ ! -d /tmp/android-sdk/build-tools/34.0.0 ]; then
  mkdir -p /tmp/android-sdk/cmdline-tools
  cd /tmp/android-sdk/cmdline-tools
  curl -sO https://dl.google.com/android/repository/commandlinetools-linux-11076708_latest.zip
  unzip -q commandlinetools-linux-11076708_latest.zip
  mv cmdline-tools latest
  cd /tmp/android-sdk
  yes | cmdline-tools/latest/bin/sdkmanager --sdk_root=/tmp/android-sdk "platforms;android-34" "build-tools;34.0.0" >/dev/null 2>&1
fi
ls /tmp/android-sdk/platforms /tmp/android-sdk/build-tools

echo "=== 2. ecj ==="
if [ ! -f /tmp/ecj.jar ]; then
  curl -sL -o /tmp/ecj.jar https://repo1.maven.org/maven2/org/eclipse/jdt/ecj/3.37.0/ecj-3.37.0.jar
fi

echo "=== 3. project ==="
rm -rf $WORK
mkdir -p $WORK/java/com/cwdecoder/app $WORK/res/mipmap-xxxhdpi $WORK/assets/www $WORK/obj $WORK/dex
cp $B/index.html $WORK/assets/www/index.html
# DeepCW 引擎资产（model.onnx / onnxruntime-web wasm）随包
cp -r $B/deepcw $WORK/assets/www/
cp $B/android/MainActivity.java $WORK/java/com/cwdecoder/app/MainActivity.java
cp $B/android/CwRxService.java $WORK/java/com/cwdecoder/app/CwRxService.java
ED=${1:-b}
VC=$(cat $B/version.txt 2>/dev/null || echo 6)
VN=$(cat $B/versionname.txt 2>/dev/null || echo 1.5)
PKG="com.cwdecoder.app"
if [ "$ED" = "n" ]; then
  VC=$((VC-100)); PKG="com.cwdecoder.app.n"
  python3 - <<PYEOF
p='$WORK/assets/www/index.html'
s=open(p,encoding='utf-8').read()
old='const EDITION="b";'
assert s.count(old)==1, "EDITION 锚点不唯一或缺失——中止"
open(p,'w',encoding='utf-8').write(s.replace(old,'const EDITION="n";'))
print("EDITION=n 注入 OK")
PYEOF
fi
VN="${VN}${ED}"
cat > $WORK/AndroidManifest.xml <<MAN
<?xml version="1.0" encoding="utf-8"?>
<manifest xmlns:android="http://schemas.android.com/apk/res/android" package="$PKG" android:versionCode="$VC" android:versionName="$VN">
    <uses-sdk android:minSdkVersion="24" android:targetSdkVersion="34"/>
    <uses-permission android:name="android.permission.RECORD_AUDIO"/>
    <uses-permission android:name="android.permission.MODIFY_AUDIO_SETTINGS"/>
    <!-- 后台持续收听——前台服务 + 麦克风类型 + 唤醒锁 + 通知（13+ 运行时授权） -->
    <uses-permission android:name="android.permission.FOREGROUND_SERVICE"/>
    <uses-permission android:name="android.permission.FOREGROUND_SERVICE_MICROPHONE"/>
    <uses-permission android:name="android.permission.WAKE_LOCK"/>
    <uses-permission android:name="android.permission.POST_NOTIFICATIONS"/>
    <application
        android:label="CW解码器"
        android:icon="@mipmap/ic_launcher"
        android:theme="@android:style/Theme.Material.NoActionBar.Fullscreen"
        android:hardwareAccelerated="true">
        <activity android:name="com.cwdecoder.app.MainActivity"
            android:exported="true"
            android:configChanges="orientation|screenSize|keyboardHidden">
            <intent-filter>
                <action android:name="android.intent.action.MAIN"/>
                <category android:name="android.intent.category.LAUNCHER"/>
            </intent-filter>
        </activity>
        <service android:name="com.cwdecoder.app.CwRxService"
            android:exported="false"
            android:foregroundServiceType="microphone"/>
    </application>
</manifest>
MAN
if [ ! -f $B/ic_launcher.png ]; then
python3 -c "
from PIL import Image, ImageDraw
img = Image.new('RGBA',(192,192),(11,15,20,255))
d = ImageDraw.Draw(img)
d.rounded_rectangle([8,8,184,184], radius=40, fill=(18,25,34,255), outline=(31,43,56,255), width=3)
d.ellipse([40,76,80,116], fill=(61,255,140,255))
d.rounded_rectangle([96,88,156,104], radius=8, fill=(61,255,140,255))
img.save('$B/ic_launcher.png')
"
fi
cp $B/ic_launcher.png $WORK/res/mipmap-xxxhdpi/ic_launcher.png

echo "=== 4. build ==="
cd $WORK
BT=/tmp/android-sdk/build-tools/34.0.0
AJ=/tmp/android-sdk/platforms/android-34/android.jar
java -jar /tmp/ecj.jar -1.8 -encoding UTF-8 -bootclasspath $AJ -d obj java/com/cwdecoder/app/*.java
echo "ecj OK"
$BT/aapt2 compile --dir res -o compiled.zip
$BT/aapt2 link -o app-unaligned.apk -I $AJ --manifest AndroidManifest.xml -A assets compiled.zip
$BT/d8 --lib $AJ --release obj/com/cwdecoder/app/*.class --output dex
cd dex && zip -q $WORK/app-unaligned.apk classes.dex && cd $WORK
$BT/zipalign -f 4 app-unaligned.apk app-aligned.apk
# 签名私钥不入库（.gitignore）；首次构建自动生成开发用 keystore，正式发布请自行替换
if [ ! -f $B/cw.keystore ]; then
  KEYTOOL=$(command -v keytool || echo /usr/lib/jvm/java-17-openjdk-amd64/bin/keytool)
  $KEYTOOL -genkeypair -keystore $B/cw.keystore -alias cwkey -keyalg RSA -keysize 2048 -validity 10000 -storepass cwdecoder -keypass cwdecoder -dname "CN=CW Decoder, OU=Amateur Radio, O=BG4ICC, L=CN, ST=CN, C=CN"
fi
$BT/apksigner sign --ks $B/cw.keystore --ks-pass pass:cwdecoder --key-pass pass:cwdecoder --out cwdecoder.apk app-aligned.apk
$BT/apksigner verify cwdecoder.apk
mkdir -p $B/out
cp -f cwdecoder.apk "$B/out/CW解码器_v${VN}.apk"

echo "=== 5. verify content ==="
cd /tmp && rm -rf verify && mkdir verify && cd verify
unzip -q "$B/out/CW解码器_v${VN}.apk"
sha256sum assets/www/index.html $WORK/assets/www/index.html assets/www/deepcw/dcwworker.js $B/deepcw/dcwworker.js
$BT/aapt2 dump badging "$B/out/CW解码器_v${VN}.apk" | grep -E "^package"
ls -la "$B/out/CW解码器_v${VN}.apk"
echo "ALL_DONE"
