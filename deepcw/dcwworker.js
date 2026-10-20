/* DeepCW 直通 Worker（v1.85-P15b-3）——官方词隙锚定流式提交协议的忠实移植。
   协议来源：web-deep-cw-decoder useStreamingDecode.ts + textDecoder.ts（逐语义对齐）。
   预处理：dcwpre.js（P15b-2 金样锁定：3200Hz/FFT256/hop48/log1p，位级一致）。
   模型：deepcw-engine 开源内核 model.onnx（42 类，blank=41，stride=1）。

   主→Worker：
     {type:"init", base}        加载 dcwpre/ort/模型 → 回 ready / error
     {type:"start", sr}         新收听会话：清空流状态
     {type:"audio", sr, buf}    追加原生采样率 PCM（ArrayBuffer, float32）
     {type:"cfg", narrow, fc}     v1.86-A′：窄带开关（锁定后生效）+ 锁频随行，3200Hz 域 FIR
     {type:"flush"}             停止收听：allowNearEnd 强制提交全部残留
   Worker→主：
     {type:"ready", ms}  {type:"error", where, msg}
     {type:"commit", text}      词隙锚定提交段（追加定稿）
     {type:"tnt", text}         当前待确认窗口全文（暂定尾预览）
     {type:"stat", ...}         推理耗时遥测
     {type:"flushed"}           flush 完成 */
"use strict";

/* ---- 词表（model.onnx.json 锁定）---- */
const VOC=[",",".","/","0","1","2","3","4","5","6","7","8","9","?",
  "A","B","C","D","E","F","G","H","I","J","K","L","M","N","O","P","Q","R","S","T","U","V","W","X","Y","Z"," "];
const BLANK=41, SPACE=40, NC=42;

/* ---- v2.06（方案7-M2）：EdNet 受限编辑标注器（影子模式专用，不上屏）----
   字符级 Transformer Encoder（882,163 参数，opset18 单文件 ONNX，动态长度）。
   词表 40：空格+A-Z+0-9+/?- ；rep 头 42 类（0=保持 1=删 2+k=替换为词表k），
   ins 头 41 类（0=无 1+k=在位置后插入词表k）。τ=0.9（M1 摆锤曲线最优点：
   低置信编辑几乎全错，高 τ 收益最大且身份误伤 0.10%/字符）。 */
const ED_VOCAB=" ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789/?-";
const ED_C2I={}; for(let _i=0;_i<ED_VOCAB.length;_i++) ED_C2I[ED_VOCAB[_i]]=_i;
let edSess=null, edReady=false;
function _softmax(a, off, n){
  let mx=-Infinity; for(let i=0;i<n;i++) mx=Math.max(mx,a[off+i]);
  let s=0; const o=new Float64Array(n);
  for(let i=0;i<n;i++){ o[i]=Math.exp(a[off+i]-mx); s+=o[i]; }
  for(let i=0;i<n;i++) o[i]/=s;
  return o;
}
/* v2.07（方案7-M1.5b）：推理护栏——与 guard_m15.py 逐条同义（改动必须双侧同步）。
   G1 受保护词（词表/RST形态）内部字符禁替换/删除；G2 受保护词内部禁插入；
   G3 空格删除遇受保护词或单字母词邻域即否决。只否决不提议。 */
const ED_PROTECT=new Set(["K","R","BK","KN","AR","SK","AS","TU","ES","GL","GB","FB",
  "OM","YL","OK","DE","BT","E","T","U","QSL","AGN","CFM","PSE","HW","DR","UR","NIL",
  "QRM","QRN","QSB","QRP","QRZ","SRI","TNX","HPE","CUL","GM","GA","GE","GD","WX","TEMP",
  "NAME","QTH","RIG","ANT","PWR","HR","IS","MY","AND","FER","BCK","TO","VY","CQ","TEST",
  "DX","EE","SM","NW","C","S","N","W","73","59","599","5NN","5N9","559","579","589",
  "569","549","539","529","449","339"]);
const ED_RST_RE=/^[1-5][N0-9][N0-9]$/;
function edVeto(text, edits){
  const protC=new Uint8Array(text.length), protS=new Uint8Array(text.length);
  const re=/\S+/g; let mt;
  while((mt=re.exec(text))!==null){
    const w=mt[0], s=mt.index, e=s+w.length;
    const prot=ED_PROTECT.has(w)||ED_RST_RE.test(w);
    if(prot) for(let i=s;i<e;i++) protC[i]=1;
    if(prot||w.length===1){
      if(s>0&&text[s-1]===" ") protS[s-1]=1;
      if(e<text.length&&text[e]===" ") protS[e]=1;
    }
  }
  return edits.filter(ed=>{
    if(ed.t==="del"){
      if(protC[ed.p]) return false;                        /* G1 */
      if(text[ed.p]===" "&&protS[ed.p]) return false;     /* G3 */
    }else if(ed.t==="sub"){
      if(protC[ed.p]) return false;                        /* G1 */
    }else{                                                 /* ins */
      if(protC[ed.p]&&protC[ed.p+1]) return false;        /* G2 */
    }
    return true;
  });
}
function edApply(text, rep, ins, tau){
  /* rep/ins: Float32Array  logits [1,L,42]/[1,L,41]。提议→护栏→重建。 */
  const L=text.length;
  const edits=[];
  for(let i=0;i<L;i++){
    const pr=_softmax(rep, i*42, 42), ps=_softmax(ins, i*41, 41);
    if(pr[1]>tau){ edits.push({p:i,t:"del"}); }                            /* 删 */
    else{
      let bi=0, bp=0;
      for(let k=0;k<40;k++) if(pr[2+k]>bp){ bp=pr[2+k]; bi=k; }
      if(bp>tau) edits.push({p:i,t:"sub",c:ED_VOCAB[bi]});                 /* 换 */
    }
    let ii=-1, ip=0;
    for(let k=0;k<40;k++) if(ps[1+k]>ip){ ip=ps[1+k]; ii=k; }
    if(ip>tau) edits.push({p:i,t:"ins",c:ED_VOCAB[ii]});                   /* 插 */
  }
  const kept=edVeto(text, edits);
  const dels=new Set(), subs={}, insAt={};
  for(const ed of kept){
    if(ed.t==="del") dels.add(ed.p);
    else if(ed.t==="sub") subs[ed.p]=ed.c;
    else (insAt[ed.p]=insAt[ed.p]||[]).push(ed.c);
  }
  let out="";
  for(let i=0;i<L;i++){
    if(dels.has(i)){ /* 删 */ }
    else out+=subs[i]!==undefined?subs[i]:text[i];
    if(insAt[i]) out+=insAt[i].join("");
  }
  return out;
}

/* ---- 流式协议常量（官方值，3200Hz 域）----
   官方：最大段 30s / 尾护 1.25s / 最小待解码 2s / 最小确认 2s / 推理最小间隔 1s */
const SR=3200, FFT=256, HOP=48;
const MAX_SEG=30*SR, TAIL_GUARD=Math.round(1.25*SR);
const MIN_PENDING=2*SR, MIN_CONFIRMED=2*SR;
let MIN_INTERVAL=1000;   /* v1.89：改 let——cfg.mint 仅供离线回放确定性测试（每块必推理，消除墙钟竞态）；产品主线程永不下发 */

/* ---- v2.04（方案6 A1）：WPM 动态缩放提交颗粒度 ----
   病灶（v1.96/v1.98 取证留档）：官方固定常量不随码速缩放——高速码词隙锚荒 +
   固定 2s/1.25s 颗粒相对字速过粗 → 提交停滞爆发（ctc9 实测 24s 零提交一吞 49 字符）、
   落位漂移（"TU TUEE|GL" 晚一词）、commit 批处理方差（同码同音 1/16 刀缺失）。
   估计器：token 速率路（10×tok/s，含空格 token），指数衰减累积（0.92/窗 ≈ 慢变中位）。
   【双峰直方图路已否决留档】v2.04 探针台实抓 ctc9v2（48WPM）：15ms 帧分辨率把
   字内隙（1.7 帧）压进 1-2 帧量化噪声，平滑 argmax 被字间隙（3dit≈5 帧）俘获——
   读数恒 3 倍慢（实抓 16 vs 真值 48），k 反向打到 1.5 夹紧、常量全顶宽、提交更迟。
   合成流双峰可用纯属测试假象。帧分辨率绝症，纯 token 速率路分辨率免疫。
   缩放律：k=clamp(25/WPM,0.45,1.0)——【单向收紧，永不放宽】（v2.04 取证定案：wt 含解码损耗
   天然低估≈CER 比例，放宽方向会把常量顶到夹紧上限反增停滞；官方常量即安全默认，
   只在有效速率 >25WPM 的干净高速信号上收紧颗粒）。25WPM 锚点 k=1 逐字节等于旧行为；
   MIN_INTERVAL（CPU 节拍）与 MAX_SEG（内存上限）不动。
   冻结：无 token 窗口不喂估计器（静默期估计保持）；会话开始复位。
   闸口：cfg.scale 与 beam 同闸下发——推荐档 scale=1，nnd 恒 0=官方常量（固定刻度裁判）；
   scale=0 时 curPC≡BASE_PC 逐字节等于 v2.03（金样锁定探针 W4 强制）。 */
const BASE_PC={minPend:2*SR, minConf:2*SR, tailGuard:Math.round(1.25*SR)};
const curPC={minPend:BASE_PC.minPend, minConf:BASE_PC.minConf, tailGuard:BASE_PC.tailGuard};
let scaleOn=false;
const wpmE={chars:0, frames:0, wpm:0};
let fbN=0;   /* v2.23：降级锚定触发计数（遥测，诊断读取） */
const FB_OVL=3200;   /* v2.23 Fix-B：降级缝窗首重叠（样本，1.0s@3200）。seamtest：缝后首字符正确需窗深≥825ms，ln=10 隙下 M=3200 保底 1075ms */
let headSkipF=0;     /* v2.23 Fix-B：当前窗重叠修剪线（帧），startFrame 小于此线的 span 为已提交内容 */
function csHead(cs){ return headSkipF>0?cs.filter(s=>s.startFrame>=headSkipF):cs; }
/* v2.26（方案13）：提交饥饿滚动强切常量——锚荒（SPACE 锚绝迹）且降级缝也不出刀（空白段全 <150ms）
   时段长滚到 MAX_SEG(30s) 才强制倾倒，文本冻结数十秒（v2.25 真机 ctc-9 实证：积压 19.8s 平台期、
   CPU 富余 7 倍、30s 一次性倾泻）。8s 墙钟 + 8s 积压双门槛强切，堵字上限 30s→8s。 */
const STARVE_MS=8000, STARVE_PEND=8*SR, STARVE_BLANK=6;   /* 6帧=90ms：符间隙级空白 */
let lastCommitT=0, stvN=0;   /* 上次提交墙钟（会话起点初始化）/ 饥饿强切计数（遥测） */
/* v2.24（方案2）：自适应降载——worker 自测实时倍率 k=推理耗时/窗长（EMA 0.3），
   连续3拍 k>0.35 进慢机模式：推理间隔×2（砍半 O(n²) 重推）、降级缝阈值 3×→2.5×minPend
   （提交更早、窗更小）、分析窗上限 12s（堵强制倾倒的推理黑洞）；k<0.25 恢复（回差防抖）。
   病灶：慢机 WASM（真机实测净吞吐 ~0.6×实时）积压滚雪球 → MAX_SEG 螺旋+环形丢块→覆盖率塌。
   快机 k≈0.05 远低于触发线 → 全程不触发 → 行为字节级不变（回归锁定）。
   cfg.mint 测试钩优先：MIN_INTERVAL≠1000 时慢机模式不覆盖推理间隔（离线回放确定性）。 */
const DCW_BUILD="v2.26-b126";
let kEMA=0, slowOn=false, kOverN=0;
let bgdN=0;   /* v2.25（方案16）：后台弃积压累计秒数（遥测）；方案14 缝位空格臂净负否决——改由主线程 cwBoundaryFix 缝位窄签名拼合（14c），worker 零改动 */
const K_HI=0.35, K_LO=0.25, K_EMA=0.3, K_NEED=3, SLOW_MINT=2000, SLOW_ANA=12*SR;
const BG_CAP=12*SR, BG_KEEP=6*SR;   /* v2.25（方案16）：后台积压硬顶——超 12s 弃旧保新只留 6s（前台永不下发 bg 标记，口径不变） */
function kFeed(ms, winLen){
  if(winLen<SR) return;   /* 窗<1s 不计——小窗调度噪声大 */
  const k=ms/(winLen/SR*1000);
  kEMA=kEMA?kEMA+(k-kEMA)*K_EMA:k;
  if(!slowOn){ if(kEMA>K_HI){ if(++kOverN>=K_NEED) slowOn=true; } else kOverN=0; }
  else if(kEMA<K_LO){ slowOn=false; kOverN=0; }
}
function mintEff(){ return (slowOn&&MIN_INTERVAL===1000)?SLOW_MINT:MIN_INTERVAL; }
const clampF=(v,a,b)=>Math.max(a,Math.min(b,v));
function updWpm(){
  const FRW=SR/HOP;   /* 66.67 帧/秒 */
  if(wpmE.frames>=2*FRW&&wpmE.chars>=25){   /* 置信门：≥2s 有字流 & ≥25 token 才启用（开局稀疏幻觉token不污染） */
    const wt=10*wpmE.chars/(wpmE.frames/FRW);
    if(wt>0) wpmE.wpm=wt;
  }
}
/* v2.05（方案6b）：分母换源——键控跨度帧。v2.04 用窗口全帧，换手/守听静默稀释占空比，
   ctc9v2 实测：48WPM 键控被均成 ~16 → k=1 缩放在唯一目标场景不触发（取证留档 f204*）。
   改法：分母=总帧−≥1.0s 长静默帧。词隙（25WPM=336ms，12WPM=700ms）<1s 保留在分母里
   → 残余稀释只偏慢不偏快（单向收紧结构的安全方向）；25WPM 连续键控锚点 k=1 字节锁不变。 */
function keySpanFrames(audio){
  const nF=Math.floor(audio.length/HOP);
  if(!nF) return 0;
  const e=new Float32Array(nF); let peak=0;
  for(let f=0;f<nF;f++){ let s=0; const o=f*HOP;
    for(let i=0;i<HOP;i++){ const v=audio[o+i]; s+=v*v; }
    e[f]=Math.sqrt(s/HOP); if(e[f]>peak) peak=e[f]; }
  if(peak<=0) return 0;
  const srt=Float32Array.from(e).sort(); const med=srt[nF>>1];
  const th=(peak+med)*0.5;                 /* 峰-中位中点：占空<50% 时 med=噪声，>50% 时 med=音，均稳 */
  const MIN_SIL=Math.round(SR/HOP);        /* 1.0s：长于任何词隙（≥10WPM），短于典型换手停顿 */
  let sil=0, run=0;
  for(let f=0;f<=nF;f++){
    if(f<nF&&e[f]<th){ run++; continue; }
    if(run>=MIN_SIL) sil+=run;
    run=0;
  }
  return nF-sil;
}
function wpmFeed(idx,T,kf){   /* 常开轻量；tick 每窗调用。无 token/稀疏窗整体跳过（冻结+防幻觉污染） */
  let toks=0, prev=BLANK;
  for(let t=0;t<T;t++){
    const v=idx[t];
    if(v!==BLANK){ if(v!==prev) toks++; }
    prev=v;
  }
  /* v2.05b（f205/f205b A/B 取证定版）：单窗 toks 钳 6 tok/s（=60WPM 物理上限，超此必为解码
     噪声——f205 曾见 243 尖峰），钳在进入衰减累积之前 → 全程估计 ≤60。短窗（kf<1s）保留：
     f205b 取证证明长窗 CTC 高速塌缩把 48WPM 读成 ~20（过度保守=缩放不触发），短窗才是
     真实跟速读数；尖峰经钳位后最坏=k 下限 0.45，且 f205 实证尖峰下文本硬指标仍全绿
     （LM 五件套+锚点门在下游把关）。真实 25WPM=2.5tok/s 远低于钳位，锚点不受影响。 */
  const FRW=SR/HOP;
  if(toks>=3&&kf>0){
    const cap=Math.round(6*kf/FRW);
    wpmE.chars=wpmE.chars*0.92+Math.min(toks,cap);
    wpmE.frames=wpmE.frames*0.92+kf; updWpm();
  }
}
function applyScale(){
  if(!scaleOn||!wpmE.wpm){
    curPC.minPend=BASE_PC.minPend; curPC.minConf=BASE_PC.minConf; curPC.tailGuard=BASE_PC.tailGuard;
    return;
  }
  const k=clampF(25/wpmE.wpm,0.45,1.0);
  curPC.minConf=Math.round(clampF(2*SR*k,0.8*SR,2*SR));   /* 上界=BASE：单向收紧（k≤1） */
  curPC.tailGuard=Math.round(clampF(1.25*SR*k,0.5*SR,1.25*SR));
  curPC.minPend=Math.round(clampF(2*SR*k,0.8*SR,2*SR));
}

let sess=null, ready=false;
let pend=new Float32Array(0);       // 待确认 3200Hz 音频（增长缓冲，非滑窗）
let lastInf=0, inferring=false;

/* ---- 流式重采样（与 dcwpre.resampleTo3200 全文件版逐位对齐）----
   全文件版：x 对称零延拓 M/2 → FIR → c[0..N-1] → out[t]=c[t*R]（整数网格/线性插值）。
   流式版：rh 保 M 样本历史，每块算出 c 的新增段追入 fq；抽取游标 dp 为 fq 局部坐标，
   消费后丢弃头部并同步 dp。起点零延拓与全文件版一致 → 同序同值；末端 mid 个滤波
   样本在 flush 补零后产出。非整数 R 的插值右邻必须真实存在（不夹逼）才产出，
   与全文件版逐位一致。 */
const rs={sr:0, h:null, M:0, mid:0, R:1, rh:null, fq:new Float32Array(0), dp:0, first:true};
function rsReset(sr){
  rs.sr=sr; rs.dp=0; rs.fq=new Float32Array(0); rs.first=true;
  if(sr===SR){ rs.h=null; rs.rh=null; rs.R=1; rs.M=0; rs.mid=0; return; }
  rs.h=DCWPRE.designLP(sr); rs.M=rs.h.length-1; rs.mid=rs.M/2; rs.R=sr/SR;
  rs.rh=new Float32Array(rs.M);
}
function rsDrain(final){   // 从 fq 抽取 3200Hz 样本；final=true 时允许右缘夹逼
  const out=[]; const intR=Number.isInteger(rs.R);
  for(;;){
    const p=rs.dp, n=rs.fq.length;
    if(intR){ if(p>n-1) break; out.push(rs.fq[Math.min(Math.round(p),n-1)]); }
    else{
      const i0=Math.floor(p);
      if(i0+1>n-1){ if(!final||i0>n-1) break; out.push(rs.fq[n-1]); rs.dp+=rs.R; continue; }
      const f=p-i0, a=rs.fq[i0], b2=rs.fq[i0+1]; out.push(a+(b2-a)*f);
    }
    rs.dp+=rs.R;
  }
  const dropN=Math.max(0,Math.min(rs.fq.length, Math.floor(rs.dp)-1));
  if(dropN>0){ rs.fq=rs.fq.slice(dropN); rs.dp-=dropN; }
  return Float32Array.from(out);
}
function rsPush(chunk){   // → 新产出的 3200Hz 样本
  if(rs.sr===SR) return chunk;
  /* ext=rh(M)+chunk；cf[i]=c[g+i-mid]，i 满程 [0,chunk.length) 保证跨块连续。
     首块 cf[0..mid-1] 对应 c[-mid..-1]（零延拓区，全文件版不发射）→ 跳过。 */
  const ext=new Float32Array(rs.M+chunk.length);
  ext.set(rs.rh,0); ext.set(chunk,rs.M);
  const nC=chunk.length;
  const cf=new Float32Array(nC);
  for(let i=0;i<nC;i++){
    let acc=0; const b=i;
    for(let j=0;j<=rs.M;j++) acc+=rs.h[j]*ext[b+j];
    cf[i]=acc;
  }
  rs.rh.set(ext.subarray(ext.length-rs.M));
  const skip=rs.first?rs.mid:0; rs.first=false;
  if(skip<nC){
    const cfv=cf.subarray(skip);
    const nf=new Float32Array(rs.fq.length+cfv.length);
    nf.set(rs.fq,0); nf.set(cfv,rs.fq.length); rs.fq=nf;
  }
  return rsDrain(false);
}
function rsFlush(){   // 补 mid 个零收尾，产出末端滤波样本对应的剩余抽取点
  if(rs.sr===SR) return new Float32Array(0);
  const ext=new Float32Array(rs.M+rs.mid);
  ext.set(rs.rh,0);
  const cf=new Float32Array(rs.mid);
  for(let i=0;i<rs.mid;i++){
    let acc=0; for(let j=0;j<=rs.M;j++) acc+=rs.h[j]*ext[i+j];
    cf[i]=acc;
  }
  const nf=new Float32Array(rs.fq.length+rs.mid);
  nf.set(rs.fq,0); nf.set(cf,rs.fq.length); rs.fq=nf;
  return rsDrain(true);
}

/* ---- v1.86-A′：窄带带通（3200Hz 域；与主线程 mkFirBp 同一 Blackman-sinc 设计，201 阶 fc±200Hz）----
   主线程统一判定生效条件（窄带开关开 + 已锁定）经 cfg 下发；开关/频率变化冷启动清零防旧状态冲击。
   3200Hz 域 201 阶 ≈ 63ms 窗；开销 0.64M MAC/s，可忽略。 */
const bp={h:null, ring:new Float32Array(256), rp:0, fc:0, on:false};
function bpDesign(fc){
  const N=201, M=(N-1)/2;
  const fL=Math.max(40,fc-200)/SR, fH=Math.min(1500,fc+200)/SR;
  const h=new Float32Array(N);
  for(let n=0;n<N;n++){
    const k=n-M;
    const sH=(k===0)?2*fH:Math.sin(2*Math.PI*fH*k)/(Math.PI*k);
    const sL=(k===0)?2*fL:Math.sin(2*Math.PI*fL*k)/(Math.PI*k);
    const w=0.42-0.5*Math.cos(2*Math.PI*n/(N-1))+0.08*Math.cos(4*Math.PI*n/(N-1)); // Blackman
    h[n]=(sH-sL)*w;
  }
  return h;
}
function bpReset(on,fc){ bp.on=on; bp.fc=fc; bp.h=on?bpDesign(fc):null; bp.ring=new Float32Array(256); bp.rp=0; }

/* ---- v1.89-P15c：CTC 前缀束搜索 + CWLM（移植自主线程 ctcBeam 语义，适配 DeepCW 42 类）----
   设计要点：
   ① 贪心=每帧 argmax 当场拍板，犹豫帧（5/T、空格/空白概率接近）选错即固化；
     束搜索同时养 BW 条候选前缀，声学 logP 累计 + 追加字符时加 α·cwLmLP 语言模型分，
     全局高分路径反超——治词界粘连与犹豫帧字符混淆。
   ② 返回"最优路径的逐帧类别序列"（持久链表对齐，与 predIndices 同型），
     下游 span 提取/词隙锚定提交/灰色暂定尾协议零改动。
   ③ 提交后 pend 截断，每窗独立 → 无跨窗前缀收敛问题；LM 词内上下文窗首重置。
   ④ CWLM 无空格字符：空格扩展不打 LM 分（中性），空格后首字符用 "^^^" 词首上下文。
      【β 空格加分已否决留档】ctc9 β=0/1.0/2.0 → CER 39.1/39.8/40.1 单调变差：
      粘连段空格类概率被空白碾压且常跌出 topK=8，固定加分翻不了盘只会过切分。β 默认 0。
   ⑤ 安全绳：α=0+束宽1+topK1 逐字节退化为贪心（金样锁定）；默认由离线 A/B 定。 */
const beamCfg={on:false, alpha:0.3, BW:16, topK:8, beta:0, fsp:false};   /* beta：空格加分（nat），治词界粘连；fsp：诊断钩——强制空格入候选 */
/* v1.89-诊断钩（测试专用，产品永不下发）：dbg 统计"空白强势帧"（top1=BLANK 且 lp>-0.7）
   上 SPACE 类的排名分布与平均 lp——量化词隙帧空格被压到什么程度，决定 topK 截断/
   强制入候选/β 加分哪种补救有效。cfg.dbg=1 采集，cfg.dbg=2 回报并清零。 */
const dbg={on:false, bt:0, rankHist:new Array(43).fill(0), lpSum:0, lpN:0, runHist:new Array(121).fill(0), runs:0};
function beamIndices(lp, T, alpha, BW, topK){
  const NEG=-1e30;
  const lae=(a,b)=>{ if(a<b){const t=a;a=b;b=t;} return a+Math.log1p(Math.exp(b-a)); };
  /* 束条目：{pb(末帧blank),pn(末帧非blank),nB,nN(两态各自对齐链节点)} 节点={prev,cls} */
  let beams=new Map([["",{pb:0,pn:NEG,nB:null,nN:null}]]);
  const ord=new Int32Array(NC);
  const lmOf=(pfx,ch)=>{ const ws=pfx.lastIndexOf(" "); return cwLmLP(ch,"^^^"+pfx.slice(ws+1)); };
  for(let t=0;t<T;t++){
    const row=t*NC;
    for(let c=0;c<NC;c++) ord[c]=c;
    const arr=Array.from(ord).sort((a,b)=>{const d=lp[row+b]-lp[row+a]; return d!==0?d:a-b;}).slice(0,topK);
    if(beamCfg.fsp&&!arr.includes(SPACE)) arr.push(SPACE);   /* 诊断钩：空格强制入候选（topK 截断豁免） */
    const nb=new Map();
    const add=(p,st,v,prev,cls)=>{
      const nd={prev,cls};
      const e=nb.get(p);
      if(!e){ const o={pb:NEG,pn:NEG,nB:null,nN:null};
        if(st){o.pn=v;o.nN=nd;}else{o.pb=v;o.nB=nd;} nb.set(p,o); return; }
      if(st){ if(v>e.pn){ e.pn=lae(e.pn,v); e.nN=nd; } else e.pn=lae(e.pn,v); }
      else{ if(v>e.pb){ e.pb=lae(e.pb,v); e.nB=nd; } else e.pb=lae(e.pb,v); }
    };
    for(const [pfx,e] of beams){
      const tot=lae(e.pb,e.pn);
      const prevBest=e.pb>=e.pn?e.nB:e.nN;
      const last=pfx.length?pfx[pfx.length-1]:"";
      for(const c of arr){
        const lc=lp[row+c];
        if(c===BLANK){ add(pfx,0,tot+lc,prevBest,c); continue; }
        if(c===SPACE){
          if(last===" ") add(pfx,1,e.pn+lc,prevBest,c);          /* 重复空格合并 */
          else add(pfx+" ",1,tot+lc+beamCfg.beta,prevBest,c);    /* 词界：LM 中性 + β 加分（词隙帧空白概率碾压空格时翻盘用） */
          continue;
        }
        const ch=VOC[c];
        if(ch===last){
          add(pfx,1,e.pn+lc,prevBest,c);                                  /* 同字符延续 */
          add(pfx+ch,1,e.pb+lc+alpha*lmOf(pfx,ch),prevBest,c);            /* 隔 blank 重复字母 */
        }else{
          add(pfx+ch,1,tot+lc+alpha*lmOf(pfx,ch),prevBest,c);
        }
      }
    }
    if(nb.size>BW){
      const es=[...nb.entries()].sort((a,b)=>lae(b[1].pb,b[1].pn)-lae(a[1].pb,a[1].pn)).slice(0,BW);
      beams=new Map(es);
    }else beams=nb;
  }
  let best=null,bv=NEG;
  for(const e of beams.values()){
    if(e.pb>=e.pn){ if(e.pb>bv){bv=e.pb;best=e.nB;} } else if(e.pn>bv){bv=e.pn;best=e.nN;}
  }
  const idx=new Uint8Array(T);
  let n=best, t=T-1;
  while(n&&t>=0){ idx[t--]=n.cls; n=n.prev; }
  return idx;
}
/* ---- v1.88-P15b-4：频率搬移 800Hz（FFT 窗重心法）【已否决留档，永不开启】----
   cfg.shift 开且锁频已知时，specFrom3200 取 binC±32（binC=round(fc/12.5)），
   音调恒落模型输入窗正中央（等效 800Hz）。默认关=旧行为（固定 bin32..96）。
   否决依据（v1.88 离线回放同条件 A/B，nb=1，shift 0→1 CER）：
     ctc9(671Hz) 42.3→41.2 持平；no1(692Hz) 23.8→23.3 持平；v4 实机轨 逐词持平；
     ctc8(700Hz+197Hz哼声) 38.5→41.8 变差（开头 TEU/DN/EHNY 插字符）。
   结论：DeepCW 训练已具频率不变性，重心化无收益且个别样本净负 → 主线程不下发
   shift:true，本通路仅作实验保留（replay186.html?shift=1 可复测）。 */
const shf={on:false, binC:64};
function shfSet(on,fc){
  shf.on=!!on;
  shf.binC=on?Math.max(32,Math.min(96,Math.round(fc/12.5))):64;
}
function bpApply(x){
  if(!bp.on||!bp.h||!x.length) return x;
  const h=bp.h, N=h.length, ring=bp.ring; let rp=bp.rp;
  const out=new Float32Array(x.length);
  for(let i=0;i<x.length;i++){
    ring[rp]=x[i]; let acc=0;
    for(let j=0;j<N;j++) acc+=h[j]*ring[(rp-j+512)&255];
    out[i]=acc; rp=(rp+1)&255;
  }
  bp.rp=rp; return out;
}

/* ---- CTC 贪心 + span 提取（textDecoder.ts 逐语义移植）---- */
function predIndices(lp, T){
  const idx=new Uint8Array(T);
  for(let t=0;t<T;t++){
    let mv=-Infinity, mi=0; const o=t*NC;
    for(let c=0;c<NC;c++){ const v=lp[o+c]; if(v>mv){mv=v;mi=c;} }
    idx[t]=mi;
  }
  return idx;
}
function getWordSpaceSpans(idx){   // 词隙 = space token 连续段（非 blank 段）
  const spans=[]; let st=-1;
  for(let f=0;f<idx.length;f++){
    if(idx[f]===SPACE){ if(st<0) st=f; }
    else if(st>=0){ spans.push({startFrame:st, endFrame:f-1}); st=-1; }
  }
  if(st>=0) spans.push({startFrame:st, endFrame:idx.length-1});
  return spans;
}
function getCharacterSpans(idx){   // 最优路径连续段（重复延展 endFrame）
  const spans=[]; let prev=-1, act=-1;
  for(let f=0;f<idx.length;f++){
    const ix=idx[f];
    if(ix===BLANK){ prev=-1; act=-1; continue; }
    if(ix===prev){ if(act>=0) spans[act].endFrame=f; continue; }
    prev=ix;
    const ch=VOC[ix]; if(!ch){ act=-1; continue; }
    spans.push({char:ch, startFrame:f, endFrame:f}); act=spans.length-1;
  }
  return spans;
}
function normalizeSpans(cs){   // normalizeDetailedResult：空白折叠、去头尾空白
  const chars=[]; let wS=null, wE=null; const spans=[];
  for(const sp of cs){
    if(sp.char===" "){
      if(!chars.length) continue;
      if(wS==null) wS=sp.startFrame;
      wE=sp.endFrame; continue;
    }
    if(wS!=null){ chars.push(" "); spans.push({char:" ",startFrame:wS,endFrame:wE}); wS=wE=null; }
    chars.push(sp.char); spans.push(sp);
  }
  return {text:chars.join(""), spans};
}
function trimSpansToFrame(cs, endFrame){ return cs.filter(s=>s.endFrame<=endFrame); }

/* ---- v1.89-诊断钩：词隙时序补切（产品永不下发，实验验证中）----
   DeepCW 模型级弱点（ctc9 实测）：词隙帧空白概率碾压空格（SPACE 平均 lp≈-19 nat，
   即便 topK 豁免+β=2.0 也翻不了盘），"BG2MCUBG2MCM" 式粘连贪心/束搜索均有，
   deepcw.cc 官方站同源同样粘连——非移植偏差。
   补救思路：字符 span 之间对齐序列（idx）上的纯空白段长度≈真实静默时长
   （键起判空白），字内隙≈3 点、词隙≈7 点呈双峰 → 空白段≥wgap 帧时注入合成空格 span。
   【span 间距方案已否决留档】span 间距含字符尾部键down，长字符间距天然大，
   ctc9 阈值 28/36/44 帧 → 空格 189/169/129 个（ref 75）全过切分，CER 64~79 全劣。
   cfg.wgap=帧阈值（15ms/帧），0=关。生效时词隙锚定提交同步用合成词隙。 */
const gapCfg={wgap:0};
function injectGaps(idx, cs, wgap){
  const out=[]; let prev=null;
  for(const sp of cs){
    if(sp.char!==" "){
      if(prev){
        let run=0;
        for(let f=prev.endFrame+1;f<sp.startFrame;f++) if(idx[f]===BLANK) run++;
        if(run>=wgap) out.push({char:" ", startFrame:prev.endFrame+1, endFrame:sp.startFrame-1});
      }
      prev=sp;
    }
    out.push(sp);
  }
  return out;
}
function wssFromSpans(cs){   /* 合成词隙段（合并连续空格 span） */
  const out=[]; let s=null;
  for(const sp of cs){
    if(sp.char===" "){ if(!s) s={startFrame:sp.startFrame, endFrame:sp.endFrame}; else s.endFrame=sp.endFrame; }
    else if(s){ out.push(s); s=null; }
  }
  if(s) out.push(s);
  return out;
}
function spansOf(idx){ const cs=getCharacterSpans(idx); return gapCfg.wgap>0?injectGaps(idx,cs,gapCfg.wgap):cs; }

/* ---- 词隙锚定提交（useStreamingDecode.ts 逐语义移植）---- */
function getSpanSplitSample(span){
  const midFrame=(span.startFrame+span.endFrame)/2;
  return Math.round(midFrame*HOP+FFT/2);
}
function getConfirmedSplitPoint(spans, analysisLength, allowNearEnd){
  const maxC=allowNearEnd?analysisLength:Math.max(curPC.minConf, analysisLength-curPC.tailGuard);   /* v2.04：MIN_CONFIRMED/TAIL_GUARD→curPC（scale=0 时恒等于官方常量） */
  for(let i=spans.length-1;i>=0;i--){
    const s=getSpanSplitSample(spans[i]);
    if(s>=curPC.minConf&&s<=maxC) return {sample:s, endFrame:spans[i].endFrame};
  }
  return null;
}

/* ---- 推理 + 提交主流程 ---- */
async function inferOnce(audio){
  const spec=DCWPRE.specFrom3200(audio, shf.binC);   /* v1.88：binC 默认64=旧行为；shift 开时随锁频重心化 */
  const x=new ort.Tensor("float32", spec.data, [1,1,spec.T,65]);
  const t0=performance.now();
  const out=await sess.run({spectrogram:x});
  const dt=performance.now()-t0;
  const lp=out.log_probs.data, T=out.log_probs.dims[1];
  if(dbg.on){   /* 诊断钩：空白强势帧上 SPACE 类的排名/lp 采集 */
    for(let t=0;t<T;t++){
      const o=t*NC; let mx=-Infinity, mi=0;
      for(let c=0;c<NC;c++){ const v=lp[o+c]; if(v>mx){mx=v;mi=c;} }
      if(mi!==BLANK||mx<-0.7) continue;
      const ls=lp[o+SPACE]; let rk=1;
      for(let c=0;c<NC;c++) if(lp[o+c]>ls) rk++;
      dbg.bt++; dbg.rankHist[Math.min(rk,42)]++; dbg.lpSum+=ls; dbg.lpN++;
    }
  }
  /* v1.89-P15c：束搜索开 → 最优路径逐帧对齐（同型替换贪心）；LM 缺失时回落贪心 */
  const idx=(beamCfg.on&&typeof cwLmLP==="function")
    ? beamIndices(lp,T,beamCfg.alpha,beamCfg.BW,beamCfg.topK)
    : predIndices(lp,T);
  if(dbg.on){   /* 诊断钩：字符间纯空白段长度直方图（词隙时序补切阈值取证） */
    let run=0;
    for(let t=0;t<=T;t++){
      if(t<T&&idx[t]===BLANK){ run++; continue; }
      if(run>0){ dbg.runHist[Math.min(run,120)]++; dbg.runs++; }
      run=0;
    }
  }
  return {idx, T, dt};
}
async function tick(force){
  if(!ready||inferring) return;
  applyScale();   /* v2.04：cfg 变更即时反映（估计未建立/scale=0 → 官方常量） */
  if(pend.length<curPC.minPend){ postMessage({type:"tnt",text:""}); return; }   /* v2.04：MIN_PENDING→curPC */
  const now=performance.now();
  if(!force&&lastInf>0&&now-lastInf<mintEff()) return;   /* v2.24：MIN_INTERVAL→mintEff（慢机×2，测试钩优先） */
  inferring=true;
  try{
    const analysisLength=Math.min(pend.length, slowOn?SLOW_ANA:MAX_SEG);   /* v2.24：慢机模式分析窗上限 12s */
    const audio=pend.slice(0,analysisLength);
    lastInf=performance.now();
    const r=await inferOnce(audio);
    kFeed(r.dt, audio.length);   /* v2.24：实时倍率采样 */
    wpmFeed(r.idx,r.T,keySpanFrames(audio)); applyScale();   /* v2.05（方案6b）：分母=键控跨度帧；无 token 窗口自然冻结 */
    const wss=gapCfg.wgap>0?wssFromSpans(spansOf(r.idx)):getWordSpaceSpans(r.idx), cs=spansOf(r.idx);
    /* 暂定尾 = 当前窗口全文（官方 pendingText 同口径） */
    postMessage({type:"tnt", text:normalizeSpans(csHead(cs)).text});   /* v2.23 Fix-B：重叠区重发字符不进预览 */
    postMessage({type:"stat", frames:r.T, ms:+r.dt.toFixed(1), pendS:+(pend.length/SR).toFixed(1),
      wpm:+wpmE.wpm.toFixed(1), pc:[curPC.minPend,curPC.minConf,curPC.tailGuard], fb:fbN, bgd:bgdN, stv:stvN,
      k:+kEMA.toFixed(2), slow:slowOn?1:0, build:DCW_BUILD});   /* v2.04：缩放遥测；v2.23：降级锚计数；v2.24：实时倍率/降载态/构建号；v2.25：后台弃积压秒数（纯遥测字段，主线程仅诊断读取） */

    let split=getConfirmedSplitPoint(wss, analysisLength, force||pend.length>=MAX_SEG);
    /* v2.23（方案12 降级锚定 + Fix-B 深重叠）：词隙锚荒兜底——无 SPACE 锚且非强制路径、
       积压≥3×minPend 时，取可提交区 [minConf, ana−tailGuard] 内最长空白段（≥10帧=150ms，
       不切字内点划隙）作切点。只切不注空格（wgap 文本注入否决路径不动）：跨提交空格由主线程
       段间分隔（joinSegmentText 同口径）补齐，粘连文本仍走 LM 三件套分词。
       病灶留档：模型词隙粘连（SPACE lp≈-19nat）+声学混响填平词隙 → nWss 长达 30s+ 为零，
       提交饥饿到 MAX_SEG/flush 才倾倒（ctc9 空外实测全程零锚）。
       缝误码机制留档（seamtest 离线分离）：冷窗起切缝后首字符翻转（2→W），贪心同翻 → 声学
       左上下文不足，非 LM 冷启动（热种子无效）；重叠≥960 样本（窗深≥825ms）复原 → Fix-B。 */
    if(!split && !force && pend.length<MAX_SEG && pend.length>=curPC.minPend*(slowOn?2.5:3)){   /* v2.24：慢机模式降级缝阈值收紧 */
      const lo=curPC.minConf, hi=analysisLength-curPC.tailGuard;
      let br=null, brs=-1;
      for(let f=0;f<=r.T;f++){
        if(f<r.T&&r.idx[f]===BLANK){ if(brs<0) brs=f; continue; }
        if(brs>=0){
          const ef=f-1, ln=f-brs, ss=Math.round(((brs+ef)/2)*HOP+FFT/2);
          if(ln>=10&&ss>=lo&&ss<=hi&&(!br||ln>br.ln)) br={ln,ss,ef:brs-1};
          brs=-1;
        }
      }
      if(br&&br.ef>=0){ split={sample:br.ss, endFrame:br.ef, fb:1}; fbN++; }
    }
    /* v2.26（方案13）：提交饥饿滚动强切——距上次提交 >8s 墙钟且积压 ≥8s 时，在可提交区
       [minConf, ana−tailGuard] 内找最长空白 ≥6帧（90ms，符间/字间隙）切之，缝按 fb 同构
       FB_OVL 重叠保声学左上下文（切点在空白内，两侧字符不跨缝，修剪线无吞字风险）；
       连 90ms 空白都没有（连续键流/噪声填满，病态兜底）则在 hi 处硬切——跨缝字符随修剪
       丢弃（干净缺失）优于 ovl=0 残留半字符重解码出垃圾。切点文本粘连由主线程缝窗
       重分词（方案17）兜底。墙钟判定而非积压长度：桌面快速回放积压瞬时 >8s 属常态，
       不会误触（回放提交频繁、lastCommitT 恒新鲜）；真机锚荒 8s 无提交才启动。
       正常段有锚走锚、有 150ms 空白走降级缝，本分支摸不到，行为零变化。 */
    if(!split && !force && pend.length<MAX_SEG && pend.length>=STARVE_PEND && now-lastCommitT>STARVE_MS){
      const lo=curPC.minConf, hi=analysisLength-curPC.tailGuard;
      let br=null, brs=-1;
      for(let f=0;f<=r.T;f++){
        if(f<r.T&&r.idx[f]===BLANK){ if(brs<0) brs=f; continue; }
        if(brs>=0){
          const ef=f-1, ln=f-brs, ss=Math.round(((brs+ef)/2)*HOP+FFT/2);
          if(ln>=STARVE_BLANK&&ss>=lo&&ss<=hi&&(!br||ln>br.ln)) br={ln,ss,ef:brs-1};
          brs=-1;
        }
      }
      if(br&&br.ef>=0){ split={sample:br.ss, endFrame:br.ef, fb:1}; stvN++; }
      else{
        const hf=Math.floor((hi-FFT/2)/HOP);
        if(hf>0){ split={sample:hi, endFrame:hf, fb:1}; stvN++; }
      }
    }
    const confirmedLen=split?split.sample:(pend.length>=MAX_SEG?(slowOn?SLOW_ANA:MAX_SEG):(force?analysisLength:null));   /* v2.24：慢机模式强制倾倒同样限 12s， bounded 推理黑洞 */
    if(confirmedLen==null||confirmedLen<curPC.minConf) return;   /* v2.04：MIN_CONFIRMED→curPC */

    let commitText="";
    if(split){
      commitText=normalizeSpans(trimSpansToFrame(csHead(cs), split.endFrame)).text;
    }else{
      /* 无词隙可用（30s 强制 / flush 兜底）：官方对确认段重跑一次推理取全文 */
      if(confirmedLen===analysisLength){ commitText=normalizeSpans(csHead(cs)).text; }
      else{
        const r2=await inferOnce(pend.slice(0,confirmedLen));
        kFeed(r2.dt, confirmedLen);   /* v2.24：强制路径重推理同样计入实时倍率 */
        commitText=normalizeSpans(csHead(spansOf(r2.idx))).text;
      }
    }
    if(commitText){ postMessage({type:"commit", text:commitText}); lastCommitT=performance.now(); }   /* v2.26（方案13）：提交墙钟刷新——饥饿强切的武装基准 */
    /* v2.23 Fix-B：降级缝重叠回切——新窗首回退 FB_OVL 样本（声学左上下文），
       headSkipF 记修剪线（帧），重叠区重发 span 不进下次提交/预览；SPACE 锚/强制路径不动 */
    const ovl=(split&&split.fb)?Math.min(FB_OVL, confirmedLen):0;
    pend=pend.slice(confirmedLen-ovl);
    headSkipF=ovl?Math.ceil(ovl/HOP):0;
    postMessage({type:"tnt", text:""});   /* 官方：提交后立即清预览，防双显 */
  }catch(e){
    postMessage({type:"error", where:"tick", msg:String(e&&e.message||e)});
  }finally{
    inferring=false;
  }
}

self.onmessage=async ev=>{
  const m=ev.data;
  try{
    if(m.type==="init"){
      const base=m.base;
      importScripts(base+"dcwpre.js");
      importScripts(base+"dcwlm.js");   /* v1.89-P15c：束搜索 LM（缺失时束搜索自动回落贪心） */
      importScripts(base+"ort.wasm.min.js");
      ort.env.wasm.wasmPaths=base; ort.env.wasm.numThreads=1;
      const t0=performance.now();
      const buf=await (await fetch(base+"model.onnx")).arrayBuffer();
      if(buf.byteLength!==15139839) throw new Error("model size "+buf.byteLength);
      sess=await ort.InferenceSession.create(new Uint8Array(buf));
      /* v2.06（方案7-M2）：EdNet 影子模型——fail-soft，加载失败仅影子不可用，主解码零影响 */
      try{
        const ebuf=await (await fetch(base+"ednet.onnx")).arrayBuffer();
        if(ebuf.byteLength!==3820081) throw new Error("ednet size "+ebuf.byteLength);   /* v2.07：M1.5 ep5 */
        edSess=await ort.InferenceSession.create(new Uint8Array(ebuf));
        edReady=true;
      }catch(ee){ edSess=null; edReady=false; }
      ready=true;
      postMessage({type:"ready", ms:+(performance.now()-t0).toFixed(0), edReady, build:DCW_BUILD});   /* v2.24：构建号随就绪上报 */
      return;
    }
    if(m.type==="edfix"){   /* v2.06（方案7-M2）：影子试算——只回投记录，不参与任何提交/预览 */
      const id=m.id|0, txt=String(m.text||"");
      if(!edReady||!edSess||!txt||txt.length>224){
        postMessage({type:"edfix", id, bad:txt, pred:null, ms:0}); return;
      }
      const et0=performance.now();
      try{
        const ids=new BigInt64Array(txt.length);
        for(let i=0;i<txt.length;i++){
          const c=ED_C2I[txt[i]];
          ids[i]=BigInt(c===undefined?0:c);   /* 表外字符（,.等）→空格位 */
        }
        const out=await edSess.run({x:new ort.Tensor("int64",ids,[1,txt.length])});
        const pred=edApply(txt, out.rep_logits.data, out.ins_logits.data,
          (m.tau>0&&m.tau<1)?+m.tau:0.9);
        postMessage({type:"edfix", id, bad:txt, pred,
          ms:+(performance.now()-et0).toFixed(1)});
      }catch(ee){
        postMessage({type:"edfix", id, bad:txt, pred:null, ms:0,
          err:String(ee&&ee.message||ee)});
      }
      return;
    }
    if(m.type==="start"){
      pend=new Float32Array(0); lastInf=0;
      rsReset(m.sr||48000);
      bpReset(false,0);   /* v1.86：会话复位——窄带等主线程 cfg 再激活 */
      shfSet(false,0);    /* v1.88：会话复位——搬移同口径复位 */
      beamCfg.on=false;   /* v1.89：会话复位——束搜索等主线程 cfg 再激活 */
      beamCfg.topK=8; beamCfg.fsp=false;   /* 诊断钩复位：防跨会话残留 */
      gapCfg.wgap=0;                       /* 诊断钩复位：词隙补切默认关 */
      scaleOn=false;                       /* v2.04：会话复位——动态缩放等主线程 cfg 再激活 */
      wpmE.chars=0; wpmE.frames=0; wpmE.wpm=0; fbN=0; headSkipF=0;   /* v2.23 */
      bgdN=0;   /* v2.25（方案16）：后台弃积压计数随会话复位 */
      lastCommitT=performance.now(); stvN=0;   /* v2.26（方案13）：提交墙钟/饥饿计数随会话复位 */
      kEMA=0; slowOn=false; kOverN=0;   /* v2.24：降载状态随会话复位 */
      curPC.minPend=BASE_PC.minPend; curPC.minConf=BASE_PC.minConf; curPC.tailGuard=BASE_PC.tailGuard;
      dbg.on=false; dbg.bt=0; dbg.rankHist.fill(0); dbg.lpSum=0; dbg.lpN=0;
      dbg.runHist.fill(0); dbg.runs=0;
      return;
    }
    if(m.type==="audio"){
      if(!ready) return;
      const chunk=new Float32Array(m.buf);
      if(rs.sr!==m.sr) rsReset(m.sr);   /* 采样率变化（重开会话）自适应 */
      const out=bpApply(rsPush(chunk));   /* v1.86-A′：窄带 FIR（cfg 激活后生效，未激活直通） */
      if(out.length){
        const np=new Float32Array(pend.length+out.length);
        np.set(pend,0); np.set(out,pend.length); pend=np;
      }
      /* v2.25（方案16 安全阀）：后台积压硬顶。worker 队列无上限——后台 CPU 被系统降级时
         慢机降载（方案2）也可能追不上，积压滚雪球即成"文字炸弹"（真机实证 47.1s）。
         bg 随行时积压超 BG_CAP(12s) 弃旧保新只留 BG_KEEP(6s) 从当前解码，弃秒数入遥测 bgd；
         前台主线程永不下发 bg 标记，前台口径零变化。 */
      if(m.bg&&pend.length>BG_CAP){
        const drop=pend.length-BG_KEEP;
        pend=pend.slice(drop); headSkipF=0;   /* 断档后旧重叠修剪线失效归零 */
        bgdN+=Math.round(drop/SR);
      }
      await tick(false);
      return;
    }
    if(m.type==="cfg"){   /* v1.86-A′：窄带开关/锁频随行（生效条件由主线程统一判定） */
      const on=!!m.narrow, fc=Math.round(+m.fc||700);
      if(on!==bp.on||fc!==bp.fc) bpReset(on,fc);
      shfSet(m.shift, fc);   /* v1.88-P15b-4：频率搬移（默认关；主线程按 lock/trial 判定下发） */
      if(m.beam!==undefined) beamCfg.on=!!m.beam;            /* v1.89-P15c */
      if(m.scale!==undefined) scaleOn=!!m.scale;             /* v2.04（方案6）：动态缩放闸——与 beam 同闸下发，nnd 恒 0=官方常量 */
      if(m.alpha!=null) beamCfg.alpha=+m.alpha;
      if(m.bw!=null) beamCfg.BW=Math.max(1,Math.min(64,+m.bw|0));
      if(m.beta!=null) beamCfg.beta=+m.beta;
      if(m.mint!=null) MIN_INTERVAL=+m.mint;                 /* v1.89：测试钩，产品不下发 */
      if(m.topk!=null) beamCfg.topK=Math.max(1,Math.min(42,+m.topk|0));   /* 诊断钩，产品不下发 */
      if(m.fsp!=null) beamCfg.fsp=!!m.fsp;                                /* 诊断钩，产品不下发 */
      if(m.wgap!=null) gapCfg.wgap=Math.max(0,Math.min(200,+m.wgap|0));   /* 诊断钩：词隙补切帧阈，产品不下发 */
      if(m.dbg!=null){                                                    /* 诊断钩：1=采集 2=回报并清零 */
        if(m.dbg===2){ postMessage({type:"dbg", bt:dbg.bt, hist:dbg.rankHist.slice(1),
          lpm:dbg.lpN?+(dbg.lpSum/dbg.lpN).toFixed(3):null,
          runs:dbg.runs, runHist:Array.from(dbg.runHist)});
          dbg.bt=0; dbg.rankHist.fill(0); dbg.lpSum=0; dbg.lpN=0;
          dbg.runHist.fill(0); dbg.runs=0; }
        else dbg.on=!!m.dbg;
      }
      return;
    }
    if(m.type==="flush"){
      if(!ready){ postMessage({type:"flushed"}); return; }
      const tail=bpApply(rsFlush());   /* v1.86-A′：尾段同口径 */
      if(tail.length){
        const np=new Float32Array(pend.length+tail.length);
        np.set(pend,0); np.set(tail,pend.length); pend=np;
      }
      lastInf=0;
      await tick(true);
      /* flush 后仍残留的（不足最小确认长度或无词隙的尾巴）整段提交 */
      if(pend.length>0){
        try{
          const r=await inferOnce(pend.slice(0));
          const t=normalizeSpans(spansOf(r.idx)).text;
          if(t) postMessage({type:"commit", text:t});
        }catch(e){}
        pend=new Float32Array(0);
      }
      postMessage({type:"tnt", text:""});
      postMessage({type:"flushed"});
      return;
    }
  }catch(e){
    postMessage({type:"error", where:m.type, msg:String(e&&e.message||e)});
  }
};
