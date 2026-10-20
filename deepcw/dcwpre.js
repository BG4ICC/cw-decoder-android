/* DeepCW 预处理模块（v1.84-P15b-2）——与 Python 参考实现对齐，金样比对通过后才进产品。
   输入：单声道 Float32 PCM（任意常见采样率）；输出：log1p 幅度谱 [T×65]。
   管线：抗混叠低通(FIR,fc≈1500Hz) → 重采样至 3200Hz → 256点Hann/hop48 → bin32..96 → log1p。 */
const DCWPRE=(()=>{

  // ── 256 点迭代 radix-2 FFT（原地，re/im 长度 256）──
  const REV=new Uint8Array(256), COS=new Float32Array(128), SIN=new Float32Array(128);
  (()=>{ for(let i=0;i<256;i++){ let r=0,x=i; for(let b=0;b<8;b++){ r=(r<<1)|(x&1); x>>=1; } REV[i]=r; }
         for(let k=0;k<128;k++){ COS[k]=Math.cos(2*Math.PI*k/256); SIN[k]=-Math.sin(2*Math.PI*k/256); } })();
  function fft256(re,im){
    for(let i=0;i<256;i++){ const j=REV[i]; if(j>i){ let t=re[i];re[i]=re[j];re[j]=t; t=im[i];im[i]=im[j];im[j]=t; } }
    for(let len=2;len<=256;len<<=1){
      const half=len>>1, step=256/len;
      for(let i=0;i<256;i+=len){
        for(let k=0,j=0;k<half;k++,j+=step){
          const a=i+k, b=a+half;
          const wr=COS[j], wi=SIN[j];
          const xr=re[b]*wr-im[b]*wi, xi=re[b]*wi+im[b]*wr;
          re[b]=re[a]-xr; im[b]=im[a]-xi; re[a]+=xr; im[a]+=xi;
        }
      }
    }
  }

  // ── Hann 窗：np.hanning(257)[:-1] 等价 0.5-0.5cos(2πn/256) ──
  const HANN=new Float32Array(256);
  for(let n=0;n<256;n++) HANN[n]=0.5-0.5*Math.cos(2*Math.PI*n/256);

  // ── 3200Hz 频谱：reflect pad 128 → 帧 hop48 → |rfft| 65 bin → log1p ──
  /* v1.88-P15b-4：binC 窗重心（频率搬移）。默认 64 = 取 bin32..96（400..1200Hz），
     与 v1.84 以来行为逐字节一致；binC=round(fc/12.5) 时取 binC-32..binC+32，
     音调恰好落模型输入窗正中央（等效 800Hz），量化误差 ≤6.25Hz，零插值零混叠。 */
  const FRE=new Float32Array(256), FIM=new Float32Array(256);
  function specFrom3200(au, binC){
    const b0=(binC||64)-32;   // 窗首 bin（默认 32）
    const N=au.length;
    const T=1+Math.floor(N/48);   // np 语义：1+floor((N+256-256)/48)=1+floor(N/48)
    const out=new Float32Array(T*65);
    for(let f=0;f<T;f++){
      const base=f*48-128;   // reflect pad 后第 f 帧起点对应原序列 base..base+255
      for(let n=0;n<256;n++){
        let idx=base+n;
        if(idx<0) idx=-idx;              // np.pad reflect 左缘：a[1..128] 镜像 → 下标 -idx（idx=-1→1? 见验证）
        else if(idx>=N) idx=2*N-2-idx;   // 右缘镜像
        if(idx<0) idx=0; if(idx>N-1) idx=N-1;
        FRE[n]=au[idx]*HANN[n]; FIM[n]=0;
      }
      fft256(FRE,FIM);
      const o=f*65;
      for(let b=b0;b<=b0+64;b++) out[o+b-b0]=Math.log1p(Math.hypot(FRE[b],FIM[b]));
    }
    return {data:out, T};
  }

  // ── 抗混叠低通 + 重采样至 3200Hz ──
  // Kaiser(β=5) 加窗 sinc 低通，fc=1500Hz（<1600 Nyquist@3.2k），长度随源率自适应（≈6.3ms 过渡带）
  function designLP(sr){
    const fc=1500/sr, L=Math.max(97,Math.round(sr/160))|1;  // 奇数长度（中心必须为整数样本，否则半样本相移）
    const M=L-1, h=new Float32Array(L), mid=M/2, beta=5.0;
    const i0=x=>{ let s=1,y=x*x/4,t=1; for(let k=1;k<30;k++){ t*=y/(k*k); s+=t; } return s; };
    const ib=i0(beta);
    for(let n=0;n<=M;n++){
      const d=n-mid;
      const sinc=d===0?2*fc:Math.sin(2*Math.PI*fc*d)/(Math.PI*d);
      const r=2*d/M, w=i0(beta*Math.sqrt(Math.max(0,1-r*r)))/ib;
      h[n]=sinc*w;
    }
    let sum=0; for(let n=0;n<=M;n++) sum+=h[n];
    for(let n=0;n<=M;n++) h[n]/=sum;
    return h;
  }
  let _lpCache={};
  function resampleTo3200(x, sr){
    if(sr===3200) return x;
    const h=_lpCache[sr]||(_lpCache[sr]=designLP(sr));
    const M=h.length-1, y=new Float32Array(x.length+M);
    y.set(x, M/2);   // 对称零延拓，卷积后居中截取
    const c=new Float32Array(x.length);
    for(let i=0;i<x.length;i++){
      let acc=0; const b=i;
      for(let j=0;j<=M;j++) acc+=h[j]*y[b+j];
      c[i]=acc;
    }
    const R=sr/3200, T=Math.floor(x.length/R);
    const out=new Float32Array(T);
    if(Number.isInteger(R)){   // 整数抽取（48k→3.2k=15）走精确网格，消除插值相位误差
      for(let t=0;t<T;t++) out[t]=c[Math.min(t*R,c.length-1)];
    }else{
      for(let t=0;t<T;t++){
        const p=t*R, i0=Math.floor(p), f=p-i0;
        const a=c[Math.min(i0,c.length-1)], b2=c[Math.min(i0+1,c.length-1)];
        out[t]=a+(b2-a)*f;
      }
    }
    return out;
  }

  return {fft256, specFrom3200, resampleTo3200, designLP};
})();
if(typeof module!=="undefined") module.exports=DCWPRE;
