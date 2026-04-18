/*
 * Scorbits (SCO) GPU Miner — Linux
 * Supports: GTX 1060/1070 (sm_61), RTX 3060/3070/3080 (sm_86), and more
 * Auto-detects GPU architecture
 *
 * ── QUICK START (vast.ai or any Linux with CUDA) ──────────────────────────
 *
 *   # 1. Install (one-time):
 *   apt-get update && apt-get install -y build-essential curl
 *
 *   # 2. Build:
 *   nvcc -O3 --use_fast_math -o scorbits_miner scorbits_miner.cu -lcurl
 *
 *   # 3. Run:
 *   ./scorbits_miner --address SCO7e38c262a4a26f838a5eb2c9a7876efd
 *
 * ── OPTIONS ───────────────────────────────────────────────────────────────
 *   --address SCO...    Your SCO wallet address (required)
 *   --node URL          Node URL (default: http://51.91.122.48:8080)
 *   --gpu N             GPU device index (default: 0)
 *
 * ── HASH FORMAT ───────────────────────────────────────────────────────────
 *   SHA256(index + timestamp + txData + prevHash + nonce + minerAddr)
 *   Matches blockchain/block.go CalculateHash() exactly
 */

#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include <time.h>
#include <curl/curl.h>

/* ── SHA-256 macros ──────────────────────────────────────────────────────── */

#define ROTR32(x,n) (((x)>>(n))|((x)<<(32-(n))))
#define CH(x,y,z)   (((x)&(y))^((~(x))&(z)))
#define MAJ(x,y,z)  (((x)&(y))^((x)&(z))^((y)&(z)))
#define EP0(x)      (ROTR32(x,2) ^ROTR32(x,13)^ROTR32(x,22))
#define EP1(x)      (ROTR32(x,6) ^ROTR32(x,11)^ROTR32(x,25))
#define SIG0(x)     (ROTR32(x,7) ^ROTR32(x,18)^((x)>>3))
#define SIG1(x)     (ROTR32(x,17)^ROTR32(x,19)^((x)>>10))

/* ── GPU SHA-256 ─────────────────────────────────────────────────────────── */

__device__ void sha256_gpu(const uint8_t* data, uint32_t len, uint8_t out[32])
{
    /* Inline K constants — avoids __constant__ memory issues */
    const uint32_t K[64]={
        0x428a2f98u,0x71374491u,0xb5c0fbcfu,0xe9b5dba5u,
        0x3956c25bu,0x59f111f1u,0x923f82a4u,0xab1c5ed5u,
        0xd807aa98u,0x12835b01u,0x243185beu,0x550c7dc3u,
        0x72be5d74u,0x80deb1feu,0x9bdc06a7u,0xc19bf174u,
        0xe49b69c1u,0xefbe4786u,0x0fc19dc6u,0x240ca1ccu,
        0x2de92c6fu,0x4a7484aau,0x5cb0a9dcu,0x76f988dau,
        0x983e5152u,0xa831c66du,0xb00327c8u,0xbf597fc7u,
        0xc6e00bf3u,0xd5a79147u,0x06ca6351u,0x14292967u,
        0x27b70a85u,0x2e1b2138u,0x4d2c6dfcu,0x53380d13u,
        0x650a7354u,0x766a0abbu,0x81c2c92eu,0x92722c85u,
        0xa2bfe8a1u,0xa81a664bu,0xc24b8b70u,0xc76c51a3u,
        0xd192e819u,0xd6990624u,0xf40e3585u,0x106aa070u,
        0x19a4c116u,0x1e376c08u,0x2748774cu,0x34b0bcb5u,
        0x391c0cb3u,0x4ed8aa4au,0x5b9cca4fu,0x682e6ff3u,
        0x748f82eeu,0x78a5636fu,0x84c87814u,0x8cc70208u,
        0x90befffau,0xa4506cebu,0xbef9a3f7u,0xc67178f2u
    };
    uint32_t h0=0x6a09e667u,h1=0xbb67ae85u,h2=0x3c6ef372u,h3=0xa54ff53au;
    uint32_t h4=0x510e527fu,h5=0x9b05688cu,h6=0x1f83d9abu,h7=0x5be0cd19u;
    uint8_t blk[256];
    uint32_t nb=(len+9+63)/64;
    uint32_t i;
    for(i=0;i<nb*64;i++) blk[i]=0;
    for(i=0;i<len;i++) blk[i]=data[i];
    blk[len]=0x80u;
    uint64_t bits=(uint64_t)len*8ULL;
    uint32_t last=nb*64;
    blk[last-1]=(uint8_t)(bits);    blk[last-2]=(uint8_t)(bits>>8);
    blk[last-3]=(uint8_t)(bits>>16);blk[last-4]=(uint8_t)(bits>>24);
    blk[last-5]=(uint8_t)(bits>>32);blk[last-6]=(uint8_t)(bits>>40);
    blk[last-7]=(uint8_t)(bits>>48);blk[last-8]=(uint8_t)(bits>>56);
    for(uint32_t bn=0;bn<nb;bn++){
        uint8_t* b=blk+bn*64;
        uint32_t w[64];
        for(i=0;i<16;i++)
            w[i]=((uint32_t)b[i*4]<<24)|((uint32_t)b[i*4+1]<<16)|
                 ((uint32_t)b[i*4+2]<<8)|b[i*4+3];
        for(i=16;i<64;i++)
            w[i]=SIG1(w[i-2])+w[i-7]+SIG0(w[i-15])+w[i-16];
        uint32_t a=h0,b2=h1,c=h2,d=h3,e=h4,f=h5,g=h6,hh=h7,t1,t2;
        for(i=0;i<64;i++){
            t1=hh+EP1(e)+CH(e,f,g)+K[i]+w[i];
            t2=EP0(a)+MAJ(a,b2,c);
            hh=g;g=f;f=e;e=d+t1;d=c;c=b2;b2=a;a=t1+t2;
        }
        h0+=a;h1+=b2;h2+=c;h3+=d;h4+=e;h5+=f;h6+=g;h7+=hh;
    }
#define WB(n,v) out[n*4]=(v>>24)&0xFF;out[n*4+1]=(v>>16)&0xFF;\
                out[n*4+2]=(v>>8)&0xFF;out[n*4+3]=v&0xFF;
    WB(0,h0)WB(1,h1)WB(2,h2)WB(3,h3)WB(4,h4)WB(5,h5)WB(6,h6)WB(7,h7)
#undef WB
}

/* ── GPU itoa ────────────────────────────────────────────────────────────── */

__device__ int gpu_itoa(char* buf, long long v)
{
    if(v==0){buf[0]='0';return 1;}
    char tmp[20]; int i=0;
    long long u=(v<0)?-v:v;
    while(u>0){tmp[i++]='0'+(int)(u%10);u/=10;}
    if(v<0) tmp[i++]='-';
    for(int j=0;j<i;j++) buf[j]=tmp[i-1-j];
    return i;
}

/* ── Mining kernel ───────────────────────────────────────────────────────── */
/*
 * Hash = SHA256(prefix + nonce_decimal + address)
 * prefix = index_str + ts_str + txData + prevHash  (built on CPU, sent to GPU)
 * address = miner address                           (built on CPU, sent to GPU)
 */
__global__ void mine_kernel(
    const char* prefix, int plen,
    const char* address, int alen,
    long long   base_nonce,
    int         difficulty,
    long long*  found_nonce,
    long long*  found_ts_out,
    long long   current_ts,
    char*       found_hash_hex)
{
    long long nonce = base_nonce +
        (long long)(blockIdx.x*(int)blockDim.x+(int)threadIdx.x);

    char input[512];
    int pos=0, i;
    for(i=0;i<plen;i++) input[pos++]=prefix[i];
    char ns[22];
    int nl=gpu_itoa(ns,nonce);
    for(i=0;i<nl;i++) input[pos++]=ns[i];
    for(i=0;i<alen;i++) input[pos++]=address[i];

    uint8_t hash[32];
    sha256_gpu((const uint8_t*)input,(uint32_t)pos,hash);

    /* Check difficulty: N leading hex zeros */
    int full=difficulty/2;
    bool ok=true;
    for(i=0;i<full&&ok;i++) if(hash[i]!=0x00u) ok=false;
    if(ok&&(difficulty&1)) if((hash[full]>>4)!=0) ok=false;

    if(ok){
        unsigned long long prev=atomicCAS(
            (unsigned long long*)found_nonce,
            (unsigned long long)(-1LL),
            (unsigned long long)nonce);
        if(prev==(unsigned long long)(-1LL)){
            *found_ts_out=current_ts;
            const char hx[]="0123456789abcdef";
            for(i=0;i<32;i++){
                found_hash_hex[i*2  ]=hx[hash[i]>>4];
                found_hash_hex[i*2+1]=hx[hash[i]&0xF];
            }
            found_hash_hex[64]='\0';
        }
    }
}

/* ── CPU SHA-256 for verification ───────────────────────────────────────── */

static void sha256_cpu(const char* input, char* hex_out)
{
    static const uint32_t Kh[64]={
        0x428a2f98u,0x71374491u,0xb5c0fbcfu,0xe9b5dba5u,
        0x3956c25bu,0x59f111f1u,0x923f82a4u,0xab1c5ed5u,
        0xd807aa98u,0x12835b01u,0x243185beu,0x550c7dc3u,
        0x72be5d74u,0x80deb1feu,0x9bdc06a7u,0xc19bf174u,
        0xe49b69c1u,0xefbe4786u,0x0fc19dc6u,0x240ca1ccu,
        0x2de92c6fu,0x4a7484aau,0x5cb0a9dcu,0x76f988dau,
        0x983e5152u,0xa831c66du,0xb00327c8u,0xbf597fc7u,
        0xc6e00bf3u,0xd5a79147u,0x06ca6351u,0x14292967u,
        0x27b70a85u,0x2e1b2138u,0x4d2c6dfcu,0x53380d13u,
        0x650a7354u,0x766a0abbu,0x81c2c92eu,0x92722c85u,
        0xa2bfe8a1u,0xa81a664bu,0xc24b8b70u,0xc76c51a3u,
        0xd192e819u,0xd6990624u,0xf40e3585u,0x106aa070u,
        0x19a4c116u,0x1e376c08u,0x2748774cu,0x34b0bcb5u,
        0x391c0cb3u,0x4ed8aa4au,0x5b9cca4fu,0x682e6ff3u,
        0x748f82eeu,0x78a5636fu,0x84c87814u,0x8cc70208u,
        0x90befffau,0xa4506cebu,0xbef9a3f7u,0xc67178f2u
    };
    uint32_t len=(uint32_t)strlen(input);
    uint32_t h0=0x6a09e667u,h1=0xbb67ae85u,h2=0x3c6ef372u,h3=0xa54ff53au;
    uint32_t h4=0x510e527fu,h5=0x9b05688cu,h6=0x1f83d9abu,h7=0x5be0cd19u;
    uint32_t nb=(len+9+63)/64;
    uint8_t* padded=(uint8_t*)calloc(nb*64,1);
    memcpy(padded,input,len);
    padded[len]=0x80;
    uint64_t bits=(uint64_t)len*8;
    for(int i=0;i<8;i++) padded[nb*64-1-i]=(uint8_t)(bits>>(i*8));
    for(uint32_t bn=0;bn<nb;bn++){
        uint8_t* blk=padded+bn*64;
        uint32_t w[64];
        for(int i=0;i<16;i++)
            w[i]=((uint32_t)blk[i*4]<<24)|((uint32_t)blk[i*4+1]<<16)|
                 ((uint32_t)blk[i*4+2]<<8)|blk[i*4+3];
        for(int i=16;i<64;i++)
            w[i]=SIG1(w[i-2])+w[i-7]+SIG0(w[i-15])+w[i-16];
        uint32_t a=h0,b=h1,c=h2,d=h3,e=h4,f=h5,g=h6,hh=h7;
        for(int i=0;i<64;i++){
            uint32_t t1=hh+EP1(e)+CH(e,f,g)+Kh[i]+w[i];
            uint32_t t2=EP0(a)+MAJ(a,b,c);
            hh=g;g=f;f=e;e=(d+t1);d=c;c=b;b=a;a=(t1+t2);
        }
        h0+=a;h1+=b;h2+=c;h3+=d;h4+=e;h5+=f;h6+=g;h7+=hh;
    }
    free(padded);
    uint32_t hv[8]={h0,h1,h2,h3,h4,h5,h6,h7};
    const char hx[]="0123456789abcdef";
    for(int i=0;i<8;i++){
        hex_out[i*8  ]=hx[(hv[i]>>28)&0xF]; hex_out[i*8+1]=hx[(hv[i]>>24)&0xF];
        hex_out[i*8+2]=hx[(hv[i]>>20)&0xF]; hex_out[i*8+3]=hx[(hv[i]>>16)&0xF];
        hex_out[i*8+4]=hx[(hv[i]>>12)&0xF]; hex_out[i*8+5]=hx[(hv[i]>>8 )&0xF];
        hex_out[i*8+6]=hx[(hv[i]>>4 )&0xF]; hex_out[i*8+7]=hx[(hv[i]    )&0xF];
    }
    hex_out[64]='\0';
}

/* ── HTTP with libcurl ───────────────────────────────────────────────────── */

typedef struct { char* data; size_t size; } CurlBuf;

static size_t curl_cb(void* ptr, size_t size, size_t nmemb, void* ud)
{
    CurlBuf* b=(CurlBuf*)ud;
    size_t n=size*nmemb;
    b->data=(char*)realloc(b->data,b->size+n+1);
    memcpy(b->data+b->size,ptr,n);
    b->size+=n;
    b->data[b->size]='\0';
    return n;
}

static int http_get(const char* url, char* out, int outsz)
{
    CURL* c=curl_easy_init(); if(!c) return -1;
    CurlBuf b={strdup(""),0};
    curl_easy_setopt(c,CURLOPT_URL,url);
    curl_easy_setopt(c,CURLOPT_WRITEFUNCTION,curl_cb);
    curl_easy_setopt(c,CURLOPT_WRITEDATA,&b);
    curl_easy_setopt(c,CURLOPT_TIMEOUT,15L);
    CURLcode rc=curl_easy_perform(c);
    curl_easy_cleanup(c);
    if(rc!=CURLE_OK){free(b.data);return -1;}
    strncpy(out,b.data,outsz-1); out[outsz-1]='\0';
    free(b.data); return (int)strlen(out);
}

static int http_post(const char* url, const char* body, char* out, int outsz, long* status)
{
    CURL* c=curl_easy_init(); if(!c) return -1;
    CurlBuf b={strdup(""),0};
    struct curl_slist* h=NULL;
    h=curl_slist_append(h,"Content-Type: application/json");
    curl_easy_setopt(c,CURLOPT_URL,url);
    curl_easy_setopt(c,CURLOPT_POSTFIELDS,body);
    curl_easy_setopt(c,CURLOPT_HTTPHEADER,h);
    curl_easy_setopt(c,CURLOPT_WRITEFUNCTION,curl_cb);
    curl_easy_setopt(c,CURLOPT_WRITEDATA,&b);
    curl_easy_setopt(c,CURLOPT_TIMEOUT,15L);
    CURLcode rc=curl_easy_perform(c);
    if(status) curl_easy_getinfo(c,CURLINFO_RESPONSE_CODE,status);
    curl_slist_free_all(h);
    curl_easy_cleanup(c);
    if(rc!=CURLE_OK){free(b.data);return -1;}
    strncpy(out,b.data,outsz-1); out[outsz-1]='\0';
    free(b.data); return (int)strlen(out);
}

/* ── JSON helpers ────────────────────────────────────────────────────────── */

static int jstr(const char* js, const char* key, char* out, int sz)
{
    char pat[128]; snprintf(pat,sizeof(pat),"\"%s\":",key);
    const char* p=strstr(js,pat); if(!p) return 0;
    p+=strlen(pat); while(*p==' ')p++;
    int i=0;
    if(*p=='"'){p++;while(*p&&*p!='"'&&i<sz-1)out[i++]=*p++;}
    else{while(*p&&((*p>='0'&&*p<='9')||*p=='-')&&i<sz-1)out[i++]=*p++;}
    out[i]='\0'; return i>0;
}

static int jbool(const char* js, const char* key)
{
    char pat[128]; snprintf(pat,sizeof(pat),"\"%s\":",key);
    const char* p=strstr(js,pat); if(!p) return -1;
    p+=strlen(pat); while(*p==' ')p++;
    if(strncmp(p,"true",4)==0) return 1;
    if(strncmp(p,"false",5)==0) return 0;
    return -1;
}

/* ── Work template ───────────────────────────────────────────────────────── */

typedef struct {
    int   block_index;
    char  previous_hash[128];
    int   difficulty;
    int   reward;
    long long timestamp;
    long long last_timestamp;
    char  transactions[512];
} Work;

static int fetch_work(const char* node, Work* w)
{
    char url[256]; snprintf(url,sizeof(url),"%s/mining/work",node);
    char resp[4096]={0};
    if(http_get(url,resp,sizeof(resp))<10) return 0;
    char val[64];
    if(!jstr(resp,"block_index",val,sizeof(val))) return 0;
    w->block_index=atoi(val);
    jstr(resp,"previous_hash",w->previous_hash,sizeof(w->previous_hash));
    jstr(resp,"difficulty",val,sizeof(val)); w->difficulty=atoi(val);
    jstr(resp,"reward",val,sizeof(val)); w->reward=atoi(val);
    jstr(resp,"timestamp",val,sizeof(val)); w->timestamp=atoll(val);
    jstr(resp,"last_timestamp",val,sizeof(val)); w->last_timestamp=atoll(val);
    /* parse transactions array */
    const char* ta=strstr(resp,"\"transactions\":");
    if(ta){
        const char* br=strchr(ta,'[');
        if(br){
            br++; char items[512]={0}; int ilen=0;
            while(*br&&*br!=']'){
                while(*br==' '||*br==','||*br=='\n') br++;
                if(*br=='"'){
                    br++;
                    while(*br&&*br!='"'&&ilen<(int)sizeof(items)-2)
                        items[ilen++]=*br++;
                    if(*br=='"') br++;
                    while(*br==' ') br++;
                    if(*br==',') items[ilen++]=';';
                }
            }
            items[ilen]='\0';
            strncpy(w->transactions,items,sizeof(w->transactions)-1);
        }
    }
    if(!w->transactions[0]) strcpy(w->transactions,"empty-block");
    return 1;
}

typedef struct {
    int   success;
    int   block_index;
    int   reward;
    char  hash[128];
    char  error[256];
    long  http_status;
} SubmitResult;

static void submit_block(const char* node, const Work* w,
                         long long nonce, long long ts,
                         const char* hash_hex, const char* address,
                         SubmitResult* sr)
{
    /* Build transactions JSON array */
    char tx_json[512]; int jpos=0;
    char tx_copy[512]; strncpy(tx_copy,w->transactions,sizeof(tx_copy)-1);
    tx_json[jpos++]='[';
    char* tok=strtok(tx_copy,";"); int first=1;
    while(tok){
        if(!first) tx_json[jpos++]=',';
        tx_json[jpos++]='"';
        while(*tok&&jpos<(int)sizeof(tx_json)-4) tx_json[jpos++]=*tok++;
        tx_json[jpos++]='"'; first=0; tok=strtok(NULL,";");
    }
    tx_json[jpos++]=']'; tx_json[jpos]='\0';

    char body[2048];
    snprintf(body,sizeof(body),
        "{\"block_index\":%d,\"nonce\":%lld,\"hash\":\"%s\","
        "\"miner_address\":\"%s\",\"timestamp\":%lld,\"transactions\":%s}",
        w->block_index,(long long)nonce,hash_hex,address,(long long)ts,tx_json);

    char url[256]; snprintf(url,sizeof(url),"%s/mining/submit",node);
    char resp[2048]={0};
    sr->success=0; sr->error[0]='\0'; sr->http_status=0;
    http_post(url,body,resp,sizeof(resp),&sr->http_status);
    if(jbool(resp,"success")==1){
        sr->success=1;
        char val[64];
        jstr(resp,"block_index",val,sizeof(val)); sr->block_index=atoi(val);
        jstr(resp,"reward",val,sizeof(val)); sr->reward=atoi(val);
        jstr(resp,"hash",sr->hash,sizeof(sr->hash));
    } else {
        jstr(resp,"error",sr->error,sizeof(sr->error));
        if(!sr->error[0]) strncpy(sr->error,resp,sizeof(sr->error)-1);
    }
}

/* ── Timing ──────────────────────────────────────────────────────────────── */

static double get_time()
{
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC,&ts);
    return ts.tv_sec+ts.tv_nsec*1e-9;
}

/* ── Main ────────────────────────────────────────────────────────────────── */

int main(int argc, char** argv)
{
    curl_global_init(CURL_GLOBAL_ALL);

    char address[128]="";
    char node[128]="http://51.91.122.48:8080";
    int  gpu_id=0;
    long long nonce_offset=0;

    for(int i=1;i<argc;i++){
        if(strcmp(argv[i],"--address")==0&&i+1<argc) strncpy(address,argv[++i],sizeof(address)-1);
        else if(strcmp(argv[i],"--node")==0&&i+1<argc) strncpy(node,argv[++i],sizeof(node)-1);
        else if(strcmp(argv[i],"--gpu")==0&&i+1<argc) gpu_id=atoi(argv[++i]);
        else if(strcmp(argv[i],"--nonce-offset")==0&&i+1<argc) nonce_offset=atoll(argv[++i]);
        else if(strncmp(argv[i],"SCO",3)==0) strncpy(address,argv[i],sizeof(address)-1);
    }

    if(!address[0]){
        printf("Usage: %s --address SCO... [--node http://...] [--gpu N]\n",argv[0]);
        return 1;
    }

    /* GPU setup */
    int dev_count=0; cudaGetDeviceCount(&dev_count);
    if(dev_count==0){printf("[ERROR] No CUDA GPU found!\n");return 1;}
    if(gpu_id>=dev_count){printf("[ERROR] GPU %d not found (have %d)\n",gpu_id,dev_count);return 1;}
    cudaSetDevice(gpu_id);
    cudaDeviceProp prop; cudaGetDeviceProperties(&prop,gpu_id);
    printf("\n");
    printf("╔══════════════════════════════════════════════════════╗\n");
    printf("║          Scorbits GPU Miner — Linux                  ║\n");
    printf("╠══════════════════════════════════════════════════════╣\n");
    printf("║  GPU     : %-42s║\n",prop.name);
    printf("║  Address : %-42s║\n",address);
    printf("║  Node    : %-42s║\n",node);
    printf("╚══════════════════════════════════════════════════════╝\n\n");

    /* Kernel config — auto-tune per GPU */
    int tpb=256;
    int bpg=prop.multiProcessorCount*128;
    if(bpg>32768) bpg=32768;
    long long batch=(long long)tpb*bpg;
    printf("[GPU] %d SMs | CUDA %d.%d | Batch: %lld hashes/launch\n\n",
        prop.multiProcessorCount,prop.major,prop.minor,(long long)batch);

    /* GPU buffers */
    char *d_prefix,*d_address,*d_found_hash;
    long long *d_found_nonce,*d_found_ts;
    cudaMalloc(&d_prefix,512);
    cudaMalloc(&d_address,128);
    cudaMalloc(&d_found_nonce,sizeof(long long));
    cudaMalloc(&d_found_ts,sizeof(long long));
    cudaMalloc(&d_found_hash,65);

    long long last_accepted_ts=0;
    long long total_blocks=0;
    long long global_base=nonce_offset;  /* start at offset — no overlap between GPUs */
    double session_start=get_time();

    for(;;){
        /* Fetch work */
        Work w; memset(&w,0,sizeof(w));
        printf("[Work] Fetching...\n");
        if(!fetch_work(node,&w)){
            printf("[Work] Failed — retry in 5s\n");
            sleep(5); continue;
        }
        printf("[Work] Block #%d | diff=%d | reward=%d SCO | lastTs=%lld\n",
            w.block_index,w.difficulty,w.reward,(long long)w.last_timestamp);

        if(w.last_timestamp>last_accepted_ts)
            last_accepted_ts=w.last_timestamp;

        /* Anti-spike: wait until last_ts + 125s */
        long long minTs=last_accepted_ts+125;
        long long now=(long long)time(NULL);
        if(now<minTs){
            printf("[AntiSpike] Waiting %llds...\n",(long long)(minTs-now));
            sleep((int)(minTs-now));
        }

        /* Mine with timestamp = last_ts + 125 so it's valid when submitted */
        long long ts=last_accepted_ts+125;
        now=(long long)time(NULL);
        if(now>ts) ts=now;

        /* Build prefix for GPU: index + ts + txData + prevHash */
        char prefix[512];
        int plen=snprintf(prefix,sizeof(prefix),"%d%lld%s%s",
            w.block_index,(long long)ts,w.transactions,w.previous_hash);
        int alen=(int)strlen(address);

        cudaMemcpy(d_prefix,prefix,plen,cudaMemcpyHostToDevice);
        cudaMemcpy(d_address,address,alen,cudaMemcpyHostToDevice);

        long long h_nonce=-1LL, h_found_ts=0;
        char h_hash[65]={0};
        cudaMemcpy(d_found_nonce,&h_nonce,sizeof(long long),cudaMemcpyHostToDevice);
        cudaMemcpy(d_found_ts,&h_found_ts,sizeof(long long),cudaMemcpyHostToDevice);
        cudaMemset(d_found_hash,0,65);

        printf("[Miner] Mining block #%d (diff=%d)...\n",w.block_index,w.difficulty);

        long long base=global_base;
        long long batch_hashes=0;
        double t0=get_time(),tr=t0,poll_t=t0;
        int found=0;

        for(;;){
            /* Refresh timestamp every second */
            long long new_ts=(long long)time(NULL);
            if(new_ts!=ts){
                ts=new_ts;
                plen=snprintf(prefix,sizeof(prefix),"%d%lld%s%s",
                    w.block_index,ts,w.transactions,w.previous_hash);
                cudaMemcpy(d_prefix,prefix,plen,cudaMemcpyHostToDevice);
            }

            mine_kernel<<<bpg,tpb>>>(d_prefix,plen,d_address,alen,
                base,w.difficulty,d_found_nonce,d_found_ts,ts,d_found_hash);
            cudaDeviceSynchronize();
            batch_hashes+=batch; base+=batch; global_base=base;

            /* Check found */
            cudaMemcpy(&h_nonce,d_found_nonce,sizeof(long long),cudaMemcpyDeviceToHost);
            if(h_nonce!=-1LL){
                cudaMemcpy(h_hash,d_found_hash,64,cudaMemcpyDeviceToHost);
                cudaMemcpy(&h_found_ts,d_found_ts,sizeof(long long),cudaMemcpyDeviceToHost);
                h_hash[64]='\0'; found=1; break;
            }

            double now2=get_time();

            /* Hashrate every 5s */
            if(now2-tr>=5.0){
                double hr=batch_hashes/(now2-t0);
                printf("[GPU] #%d | %.2f MH/s | %lld H\n",
                    w.block_index,hr/1e6,(long long)batch_hashes);
                tr=now2;
            }

            /* Poll for new block every 3s */
            if(now2-poll_t>=3.0){
                poll_t=now2;
                Work fresh; memset(&fresh,0,sizeof(fresh));
                if(fetch_work(node,&fresh)&&fresh.block_index!=w.block_index){
                    printf("[Chain] #%d -> #%d — switching\n",
                        w.block_index,fresh.block_index);
                    w=fresh; break;
                }
            }
        }

        if(!found) continue;

        double el=get_time()-t0;
        printf("[Found!] Block #%d | nonce=%lld | ts=%lld | %.1fs | %.2f MH/s\n",
            w.block_index,(long long)h_nonce,(long long)h_found_ts,el,
            (double)batch_hashes/el/1e6);
        printf("[Hash  ] GPU: %s\n",h_hash);

        /* CPU verify */
        char verify_in[512];
        snprintf(verify_in,sizeof(verify_in),"%d%lld%s%s%lld%s",
            w.block_index,(long long)h_found_ts,
            w.transactions,w.previous_hash,(long long)h_nonce,address);
        char cpu_hash[65]; sha256_cpu(verify_in,cpu_hash);
        printf("[Hash  ] CPU: %s\n",cpu_hash);
        if(strcmp(h_hash,cpu_hash)!=0){
            printf("[ERROR] Hash mismatch — resetting\n");
            h_nonce=-1LL;
            cudaMemcpy(d_found_nonce,&h_nonce,sizeof(long long),cudaMemcpyHostToDevice);
            cudaMemset(d_found_hash,0,65);
            continue;
        }
        printf("[Verify] OK!\n");

        /* Submit with retry */
        double retry_start=get_time();
        for(;;){
            if(get_time()-retry_start>90.0){
                printf("[Submit] Timeout\n"); break;
            }
            SubmitResult sr; memset(&sr,0,sizeof(sr));
            submit_block(node,&w,h_nonce,h_found_ts,h_hash,address,&sr);
            if(sr.success){
                total_blocks++;
                last_accepted_ts=(long long)time(NULL);
                printf("[Accepted] Block #%d | +%d SCO | Session total: %lld\n",
                    sr.block_index,sr.reward,(long long)total_blocks);
                printf("[Stats] Time: %.0fs\n",get_time()-session_start);
                break;
            } else if(sr.http_status==409){
                printf("[Stale] Chain moved\n"); break;
            } else if(sr.http_status==429){
                printf("[RateLimit] Wait 30s\n"); sleep(30); break;
            } else if(strstr(sr.error,"rapide")||strstr(sr.error,"spike")){
                printf("[Rejected] %s\n",sr.error); break;
            } else {
                printf("[Rejected] HTTP=%ld | %s\n",sr.http_status,sr.error); break;
            }
        }

        /* Reset for next block */
        h_nonce=-1LL;
        cudaMemcpy(d_found_nonce,&h_nonce,sizeof(long long),cudaMemcpyHostToDevice);
        cudaMemset(d_found_hash,0,65);
    }

    cudaFree(d_prefix); cudaFree(d_address);
    cudaFree(d_found_nonce); cudaFree(d_found_ts); cudaFree(d_found_hash);
    curl_global_cleanup();
    return 0;
}
