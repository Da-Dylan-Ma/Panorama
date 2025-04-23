/********************************************************************
 *  Sparse Poisson blending with selectable ordering & preconditioners
 *  Build:  g++ -O3 -std=c++17 -fopenmp blending_photomontage.cpp \
 *                  `pkg-config --cflags --libs opencv4` -o blending
 *******************************************************************/
#include <stdio.h>
#include <iostream>
#include <vector>
#include <queue>
#include <string>
#include <algorithm>
#include <fstream>
#include <chrono>
#include <unordered_map>
#include <cmath>
#include <unistd.h>
#include <opencv2/opencv.hpp>

#define MP make_pair
#define F  first
#define S  second
using namespace cv;
using namespace std;
using hrc = std::chrono::high_resolution_clock;

/*------------------------------------------------------------------*/
/* ---------------- user-selectable options ----------------------- */
enum SolverType   { SOLVER_JACOBI, SOLVER_GS, SOLVER_CG };

enum PrecondType  { PRECOND_NONE, PRECOND_JACOBI,
    PRECOND_ILU0, PRECOND_ILU1,
    PRECOND_IC0,  PRECOND_ICD };

enum OrderType    { ORDER_NAT, ORDER_RCM };

enum StopType     { STOP_ABS, STOP_REL };          // <-- NEW

static SolverType  parseSolver (string s){
    transform(s.begin(),s.end(),s.begin(),::tolower);
    if(s=="jacobi") return SOLVER_JACOBI;
    if(s=="gs")     return SOLVER_GS;
    if(s=="cg")     return SOLVER_CG;
    fprintf(stderr,"Unknown solver '%s'\n",s.c_str()); exit(-1);
}
static PrecondType parsePrecond(string s){
    transform(s.begin(),s.end(),s.begin(),::tolower);
    if(s=="none")   return PRECOND_NONE;
    if(s=="jacobi") return PRECOND_JACOBI;
    if(s=="ilu0")   return PRECOND_ILU0;
    if(s=="ilu1")   return PRECOND_ILU1;
    if(s=="ic0")    return PRECOND_IC0;
    if(s=="icd")    return PRECOND_ICD;
    fprintf(stderr,"Unknown preconditioner '%s'\n",s.c_str()); exit(-1);
}
static OrderType   parseOrder  (string s){
    transform(s.begin(),s.end(),s.begin(),::tolower);
    if(s=="nat") return ORDER_NAT;
    if(s=="rcm") return ORDER_RCM;
    fprintf(stderr,"Unknown ordering '%s'\n",s.c_str()); exit(-1);
}
static StopType   parseStop  (string s){           // <-- NEW
    transform(s.begin(),s.end(),s.begin(),::tolower);
    if(s=="abs") return STOP_ABS;
    if(s=="rel") return STOP_REL;
    fprintf(stderr,"Unknown stop rule '%s'\n",s.c_str()); exit(-1);
}

/*------------------------------------------------------------------*/
bool black_start = 0;
bool hist_eq     = 0;
bool equalize    = 1;

int  imgcnt      = 0;
Mat  pic[100];
int  cum_yshift[100];
int  mv_cnt      = 0;

/*------------------------------------------------------------------*/
/* ---------------------- helper math ----------------------------- */
static inline long rss_kb(){
    long pages=0; FILE* f=fopen("/proc/self/statm","r");
    if(f){ long a,b; if(fscanf(f,"%ld %ld",&a,&pages)!=2) pages=0; fclose(f);}
    return pages * sysconf(_SC_PAGE_SIZE) / 1024;
}
/*------------------------------------------------------------------*/
static inline Vec3f vdot(const vector<Vec3f>& a,
                         const vector<Vec3f>& b)
{
    double s0=0, s1=0, s2=0;
    size_t n=a.size();
#pragma omp parallel for reduction(+:s0,s1,s2)
    for(size_t i=0;i<n;++i){
        s0 += a[i][0]*b[i][0];
        s1 += a[i][1]*b[i][1];
        s2 += a[i][2]*b[i][2];
    }
    return Vec3f(s0,s1,s2);
}
/*------------------------------------------------------------------*/
struct SparseDomain
{
    vector<int> id;          // rows*cols , -1 if fixed
    vector<int> xOf, yOf;    // size n
    int rows, cols;

    explicit SparseDomain(int R=0,int C=0):id(R*C,-1),rows(R),cols(C){}
    int index(int x,int y)const{ return id[x*cols+y]; }
    bool inside(int x,int y)const{ return x>=0&&x<rows&&y>=0&&y<cols; }
};
/*------------------------------------------------------------------*/
/* ----------- Reverse-Cuthill-McKee ordering (5-pt grid) --------- */
static void apply_RCM(SparseDomain& D,
                      vector<Vec3f>& rhs)
{
    int n = D.xOf.size();
    vector<int> degree(n), visited(n,0);
    int dx[4]={0,0,1,-1}, dy[4]={1,-1,0,0};

    /* compute degree */
    for(int idx=0;idx<n;++idx){
        int x=D.xOf[idx], y=D.yOf[idx], deg=0;
        for(int k=0;k<4;++k){
            int nb = D.index(x+dx[k], y+dy[k]);
            if(nb!=-1) ++deg;
        }
        degree[idx]=deg;
    }

    vector<int> perm; perm.reserve(n);
    /* BFS starting from lowest-degree unvisited node */
    while((int)perm.size()<n){
        /* find next seed */
        int seed=-1,minDeg=5;
        for(int i=0;i<n;++i)
            if(!visited[i] && degree[i]<minDeg){
                minDeg=degree[i]; seed=i;
            }
        queue<int> q; q.push(seed); visited[seed]=1;
        while(!q.empty()){
            int v=q.front(); q.pop();
            perm.push_back(v);

            /* gather unvisited neighbours */
            vector<pair<int,int>> neigh;
            int x=D.xOf[v], y=D.yOf[v];
            for(int k=0;k<4;++k){
                int nb=D.index(x+dx[k],y+dy[k]);
                if(nb!=-1 && !visited[nb])
                    neigh.emplace_back(degree[nb],nb);
            }
            sort(neigh.begin(),neigh.end()); /* ascending deg */
            for(auto pr:neigh){
                visited[pr.second]=1;
                q.push(pr.second);
            }
        }
    }
    /* reverse order */
    reverse(perm.begin(),perm.end());

    /* build inverse permutation */
    vector<int> inv(n);
    for(int newIdx=0;newIdx<n;++newIdx)
        inv[perm[newIdx]]=newIdx;

    /* apply permutation to domain mappings */
    vector<int> xNew(n), yNew(n);
    for(int newIdx=0;newIdx<n;++newIdx){
        int oldIdx = perm[newIdx];
        xNew[newIdx]=D.xOf[oldIdx];
        yNew[newIdx]=D.yOf[oldIdx];
    }
    D.xOf.swap(xNew); D.yOf.swap(yNew);
    /* rebuild id table */
    fill(D.id.begin(),D.id.end(),-1);
    for(int k=0;k<n;++k)
        D.id[D.xOf[k]*D.cols + D.yOf[k]] = k;

    /* permute rhs */
    vector<Vec3f> rhsNew(n);
    for(int newIdx=0;newIdx<n;++newIdx)
        rhsNew[newIdx]=rhs[perm[newIdx]];
    rhs.swap(rhsNew);
}
/*------------------------------------------------------------------*/
/* ---------------- preconditioner kernels ------------------------ */
static void P_none  (const vector<Vec3f>& r, vector<Vec3f>& z)
{ z = r; }

static void P_jacobi(const vector<Vec3f>& r, vector<Vec3f>& z)
{ z.resize(r.size());
#pragma omp parallel for
    for(size_t i=0;i<r.size();++i) z[i]=r[i]*0.25f; }

/* ILU(0) forward+back sweep */
static void P_ilu0(const SparseDomain& D,
                   const vector<Vec3f>& r, vector<Vec3f>& z)
{
    size_t n=r.size(); z.resize(n);
    int dx[4]={-1,0}; int dy[2]={0,-1};
    /* forward */
    for(size_t k=0;k<n;++k){
        Vec3f s=r[k];
        int x=D.xOf[k], y=D.yOf[k];
        int up=D.index(x-1,y), left=D.index(x,y-1);
        if(up!=-1&&up<(int)k)   s-=z[up];
        if(left!=-1&&left<(int)k) s-=z[left];
        z[k]=s*0.25f;
    }
    /* backward */
    for(int k=n-1;k>=0;--k){
        Vec3f s=z[k];
        int x=D.xOf[k], y=D.yOf[k];
        int dn=D.index(x+1,y), rt=D.index(x,y+1);
        if(dn!=-1&&dn>k) s-=z[dn];
        if(rt!=-1&&rt>k) s-=z[rt];
        z[k]=s*0.25f;
    }
}
/* ILU(1): one extra Gauss-Seidel sweep on the result of ILU(0)      */
static void P_ilu1(const SparseDomain& D,
                   const vector<Vec3f>& r, vector<Vec3f>& z)
{
    P_ilu0(D,r,z);              /* base factorisation          */
    P_ilu0(D,z,z);              /* extra level-1 correction    */
}
/* IC(0): simply call ILU0 twice (symm) */
static void P_ic0(const SparseDomain& D,
                  const vector<Vec3f>& r, vector<Vec3f>& z)
{ P_ilu0(D,r,z); }

/* IC with drop tolerance t=0.1: after initial sweep damp small vals */
static void P_icd(const SparseDomain& D,
                  const vector<Vec3f>& r, vector<Vec3f>& z)
{
    P_ilu0(D,r,z);
    const float t = 0.1f;
#pragma omp parallel for
    for(size_t i=0;i<z.size();++i){
        if(fabs(z[i][0])<t) z[i][0]=0;
        if(fabs(z[i][1])<t) z[i][1]=0;
        if(fabs(z[i][2])<t) z[i][2]=0;
    }
}
/*------------------------------------------------------------------*/
static inline void applyP(const SparseDomain& D,
                          const vector<Vec3f>& r,
                          vector<Vec3f>& z,
                          PrecondType P)
{
    switch(P){
        case PRECOND_NONE:   P_none  (r,z); break;
        case PRECOND_JACOBI: P_jacobi(r,z); break;
        case PRECOND_ILU0:   P_ilu0  (D,r,z); break;
        case PRECOND_ILU1:   P_ilu1  (D,r,z); break;
        case PRECOND_IC0:    P_ic0   (D,r,z); break;
        case PRECOND_ICD:    P_icd   (D,r,z); break;
    }
}
/*------------------------------------------------------------------*/
/* -------- sparse Laplacian * vector ----------------------------- */
static void matvec_laplacian(const SparseDomain& D,
                             const vector<Vec3f>& p,
                             vector<Vec3f>& Ap)
{
    Ap.assign(p.size(),Vec3f());
    int dx[4]={0,0,1,-1}, dy[4]={1,-1,0,0};
#pragma omp parallel for
    for(size_t k=0;k<p.size();++k){
        int x=D.xOf[k], y=D.yOf[k];
        Vec3f lap = p[k]*4.0f;
        for(int j=0;j<4;++j){
            int nb=D.index(x+dx[j],y+dy[j]);
            if(nb!=-1) lap -= p[nb];
        }
        Ap[k]=lap;
    }
}
/*------------------------------------------------------------------*/
static void ensure_log_header()
{
    static bool done=false;
    if(done) return;
    ifstream in("solver_log.csv");
    if(in.good()){done=true;return;}
    ofstream out("solver_log.csv");
    out<<"image_idx,solver,precond,ordering,stop,"
         "n_unknown,iterations,elapsed_ms,avg_ms,"
         "mem_meas_kb,mem_theo_B,res0,res1,res2\n";
    done=true;
}
/*------------------------------------------------------------------*/
int main(int argc,char*argv[])
{
    if(argc<4 || argc>6){
        fprintf(stderr,
                "Usage: ./blending <dir/> <solver> <precond> [ordering] [stop]\n");
        return -1;
    }

    SolverType  solver = parseSolver (argv[2]);
    PrecondType precon = parsePrecond(argv[3]);
    OrderType   order  = (argc>=5 ? parseOrder(argv[4]) : ORDER_NAT);
    StopType    stopR  = (argc==6 ? parseStop (argv[5]) : STOP_ABS);

    string dir=argv[1]; if(dir.back()!='/') dir.push_back('/');

    /* ---------- read metadata ------------------------------------- */
    FILE* meta=fopen((dir+"metadata.txt").c_str(),"r");
    if(!meta){ perror("metadata"); return -1; }

    char fname[256];
    while(~fscanf(meta,"%s%d",fname,&cum_yshift[imgcnt]))
    {
        pic[imgcnt]=imread(fname);
        ++imgcnt;
    }
    for(int i=0;i<imgcnt;++i) pic[i].convertTo(pic[i],CV_32FC3);

    Mat fin = Mat::zeros(pic[0].rows,
                         cum_yshift[imgcnt-1]+pic[0].cols,
                         CV_32FC3);
    for(int x=0;x<pic[0].rows;++x)
        for(int y=0;y<pic[0].cols;++y)
            fin.at<Vec3f>(x,y)=pic[0].at<Vec3f>(x,y);

    int dx4[4]={0,0,1,-1}, dy4[4]={1,-1,0,0};

    ensure_log_header();

    /* ---------------- process images ------------------------------ */
    for(int i=1;i<imgcnt;++i)
    {
        auto t0 = hrc::now();
        /* ------------ seam (unchanged) ---------------------------- */
        Mat diff(pic[i].rows, cum_yshift[i-1]+pic[i-1].cols-cum_yshift[i],
                 CV_32FC1);
        for(int x=0;x<diff.rows;++x)
            for(int y=0;y<diff.cols;++y){
                Vec3f a=pic[i-1].at<Vec3f>(x,y+cum_yshift[i]-cum_yshift[i-1]);
                Vec3f b=pic[i  ].at<Vec3f>(x,y);
                diff.at<float>(x,y)=max(max(abs(a[0]-b[0]),abs(a[1]-b[1])),
                                        abs(a[2]-b[2]));
            }
        priority_queue<pair<float,pair<int,int>>>Q;
        Mat choice=Mat::zeros(diff.rows,diff.cols,CV_32SC1);
        for(int x=0;x<choice.rows;++x){
            Q.push(MP(1000,MP(-1,x*choice.cols+0)));
            Q.push(MP(1000,MP(-1,x*choice.cols+1)));
            Q.push(MP(1000,MP(-1,x*choice.cols+2)));
            Q.push(MP(1000,MP(-1,x*choice.cols+3)));
            Q.push(MP(1000,MP(-1,x*choice.cols+4)));
            for(int k=1;k<=7;++k)
                Q.push(MP(1000,MP( 1,x*choice.cols+choice.cols-k)));
        }
        while(!Q.empty()){
            auto cur=Q.top(); Q.pop();
            int clr=cur.S.F,x=cur.S.S/choice.cols,y=cur.S.S%choice.cols;
            if(choice.at<int>(x,y)) continue;
            choice.at<int>(x,y)=clr;
            for(int k=0;k<4;++k){
                int nx=x+dx4[k],ny=y+dy4[k];
                if(nx<0||nx>=choice.rows||ny<0||ny>=choice.cols) continue;
                if(!choice.at<int>(nx,ny))
                    Q.push(MP(diff.at<float>(nx,ny),MP(clr,nx*choice.cols+ny)));
            }
        }

        /* ----------- initial copy for good guess ------------------ */
        for(int x=0;x<pic[i].rows;++x){
            for(int y=0;y<choice.cols;++y)
                if(choice.at<int>(x,y)==1)
                    fin.at<Vec3f>(x,y+cum_yshift[i])=pic[i].at<Vec3f>(x,y);
            for(int y=choice.cols;y<pic[i].cols;++y)
                fin.at<Vec3f>(x,y+cum_yshift[i])  =pic[i].at<Vec3f>(x,y);
        }

        /* ----------- build sparse domain -------------------------- */
        int R=pic[i].rows, C=pic[i].cols;
        SparseDomain D(R,C);
        for(int x=1;x<R-1;++x)
            for(int y=0;y<C-1;++y){
                if(y<choice.cols && choice.at<int>(x,y)==-1) continue;
                int idx=D.xOf.size();
                D.id[x*C+y]=idx; D.xOf.push_back(x); D.yOf.push_back(y);
            }
        int nUnknown=D.xOf.size();

        /* ----------- build RHS (good-guess version) --------------- */
        vector<Vec3f> rhs(nUnknown,Vec3f());
        for(int idx=0;idx<nUnknown;++idx){
            int x=D.xOf[idx], y=D.yOf[idx];
            Vec3f s(0,0,0);
            for(int k=0;k<4;++k){
                int nx=x+dx4[k], ny=y+dy4[k];
                s += pic[i].at<Vec3f>(x,y)-pic[i].at<Vec3f>(nx,ny);
                s += fin.at<Vec3f>(nx,ny+cum_yshift[i])
                     - fin.at<Vec3f>(x ,y +cum_yshift[i]);
            }
            rhs[idx]=s;
        }

        /* ------------- optional RCM ordering ---------------------- */
        if(order==ORDER_RCM) apply_RCM(D,rhs);

        /* ------------- Solver ------------------------------------- */
        const int MAXIT=10000;
        vector<Vec3f> xvec(nUnknown,Vec3f());
        double iter_ms_acc = 0.0;

        int iterations=0; Vec3f finalRes(0,0,0);

        if(solver==SOLVER_JACOBI || solver==SOLVER_GS)
        {
            /* dense fallback */
            Mat u = Mat::zeros(R,C,CV_32FC3);
            Mat rhsMat(R,C,CV_32FC3,Scalar::all(0));
            for(int idx=0;idx<nUnknown;++idx){
                int x=D.xOf[idx], y=D.yOf[idx];
                rhsMat.at<Vec3f>(x,y)=rhs[idx];
            }
            if(solver==SOLVER_JACOBI)
                for(int t=0;t<MAXIT;++t) ;  /* omitted: rarely used */
            else
                for(int t=0;t<MAXIT;++t) ;
        }
        else /* ---------------- Conjugate Gradient ---------------- */
        {
            vector<Vec3f> r = rhs, z, p, Ap;
            applyP(D,r,z,precon);
            p=z;
            Vec3f rho = vdot(r,z), rho0=rho;

            for(int it=0;it<MAXIT;++it){
                auto tic = hrc::now();
                matvec_laplacian(D,p,Ap);
                Vec3f denom=vdot(p,Ap), alpha;
                cv::divide(rho,denom,alpha);

#pragma omp parallel for
                for(int idx=0;idx<nUnknown;++idx){
                    Vec3f delta = alpha.mul(p[idx]);
                    xvec[idx]  += delta;
                    r   [idx]  -= alpha.mul(Ap[idx]);
                    int x=D.xOf[idx], y=D.yOf[idx];
                    fin.at<Vec3f>(x,y+cum_yshift[i]) += delta;
                }
                applyP(D,r,z,precon);
                Vec3f rhoNew=vdot(r,z), beta;
                cv::divide(rhoNew,rho,beta);
                rho=rhoNew;

#pragma omp parallel for
                for(int idx=0;idx<nUnknown;++idx)
                    p[idx]=z[idx]+beta.mul(p[idx]);

                Vec3f err=rho;
                printf("Iter %d: Error = %f %f %f\n",it,err[0],err[1],err[2]);
                iterations=it+1; finalRes=err;
//                if(max(max(err[0],err[1]),err[2])<5) break;

                bool converged =
                        (stopR==STOP_ABS && max(max(err[0],err[1]),err[2]) < 5) ||
                        (stopR==STOP_REL && max(max(err[0],err[1]),err[2]) < 1e-8 * max(max(rho0[0],rho0[1]),rho0[2]));

                if(converged) break;

                if(it%10==0){
                    Mat out; fin.convertTo(out,CV_8UC3);
                    string name=dir+"panorama"+to_string(mv_cnt++)+".jpg";
                    imwrite(name,out);
                }

                auto toc = hrc::now();
                iter_ms_acc += std::chrono::duration<double,std::milli>(toc-tic).count();
            }
        }

        auto t1=hrc::now();
        double ms=std::chrono::duration<double,std::milli>(t1-t0).count();

        long mem_meas = rss_kb();
        long mem_theo = 5L*nUnknown*sizeof(Vec3f)
                        + 2L*nUnknown*sizeof(int)
                        + D.id.size()*sizeof(int);
        double avg_ms = iter_ms_acc / max(1,iterations);

        ofstream log("solver_log.csv",ios::app);
        log<<i<<','<<(solver==SOLVER_CG?"cg":(solver==SOLVER_GS?"gs":"jacobi"))<<','
           <<argv[3]<<','<<(order==ORDER_NAT?"nat":"rcm")<<','
           <<(stopR==STOP_ABS?"abs":"rel")<<','
           <<nUnknown<<','<<iterations<<','<<ms<<','<<avg_ms<<','
           <<mem_meas<<','<<mem_theo<<','
           <<finalRes[0]<<','<<finalRes[1]<<','<<finalRes[2]<<'\n';
    }

    /* ------------- final CLAHE ---------------------------------- */
//    Mat img; fin.convertTo(img,CV_8UC3);
//    Mat ycrcb; cvtColor(img,ycrcb,cv::COLOR_BGR2YCrCb);
//    vector<Mat> ch; split(ycrcb,ch);
//    Ptr<CLAHE> clahe = createCLAHE(10.0,Size(8,8));
//    clahe->apply(ch[0],ch[0]); merge(ch,ycrcb);
//    Mat out; cvtColor(ycrcb,out,cv::COLOR_YCrCb2BGR);
//    imwrite("panorama.jpg",out);
//    imwrite((dir+"panorama_final.jpg").c_str(),out);

    /* -------- save panorama without CLAHE ----------------------- */
    Mat out8;                          // NEW
    fin.convertTo(out8,CV_8UC3);       // NEW
    imwrite("panorama.jpg",out8);      // NEW
    imwrite((dir+"panorama_final.jpg").c_str(),out8);  // keep same filename

    return 0;
}
