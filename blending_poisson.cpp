#include <stdio.h>
#include <iostream>
#include <vector>
#include <opencv2/opencv.hpp>

#define MP make_pair
#define F first
#define S second

using namespace cv;
using namespace std;

bool equalize = 1;
int imgcnt = 0;

Mat pic[100];
int cum_yshift[100];

int dx[4] = {0, 0, 1, -1};
int dy[4] = {1, -1, 0, 0};

int main(int argc, char *argv[])
{
    if(argc != 2)
    {
        fprintf(stderr, "Usage: ./blending_poisson input_directory/\n");
        exit(-1);
    }

    string dir = string(argv[1]);
    if(dir.back() != '/') dir.push_back('/');

    string metadata_nme = dir + "metadata.txt";
    FILE *metadata = fopen(metadata_nme.c_str(), "r");

    imgcnt = 0;
    char fname[100];
    while(~fscanf(metadata, "%s%d", fname, &cum_yshift[imgcnt]))
    {
        pic[imgcnt] = imread(fname);
        if(equalize && imgcnt > 0)
        {
            int overlap = cum_yshift[imgcnt-1] + pic[imgcnt-1].cols - cum_yshift[imgcnt];

            float inten_del = 0, cnt = 0;
            for(int x = 0; x < pic[imgcnt].rows; x++)
            {
                if(x < 2 || x >= pic[imgcnt].rows - 2)
                {
                    for(int y = 0; y < overlap; y++)
                    {
                        Vec3b my_clr = pic[imgcnt].at<Vec3b>(x, y);
                        Vec3b fin_clr = pic[imgcnt-1].at<Vec3b>(x, y + cum_yshift[imgcnt] - cum_yshift[imgcnt-1]);

                        float my_inten = 0.114 * my_clr[0] + 0.587 * my_clr[1] + 0.299 * my_clr[2];
                        float fin_inten = 0.114 * fin_clr[0] + 0.587 * fin_clr[1] + 0.299 * fin_clr[2];

                        inten_del += my_inten - fin_inten;
                        cnt++;
                    }
                }
            }
            inten_del /= cnt;
            pic[imgcnt].convertTo(pic[imgcnt], -1, 1, -inten_del);
        }
        imgcnt++;
    }

    for(int i = 0; i < imgcnt; i++)
        pic[i].convertTo(pic[i], CV_32FC3);

    Mat fin = Mat::zeros(pic[0].rows, cum_yshift[imgcnt-1] + pic[0].cols, CV_32FC3);
    for(int x = 0; x < pic[0].rows; x++)
        for(int y = 0; y < pic[0].cols; y++)
            fin.at<Vec3f>(x, y) = pic[0].at<Vec3f>(x, y);

    for(int i = 1; i < imgcnt; i++)
    {
        int overlap = cum_yshift[i-1] + pic[i-1].cols - cum_yshift[i];
        int T = 10000;

        for(int y = overlap; y < pic[i].cols; y++) {
            fin.at<Vec3f>(0, y + cum_yshift[i]) = pic[i].at<Vec3f>(0, y);
            fin.at<Vec3f>(pic[i].rows - 1, y + cum_yshift[i]) = pic[i].at<Vec3f>(pic[i].rows - 1, y);
        }
        for(int x = 0; x < pic[i].rows; x++)
            fin.at<Vec3f>(x, pic[i].cols - 1 + cum_yshift[i]) = pic[i].at<Vec3f>(x, pic[i].cols - 1);

        Mat x_mat = Mat::zeros(pic[i].rows, pic[i].cols, CV_32FC3);
        Mat r = Mat::zeros(pic[i].rows, pic[i].cols, CV_32FC3);
        Mat p = Mat::zeros(pic[i].rows, pic[i].cols, CV_32FC3);
        Mat Ap = Mat::zeros(pic[i].rows, pic[i].cols, CV_32FC3);
        Vec3f M_inv(0.25f, 0.25f, 0.25f);

        // Residual and initial search direction
        for(int x = 1; x < pic[i].rows - 1; x++) {
            for(int y = overlap; y < pic[i].cols - 1; y++) {
                Vec3f b(0, 0, 0);
                for(int k = 0; k < 4; k++) {
                    int nx = x + dx[k];
                    int ny = y + dy[k];
                    b += pic[i].at<Vec3f>(x, y) - pic[i].at<Vec3f>(nx, ny);
                    b += fin.at<Vec3f>(nx, ny + cum_yshift[i]) - fin.at<Vec3f>(x, y + cum_yshift[i]);
                }
                r.at<Vec3f>(x, y) = b;
                p.at<Vec3f>(x, y) = b.mul(M_inv);
            }
        }

        float init_error = 0;
        for(int t = 0; t < T; t++)
        {
            for(int x = 1; x < pic[i].rows - 1; x++)
                for(int y = overlap; y < pic[i].cols - 1; y++) {
                    Vec3f v = 4 * p.at<Vec3f>(x, y);
                    for(int k = 0; k < 4; k++) {
                        int nx = x + dx[k];
                        int ny = y + dy[k];
                        v -= p.at<Vec3f>(nx, ny);
                    }
                    Ap.at<Vec3f>(x, y) = v;
                }

            Vec3f rz(0, 0, 0), pAp(0, 0, 0);
            for(int x = 1; x < pic[i].rows - 1; x++)
                for(int y = overlap; y < pic[i].cols - 1; y++) {
                    Vec3f rp = r.at<Vec3f>(x, y).mul(M_inv);
                    rz += rp.mul(r.at<Vec3f>(x, y));
                    pAp += p.at<Vec3f>(x, y).mul(Ap.at<Vec3f>(x, y));
                }
            Vec3f alpha; divide(rz, pAp, alpha);

            for(int x = 1; x < pic[i].rows - 1; x++)
                for(int y = overlap; y < pic[i].cols - 1; y++)
                    x_mat.at<Vec3f>(x, y) += alpha.mul(p.at<Vec3f>(x, y));

            for(int x = 1; x < pic[i].rows - 1; x++)
                for(int y = overlap; y < pic[i].cols - 1; y++)
                    r.at<Vec3f>(x, y) -= alpha.mul(Ap.at<Vec3f>(x, y));

            Vec3f rnorm2(0, 0, 0);
            for(int x = 1; x < pic[i].rows - 1; x++)
                for(int y = overlap; y < pic[i].cols - 1; y++)
                    rnorm2 += r.at<Vec3f>(x, y).mul(r.at<Vec3f>(x, y));
            if(t == 0) init_error = max({rnorm2[0], rnorm2[1], rnorm2[2]});
            printf("Iter %d: Error = %.6f %.6f %.6f\n", t, rnorm2[0], rnorm2[1], rnorm2[2]);
            if (max({rnorm2[0], rnorm2[1], rnorm2[2]}) < 1e-4 * init_error) break;

            Vec3f rz_new(0, 0, 0);
            for(int x = 1; x < pic[i].rows - 1; x++)
                for(int y = overlap; y < pic[i].cols - 1; y++)
                    rz_new += r.at<Vec3f>(x, y).mul(M_inv).mul(r.at<Vec3f>(x, y));
            Vec3f beta; divide(rz_new, rz, beta);
            for(int x = 1; x < pic[i].rows - 1; x++)
                for(int y = overlap; y < pic[i].cols - 1; y++) {
                    Vec3f z = r.at<Vec3f>(x, y).mul(M_inv);
                    p.at<Vec3f>(x, y) = z + beta.mul(p.at<Vec3f>(x, y));
                }
        }

        for(int x = 1; x < pic[i].rows - 1; x++)
            for(int y = overlap; y < pic[i].cols - 1; y++)
                fin.at<Vec3f>(x, y + cum_yshift[i]) = x_mat.at<Vec3f>(x, y);
    }

    Mat out;
    fin.convertTo(out, CV_8UC3);
    imwrite("panorama.jpg", out);
    return 0;
}
