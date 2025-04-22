#include <stdio.h>
#include <iostream>
#include <vector>
#include <opencv2/opencv.hpp>
#include <Eigen/Sparse>
#include <Eigen/Dense>

using namespace cv;
using namespace std;
using namespace Eigen;

typedef SparseMatrix<float> SpMat;
typedef Triplet<float> T;

int imgcnt = 0;
Mat pic[100];
int cum_yshift[100];

int main(int argc, char *argv[])
{
    if(argc != 2)
    {
        fprintf(stderr, "Usage: ./blending_pcg input_directory/\n");
        exit(-1);
    }

    string dir = string(argv[1]);
    if(dir.back() != '/') dir.push_back('/');

    string metadata_nme = dir + "metadata.txt";
    FILE *metadata = fopen(metadata_nme.c_str(), "r");

    imgcnt = 0;
    char fname[100];
    while (~fscanf(metadata, "%s%d", fname, &cum_yshift[imgcnt]))
    {
        pic[imgcnt] = imread(fname);
        if (pic[imgcnt].empty()) {
            cerr << "Failed to read image: " << fname << endl;
            exit(-1);
        }
        imgcnt++;
    }
    fclose(metadata);

    // --- PCG-based refinement of alignment (x-offsets only) ---
    int dof = imgcnt;
    VectorXf xshift(dof);
    xshift.setZero();

    vector<T> triplets;
    vector<float> rvec;
    int eqn = 0;

    // Construct residuals from already good cum_yshift[]
    for (int i = 0; i < imgcnt - 1; i++) {
        triplets.emplace_back(eqn, i, -1);
        triplets.emplace_back(eqn, i + 1, 1);
        rvec.push_back(cum_yshift[i + 1] - cum_yshift[i]);
        eqn++;
    }

    SpMat J(eqn, dof);
    J.setFromTriplets(triplets.begin(), triplets.end());

    VectorXf rhs(eqn);
    for (int i = 0; i < eqn; i++) rhs[i] = rvec[i];

    VectorXf b = -J.transpose() * rhs;
    SpMat A = J.transpose() * J;

    ConjugateGradient<SpMat, Lower|Upper> cg;
    cg.compute(A);
    VectorXf delta = cg.solve(b);

    cout << "CG iterations: " << cg.iterations() << ", error: " << cg.error() << endl;

    // Anchor to first image (shift[0] = 0)
    delta = delta.array() - delta[0];
    for (int i = 0; i < imgcnt; i++)
        cum_yshift[i] = static_cast<int>(round(delta[i]));

    // Ensure all shifts are non-negative
    int min_shift = *min_element(cum_yshift, cum_yshift + imgcnt);
    for (int i = 0; i < imgcnt; i++)
        cum_yshift[i] -= min_shift;

    // Compute canvas size
    int max_width = 0;
    for (int i = 0; i < imgcnt; i++)
        max_width = max(max_width, cum_yshift[i] + pic[i].cols);

    Mat fin(pic[0].rows, max_width, CV_8UC3, Scalar(0,0,0));
    for (int x = 0; x < pic[0].rows; x++)
        for (int y = 0; y < pic[0].cols; y++)
            fin.at<Vec3b>(x, y) = pic[0].at<Vec3b>(x, y);

    for (int i = 1; i < imgcnt; i++)
    {
        int overlap = cum_yshift[i-1] + pic[i-1].cols - cum_yshift[i];
        for (int x = 0; x < pic[i].rows; x++)
        {
            for (int y = overlap; y < pic[i].cols; y++)
            {
                int fx = x, fy = y + cum_yshift[i];
                if (fy < fin.cols)
                    fin.at<Vec3b>(fx, fy) = pic[i].at<Vec3b>(x, y);
            }
        }
    }

    imwrite("panorama_ours.jpg", fin);
    return 0;
}
