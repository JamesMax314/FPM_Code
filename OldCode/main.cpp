#include <iostream>
#include "optim.h"
#include <numeric>
#include <vector>
#include <opencv2/opencv.hpp>
#include <complex>
#include <pybind11/complex.h>
#include <pybind11/stl.h>
#include <ctime>
#include <pybind11/numpy.h>
//#include <opencv2/core/core.hpp>
#include <stdexcept>
#include <string>
#include <cassert>
#include <opencv2/highgui/highgui.hpp>
//#include "translate.h"
#include "methods.h"
#include "translate.h"
#include "casters.h"
#include "calibrate.h"
#include "library.h"
#include "phasecorr.h"

using namespace cv;
using namespace std;
using namespace concurrency;

cv::Mat test(const Mat& in1, const Mat& in2){
    //Mat out = read_image(".", "sample1.jpg");

    Mat out = in1.clone();//complex_mult(in1, in2);
    /*
    clock_t time_req;
    time_req = clock();
    mulSpectrums(in1, in2, out, 0);
    time_req = clock() - time_req;
    cout << "mulSpectrum: " << (float)time_req/CLOCKS_PER_SEC << " seconds" << endl;
    time_req = clock();
    out = complex_mult(in1, in2);
    time_req = clock() - time_req;
    cout << "complex_mult: " << (float)time_req/CLOCKS_PER_SEC << " seconds" << endl;
    time_req = clock();
    out = pad(out, 1000, 1000);
    time_req = clock() - time_req;
    cout << "pad: " << (float)time_req/CLOCKS_PER_SEC << " seconds" << endl;
     */
    clock_t time_req;
    time_req = clock();
    Mat slice[3];
    split(out, slice);
    time_req = clock() - time_req;
    cout << "c++: " << (double)time_req/CLOCKS_PER_SEC << " seconds" << endl;
    return slice[1];

    if (!out.isContinuous()) {
        out = out.clone();
    }
    return out;
}

void test1(Matrix& mat){
    cout << mat({0,0,0,0}) << endl;
}

Mat testTrans(Mat& mat, int dx, int dy){
    Mat out = translate(mat, dx, dy);
    return out;
}

Mat fft(const Mat& arr){
    clock_t time_req;
    time_req = clock();
    Mat out = arr.clone();
    dft(arr, out);
    //fct_fftshift(out);
    time_req = clock() - time_req;
    cout << "fft: " << (double)time_req/CLOCKS_PER_SEC << " seconds" << endl;
    //Mat temp = abs(out);
    return out;
}

Mat ifft(const Mat& arr){
    Mat out = arr.clone();
    idft(arr, out, cv::DFT_SCALE);
    return out;
}

Mat sqrtTest(const Mat& arr){
    Mat in;
    arr.convertTo(in, CV_32F);
    Mat out = in.clone();
    //sqrt1(in.data);
    sqrt(in, out);
    return out;
}

Mat absTest(Mat& arr){
    clock_t time_req;
    time_req = clock();
    parallel_for_(Range(0,arr.size[0]*arr.size[1]), abs1(arr.data));
    time_req = clock() - time_req;
    cout << "absTest: " << (double)time_req/CLOCKS_PER_SEC << " seconds" << endl;
    time_req = clock();
    abs4(arr);
    time_req = clock() - time_req;
    cout << "abs4: " << (double)time_req/CLOCKS_PER_SEC << " seconds" << endl;
    //cout << out.at<double>(0, 0) << endl;
    /*
    for (int i=0; i<arr.size[0]; i++){
        for (int j=0; j<arr.size[1]; j++) {
            out.at<double>(i, j) = sqrt(pow(arr.at<complex<float>>(i, j).real(), 2) +
                    pow(arr.at<complex<float>>(i, j).imag(), 2));
        }
    }
    */
    return arr;
}

Mat conjTest(Mat arr1){
    clock_t time_req;
    time_req = clock();
    parallel_for_( Range(0, arr1.size[0]*arr1.size[1]), conjug(arr1.data));
    time_req = clock() - time_req;
    cout << "Conj: " << (double)time_req/CLOCKS_PER_SEC << " seconds" << endl;
    time_req = clock();
    conjug4(arr1);
    time_req = clock() - time_req;
    cout << "conjug4: " << (double)time_req/CLOCKS_PER_SEC << " seconds" << endl;
    return arr1;
}

void testRead(const string& img){
    Mat img1 = read_image(".", img);
    Mat slice[3];
    split(img1, slice);
    vector<int> col = {2, 1, 0};
    imshow("R", slice[col[0]]/255);
    imshow("G", slice[col[1]]/255);
    imshow("B", slice[col[2]]/255);
    waitKey(0);
}

Mat testDiv(Mat& in1, Mat& in2){
    Mat out = in1.clone();
    clock_t time_req;
    time_req = clock();
    parallel_for_( Range(0, out.size[0]*out.size[1]), div1(in1.data, in2.data, out.data));
    time_req = clock() - time_req;
    cout << "div1: " << (double)time_req/CLOCKS_PER_SEC << " seconds" << endl;
    time_req = clock();
    div2(in1, in2, out);
    time_req = clock() - time_req;
    cout << "div2: " << (double)time_req/CLOCKS_PER_SEC << " seconds" << endl;
    time_req = clock();
    div4(in1, in2, out);
    time_req = clock() - time_req;
    cout << "div3: " << (double)time_req/CLOCKS_PER_SEC << " seconds" << endl;
    //divide(in1, in2, out);
    return out;
}


Mat updateS_1(Mat& P, Mat& PHI, Mat& PSI, vector<int> coords) {
    // Returns the update step for S Note that the fitting parameter alpha is not included
    Mat DELTA = PHI - PSI;
    Mat DELTATrans = translate(DELTA, - coords[0], -coords[1]);
    Mat PTrans = translate(P, -coords[0], -coords[1]);
    Mat temp = PTrans.clone();

    //parallel_for_( Range(0, temp.size[0]*temp.size[1]), conjug(temp.data));
    conjug4(temp);
    Mat numerator = DELTATrans.clone();
    mulSpectrums(temp, DELTATrans, numerator, 0);

    double min, max;
    //parallel_for_(Range(0, PTrans.size[0]*PTrans.size[1]), abs1(PTrans.data));
    abs4(PTrans);
    minMaxIdx(PTrans, &min, &max);
    double denominator = pow(max, 2);

    Mat frac = numerator / denominator;
    return frac;
}

Mat updateP_1(Mat& S, Mat& PHI, Mat& PSI, vector<int> coords) {
    // Returns the update step for S Note that the fitting parameter alpha is not included
    Mat DELTA = PHI - PSI;
    Mat STrans = translate(S, coords[0], coords[1]);
    Mat temp = STrans.clone();
    //parallel_for_( Range(0, temp.size[0]*temp.size[1]), conjug(temp.data));
    conjug4(temp);
    Mat numerator = DELTA.clone();
    mulSpectrums(temp, DELTA, numerator, 0);
    double min, max;
    //parallel_for_(Range(0, STrans.size[0]*STrans.size[1]), abs1(STrans.data));
    abs4(STrans);
    minMaxIdx(abs(STrans), &min, &max);
    double denominator = pow(max, 2);
    Mat frac = numerator / denominator;
    return frac;
}

/*
void mainFPM(subSys section, const subImage& subImg, const int& numImgs){
    //clock_t time_req0;
    //time_req0 = clock();
    Mat split_img[3];
    split(subImg.image, split_img);
    Mat split_bayer[3];
    split(section.bayer, split_bayer);
    Mat split_invBayer[3];
    split(section.bayer, split_invBayer);
    //cout << "main-1: " << (float)(time_req0-clock())/CLOCKS_PER_SEC << " seconds" << endl;
    //parallel_for (int(0), 3, [&](int i) {
    for (int i = 0; i < 3; i++) {
        //clock_t time_req;
        //time_req = clock();
        Mat img = split_img[i];
        Mat S = section.S[i].clone();
        Mat P = section.P[i].clone();
        Mat bayer;
        split_bayer[i].convertTo(bayer, CV_32F);
        Mat invBayer;
        split_invBayer[i].convertTo(invBayer, CV_32F);
        vector<int> LED = subImg.LEDPos;
        cout << "S: " << S.at<complex<float>>(200, 200) << endl;
        cout << "P: " << P.at<complex<float>>(200, 200) << endl;


        int dx = (int)section.coords({i, LED[0] + int(pow(numImgs, 0.5)/2 - 1), LED[1] +
                                      int(pow(numImgs, 0.5)/2 - 1), 0}) - (int)section.subSize[0] / 2;
        int dy = (int)section.coords({i, LED[0] + int(pow(numImgs, 0.5)/2 - 1), LED[1] +
                                      int(pow(numImgs, 0.5)/2 - 1), 1}) - (int)section.subSize[0] / 2;
        //cout << "main0: " << (float)(time_req-clock())/CLOCKS_PER_SEC << " seconds" << endl;


        //clock_t time_req1;
        //time_req1 = clock();
        Mat STrans = translate(S, dx, dy);
        Mat SCrop = crop(STrans, section.subSize);
        Mat phi = P.clone();
        mulSpectrums(P, SCrop, phi, 0);
        Mat PHI = phi.clone();
        fct_fftshift(phi);
        idft(phi, PHI, cv::DFT_REAL_OUTPUT);
        fct_ifftshift(PHI);

        double min, max;
        minMaxIdx(phi, &min, &max);
        cout << "STrans: " << STrans.at<complex<float>>(200, 200) << endl;
        cout << "SCrop: " << SCrop.at<complex<float>>(200, 200) << endl;
        cout << "phi: " << phi.at<complex<float>>(200, 200) << endl;
        cout << "phi: " << min << ", " << max <<  endl;
        cout << "PHI: " << PHI.at<float>(200, 200) << endl;


        //Mat img1;
        //img.convertTo(img1, CV_32F);
        Mat amplitude = img.clone();
        sqrt(img.mul(bayer), amplitude);
        cout << "amplitude: " << amplitude.at<float>(200, 200) << endl;
        cout << "img: " << img.at<float>(200, 200) << endl;
        img.mul(bayer);
        cout << "imgbayer: " << img.at<float>(200, 200) << endl;
        //cout << "main1: " << (float)(time_req1-clock())/CLOCKS_PER_SEC << " seconds" << endl;
        //invBayer.convertTo(invBayer, CV_32FC2);


        //clock_t time_req2;
        //time_req2 = clock();
        Mat temp = PHI.clone();
        mulSpectrums(PHI, invBayer, temp, 0);
        Mat a = abs(temp);
        Mat bayerCorrectedAmp = amplitude + a;
        bayerCorrectedAmp.convertTo(bayerCorrectedAmp, CV_32FC2);
        cout << "bayercorramp: " << bayerCorrectedAmp.at<float>(200, 200) << endl;
        //cout << "main2: " << (float)(time_req2-clock())/CLOCKS_PER_SEC << " seconds" << endl;


        //clock_t time_req3;
        //time_req3 = clock();
        Mat PPNum = PHI.clone();
        mulSpectrums(bayerCorrectedAmp, PHI, PPNum, 0);
        Mat PPDen = abs(PHI);
        Mat PHIPrimed;
        divide(PPNum, PPDen, PHIPrimed);
        //cout << "main3: " << (float)(time_req3-clock())/CLOCKS_PER_SEC << " seconds" << endl;

        //clock_t time_req4;
        //time_req4 = clock();
        Mat phiPrimed;
        fct_ifftshift(PHIPrimed);
        dft(PHIPrimed, phiPrimed, cv::DFT_COMPLEX_OUTPUT);
        fct_fftshift(phiPrimed);
        //cout << "main4: " << (float)(time_req4-clock())/CLOCKS_PER_SEC << " seconds" << endl;


        //clock_t time_req5;
        //time_req5 = clock();
        Mat updateStepS = updateS_1(P, phiPrimed, phi, {dx, dy});
        cout << "update: " << updateStepS.at<complex<float>>(200, 200).real() << endl;
        S = S + pad(updateStepS, section.bigSize[0], section.bigSize[1]);

        Mat SCrop1 = crop(S, section.subSize);
        Mat updateStepP = updateP_1(SCrop1, phiPrimed, phi, {dx, dy});
        P = P + updateStepP;

        section.S[i] = S;
        section.P[i] = P;

        //cout << "main5: " << (float)(time_req5-clock())/CLOCKS_PER_SEC << " seconds" << endl;
    }//);
}*/

/*
void mainFPM1(subSys section, const subImage& subImg, const int& numImgs){
    Mat split_img[3];
    split(subImg.image, split_img);
    Mat split_bayer[3];
    split(section.bayer, split_bayer);
    Mat split_invBayer[3];
    split(section.bayer, split_invBayer);
    //cout << "main-1: " << (float)(time_req0-clock())/CLOCKS_PER_SEC << " seconds" << endl;
    //parallel_for (int(0), 3, [&](int i) {
    for (int i = 0; i < 3; i++) {
        Mat img = split_img[i];
        Mat S = section.S[i].clone();
        Mat P = section.P[i].clone();
        Mat bayer;
        split_bayer[i].convertTo(bayer, CV_32FC1);
        Mat invBayer;

        split_invBayer[i].convertTo(split_invBayer[i], CV_32FC1);
        Mat ivBtmp[] = {split_invBayer[i], Mat::zeros(split_invBayer[i].size(), CV_32FC1)};
        merge(ivBtmp, 2, invBayer);
        vector<int> LED = subImg.LEDPos;

        int dx = (int) section.coords({i, LED[0] + int(pow(numImgs, 0.5) / 2 - 1), LED[1] +
                                                                                   int(pow(numImgs, 0.5) / 2 - 1), 0}) -
                 (int) section.subSize[0] / 2;
        int dy = (int) section.coords({i, LED[0] + int(pow(numImgs, 0.5) / 2 - 1), LED[1] +
                                                                                   int(pow(numImgs, 0.5) / 2 - 1), 1}) -
                 (int) section.subSize[0] / 2;

        Mat STrans = translate(S, dx, dy);
        Mat SCrop = crop(STrans, section.subSize);
        Mat phi = P.clone();
        mulSpectrums(P, SCrop, phi, 0);
        Mat PHI = phi.clone();
        fct_fftshift(phi);
        idft(phi, PHI, cv::DFT_COMPLEX_INPUT | cv::DFT_COMPLEX_OUTPUT);
        fct_ifftshift(PHI);

        Mat amplitude = img.clone();
        sqrt(img.mul(bayer), amplitude);


        Mat temp = PHI.clone();
        mulSpectrums(PHI, invBayer, temp, 0);
        Mat a = abs(temp);
        Mat ampComp;
        Mat aCtmp[] = {amplitude, Mat::zeros(amplitude.size(), CV_32FC1)};
        merge(aCtmp, 2, ampComp);
        Mat bayerCorrectedAmp = ampComp + a;

        Mat PPNum = PHI.clone();
        mulSpectrums(bayerCorrectedAmp, PHI, PPNum, 0);
        Mat PPDen = abs(PHI);
        Mat PHIPrimed;
        divide(PPNum, PPDen, PHIPrimed);

        Mat phiPrimed;
        fct_ifftshift(PHIPrimed);
        dft(PHIPrimed, phiPrimed, cv::DFT_COMPLEX_OUTPUT | cv::DFT_COMPLEX_INPUT);
        fct_fftshift(phiPrimed);


        Mat updateStepS = updateS_1(P, phiPrimed, phi, {dx, dy});
        S = S + pad(updateStepS, section.bigSize[0], section.bigSize[1]);
        Mat splitS1[2];
        split(S, splitS1);
        cout << pad(updateStepS, section.bigSize[0], section.bigSize[1]).at<complex<float>>(200, 200) << endl;

        Mat SCrop1 = crop(S, section.subSize);
        Mat updateStepP = updateP_1(SCrop1, phiPrimed, phi, {dx, dy});
        P = P + updateStepP;

        section.S[i] = S;
        section.P[i] = P;
    }
}*/

void mainFPM2(subSys section, const subImage& subImg, const int& numImgs){
    /// Split images into three colour channels
    Mat split_img[3];
    split(subImg.image, split_img);
    Mat split_bayer[3];
    split(section.bayer, split_bayer);
    Mat split_invBayer[3];
    split(section.invBayer, split_invBayer);
    vector<int> col = {2, 1, 0};

    //for (int ind=0; ind<3; ind++) {
    parallel_for (int(0), 3, [&](int ind) {
        int i = col[ind];

        Mat img;
        split_img[i].convertTo(img, CV_32FC1);
        Mat S = section.S[i].clone();
        Mat P = section.P[i].clone();
        //cout << "1: " << S.at<complex<float>>(150, 150) << endl;

        /// Extract bayer matrix corresponding to ith colour channel
        Mat bayer;
        split_bayer[ind].convertTo(bayer, CV_32FC1); /// might be incorrect i
        //cout << "Bayer: " << bayer.at<float>(0, 0) << bayer.at<float>(0, 1) << bayer.at<float>(1, 0)
        //     << bayer.at<float>(1, 1) << endl;
        Mat invBayer;

        /// Convert invBayer to complex matrix
        split_invBayer[ind].convertTo(split_invBayer[ind], CV_32FC1);
        Mat ivBtmp[] = {split_invBayer[ind], Mat::zeros(split_invBayer[ind].size(), CV_32FC1)};
        merge(ivBtmp, 2, invBayer);
        //cout << "invBayer: " << invBayer.at<complex<float>>(0, 0) << invBayer.at<complex<float>>(0, 1)
        //     << invBayer.at<complex<float>>(1, 0) << invBayer.at<complex<float>>(1, 1) << endl;


        /// Extract pixel coordinates
        vector<int> LED = subImg.LEDPos;
        int dx = (int) section.coords({ind, LED[0] + int(pow(numImgs, 0.5) / 2 - 1), LED[1] +
                                                                                     int(pow(numImgs, 0.5) / 2 - 1),
                                       0}) - (int) section.subSize[0] / 2;
        int dy = (int) section.coords({ind, LED[0] + int(pow(numImgs, 0.5) / 2 - 1), LED[1] +
                                                                                     int(pow(numImgs, 0.5) / 2 - 1),
                                       1}) - (int) section.subSize[0] / 2;


        //cout << "LED: " << LED[0] << ", " << LED[1] << endl;
        //cout << "LEDPos: " << LED[0] + int(pow(numImgs, 0.5) / 2 - 1) <<
        //     ", " << LED[1] + int(pow(numImgs, 0.5) / 2 - 1) << endl;
        //cout << "dx, dy: " <<
        //     section.coords({i, LED[0] + int(pow(numImgs, 0.5) / 2 - 1), LED[1] +
        //                                                                 int(pow(numImgs, 0.5) / 2 - 1), 0}) << ", " <<
        //     section.coords({i, LED[0] + int(pow(numImgs, 0.5) / 2 - 1), LED[1] +
        //                                                                 int(pow(numImgs, 0.5) / 2 - 1), 1}) << endl;
        //cout << "dx, dy act: " << dx << ", " << dy << endl;


        /// P(u)S(u-Un)
        Mat STrans = translate(S, dx, dy);
        Mat SCrop = crop(STrans, section.subSize);
        Mat phi = P.clone();
        mulSpectrums(P, SCrop, phi, 0);
        //cout << "2: " << phi.at<complex<float>>(200, 200) << endl;

        /// F^-1(phi(u))
        Mat PHI = phi.clone();
        fct_fftshift(PHI);
        idft(phi, PHI, cv::DFT_COMPLEX_INPUT | cv::DFT_COMPLEX_OUTPUT | cv::DFT_SCALE);
        //PHI = PHI;
        fct_ifftshift(PHI);
        //cout << "3: " << PHI.at<complex<float>>(200, 200) << endl;

        /// amplitude = sqrt(img*bayer) bayer correction
        Mat amplitude = img.clone();
        sqrt(img.mul(bayer), amplitude); //
        //cout << "ok" << endl;

        /// Bayer corrected amplitude
        Mat a = PHI.clone();
        mulSpectrums(PHI, invBayer, a, 0);
        //parallel_for_(Range(0, a.size[0] * a.size[1]), abs1(a.data));
        abs4(a);
        Mat ampComp;
        // Make amplitude complex
        Mat aCtmp[] = {amplitude, Mat::zeros(amplitude.size(), CV_32FC1)};
        merge(aCtmp, 2, ampComp);
        Mat bayerCorrectedAmp = ampComp + a;
        //cout << "5: " << bayerCorrectedAmp.at<complex<float>>(200, 200) << endl;

        /// Impose intensity constraints
        Mat PPNum = PHI.clone();
        mulSpectrums(bayerCorrectedAmp, PHI, PPNum, 0);
        Mat PPDen = PHI.clone();
        //parallel_for_(Range(0, PPDen.size[0] * PPDen.size[1]), abs1(PPDen.data));
        abs4(PPDen);
        Mat PHIPrimed = PPNum.clone();
        //parallel_for_(Range(0, PPNum.size[0] * PPNum.size[1]),
        //              div1(PPNum.data, PPDen.data, PHIPrimed.data));
        div4(PPNum, PPDen, PHIPrimed);
        //divide(PPNum, PPDen, PHIPrimed);
        //cout << "6: " << PPDen.at<complex<float>>(200, 200) << endl;

        /// phi'(u) = F(PHI'(r))
        Mat phiPrimed;
        fct_ifftshift(PHIPrimed);
        dft(PHIPrimed, phiPrimed, cv::DFT_COMPLEX_OUTPUT | cv::DFT_COMPLEX_INPUT);
        //phiPrimed = phiPrimed;
        //fct_fftshift(phiPrimed);
        //cout << "7: " << phiPrimed.at<complex<float>>(200, 200) << endl;

        /// Update step S
        /*
        Mat absS = S.clone();
        parallel_for_(Range(0, absS.size[0]*absS.size[1]), abs1(absS.data));
        Mat sl[2];
        split(absS, sl);
        double mi, ma;
        minMaxIdx(sl[0], &mi, &ma);
        cout << "Max var: " << ma << endl;*/

        Mat updateStepS = updateS_1(P, phiPrimed, phi, {dx, dy});
        S = S + pad(updateStepS, section.bigSize[0], section.bigSize[1]);
        Mat splitS1[2];
        split(S, splitS1);
        //cout << "8: " << updateStepS.at<complex<float>>(200, 200) << endl;

        Mat absS = phiPrimed.clone();
        //parallel_for_(Range(0, absS.size[0] * absS.size[1]), abs1(absS.data));
        abs4(absS);
        Mat sl[2];
        split(absS, sl);
        //double mi, ma;
        //minMaxIdx(sl[0], &mi, &ma);
        //cout << "Max phiPrimed: " << ma << endl;

        /*
        absS = updateStepS.clone();
        parallel_for_(Range(0, absS.size[0]*absS.size[1]), abs1(absS.data));
        split(absS, sl);
        minMaxIdx(sl[0], &mi, &ma);
        cout << "Max var: " << ma << endl;*/

        //cout << pad(updateStepS, section.bigSize[0], section.bigSize[1]).at<complex<float>>(200, 200) << endl;

        /// Update step P
        Mat SCrop1 = crop(S, section.subSize);
        Mat updateStepP = updateP_1(SCrop1, phiPrimed, phi, {dx, dy});
        P = P + updateStepP;

        /// Update matrices
        S.copyTo(section.S[i]);
        //section.S[i] = S;
        P.copyTo(section.P[i]);
        //section.P[i] = P;
    });
}

void meanFFT(fullSys FPMSys){
    clock_t time_req;
    time_req = clock();
    string dir = FPMSys.strParams["dir"];
    vector<string> images = FPMSys.images;
    int numSplits = stoi(FPMSys.strParams["numFiles"]);
    int divisor = stoi(FPMSys.strParams["divisor"]);
    int numImgs = stoi(FPMSys.strParams["numImgs"]);
    vector<int> col = {2, 1, 0};
    for (int i0=0; i0 < numImgs; i0++) {
    //parallel_for(int(0), numImgs, [&](int i0) {
        string imgName = images[i0];
        splitImage subImgs = splitImage(dir, imgName, numSplits, divisor);
        parallel_for(int(0), numSplits, [&](int j) {
        //for (int j=0; j < numSplits; j++) {
            parallel_for(int(0), numSplits, [&](int k) {
            //for (int k=0; k < numSplits; k++) {
                vector<int> subPos{j, k};
                subImage subImg = subImgs.subImg[subPos];
                Mat split_img[3];
                split(subImg.image, split_img);
                parallel_for(int(0), 3, [&](int ind) {
                    int i = col[ind];
                    Mat img;
                    split_img[i].convertTo(img, CV_32FC1);
                    Mat FT;
                    dft(img, FT, cv::DFT_COMPLEX_OUTPUT);
                    fct_ifftshift(FT);
                    abs4(FT);
                    FPMSys.sub[subPos].meanFFT[i] = FPMSys.sub[subPos].meanFFT[i] * FPMSys.sub[subPos].num;
                    FPMSys.sub[subPos].meanFFT[i] = FPMSys.sub[subPos].meanFFT[i] + FT;
                    FPMSys.sub[subPos].meanFFT[i] = FPMSys.sub[subPos].meanFFT[i] / FPMSys.sub[subPos].num;
                });
                FPMSys.sub[subPos].num += 1;
            });
        });
    }//);
    time_req = clock() - time_req;
    cout << "meanFFT: " << (double)time_req/CLOCKS_PER_SEC << " seconds" << endl;
}

vector<vector<int>> E(Mat I, const int& dx , const int& dy, const int& Rad) {
    /// ~~~ Adds up values in a circle ~~~
    int scale = 1;
    int delta = 0;
    int ddepth = CV_32FC1;
    Mat grad_x, grad_y, grad, EMat, circKernal; /// Init variables
    /// Sobel gradient computation in both dimensions
    Sobel(I, grad_x, ddepth, 1, 0, 3, scale, delta, BORDER_DEFAULT);
    Sobel(I, grad_y, ddepth, 0, 1, 3, scale, delta, BORDER_DEFAULT);
    /// Combine grad elements
    addWeighted(grad_x, 0.5, grad_y, 0.5, 0, grad);
    /// Init circle circumfrance kernal
    circKernal = Mat::zeros(2*Rad, 2*Rad, CV_32FC1);
    circle(circKernal, Point(Rad, Rad), Rad, Scalar(1), 1, 8);
    /// Convolves afformentioned circle kernal with gradient image
    filter2D(grad, EMat, ddepth , circKernal, Point(-1, -1), delta, BORDER_DEFAULT);
    imshow("conv", EMat);
    waitKey(0);
    /// find stdDev in EMat value
    float stdD = stdDev(EMat);
    /// find max point i.e. centre of circle
    double min, max;
    minMaxIdx(EMat, &min, &max);
    /// good points are within 0.1 stdDevs of max point
    vector<vector<int>> goodPoints;
    /// for each x pixel in image do ...
    for (int i=0; i<I.size[0]; i++){
        /// for each y pixel in image do ...
        for (int j=0; j<I.size[1]; j++){
            /// if point sattisfies goodness cryterion
            if (max - EMat.at<float>(Point(i, j)) < 0.1 * stdD){
                /// append good pixel coords to goodPoints
                goodPoints.push_back(vector<int>{i, j});
                cout << "approx centre: " << i << ", " << j << endl;
            }
        } /// ~~~ end of y for
    } /// ~~~ end of x for
    return goodPoints;
}

int E1(Mat I, const int& dx , const int& dy, const int& Rad) {
    int scale = 1;
    int delta = 0;
    int ddepth = CV_32FC1;
    Mat grad_x, grad_y, grad, EMat, circKernal;
    Sobel(I, grad_x, ddepth, 1, 0, 3, scale, delta, BORDER_DEFAULT);
    Sobel(I, grad_y, ddepth, 0, 1, 3, scale, delta, BORDER_DEFAULT);
    addWeighted(grad_x, 0.5, grad_y, 0.5, 0, grad);
    Rect source = Rect(I.size[0]/2-Rad+dx, I.size[1]/2-Rad+dy, 2*Rad, 2*Rad);
    Mat smallI = Mat::zeros(2*Rad, 2*Rad, CV_32FC1);
    I(source).copyTo(smallI);
    vector<float> sumOut;
    vector<int> rads;
    int rad = Rad - int(Rad/2);
    for (int i=0; i<int(Rad); i++){
        rad += 1;
        circKernal = Mat::zeros(2*rad, 2*rad, CV_32FC1);
        circle(circKernal, Point(rad, rad), rad, Scalar(1), 1, 8);
        Mat out = I * circKernal;
        sumOut.push_back(sumMat(out));
        rads.push_back(rad);
    }
    double min, max;
    minMaxIdx(sumOut, &min, &max);
    auto ptrTmp = find(sumOut.begin(), sumOut.end(), max);
    int index = int(ptrTmp - sumOut.begin());
    int outRad = rads[index];
    return outRad;
}

void calibR(fullSys FPMSys){
    cout << "callib R" << endl;
    string dir = FPMSys.strParams["dir"];
    vector<string> images = FPMSys.images;
    int numSplits = stoi(FPMSys.strParams["numFiles"]);
    int divisor = stoi(FPMSys.strParams["divisor"]);
    int numImgs = stoi(FPMSys.strParams["numImgs"]);
    vector<int> col = {2, 1, 0};
    int sigma = 2;
    for (int j=0; j < numSplits; j++) {
        //parallel_for(int(0), numSplits, [&](int j) {
        for (int k = 0; k < numSplits; k++) {
            //parallel_for(int(0), numSplits, [&](int k) {
            vector<int> subPos{j, k};
            vector<double> avRad(3);
            int num = 0;
            for (int i0 = 0; i0 < numImgs; i0++) {
                //parallel_for(int(0), numImgs, [&](int i0) {
                string imgName = images[i0];
                splitImage subImgs = splitImage(dir, imgName, numSplits, divisor);
                vector<int> subPos{j, k};
                Matrix BF = FPMSys.sub[subPos].isBF;
                subImage subImg = subImgs.subImg[subPos];
                vector<int> LED = subImg.LEDPos;
                vector<double> BFs = {BF({0, LED[0] + int(pow(numImgs, 0.5) / 2 - 1),
                                          LED[1] + int(pow(numImgs, 0.5) / 2 - 1)}),
                                      BF({1, LED[0] + int(pow(numImgs, 0.5) / 2 - 1),
                                          LED[1] + int(pow(numImgs, 0.5) / 2 - 1)}),
                                      BF({2, LED[0] + int(pow(numImgs, 0.5) / 2 - 1),
                                          LED[1] + int(pow(numImgs, 0.5) / 2 - 1)})};
                if (find(BFs.begin(), BFs.end(), 1) != BFs.end() == 1) {
                    Mat split_img[3];
                    split(subImg.image, split_img);
                    for (int ind = 0; ind < 3; ind++) {
                        int i = col[ind];
                        vector<int> coords;
                        coords.push_back(FPMSys.sub[subPos].coords({i,
                                                                    LED[0] + int(pow(numImgs, 0.5) / 2 - 1),
                                                                    LED[1] + int(pow(numImgs, 0.5) / 2 - 1),
                                                                    0}));
                        coords.push_back(FPMSys.sub[subPos].coords({i,
                                                                    LED[0] + int(pow(numImgs, 0.5) / 2 - 1),
                                                                    LED[1] + int(pow(numImgs, 0.5) / 2 - 1),
                                                                    1}));
                        Mat img;
                        split_img[i].convertTo(img, CV_32FC1);
                        Mat FT;
                        dft(img, FT, cv::DFT_COMPLEX_OUTPUT);
                        fct_ifftshift(FT);
                        abs4(FT);
                        Mat ITilda = FT / FPMSys.sub[subPos].meanFFT[i];
                        Mat ITildaSlice[2];
                        split(ITilda, ITildaSlice);
                        Mat If = ITildaSlice[0];
                        GaussianBlur(If, If, Size(3,3),
                                     sigma, sigma, BORDER_DEFAULT);
                        double rad = E1(If, coords[0], coords[1], FPMSys.sub[subPos].fRApprox[i]);
                        avRad[i] = (avRad[i]*(num-1) + rad)/num;
                        num += 1;
                    }
                }
            }
            for (int i=0; i<3; i++) {
                FPMSys.sub[subPos].fRApprox[i] = int(avRad[i]);
            }
        }
    }
}

double BrightField(fullSys FPMSys){
    cout << "BrightField" << endl;
    /// Get data from class
    string dir = FPMSys.strParams["dir"];
    vector<string> images = FPMSys.images;
    int numSplits = stoi(FPMSys.strParams["numFiles"]);
    int divisor = stoi(FPMSys.strParams["divisor"]);
    int numImgs = stoi(FPMSys.strParams["numImgs"]);
    vector<int> col = {2, 1, 0};
    int sigma = 2;
    /// For each sub image do ...
    for (int j=0; j < numSplits; j++) {
        //parallel_for(int(0), numSplits, [&](int j) {
        /// For each sub image do ...
        for (int k=0; k < numSplits; k++) {
            /// Get vector describing which sub image is being processed
            vector<int> subPos{j, k};
            //parallel_for(int(0), numSplits, [&](int k) {
            /// For each image do ...
            for (int i0=0; i0 < numImgs; i0++) {
                cout << "i0: " << i0 << endl;
                //parallel_for(int(0), numImgs, [&](int i0) {
                string imgName = images[i0];
                splitImage subImgs = splitImage(dir, imgName, numSplits, divisor);
                Matrix BF = FPMSys.sub[subPos].isBF; /// Loads brightfield data for sub image for watch LED
                subImage subImg = subImgs.subImg[subPos]; /// Load the estimate image
                vector<int> LED = subImg.LEDPos; /// LED as given in file name
                /// Extracts Bf data for specific LED (indexed from 0, hence the long translation)
                vector<double> BFs = {BF({0, LED[0] + int(pow(numImgs, 0.5) / 2 - 1),
                                       LED[1] + int(pow(numImgs, 0.5) / 2 - 1)}),
                                   BF({1, LED[0] + int(pow(numImgs, 0.5) / 2 - 1),
                                       LED[1] + int(pow(numImgs, 0.5) / 2 - 1)}),
                                   BF({2, LED[0] + int(pow(numImgs, 0.5) / 2 - 1),
                                       LED[1] + int(pow(numImgs, 0.5) / 2 - 1)})};
                /// checks if any colour is a BF image
                if (find(BFs.begin(), BFs.end(), 1) != BFs.end() == 1){
                    cout << "is BF" << endl;
                    /// Splits into channels
                    Mat split_img[3];
                    split(subImg.image, split_img);
                    /// Initializes variable for fitting
                    vector<pair<input_vector, input_vector>> tempDatCol;
                    /// for each colour do ...
                    for (int ind=0; ind<3; ind++) {
                    //parallel_for(int(0), 3, [&](int ind) {
                        int i = col[ind]; /// Corrects for openCV reading as BGR not RGB
                        cout << "colour: " << i << endl;
                        /// cast image to appropriate dType
                        Mat img;
                        split_img[i].convertTo(img, CV_32FC1);
                        /// FT image
                        Mat FT;
                        dft(img, FT, cv::DFT_COMPLEX_OUTPUT);
                        fct_ifftshift(FT);
                        Mat Ii = FT.clone()
                        /// Find magnitude
                        abs4(FT);
                        /// Normalize with division by meanFFT
                        Mat ITilda = FT / FPMSys.sub[subPos].meanFFT[i];
                        /// Cast to a real matrix by disregarding complex part
                        Mat ITildaSlice[2];
                        split(ITilda, ITildaSlice);
                        Mat If = ITildaSlice[0];
                        /// Convolve Gaussian Kernal to smooth FT
                        GaussianBlur(If, If, Size(3,3),
                                sigma, sigma, BORDER_DEFAULT );
                        cout << "k1s" << endl;
                        /// Take 1st derivative and sum around perimetor
                        vector<vector<int>> k1s = E(If, 1, 1, FPMSys.sub[subPos].fRApprox[i]);
                        cout << "k2s" << endl;
                        /// Same but with second derivative
                        vector<vector<int>> k2s = E(If, 2, 2, FPMSys.sub[subPos].fRApprox[i]+sigma);
                        vector<vector<int>> k3s;
                        cout << "ok1" << endl;
                        /// for length of k1s do ...
                        for (int n=0; n<k1s.size(); n++){
                            /// Assigns first agreeing element from k2s to compare
                            auto compare = finder(k2s, k1s[n], {10, 10});
                            if (compare.first == 1){
                                /// if agreement addd to k3s
                                k3s.push_back(k2s[compare.second]);
                            }
                        }
                        cout << "ok2" << endl;
                        //Mat Ii;
                        //dft(img, Ii, cv::DFT_COMPLEX_OUTPUT);
                        //fct_fftshift(Ii);
                        double sBig = 0;
                        vector<int> k4;
                        cout << "ok3" << endl;
                        cout << k3s.size() << endl;
                        /// ~~~ Finds translation that maximises agreement with reality ~~~
                        /// For length of k3s do ...
                        for (int n=0; n<k3s.size(); n++){
                            cout << k3s[0][0] << ", " << k3s[0][1] << endl;
                            /// Translate a copy of the pupil function to new centre point
                            Mat PTrans = FPMSys.sub[subPos].P[i].clone();
                            PTrans = translate(PTrans, k3s[n][0], k3s[n][1]);
                            /// Multiply P by FT of image
                            Mat SP;
                            mulSpectrums(Ii, PTrans, SP, 0);
                            /// Inverse FT
                            Mat IiP;
                            idft(SP, IiP);
                            fct_ifftshift(IiP);
                            /// Absolute difference between image and filtered image
                            Mat abs = Ii - IiP;
                            abs4(abs);
                            double s = sum(abs)[0];
                            if (sBig < s){
                                sBig = s;
                                k4 = k3s[n];
                                cout << k4[0] << endl;
                            }
                        }
                        /// ~~~ Finds translation that maximises agreement with reality ~~~
                        cout << k4[0] << endl;
                        cout << "ok4" << endl;
                        //*FPMSys.sub[subPos].coords.at({ind, LED[0] + int(pow(numImgs, 0.5) / 2 - 1), LED[1] +
                        //int(pow(numImgs, 0.5) / 2 - 1), 0}) = k4[0];
                        //*FPMSys.sub[subPos].coords.at({ind, LED[0] + int(pow(numImgs, 0.5) / 2 - 1), LED[1] +
                        //int(pow(numImgs, 0.5) / 2 - 1), 0}) = k4[1];
                        /// Gets origional F-plane coordinates for given LED
                        vector<double> origCoords = {FPMSys.sub[subPos].coords(
                                {ind, LED[0] + int(pow(numImgs, 0.5) / 2 - 1),
                                 LED[1] + int(pow(numImgs, 0.5) / 2 - 1), 0}),
                                                  FPMSys.sub[subPos].coords(
                                                          {ind, LED[0] + int(pow(numImgs, 0.5) / 2 - 1),
                                                           LED[1] + int(pow(numImgs, 0.5) / 2 - 1), 1})};
                        cout << "ok5" << endl;

                        /// Casting to type for fiting
                        auto origCoords1 = vecToInVec(origCoords);
                        cout << "ok6" << endl;
                        auto k41 = vecToInVec(k4);
                        cout << "ok7" << endl;
                        /// Temp data contains one input - output data pair for fitting
                        auto tempDat = pair<input_vector, input_vector>(origCoords1, k41);
                        cout << "ok8" << endl;
                        tempDatCol.push_back(tempDat); /// colour specific

                    }//); /// ~~~ end of colour loop

                    /// Appends new triad of data pairs (3 colours)
                    FPMSys.sub[subPos].dataPairs.push_back(tempDatCol);
                    /// Appends image index, i.e. information about LED
                    FPMSys.sub[subPos].dPairsIndex.push_back(i0);
                } /// ~~~ end of BF loop
            }//); /// ~~~ end of image loop

            /// For each colour do ...
            for (int ind=0; ind<3; ind++){
                int i = col[ind]; /// correct openCV BGR
                /// Robustly fit transformation
                auto ran = RANSAC1(FPMSys.sub[subPos].dataPairs[i]);
                /// for each image do ...
                for (int i0=0; i0 < numImgs; i0++) {
                    vector<int> LED = get_LED(images[i0]);
                    /// If index has a data pair
                    if (find(FPMSys.sub[subPos].dPairsIndex.begin(), FPMSys.sub[subPos].dPairsIndex.end(), i0)
                    != FPMSys.sub[subPos].dPairsIndex.end() == 1){
                        /// Get location in dPairsIndex of i0
                        auto posInDataPs = find(FPMSys.sub[subPos].dPairsIndex.begin(),
                                FPMSys.sub[subPos].dPairsIndex.end(), i0);
                        /// convert from pointer to index
                        int posIndex = int(posInDataPs - FPMSys.sub[subPos].dPairsIndex.begin());
                        /// if inlier
                        if (ran.second[posIndex]){
                            /// Update x, y coordinates of LED offset
                            FPMSys.sub[subPos].coords.update({i, LED[0] + int(pow(numImgs, 0.5) / 2 - 1),
                                                              LED[1] + int(pow(numImgs, 0.5) / 2 - 1), 0},
                                                             inVecToVec(FPMSys.sub[subPos].dataPairs[i0][i].first)[0]);
                            FPMSys.sub[subPos].coords.update({i, LED[0] + int(pow(numImgs, 0.5) / 2 - 1),
                                                              LED[1] + int(pow(numImgs, 0.5) / 2 - 1), 1},
                                                             inVecToVec(FPMSys.sub[subPos].dataPairs[i0][i].first)[1]);
                        }
                    } else{ /// if outlier or not BF
                        /// ~~~ update coords based on transformation from origional coords ~~~
                        /// Get origional coords
                        auto indat = FPMSys.sub[subPos].dataPairs[i0][i].first;
                        /// Apply transformation
                        auto coords = model(indat, ran.first);
                        /// Update in object
                        FPMSys.sub[subPos].coords.update({i, LED[0] + int(pow(numImgs, 0.5) / 2 - 1),
                                                          LED[1] + int(pow(numImgs, 0.5) / 2 - 1), 0},
                                                         coords[0]);
                        FPMSys.sub[subPos].coords.update({i, LED[0] + int(pow(numImgs, 0.5) / 2 - 1),
                                                          LED[1] + int(pow(numImgs, 0.5) / 2 - 1), 1},
                                                         coords[1]);
                        /// ~~~ update coords based on transformation from origional coords ~~~
                    } /// ~~~ end of is inlier
                } /// ~~~ end of image loop
            } /// ~~~ end of colour loop
        }//); /// ~~~ end of sub image loop j
    }//); /// end of sub image loop i
    return 0;
}

void FPM_Feed(fullSys FPMSys) {
    string dir = FPMSys.strParams["dir"];
    vector<string> images = FPMSys.images;
    int numSplits = stoi(FPMSys.strParams["numFiles"]);
    int divisor = stoi(FPMSys.strParams["divisor"]);
    int numImgs = stoi(FPMSys.strParams["numImgs"]);
    //meanFFT(FPMSys);

    //imgIndex = arrangeImages(images);

    //tStart = time.time();
    //parallel_for (int(0), numImgs, [&](int i) {
    for (int i0=0; i0 < numImgs; i0++) {
        string imgName = images[i0];
        splitImage subImgs = splitImage(dir, imgName, numSplits, divisor);
        //parallel_for (int(0), numSplits, [&](int j) {
        for (int j=0; j < numSplits; j++) {
            //parallel_for (int(0), numSplits, [&](int k) {
            for (int k=0; k < numSplits; k++) {
                vector<int> subPos{j, k};
                subSys section = FPMSys.sub[subPos];
                subImage subImg = subImgs.subImg[subPos];
                //clock_t time_req;
                //time_req = clock();
                //Mat imgOut;
                //idft(section.S[2], imgOut, cv::DFT_REAL_OUTPUT);
                //fct_fftshift(imgOut);
                //normalize(imgOut, imgOut, 255,0);
                //Mat iii;
                //imshow("imgOut", imgOut / 1000000);
                //mainFPM(section, subImg, numImgs);
                //cout << imgOut.at<float>(200, 200) << endl;

                /****************************************************************

                /// Split images into three colour channels
                Mat split_img[3];
                split(subImg.image, split_img);
                Mat split_bayer[3];
                split(section.bayer, split_bayer);
                Mat split_invBayer[3];
                split(section.invBayer, split_invBayer);
                vector<int> col = {2, 1, 0};

                int ind = 0;
                int i = col[ind];

                Mat img;
                split_img[i].convertTo(img, CV_32FC1);
                Mat S = section.S[i].clone();
                Mat P = section.P[i].clone();
                cout << "1: " << S.at<complex<float>>(150, 150) << endl;

                /// Extract bayer matrix corresponding to ith colour channel
                Mat bayer;
                split_bayer[ind].convertTo(bayer, CV_32FC1); /// might be incorrect i
                cout << "Bayer: " << bayer.at<float>(0, 0) << bayer.at<float>(0, 1) << bayer.at<float>(1, 0) << bayer.at<float>(1, 1) << endl;
                Mat invBayer;

                /// Convert invBayer to complex matrix
                split_invBayer[ind].convertTo(split_invBayer[ind], CV_32FC1);
                Mat ivBtmp[] = {split_invBayer[ind], Mat::zeros(split_invBayer[ind].size(), CV_32FC1)};
                merge(ivBtmp, 2, invBayer);
                cout << "invBayer: " << invBayer.at<complex<float>>(0, 0) << invBayer.at<complex<float>>(0, 1) << invBayer.at<complex<float>>(1, 0) << invBayer.at<complex<float>>(1, 1) << endl;


                /// Extract pixel coordinates
                vector<int> LED = subImg.LEDPos;
                int dx = (int) section.coords({ind, LED[0] + int(pow(numImgs, 0.5) / 2 - 1), LED[1] +
                                               int(pow(numImgs, 0.5) / 2 - 1), 0}) - (int) section.subSize[0] / 2;
                int dy = (int) section.coords({ind, LED[0] + int(pow(numImgs, 0.5) / 2 - 1), LED[1] +
                                               int(pow(numImgs, 0.5) / 2 - 1), 1}) - (int) section.subSize[0] / 2;

                cout << "LED: " << LED[0] << ", " << LED[1] << endl;
                cout << "LEDPos: " << LED[0] + int(pow(numImgs, 0.5) / 2 - 1) <<
                ", " << LED[1] + int(pow(numImgs, 0.5) / 2 - 1) << endl;
                cout << "dx, dy: " <<
                section.coords({i, LED[0] + int(pow(numImgs, 0.5) / 2 - 1), LED[1] +
                int(pow(numImgs, 0.5) / 2 - 1), 0}) << ", " <<
                section.coords({i, LED[0] + int(pow(numImgs, 0.5) / 2 - 1), LED[1] +
                int(pow(numImgs, 0.5) / 2 - 1), 1}) << endl;
                cout << "dx, dy act: " << dx << ", " << dy << endl;

                /// P(u)S(u-Un)
                Mat STrans = translate(S, dx, dy);
                Mat SCrop = crop(STrans, section.subSize);
                Mat phi = P.clone();
                mulSpectrums(P, SCrop, phi, 0);
                cout << "2: " << phi.at<complex<float>>(200, 200) << endl;

                /// F^-1(phi(u))
                Mat PHI = phi.clone();
                fct_fftshift(PHI);
                idft(phi, PHI, cv::DFT_COMPLEX_INPUT | cv::DFT_COMPLEX_OUTPUT | cv::DFT_SCALE);
                //PHI = PHI;
                fct_ifftshift(PHI);
                cout << "3: " << PHI.at<complex<float>>(200, 200) << endl;

                /// amplitude = sqrt(img*bayer) bayer correction
                Mat amplitude = img.clone();
                sqrt(img.mul(bayer), amplitude); //
                cout << "ok" << endl;

                /// Bayer corrected amplitude
                Mat a = PHI.clone();
                mulSpectrums(PHI, invBayer, a, 0);
                parallel_for_(Range(0, a.size[0]*a.size[1]), abs1(a.data));
                Mat ampComp;
                // Make amplitude complex
                Mat aCtmp[] = {amplitude, Mat::zeros(amplitude.size(), CV_32FC1)};
                merge(aCtmp, 2, ampComp);
                Mat bayerCorrectedAmp = ampComp + a;
                cout << "5: " << bayerCorrectedAmp.at<complex<float>>(200, 200) << endl;

                /// Impose intensity constraints
                Mat PPNum = PHI.clone();
                mulSpectrums(bayerCorrectedAmp, PHI, PPNum, 0);
                Mat PPDen = PHI.clone();
                parallel_for_(Range(0, PPDen.size[0]*PPDen.size[1]), abs1(PPDen.data));
                Mat PHIPrimed = PPNum.clone();
                parallel_for_( Range(0, PPNum.size[0]*PPNum.size[1]),
                        div1(PPNum.data, PPDen.data, PHIPrimed.data));
                //divide(PPNum, PPDen, PHIPrimed);
                cout << "6: " << PPDen.at<complex<float>>(200, 200) << endl;

                /// phi'(u) = F(PHI'(r))
                Mat phiPrimed;
                fct_ifftshift(PHIPrimed);
                dft(PHIPrimed, phiPrimed, cv::DFT_COMPLEX_OUTPUT | cv::DFT_COMPLEX_INPUT);
                //phiPrimed = phiPrimed;
                //fct_fftshift(phiPrimed);
                cout << "7: " << phiPrimed.at<complex<float>>(200, 200) << endl;

                /// Update step S
                *
                Mat absS = S.clone();
                parallel_for_(Range(0, absS.size[0]*absS.size[1]), abs1(absS.data));
                Mat sl[2];
                split(absS, sl);
                double mi, ma;
                minMaxIdx(sl[0], &mi, &ma);
                cout << "Max var: " << ma << endl;*

                Mat updateStepS = updateS_1(P, phiPrimed, phi, {dx, dy});
                S = S + pad(updateStepS, section.bigSize[0], section.bigSize[1]);
                Mat splitS1[2];
                split(S, splitS1);
                cout << "8: " << updateStepS.at<complex<float>>(200, 200) << endl;

                ****************************
                Mat splits[2];
                Mat tp = phiPrimed.clone();
                parallel_for_(Range(0, tp.size[0]*tp.size[1]), abs1(tp.data));
                split(tp, splits);
                Mat temp12 = splits[0].clone();
                //log1(temp12.data);
                imshow("phiPrimed", temp12/1000);
                ****************************

                Mat absS = phiPrimed.clone();
                parallel_for_(Range(0, absS.size[0]*absS.size[1]), abs1(absS.data));
                Mat sl[2];
                split(absS, sl);
                double mi, ma;
                minMaxIdx(sl[0], &mi, &ma);
                cout << "Max phiPrimed: " << ma << endl;

                *
                absS = updateStepS.clone();
                parallel_for_(Range(0, absS.size[0]*absS.size[1]), abs1(absS.data));
                split(absS, sl);
                minMaxIdx(sl[0], &mi, &ma);
                cout << "Max var: " << ma << endl;*

                //cout << pad(updateStepS, section.bigSize[0], section.bigSize[1]).at<complex<float>>(200, 200) << endl;

                /// Update step P
                Mat SCrop1 = crop(S, section.subSize);
                Mat updateStepP = updateP_1(SCrop1, phiPrimed, phi, {dx, dy});
                P = P + updateStepP;

                /// Update matrices
                S.copyTo(section.S[i]);
                //section.S[i] = S;
                P.copyTo(section.P[i]);
                //section.P[i] = P;

                ****************************************************************/

                //waitKey(0);

                mainFPM2(section, subImg, numImgs);
                //cout << "main: " << (float)(time_req-clock())/CLOCKS_PER_SEC << " seconds" << endl;
            }//);
        }//);
        string Clean(20, ' ');
        cout << "\r" << "Progress: " << 100 * i0/numImgs << " %" << Clean;
    }
}

PYBIND11_MODULE(cFPM, m){
    m.def("test", &test);
    m.def("test1", &test1);
    m.def("fft", &fft);
    m.def("ifft", &ifft);
    m.def("sqrtTest", &sqrtTest);
    m.def("conjTest", &conjTest);
    m.def("FPM_Feed", &FPM_Feed);
    m.def("absTest", &absTest);
    m.def("testTrans", &testTrans);
    m.def("testRead", &testRead);
    m.def("testDiv", &testDiv);
    m.def("meanFFT", &meanFFT);
    m.def("BrightField", &BrightField);

    py::class_<subSys>(m, "subSys")
            .def(py::init<>())
            .def("add_S", &subSys::add_S)
            .def("add_P", &subSys::add_P)
            .def("add_meanFFT", &subSys::add_meanFFT)
            .def("add_wLen", &subSys::add_wLen)
            .def("add_fRApprox", &subSys::add_fRApprox)
            .def("add_coords", &subSys::add_coords)
            .def("add_subSize", &subSys::add_subSize)
            .def("add_bigSize", &subSys::add_bigSize)
            .def("add_bayer", &subSys::add_bayer)
            .def("add_invBayer", &subSys::add_invBayer)
            .def("add_isBF", &subSys::add_isBF);

    py::class_<fullSys>(m, "fullSys")
            .def(py::init<>())
            .def("list", &fullSys::list)
            .def("add_subSys", &fullSys::add_subSys)
            .def("add_images", &fullSys::add_images)
            .def("export_S", &fullSys::export_S)
            .def("export_P", &fullSys::export_P)
            .def("export_meanFFT", &fullSys::export_meanFFT);

    py::class_<splitImage>(m, "splitImage")
            .def(py::init<>());

    m.def("fill_dict_str", &fill_dict_str);
    m.def("fill_dict_d", &fill_dict_d);
}