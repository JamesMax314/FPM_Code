#ifndef METHODS
#define METHODS

#include <opencv2/opencv.hpp>
#include <iostream>
#include <sstream>
#include <ppl.h>
#include "library.h"
#include "calibrate.h"
using namespace cv;
using namespace std;
using std::string;


// Define the element type for a 3 channel complex (CV_32FC3) image
typedef Vec<complex<float>, 3> Vec3cd;

string pathAppend(const string& p1, const string& p2) {
    char sep = '/';
    string tmp = p1;

    if (p1[p1.length()] != sep) {
        tmp += sep;
        return(tmp + p2);
    }
    else
        return(p1 + p2);
}

Mat crop(const Mat& arr, const vector<int>& size){
    const vector<int> shape = {arr.rows, arr.cols};
    const vector<int> halfSize = {(int)(floor(size[0] / 2)), (int)(floor(size[1] / 2))};
    const vector<int> centre = {(int)(shape[0]/2), (int)(shape[1]/2)};
    const vector<int> posStart = {centre[0] - halfSize[0], centre[1] - halfSize[1]};
    const vector<int> posEnd = {centre[0] + halfSize[0], centre[1] + halfSize[1]};
    Mat out = arr(cv::Range(posStart[0], posEnd[0]), cv::Range(posStart[1], posEnd[1]));
    return out;
}

Mat read_image(const string& dir, const string& file){
    string path = pathAppend(dir, file);
    Mat img = imread(path);
    vector<int> shape = {img.cols, img.rows, img.channels()};
    int minLen = static_cast<int>(min_element(begin(shape), end(shape)-1)[0]);
    vector<int> size = {minLen, minLen};
    Mat cropped;
    crop(img, size).convertTo(cropped, CV_32F);
    return cropped;
}

vector<int> get_LED(const string& imgName){
    stringstream ss(imgName);
    vector<string> parts;
    string part;
    while(getline(ss, part, '_')){
        parts.push_back(part);
    }
    vector<int> out = {stoi(parts[1]), stoi(parts[2])};
    return out;
}

Mat translate(const Mat& arr, const __int32& dx, const __int32& dy) {
    Mat out = Mat::zeros(arr.size(), arr.type());
    Rect source = Rect(max(0, -dx), max(0, -dy), arr.cols-abs(dx), arr.rows-abs(dy));
    Rect target = Rect(max(0, dx), max(0, dy), arr.cols-abs(dx), arr.rows-abs(dy));
    arr(source).copyTo(out(target));
    return out;
}

Mat pad(const Mat& arr, const int& size_x, const int& size_y){
    Mat out = Mat::zeros(size_x, size_y, arr.type());
    vector<int> centre = {(int)(floor(size_x / 2)), (int)(floor(size_y / 2))};
    vector<int> len = {(int)(arr.size[0]), (int)(arr.size[1])};
    vector<int> posStart = {(int)(centre[0] - len[0] / 2), (int)(centre[1] - len[1] / 2)};
    vector<int> posEnd = {(int)(centre[0] + len[0] / 2 - 1), (int)(centre[1] + len[1] / 2 - 1)};
    if (posEnd[0] - posStart[0] <= arr.size[0])
        posEnd[0] += 1;
    if (posEnd[1] - posStart[1] <= arr.size[1])
        posEnd[1] += 1;
    Rect target = Rect(posStart[1], posStart[0], posEnd[1]-posStart[1], posEnd[0]-posStart[0]);
    arr.copyTo(out(target));
    return out;
}

void fct_fftshift(cv::Mat& src)
{
    int cx = src.cols/2;
    int cy = src.rows/2;

    cv::Mat q0(src, cv::Rect(0, 0, cx, cy));
    cv::Mat q1(src, cv::Rect(cx, 0, cx, cy));
    cv::Mat q2(src, cv::Rect(0, cy, cx, cy));
    cv::Mat q3(src, cv::Rect(cx, cy, cx, cy));

    cv::Mat tmp;
    q0.copyTo(tmp);
    q3.copyTo(q0);
    tmp.copyTo(q3);

    q1.copyTo(tmp);
    q2.copyTo(q1);
    tmp.copyTo(q2);
}

void fct_ifftshift(cv::Mat& src)
{
    int cx = src.cols/2;
    int cy = src.rows/2;

    cv::Mat q0(src, cv::Rect(cx, cy, cx, cy));
    cv::Mat q1(src, cv::Rect(0, cy, cx, cy));
    cv::Mat q2(src, cv::Rect(cx, 0, cx, cy));
    cv::Mat q3(src, cv::Rect(0, 0, cx, cy));

    cv::Mat tmp;
    q0.copyTo(tmp);
    q3.copyTo(q0);
    tmp.copyTo(q3);

    q1.copyTo(tmp);
    q2.copyTo(q1);
    tmp.copyTo(q2);
}

class conjug : public ParallelLoopBody
{
private:
    uchar *p ;
public:
    conjug(uchar* ptr) : p(ptr) {}

    virtual void operator()( const Range &r ) const{
        for (int i = r.start; i != r.end; i++)
        {
            ((complex<float>*)(p))[i] = std::conj(((complex<float>*)(p))[i]);
        }
    }
};

class sqrt1 : public ParallelLoopBody
{
private:
    uchar *p ;
public:
    sqrt1(uchar* ptr) : p(ptr) {}

    virtual void operator()( const Range &r ) const{
        for (int i = r.start; i != r.end; i++)
        {
            ((double*)(p))[i] = std::sqrt(((double*)(p))[i]);
        }
    }
};

class log1 : public ParallelLoopBody
{
private:
    uchar *p ;
public:
    log1(uchar* ptr) : p(ptr) {}

    virtual void operator()( const Range &r ) const{
        for (int i = r.start; i != r.end; i++)
        {
            ((double*)(p))[i] = std::log(((double*)(p))[i]);
        }
    }
};

class abs1 : public ParallelLoopBody
{
private:
    uchar *p ;
public:
    abs1(uchar* ptr) : p(ptr) {}

    virtual void operator()( const Range &r ) const{
        for (int i = r.start; i != r.end; i++)
        {
            ((complex<float>*)(p))[i] = std::sqrt(std::pow(((complex<float>*)(p))[i].real(), 2) +
                    std::pow(((complex<float>*)(p))[i].imag(), 2));
        }
    }
};

class div1 : public ParallelLoopBody
{
private:
    uchar *den;
    uchar *num;
    uchar *out;
public:
    div1(uchar* n, uchar* d, uchar* o) : den(d), num(n), out(o) {}

    virtual void operator()( const Range &r ) const{
        for (int i = r.start; i != r.end; i++)
        {
            ((complex<float>*)(out))[i] = ((complex<float>*)(num))[i] / ((complex<float>*)(den))[i];
        }
    }
};

void div2(Mat& n, Mat& d, Mat& o){
    for (int i=0; i<n.size[0]; i++){
        for (int j=0; j<n.size[0]; j++) {
            o.at<complex<float>>(i, j) = n.at<complex<float>>(i, j) / d.at<complex<float>>(i, j);
        }
    }
}

/*
void div3(Mat& n, Mat& d, Mat& o){
    complex<float>* ptrN = ((complex<float>*)(n.data));
    complex<float>* ptrD = ((complex<float>*)(d.data));
    complex<float>* ptrO = ((complex<float>*)(o.data));
#pragma loop(hint_parallel(8))
    for (int i=0; i<n.size[0]*n.size[1]; i++){
        *ptrO = *ptrN / *ptrD;
        ptrN++;
        ptrD++;
        ptrO++;
    }
}*/

void div4(Mat& n, Mat& d, Mat& o){
    complex<float>* ptrN = ((complex<float>*)(n.data));
    complex<float>* ptrD = ((complex<float>*)(d.data));
    complex<float>* ptrO = ((complex<float>*)(o.data));
    div_part(ptrN, ptrD, ptrO, n.size[0]*n.size[1]);
}

void abs4(Mat& arr){
    complex<float>* ptrArr = ((complex<float>*)(arr.data));
    abs_part(ptrArr, arr.size[0]*arr.size[1]);
}

void conjug4(Mat& arr){
    complex<float>* ptrArr = ((complex<float>*)(arr.data));
    conjug_part(ptrArr, arr.size[0]*arr.size[1]);
}

float stdDev(Mat& arr){
    float mean = 0;
    for (int i=0; i<arr.size[0]*arr.size[1]; i++){
        mean += float(arr.data[i]);
    }
    mean = mean / float(arr.size[0]*arr.size[1]);
    float stdPart = 0;
    for (int i=0; i<arr.size[0]*arr.size[1]; i++){
        stdPart += pow(float(arr.data[i]) - mean, 2);
    }
    auto stdD = float(pow(stdPart / float(arr.size[0]*arr.size[1] - 1), 0.5));
    return stdD;
}

float sumMat(Mat& arr){
    float sum = 0;
    for (int i=0; i<arr.size[0]*arr.size[1]; i++){
        sum += float(arr.data[i]);
    }
    return sum;
}

pair<bool, int> finder(vector<vector<int>> arr1, vector<int> val, vector<int> tolerance){
    bool found = false;
    int index = -1;
    for (int i=0; i<arr1.size(); i++){
        if ((arr1[i][0] == val[0] - tolerance[0] || arr1[i][0] == val[0] + tolerance[0]) &&
        (arr1[i][1] == val[1] - tolerance[1] || arr1[i][1] == val[1] + tolerance[1])){
            found = true;
            index = i;
            break;
        }
    }
    return {found, index};
}


/*
Mat complex_mult(const Mat& arr1, const Mat& arr2){
    const vector<int> shape = {arr1.rows, arr1.cols};
    Mat out = Mat(arr1);
    if (arr1.type() == CV_32FC2) {
        cout << "CV_32FC2" << endl;
        for (int i = 0; i < shape[0]; i++) {
            for (int j = 0; j < shape[1]; j++) {
                float real = arr1.at<complex<float>>(i, j).real() * arr2.at<complex<float>>(i, j).real() -
                             arr1.at<complex<float>>(i, j).imag() * arr2.at<complex<float>>(i, j).imag();
                float imag = arr1.at<complex<float>>(i, j).real() * arr2.at<complex<float>>(i, j).imag() -
                             arr1.at<complex<float>>(i, j).imag() * arr2.at<complex<float>>(i, j).real();
                out.at<complex<float>>(i, j) = {real, imag};
            }
        }
    } else if (arr1.type() == CV_32FC3){
        cout << "CV_32FC3" << endl;
        for (int i = 0; i < shape[0]; i++) {
            for (int j = 0; j < shape[1]; j++) {
                for (int k = 0; k < 3; k++) {
                    float real = arr1.at<Vec3cd>(i, j)[k].real() * arr2.at<Vec3cd>(i, j)[k].real() -
                                 arr1.at<Vec3cd>(i, j)[k].imag() * arr2.at<Vec3cd>(i, j)[k].imag();
                    float imag = arr1.at<Vec3cd>(i, j)[k].real() * arr2.at<Vec3cd>(i, j)[k].imag() +
                                 arr1.at<Vec3cd>(i, j)[k].imag() * arr2.at<Vec3cd>(i, j)[k].real();
                    out.at<Vec3cd>(i, j)[k] = {real, imag};
                }
            }
        }
    }
    return out;
}
*/

/* TO DO
 * write performance sensative code in unwrapped arrays and use fftw dircetly
 */


#endif // METHODS