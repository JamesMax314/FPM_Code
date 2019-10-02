#ifndef TRANSLATE
#define TRANSLATE

#include <numeric>                        // Standard library import for std::accumulate
#include "pybind11/pybind11.h"            // Pybind11 import to define Python bindings
#define FORCE_IMPORT_ARRAY                // numpy C api loading
#include <complex>
#include <utility>
#include <vector>
#include <opencv2/opencv.hpp>
#include "methods.h"
#include "casters.h"
#include <typeinfo>
#include "optim.h"


namespace py = pybind11;
using namespace std::complex_literals;
using namespace std;
using namespace cv;

Mat read_image(const string&, const string&);
vector<int> get_LED(const string&);

struct subSys{
    vector<double> wLen;
    float num;
    Mat bayer, invBayer;
    vector<int> fRApprox, subSize, bigSize;
    Matrix coords, isBF;
    Matrix isFit = Matrix(vector<ssize_t>({3, 16, 16}));
    Mat S[3], P[3], meanFFT[3];
    vector<vector<pair<input_vector, input_vector>>> dataPairs;
    vector<int> dPairsIndex;


    void add_S(Mat& s_0, Mat& s_1, Mat& s_2){
        this->S[0] = s_0.clone();
        this->S[1] = s_1.clone();
        this->S[2] = s_2.clone();
    }
    Mat get_S(const int& i){
        return S[i];
    }
    void add_P(Mat& p_0, Mat& p_1, Mat& p_2){
        this->P[0] = p_0.clone();
        this->P[1] = p_1.clone();
        this->P[2] = p_2.clone();
        this->num = 1;
    }
    Mat get_P(const int& i){
        return P[i];
    }
    void add_meanFFT(Mat& mf_0, Mat& mf_1, Mat& mf_2){
        this->meanFFT[0] = mf_0.clone();
        this->meanFFT[1] = mf_1.clone();
        this->meanFFT[2] = mf_2.clone();
    }
    Mat get_meanFFT(const int& i){
        return meanFFT[i];
    }
    void add_wLen(vector<double>& wl){
        this->wLen = wl;
    }
    void add_fRApprox(vector<int>& fr){
        this->fRApprox = fr;
    }
    void add_coords(Matrix& c){
        this->coords = c;
    }
    void add_subSize(vector<int>& ss){
        this->subSize = ss;
    }
    void add_bigSize(vector<int>& bs){
        this->bigSize = bs;
    }
    void add_bayer(Mat& bay){
        this->bayer = bay;
    }
    void add_invBayer(Mat& iBay){
        this->invBayer = iBay;
    }
    void add_isBF(Matrix& BF){
        this->isBF = BF;
    }
};

/// *******************************************************************************************************************

struct fullSys{
    vector<double> wl = {630e-9, 530e-9, 430e-9};
    map<vector<int>, subSys> sub;
    vector<string> images;
    map<string, string> strParams;
    map<string, double> dParams;
    fullSys();
    ~fullSys();
    void list();
    void add_subSys(const subSys&, const int&, const int&);
    void add_images(const vector<string>&);
    Mat export_S(const vector<int>&, const int&);
    Mat export_P(const vector<int>&, const int&);
    Mat export_meanFFT(const vector<int>&, const int&);
};

fullSys::fullSys() = default;
fullSys::~fullSys() = default;

void fullSys::list(){
    for (const auto& param : strParams)
        std::cout << param.second << std::endl;
}

void fullSys::add_subSys(const subSys& ss, const int& i, const int& j){
    vector<int> imgSect{i, j};
    sub[imgSect] = ss;
}

void fullSys::add_images(const vector<string>& imgs){
    images = imgs;
}

Mat fullSys::export_S(const vector<int>& index, const int& colour) {
    Mat S = sub[index].get_S(colour);
    return S;
}

Mat fullSys::export_P(const vector<int>& index, const int& colour) {
    Mat P = sub[index].get_P(colour);
    return P;
}

Mat fullSys::export_meanFFT(const vector<int>& index, const int& colour){
    Mat meanFFT = sub[index].get_meanFFT(colour);
    return meanFFT;
}

/// *******************************************************************************************************************

struct subImage{
    vector<int> imgPos;
    vector<int> LEDPos;
    int ok;
    Mat img;
    Mat image;
    subImage();
    subImage(const string&, const __int32&, const Mat&,
            const vector<int>& , const int&, const int&);
    ~subImage();
};

subImage::subImage() = default;

subImage::subImage(const string &dir, const __int32 &splitSize, const Mat &img, const vector<int>& LEDP,
                   const int &i, const int &j) : LEDPos(LEDP), imgPos({i, j}), ok(j){
    image = img(cv::Range(i*splitSize, (i+1)*splitSize),
                      cv::Range(j*splitSize, (j+1)*splitSize));
}

subImage::~subImage() = default;

/// *******************************************************************************************************************

struct splitImage{
    vector<int> LEDPos;
    map<vector<int>, subImage> subImg;
    Mat img;
    splitImage();
    splitImage(const string& dir, const string& imgName, const int& numSplits, const __int32& splitSize);
    ~splitImage();
};

splitImage::splitImage() = default;

splitImage::splitImage(const string& dir, const string& imgName,
        const int& numSplits, const __int32& splitSize) : LEDPos(get_LED(imgName)){
    img = read_image(dir, imgName);
    //imshow(imgName, img/255);
    for (int i=0; i < numSplits; i++){
        for (int j=0; j < numSplits; j++){
            subImage sub = subImage(dir, splitSize, img, LEDPos, i, j);
            vector<int> pos{i, j};
            subImg[pos] = sub;
        }
    }
}

splitImage::~splitImage() = default;

/// *******************************************************************************************************************

void fill_dict_str(fullSys& object, const py::dict& dict) {
    for (auto item : dict) {
        object.strParams[std::string(py::str(item.first))] = std::string(py::str(item.second));
    }
}

void fill_dict_d(fullSys& object, const py::dict& dict) {
    for (auto item : dict) {
        object.dParams[std::string(py::str(item.first))] = stod(std::string(py::str(item.second)));
    }
}

#endif



