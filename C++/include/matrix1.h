#ifndef MATRIX
#define MATRIX

#include <complex>
#include <iostream>


using std::vector;

class Matrix {
    __int32 getIndex(const vector<__int32>&);
public:
    uchar* data_start;
    vector<ssize_t> shape;
    __int32 len;
    // Definitions
    Matrix();
    Matrix(const vector<ssize_t>&, void*);
    double start();
    explicit Matrix(const vector<ssize_t>&);
    double operator()(const vector<__int32>&);
    double* at(const vector<__int32>&);
    void update(const vector<__int32>&, const double&);
    void print();
    ~Matrix();
};

__int32 Matrix::getIndex(const vector<__int32>& pos){
    int index = 0;
    for (int i=0; i<(int)(shape.size()); i++){
        int temp = pos[i];
        for (int j=(int)(shape.size())-1; j>i; j--){

            temp = temp * (int)shape[j];
        }
        index += temp;
    }
    return index;
}

Matrix::Matrix(){}

Matrix::Matrix(const vector<ssize_t>& _shape, void* _data){
    data_start = (uchar*) _data;
    shape = _shape;
    len = 1;
    for (__int64 i : shape){
        len *= (int)(i);
    }
    cout << "len " << len << endl;
}

double Matrix::start() {
    return ((double*)(data_start))[0];
}

Matrix::Matrix(const vector<ssize_t>& in_shape){
    shape = in_shape;
    len = 1;
    for (__int64 i : shape){
        len *= (int)(i);
    }
    auto temp = new uchar*[len];
    //for (__int32 i=0; i<len; i++){
    //   *temp[i] = 1.0;
    //}
    data_start = *temp;
}

double Matrix::operator()(const vector<__int32>& pos){
    auto index = getIndex(pos);
    return ((double*)(data_start))[index];
}

double* Matrix::at(const vector<__int32>& pos){
    auto index = getIndex(pos);
    return &((double*)(data_start))[index];
}

void Matrix::update(const vector<__int32>& pos, const double& val) {
    auto index = getIndex(pos);
    ((double*)(data_start))[index] = val;
}

void Matrix::print(){
    for (int i=0; i<this->len; i++){
        //std::cout << i << " ";
        std::cout << ((double*)(data_start))[i] << ", ";
    }
}

Matrix::~Matrix() = default;

#endif