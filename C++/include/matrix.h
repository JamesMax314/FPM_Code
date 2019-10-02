#ifndef MATRIX
#define MATRIX

#include <complex>
#include <iostream>


using std::vector;

class Matrix {
public:
    uchar* data_start;
    vector<ssize_t> shape;
    __int32 len;
    // Definitions
    Matrix();
    Matrix(const vector<ssize_t>&, void*);
    double start();

    explicit Matrix(const vector<ssize_t>&);
    //Matrix(const Matrix&);
    double operator()(const vector<__int32>&);
    Matrix colour(int);
    void print();

    //double& operator()(const vector<vector<__int32>>&);
    //~Matrix();
    /*
    Matrix(const Matrix &);
    // Operations
    Matrix operator+(Matrix &);
    Matrix operator-(Matrix &);
    Matrix operator+(T);
    Matrix operator-(T);
     */
};

Matrix::Matrix(){}

Matrix::Matrix(const vector<ssize_t>& _shape, void* _data){
    data_start = (uchar*) _data;
    //std::cout << "31: " << ((double*)(data_start))[0] << std::endl;
    shape = _shape;
    len = 1;
    for (int i : shape){
        len *= i;
    }
    std::cout << (double*)(data_start) << std::endl;
}

double Matrix::start() {
    //std::cout << (double*)(data_start) << std::endl;
    //std::cout << ((double*)(data_start))[10] << std::endl;
    return ((double*)(data_start))[10];
}

Matrix::Matrix(const vector<ssize_t>& in_shape){
    shape = in_shape;
    len = 1;
    for (int i : shape){
        len *= i;
    }
    auto temp = new uchar*[len];
    //for (__int32 i=0; i<len; i++){
    //   *temp[i] = 1.0;
    //}
    data_start = *temp;
    //std::cout << ((double*)(temp))[0] << std::endl;
    //std::cout << (double)(data_start)[0] << std::endl;
    //std::cout << ((double*)(data_start))[0] << std::endl;
}

/*
Matrix::Matrix(const Matrix &mat){
    shape = mat.shape;
    len = mat.len;
    auto temp = new uchar[len];
    data_start = temp;
    for (__int32 i=0; i<mat.len; i++){
        data_start[i] = mat.data_start[i];
    }
}*/


double Matrix::operator()(const vector<__int32>& pos){
    __int32 index = 0;
    std::cout << shape[0] << std::endl;
    std::cout << shape[1] << std::endl;
    for (int i=1; i<(int)(shape.size()); i++){
        index += pos[i]*(__int32)(shape[i-1]);
    }
    index += pos[0];
    //std::cout << "Index: " << index << std::endl;
    //std::cout << "3i: " << ((double*)(data_start))[index] << std::endl;
    return ((double*)(data_start))[index];
}

Matrix Matrix::colour(int color) {
    vector<ssize_t> s = {shape[0], shape[1]};
    //std::cout << s[0] << " " << s[1] << std::endl;
    Matrix out = Matrix(s);
    __int32 index;
    for (__int32 i=color; i*3+color<len; i++){
        index = color + i*3;
        out.data_start[i] = ((uchar*)(data_start))[index];
    }
    std::cout << "out" << ((double*)(out.data_start))[0] << std::endl;
    std::cout << "in" << ((double*)(data_start))[0] << std::endl;
    return out;
}

void Matrix::print(){
    for (__int32 i=0; i<this->len; i++){
        //std::cout << i << " ";
        std::cout << ((double*)(data_start))[i] << ", ";
    }
}

#endif