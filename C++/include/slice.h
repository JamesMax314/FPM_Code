#ifndef SLICE
#define SLICE

#include <complex>
#include <iostream>
#include <matrix.h>


using std::vector;


class Slice {
    /* 2d or 3d array slicing */
    vector<ssize_t> shape;
    vector<vector<__int32>> slices;
    vector<__int32> dimLen;
    uchar* start;
    uchar* start_pos;
    __int32 start_index;
    __int32 active_stride;
    __int32 dead_stride;
public:
    Slice();
    Slice(uchar*, const vector<ssize_t>&, const vector<vector<__int32>>&);
    Slice& operator=(Matrix mat);
};

Slice::Slice() {

}

Slice::Slice(uchar* in_start, const vector<ssize_t>& in_shape, const vector<vector<__int32>>& in_slices) {
    shape = in_shape;
    start = in_start;
    slices = in_slices;
    start_index = (slices[1][0] * (__int32)(in_shape[0]) + slices[0][0]) * (__int32)(in_shape[2]);
    active_stride = (slices[0][1] - slices[0][0]) * (__int32)(in_shape[2]);
    dead_stride = (__int32)(in_shape[2]*in_shape[1]) - active_stride;
}

Slice& Slice::operator=(Matrix mat){
    int num_rows = (int)(slices[1][1]-slices[1][0]);
    for (int i=0; i<num_rows; i++){
        for (__int32 j=start_index; j<active_stride; j++){
            start[start_index + j] = mat({j, i, })
        }
    }
}

#endif