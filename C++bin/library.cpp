#include "library.h"

void div_part(complex<float>* n, complex<float>* d, complex<float>* o, int len){
//#pragma loop(hint_parallel(8))
//#pragma loop(no_vector)
#pragma loop(hint_parallel(8))
#pragma omp simd
    for (int i=0; i<len; i++){
        *o = *n / *d;
        n++;
        d++;
        o++;
    }
}

void abs_part(complex<float>* ptr, int len){
//#pragma loop(hint_parallel(8))
//#pragma loop(no_vector)
#pragma loop(hint_parallel(8))
#pragma omp simd
    for (int i=0; i<len; i++){
        *ptr = std::sqrt(std::pow((*ptr).real(), 2) + std::pow((*ptr).imag(), 2));
        ptr++;
    }
}

void conjug_part(complex<float>* ptr, int len){
//#pragma loop(hint_parallel(8))
//#pragma loop(no_vector)
#pragma loop(hint_parallel(8))
#pragma omp simd
    for (int i=0; i<len; i++){
        *ptr = std::conj(*ptr);
        ptr++;
    }
}

int main(){
    return 0;
}