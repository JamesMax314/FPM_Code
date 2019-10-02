#ifndef CASTERS
#define CASTERS

#include <iostream>
#include <numeric>
#include <vector>
#include <opencv2/opencv.hpp>
#include <complex>
#include <pybind11/complex.h>
#include <pybind11/stl.h>
#include <ctime>
#include <pybind11/numpy.h>
#include "matrix1.h"

namespace pybind11 { namespace detail {
        template <> struct type_caster<cv::Mat> {
        public:

            PYBIND11_TYPE_CASTER(cv::Mat, _("numpy.ndarray"));


            bool load(handle src, bool)
            {

                array b = reinterpret_borrow<array>(src);
                buffer_info info = b.request();

                int ndims = (int)(info.ndim);

                decltype(CV_32F) dtype;

                //cout << "Type: " << info.format << endl;

                if (info.format == format_descriptor<float>::format()) {
                    if (ndims == 3) {
                        dtype = CV_32FC3;
                    } else {
                        dtype = CV_32FC1;
                    }
                } else if (info.format == format_descriptor<double>::format()) {
                    if (ndims == 3) {
                        dtype = CV_64FC3;
                    } else {
                        dtype = CV_64FC1;
                    }
                } else if (info.format == format_descriptor<unsigned char>::format()) {
                    if (ndims == 3) {
                        dtype = CV_8UC3;
                    } else {
                        dtype = CV_8UC1;
                    }
                } else if (info.format == format_descriptor<complex<float>>::format()){
                    if (ndims == 3) {
                        dtype = CV_32FC3;
                    } else {
                        dtype = CV_32FC2;
                    }
                } else if (info.format == format_descriptor<complex<double>>::format()){
                    if (ndims == 3) {
                        dtype = CV_64FC3;
                    } else {
                        dtype = CV_64FC2;
                    }
                } else {
                    std::cout << info.format << std::endl;
                    throw std::logic_error("Unsupported type");
                }
                std::vector<int> shape = {static_cast<int>(info.shape[0]), static_cast<int>(info.shape[1])};

                value = cv::Mat(cv::Size(shape[1], shape[0]), dtype, info.ptr, cv::Mat::AUTO_STEP);

                return true;
            }


            static handle cast(const cv::Mat &m, return_value_policy, handle defval)
            {
                std::string format = format_descriptor<unsigned char>::format();
                size_t elemsize = sizeof(unsigned char);
                int dim;
                //std::cout << m.type() << std::endl;
                switch(m.type()) {
                    case CV_8U:
                        format = format_descriptor<unsigned char>::format();
                        elemsize = sizeof(unsigned char);
                        dim = 2;
                        break;
                    case CV_8UC3:
                        format = format_descriptor<unsigned char>::format();
                        elemsize = sizeof(unsigned char);
                        dim = 3;
                        break;
                    case CV_32F:
                        format = format_descriptor<float>::format();
                        elemsize = sizeof(float);
                        dim = 2;
                        break;
                    case CV_64F:
                        format = format_descriptor<double>::format();
                        elemsize = sizeof(double);
                        dim = 2;
                        break;
                    case CV_64FC3:
                        format = format_descriptor<double>::format();
                        elemsize = sizeof(double);
                        dim = 3;
                        break;
                    case CV_64FC2:
                        format = format_descriptor<complex<double>>::format();
                        elemsize = sizeof(long double);
                        dim = 2;
                        break;
                    case CV_32FC2:
                        format = format_descriptor<complex<float>>::format();
                        elemsize = sizeof(double);
                        dim = 2;
                        break;
                    case CV_32FC3:
                        format = format_descriptor<complex<float>>::format();
                        elemsize = sizeof(double); // numpy complex 64 has size float in c++ but double in pybind11
                        dim = 3;
                        break;
                    default:
                        throw std::logic_error("Unsupported type");
                }
                std::vector<size_t> bufferdim;
                std::vector<size_t> strides;
                if (dim == 2) {
                    bufferdim = {(size_t) m.rows, (size_t) m.cols};
                    strides = {elemsize * (size_t) m.cols, elemsize};
                } else {
                    bufferdim = {(size_t) m.rows, (size_t) m.cols, (size_t) 3};
                    strides = {(size_t) elemsize * m.cols * 3, (size_t) elemsize * 3, (size_t) elemsize};
                }
                return array(buffer_info(
                        m.data,
                        elemsize,
                        format,
                        dim,
                        bufferdim,
                        strides
                )).release();
            }
        };
    }}
// namespace pybind11::detail


namespace pybind11 { namespace detail {
        template <> struct type_caster<Matrix> {
        public:

            PYBIND11_TYPE_CASTER(Matrix, _("numpy.ndarray"));


            bool load(handle src, bool)
            {
                array b = reinterpret_borrow<array>(src);
                buffer_info info = b.request();
                int ndims = (int)(info.ndim);
                //std::vector<int> shape = {static_cast<int>(info.shape[0]), static_cast<int>(info.shape[1])};
                //cout << "shape: " << info.shape[0] << ", " <<  info.shape[1] <<", " <<  info.shape[2] <<", " <<  info.shape[3] << endl;
                value = Matrix(info.shape, info.ptr);
                return true;
            }
            static handle cast(const cv::Mat &m, return_value_policy, handle defval)
            {
                std::string format = format_descriptor<unsigned char>::format();
                size_t elemsize = sizeof(unsigned char);
                int dim;
                //std::cout << m.type() << std::endl;
                switch(m.type()) {
                    case CV_8U:
                        format = format_descriptor<unsigned char>::format();
                        elemsize = sizeof(unsigned char);
                        dim = 2;
                        break;
                    case CV_8UC3:
                        format = format_descriptor<unsigned char>::format();
                        elemsize = sizeof(unsigned char);
                        dim = 3;
                        break;
                    case CV_32F:
                        format = format_descriptor<float>::format();
                        elemsize = sizeof(float);
                        dim = 2;
                        break;
                    case CV_64F:
                        format = format_descriptor<double>::format();
                        elemsize = sizeof(double);
                        dim = 2;
                        break;
                    case CV_64FC3:
                        format = format_descriptor<double>::format();
                        elemsize = sizeof(double);
                        dim = 3;
                        break;
                    case CV_32FC2:
                        format = format_descriptor<complex<float>>::format();
                        elemsize = sizeof(double);
                        dim = 2;
                        break;
                    case CV_32FC3:
                        format = format_descriptor<complex<float>>::format();
                        elemsize = sizeof(double); // numpy complex 64 has size float in c++ but double in pybind11
                        dim = 3;
                        break;
                    default:
                        throw std::logic_error("Unsupported type");
                }
                std::vector<size_t> bufferdim;
                std::vector<size_t> strides;
                if (dim == 2) {
                    bufferdim = {(size_t) m.rows, (size_t) m.cols};
                    strides = {elemsize * (size_t) m.cols, elemsize};
                } else {
                    bufferdim = {(size_t) m.rows, (size_t) m.cols, (size_t) 3};
                    strides = {(size_t) elemsize * m.cols * 3, (size_t) elemsize * 3, (size_t) elemsize};
                }
                return array(buffer_info(
                        m.data,
                        elemsize,
                        format,
                        dim,
                        bufferdim,
                        strides
                )).release();
            }
        };
    }}
// namespace pybind11::detail

#endif