#ifndef OPTIM
#define OPTIM

#include <dlib/optimization.h>
//#include "phasecorr.h"
#include <iostream>
#include <cmath>
#include <cstdlib>
#include <random>

typedef dlib::matrix<double,2,1> input_vector;
typedef dlib::matrix<double,3,1> parameter_vector;

template <typename T>
input_vector vecToInVec(const std::vector<T>& in) {
    input_vector out;
    for (int i = 0; i < 2; i++) {
        out(i) = double(in[i]);
    }
    return out;
}

std::vector<double> inVecToVec(const input_vector& in) {
    std::vector<double> out;
    for (int i = 0; i < 2; i++) {
        out[i] = int(in(i));
    }
    return out;
}

std::vector<int> transform(const std::vector<int>& coords, const std::vector<double>& params){
    std::vector<int> out = {static_cast<int>(coords[0]*std::cos(params[0]) - coords[01]*std::sin(params[0])),
                       static_cast<int>(coords[0]*std::sin(params[0]) + coords[01]*std::cos(params[0]))};
    out[0] += params[1];
    out[1] += params[2];
    return out;
}

std::vector<double> model(const input_vector& input, const parameter_vector& params){
    const double p0 = params(0);
    const double p1 = params(1);
    const double p2 = params(2);

    const double i0 = input(0);
    const double i1 = input(1);

    const std::vector<double> out = {i0*std::cos(p0) - i1*std::sin(p0) + p1,
                                i0*std::sin(p0) + i1*std::cos(p0) + p2};

    return out;
}

double residual(const std::pair<input_vector, input_vector>& data, const parameter_vector& params){
    double xTmp = std::pow(model(data.first, params)[0] - data.second(0), 2);
    double yTmp = std::pow(model(data.first, params)[1] - data.second(1), 2);
    return xTmp + yTmp;
}

parameter_vector residual_derivative(const std::pair<input_vector, input_vector>& data,
        const parameter_vector& params){
    parameter_vector der;

    const double p0 = params(0);
    const double p1 = params(1);
    const double p2 = params(2);

    const double i0 = data.first(0);
    const double i1 = data.first(1);

    const double tx = i0*std::cos(p0) - i1*std::sin(p0) + p1 - data.second(0);
    const double ty = i0*std::sin(p0) - i1*std::cos(p0) + p2 - data.second(1);

    der(0) = -2*tx*(i0*std::sin(p0) + i1*std::cos(p0)) + 2*ty*(i0*std::cos(p0) - i1*std::sin(p0));
    der(1) = 2*tx;
    der(2) = 2*ty;

    return der;
}

std::vector<int> getRand(const int& start, const int& end, const int& num){
    std::random_device seed;
    std::mt19937 engine(seed());
    std::uniform_int_distribution<std::mt19937::result_type> dist(start,end); // distribution in range [1, 6]
    std::vector<int> out;
    for (int i=0; i<num; i++){
        out.push_back(dist(engine));
    }
    return out;
}

std::vector<std::pair<input_vector, input_vector>> take(const std::vector<std::pair<input_vector, input_vector>>& data,
        const std::vector<int>& selectIndices){
    std::vector<std::pair<input_vector, input_vector>> out;
    for (int i=0; i<selectIndices.size(); i++){
        out.push_back(data[selectIndices[i]]);
    }
    return out;
}

std::pair<parameter_vector, std::vector<bool>> RANSAC1(
         const std::vector<std::pair<input_vector, input_vector>>& data,
         const int& threashProportion = 0.5,
         const int& maxTrials = 100,
         const int& minSamples = 3,
         const int& radiusAccepted = 20){
    std::vector<bool> inliers(data.size(), true);
    parameter_vector params = 10*dlib::randm(3,1);
    int threash = int(threashProportion * data.size());
    for (int i=0; i<maxTrials; i++) {
        std::vector<int> selectIndices = getRand(0, data.size(), minSamples);
        std::vector<std::pair<input_vector, input_vector>> DatSelect = take(data, selectIndices);
        dlib::solve_least_squares_lm(dlib::objective_delta_stop_strategy(1e-7).be_verbose(),
                               residual,
                               residual_derivative,
                               DatSelect,
                               params);

        int k = 0;
        for (int j=0; j<data.size(); j++){
            double fit = residual(data[j], params);
            if (fit <= std::pow(radiusAccepted, 2)){
                inliers[j] = true;
                k += 1;
            }else{
                inliers[j] = false;
            }
        }

        if (k >= threash){
            break;
        }
    }
    auto out = std::pair<parameter_vector, std::vector<bool>>(params, inliers);
    return out;
}



#endif // OPTIM