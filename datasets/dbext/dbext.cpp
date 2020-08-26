#include "pybind11/pybind11.h" // from source
#include "pybind11/stl.h"
#include "pybind11/numpy.h"
#include <iostream>
#include <vector>
#include <string>
#include <unordered_map>
namespace py = pybind11;
using namespace std;


template<typename T>
vector<int> find_first_last_gt(py::array_t<T> x, T value) 
{
    py::buffer_info buf = x.request();
    if (buf.ndim != 1)
        throw std::runtime_error("Number of dimensions must be one");
    auto x_r = x.template unchecked<1>();
    int first = (int)x_r.shape(0)-1, last = 0;
    for (ssize_t i = 0; i < x_r.shape(0); i++) {
        if (x_r(i) >= value) {
            first = (int)i;
            break;
        }
    }
    for (ssize_t i = x_r.shape(0)-1; i >= 0; i--) {
        if (x_r(i) >= value) {
            last = (int)i;
            break;
        }
    }
    vector<int> res;
    res.push_back(first);
    res.push_back(last);
    return res;
}


PYBIND11_MODULE(dbext, m) {
    m.doc() = "dbext plugin";
    m.def("find_first_last_gt", &find_first_last_gt<float>, ""); 
}
