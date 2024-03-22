/*
    Program:    buildHistogram.c
    Author:     Christopher Hanley
    Purpose:    Populate a 1 dimensional python object to create a histogram
*/
#include <string.h>
#include <stdio.h>
#include <float.h>
#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION
#include <Python.h>
#include "numpy/arrayobject.h"

int
populate1DHist_(float *image, int image_elements, unsigned int *histogram,
                int histogram_elements, float minValue, float maxValue,
                float binWidth)
{
    int i, idx, allow_max_eq = 0;
    float f, hist_edge, v;

    f = 1.0f / binWidth;
    hist_edge = minValue + binWidth * histogram_elements;
    if ((int) (f * (hist_edge - minValue)) == histogram_elements) {
        hist_edge *= (1.0f - FLT_EPSILON);
        allow_max_eq = 1;
    }
    if (maxValue > hist_edge) {
        maxValue = hist_edge;
    }

    if (allow_max_eq) {
        for (i = 0; i < image_elements; ++i) {
            v = image[i];
            if ((v >= minValue) && (v <= maxValue)) {
                idx = (int)(f * (v - minValue));
                ++histogram[idx];
            }
        }
    } else {
        for (i = 0; i < image_elements; ++i) {
            v = image[i];
            if ((v >= minValue) && (v < maxValue)) {
                idx = (int)(f * (v - minValue));
                ++histogram[idx];
            }
        }
    }

    return 1;
}

static PyObject *
populate1DHist(PyObject *obj, PyObject *args)
{
    PyObject *oimage, *ohistogram;
    PyArrayObject *image, *histogram;
    float minValue, maxValue, binWidth;
    int status;

    (void)obj;

    if (!PyArg_ParseTuple(args, "OOfff:populate1DHist", &oimage, &ohistogram,
                          &minValue, &maxValue, &binWidth))
        return NULL;

    image = (PyArrayObject *)PyArray_ContiguousFromObject(oimage, NPY_FLOAT32,
                                                          1, 2);

    if (!image) return NULL;

    histogram = (PyArrayObject *)PyArray_ContiguousFromObject(
        ohistogram, NPY_UINT32, 1, 1);

    if (!histogram) return NULL;

    status = populate1DHist_(
        (float *)PyArray_DATA(image), PyArray_Size((PyObject *)image),
        (unsigned int *)PyArray_DATA(histogram),
        PyArray_Size((PyObject *)histogram), minValue, maxValue, binWidth);

    Py_XDECREF(image);
    Py_XDECREF(histogram);

    return Py_BuildValue("i", status);
}

#pragma clang diagnostic push
#pragma GCC diagnostic ignored "-Wmissing-field-initializers"
static PyMethodDef buildHistogram_methods[] = {
    {"populate1DHist", populate1DHist, METH_VARARGS,
     "populate1Dhist(image, histogram, minValue, maxValue, binWidth)"},
    {0, 0} /* sentinel */
};
#pragma GCC diagnostic pop

static struct PyModuleDef moduledef = {
    PyModuleDef_HEAD_INIT,
    "buildHistogram",           /* m_name */
    "C build histogram module", /* m_doc */
    -1,                         /* m_size */
    buildHistogram_methods,     /* m_methods */
    NULL,                       /* m_reload */
    NULL,                       /* m_traverse */
    NULL,                       /* m_clear */
    NULL,                       /* m_free */
};

PyMODINIT_FUNC
PyInit_buildHistogram(void)
{
    PyObject *m;
    import_array();
    m = PyModule_Create(&moduledef);
    return m;
}
