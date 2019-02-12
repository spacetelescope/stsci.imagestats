/*
    Program:    buildHistogram.c
    Author:     Christopher Hanley
    Purpose:    Populate a 1 dimensional python object to create a histogram
*/
#include <string.h>
#include <stdio.h>
#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION
#include <Python.h>
#include "numpy/arrayobject.h"

int populate1DHist_(float *image, int image_elements,
		    unsigned int *histogram, int histogram_elements,
                    float minValue, float maxValue, float binWidth)
{
    int i, index=0;
    for (i = 0; i < image_elements; i++) {
        if ( (image[i] >= minValue) && (image[i] < maxValue) ) {
            index = (int)((double)(image[i] - minValue) / binWidth );

            /* Handle histogram population for floating point errors at end points */
            /* Case 1: Populating below index 0.*/
            if ( index < 0 ) {
                histogram[0] += 1;
            }
            /* Case 2: Populating above the maximum index value*/
            else if (index >= histogram_elements ) {
                histogram[histogram_elements - 1] +=1;
            }
            /* Case 3: Normal Case - Population of histogram occurs as expected in valid index range */
            else {
                histogram[ index ] += 1;
            }
        }
    }
    return 1;
}

static PyObject * populate1DHist(PyObject *obj, PyObject *args)
{
    PyObject *oimage, *ohistogram;
    PyArrayObject *image, *histogram;
    float minValue, maxValue, binWidth;
    int status=0;

    if (!PyArg_ParseTuple(args,"OOfff:populate1DHist",&oimage,&ohistogram,&minValue,&maxValue,&binWidth))
	    return NULL;

    image = (PyArrayObject *)PyArray_ContiguousFromObject(oimage, NPY_FLOAT32, 1, 2);

    if (!image) return NULL;

	histogram = (PyArrayObject *)PyArray_ContiguousFromObject(ohistogram, NPY_UINT32, 1, 1);

    if (!histogram) return NULL;

    status = populate1DHist_((float *)PyArray_DATA(image), PyArray_Size((PyObject*)image),
			     (unsigned int *)PyArray_DATA(histogram), PyArray_Size((PyObject*)histogram),
			     minValue, maxValue, binWidth);

    Py_XDECREF(image);
    Py_XDECREF(histogram);

    return Py_BuildValue("i", status);
}

static PyMethodDef buildHistogram_methods[] =
{
    {"populate1DHist",  populate1DHist, METH_VARARGS,
        "populate1Dhist(image, histogram, minValue, maxValue, binWidth)"},
    {0,            0}                             /* sentinel */
};

static struct PyModuleDef moduledef = {
  PyModuleDef_HEAD_INIT,
  "buildHistogram",            /* m_name */
  "C build histogram module",  /* m_doc */
  -1,                          /* m_size */
  buildHistogram_methods,      /* m_methods */
  NULL,                        /* m_reload */
  NULL,                        /* m_traverse */
  NULL,                        /* m_clear */
  NULL,                        /* m_free */
};

PyMODINIT_FUNC PyInit_buildHistogram(void)
{
	PyObject* m;
    import_array();
	m = PyModule_Create(&moduledef);
	return m;
}
