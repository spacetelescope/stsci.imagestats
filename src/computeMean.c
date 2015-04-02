/*
    Program:    computeMean.c
    Author:     Christopher Hanley
    Purpose:    Compute the mean, stddev, max, and min for a nnumarray object
                while applying some upper and lower pixel clipping values.
		
    Updated May 9, 2008   Fixed standard deviation computation, Megan Sosey
    
*/

#include <Python.h>
#include "numpy/arrayobject.h"


#include <string.h>
#include <stdio.h>
#include <math.h>


int computeMean_(float *image, int nelements, float clipmin, float clipmax, 
                               int *numGoodPixels, float *mean, float *stddev, 
                                float *minValue, float *maxValue)
{
    int i;
    float tmpMinValue, tmpMaxValue;
    double sum, sumsq, sumdiff;

    /* Initialize some local variables */
    sum = 0;
    sumsq = 0;
    sumdiff = 0;
    
    /*Initialize the tmpMinValue and tmpMaxValue so that we can find the 
      largest and smallest non-clipped values */

    tmpMinValue = (clipmin > image[0]) ? clipmin : image[0];
    tmpMaxValue = (clipmax < image[0]) ? clipmax : image[0];

    for (i = 0; i < nelements; i++) {
        if ( (image[i] >= clipmin) && (image[i] <= clipmax) ) {
            /* Find lowest value in the clipped image */
            if (image[i] <= tmpMinValue) {
                tmpMinValue = image[i];
            }

            /* Find largest value in the clipped image */
            if (image[i] >= tmpMaxValue) {
                tmpMaxValue = image[i];
            }

            /* Increment the counter  of numGoodValues (i.e. not clipped) */
            *numGoodPixels = *numGoodPixels + 1;

            /* Compute the sum of the "good" pixels */
            sum = sum + image[i];

            /* Compute the sum of the pixels squared */
            sumsq = sumsq + (image[i] * image[i]);
        }
    }
 
    *minValue = tmpMinValue;
    *maxValue = tmpMaxValue;
    *mean = (float)(sum / *numGoodPixels);
   
    for (i=0; i < nelements; i++) {
        if ( (image[i] >= *minValue) && (image[i] <= *maxValue) ) {
            sumdiff = sumdiff + ( (image[i] - *mean) * (image[i] - *mean) );
	}
    }
    
    *stddev = (float)sqrt( sumdiff  / (*numGoodPixels - 1));    

    return 1;
}

static PyObject * computeMean(PyObject *obj, PyObject *args)
{
    PyObject *oimage;
    PyArrayObject *image;
    int status=0;
    int numGoodPixels;
    float clipmin, clipmax, mean, stddev, minValue, maxValue;

    if (!PyArg_ParseTuple(args,"Off:computeMean",&oimage, &clipmin, &clipmax))
	    return NULL;

    image = (PyArrayObject *)PyArray_ContiguousFromObject(oimage, PyArray_FLOAT, 1, 2);

    if (!image) return NULL;

    mean = 0;
    stddev = 0;
    numGoodPixels = 0;
    minValue = 0;
    maxValue = 0;

    status = computeMean_((float *)image->data, PyArray_Size((PyObject*)image), 
			  clipmin, clipmax,
			  &numGoodPixels, &mean, &stddev, &minValue, &maxValue);
    Py_XDECREF(image); 

    return Py_BuildValue("iffff",numGoodPixels,mean,stddev,minValue,maxValue);
}

static PyMethodDef computeMean_methods[] =
{
    {"computeMean",  computeMean, METH_VARARGS, 
        "computeMean(image, clipmin, clipmax, numGoodPixels, mean, stddev, minValue, maxValue)"},
    {0,            0}                             /* sentinel */
};

#if PY_MAJOR_VERSION >= 3
static struct PyModuleDef moduledef = {
  PyModuleDef_HEAD_INIT,
  "computeMean",           /* m_name */
  "C compute mean module", /* m_doc */
  -1,                      /* m_size */
  computeMean_methods,     /* m_methods */
  NULL,                    /* m_reload */
  NULL,                    /* m_traverse */
  NULL,                    /* m_clear */
  NULL,                    /* m_free */
};
#endif

PyMODINIT_FUNC
#if PY_MAJOR_VERSION >= 3
PyInit_computeMean(void)
#else
initcomputeMean(void)
#endif
{
	PyObject* m;
	import_array();

#if PY_MAJOR_VERSION >= 3
	m = PyModule_Create(&moduledef);
	return m;
#else
	m = Py_InitModule3("computeMean", computeMean_methods, "C compute mean module");
	if (m == NULL)
		return;
#endif
}
