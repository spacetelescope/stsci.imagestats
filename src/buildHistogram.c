/*
    Program:    buildHistogram.c
    Author:     Christopher Hanley
    Purpose:    Populate a 1 dimensional python object to create a histogram

    Version:
            Version 0.1.0, 15-Jan-2004: Created -- CJH
            Version 0.1.1, 26-Feb-2004: Bug fix.  Py_DECREF() was not being 
                                        called for histogram.
            Version 0.1.2, 26-May-2004: Bug Fix.  When computing the index the
                difference needed to be case as type double to prevent loss of
                precision.
            Version 0.1.3, 27-May-2004: Removed Pydecref statments to avoid
                pointer errors.  -- CJH
            Version 0.1.4, 07-Jun-2004: Put Pydecref statements back into to plug 
                memory leak, corrected boundary conditions on clipping loop.
            Version 0.1.5, 21-Jun-2004: Added code to check that the population of
                histogram at boundaries is not corrupted by 
 

*/
#include <Python.h>
#include <string.h>
#include <stdio.h>


#ifdef NUMPY
    #include <numpy/arrayobject.h>
    #include <numpy/libnumarray.h>
#else
    #include <arrayobject.h>
    #include <libnumarray.h>
#endif


int populate1DHist_(float *image, int image_elements, 
		    unsigned int *histogram, int histogram_elements,
                    float minValue, float maxValue, float binWidth)
{
    int i, index=0;
    for (i = 0; i < image_elements; i++) {
        if ( (image[i] >= minValue) && (image[i] < maxValue) ) {
            index = (int)((double)(image[i] - minValue) / binWidth );

            /* Check the boundary conditions so that index of histogram remain is valid range 
	        assert((index >= 0) && (index < histogram_elements));
            if (!((index >= 0) && (index < histogram_elements))) {
              fprintf(stderr, "index: %d, image[%d]: %f, minValue: %f, maxValue: %f\n", 
                      index, i, image[i], minValue, maxValue);
              fprintf(stderr, "binWidth: %f, histogram_elements: %d\n",
                      binWidth, histogram_elements);
            }
            */

            /* Handle histogram population for floating point errors at end points */
            /* Case 1: Populating below index 0.*/
            if ( index < 0 ) {
                /*fprintf(stderr,"ERROR: trying to populate a negative index!\n");*/
                histogram[0] += 1;
            }
            /* Case 2: Populating above the maximum index value*/
            else if (index >= histogram_elements ) {
                /*fprintf(stderr,"ERROR: trying to populate past end of histogram!\n");*/
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

    image = (PyArrayObject *)NA_InputArray(oimage, tFloat32, C_ARRAY);
    if (!image) return NULL;

    histogram = (PyArrayObject *)NA_IoArray(ohistogram, tUInt32, C_ARRAY);
    if (!histogram) return NULL;
    
    status = populate1DHist_(NA_OFFSETDATA(image), NA_elements(image),
			     NA_OFFSETDATA(histogram), NA_elements(histogram),
			     minValue, maxValue, binWidth);

    Py_XDECREF(image);
    Py_XDECREF(histogram);

    return Py_BuildValue("i",status);
}

static PyMethodDef buildHistogram_methods[] =
{
    {"populate1DHist",  populate1DHist, METH_VARARGS, 
        "populate1Dhist(image, histogram, minValue, maxValue, binWidth)"},
    {0,            0}                             /* sentinel */
};

void initbuildHistogram(void) {
    Py_InitModule("buildHistogram", buildHistogram_methods);
    import_libnumarray();
}

