/* These are functions that do not rely on FLT.
   They are organized by originating file.
*/
//TODO: remove kernels that do not depend on dimension

#include <thrust/device_ptr.h>
#include <thrust/scan.h>
#include <cuComplex.h>

#include "precision_independent.h"
