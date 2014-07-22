#ifndef DEVICE_MEM_H__
#define DEVICE_MEM_H__
#ifdef __APPLE__
  #include <OpenCL/cl.h>
#else
  #include <CL/cl.h>
#endif

#define MEM_TYPE_READ_WRITE		CL_MEM_READ_WRITE
#define MEM_TYPE_WRITE_ONLY		CL_MEM_WRITE_ONLY
#define MEM_TYPE_READ_ONLY		CL_MEM_READ_ONLY
#define MEM_TYPE_USE_HOST_PTR		CL_MEM_USE_HOST_PTR
#define MEM_TYPE_ALLOC_HOST_PTR		CL_MEM_ALLOC_HOST_PTR
#define MEM_TYPE_COPY_HOST_PTR		CL_MEM_COPY_HOST_PTR
#define MEM_TYPE_HOST_WRITE_ONLY	CL_MEM_HOST_WRITE_ONLY
#define MEM_TYPE_HOST_READ_ONLY		CL_MEM_HOST_READ_ONLY
#define MEM_TYPE_HOST_NO_ACCESS		CL_MEM_HOST_NO_ACCESS
#define mem_flags cl_mem_flags

struct device_mem {
	uintptr_t 	_mem_pointer;
	size_t		_mem_size;
};

#endif /* DEVICE_MEM_H__ */
