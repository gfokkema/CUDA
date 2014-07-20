#ifndef DEVICE_MEM_H__
#define DEVICE_MEM_H__

enum permission : uint8_t {
	PERM_WRITE_ONLY,
	PERM_READ_ONLY,
	PERM_READ_WRITE
};

struct device_mem {
	uintptr_t 	_mem_pointer;
	size_t		_mem_size;

	void* operator&() {
		return (void*) _mem_pointer;
	}
};

#endif /* DEVICE_MEM_H__ */
