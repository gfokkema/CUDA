#ifndef RENDER_SESSION_H_
#define RENDER_SESSION_H_
#include "devices/device.h"
#include "scene.h"

#include <ctime>

class RenderSession {
public:
	RenderSession(Device* device, Scene* scene);
	~RenderSession();
	void render();
private:
	Device* _device;
	Scene* _scene;

	device_mem ray_dirs;
	device_mem shapes;
	device_mem buffer;
	std::chrono::time_point<std::chrono::system_clock> _start;
	unsigned frames = 0;
};

#endif /* RENDER_SESSION_H_ */
