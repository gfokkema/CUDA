#ifndef RENDER_SESSION_H_
#define RENDER_SESSION_H_
#include "devices/device.h"
#include "scene.h"

class RenderSession {
public:
	RenderSession(Device* device, Scene* scene);
	void render();
private:
	Device* _device;
	Scene* _scene;

	device_mem cam_buff;
	device_mem ray_dirs;
	device_mem shapes;
	device_mem buffer;
};

#endif /* RENDER_SESSION_H_ */
