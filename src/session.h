#ifndef RENDER_SESSION_H_
#define RENDER_SESSION_H_

class RenderSession {
public:
	RenderSession(Device* device, Scene* scene);
	void render();
private:
	Buffer* _renderbuffer;
	Device* _device;
	Scene* _scene;
};

#endif /* RENDER_SESSION_H_ */
