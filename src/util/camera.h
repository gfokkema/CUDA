#ifndef CAMERA_H_
#define CAMERA_H_

#include "vector.h"

typedef struct camera {
	int width, height;
	float4 pos;
	float4 dir;
	float4 up;
	float4 right;
} camera;

class Camera {
public:
	Camera(int width, int height, Vector pos = Vector(0,0,0), Vector dir = Vector(0,0,-1), Vector up = Vector(0,1,0))
		: _width(width), _height(height), _pos(pos), _dir(dir), _up(up) {};
	~Camera() {};

	void strafe(float velocity, float dt);
	void move(float velocity, float dt);
	void lookAt(float x, float y);

	const Vector pos() const   { return _pos; };
	const Vector dir() const   { return _dir; };
	const Vector up() const    { return _up;  };
	const Vector right() const { return _dir % _up * (float(_width) / float(_height)); };
	const int width() const    { return _width; };
	const int height() const   { return _height; };
	const camera gpu_type() const { return {	_width, _height,
												pos().gpu_type(),
												dir().gpu_type(),
												up().gpu_type(),
												right().gpu_type() }; };
private:
	int _width, _height;
	Vector _pos;
	Vector _dir;
	Vector _up;
};

#endif /* CAMERA_H_ */
