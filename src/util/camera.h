#ifndef CAMERA_H_
#define CAMERA_H_

#include <cmath>

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
	Camera(int width, int height, float angle = 45, Vector pos = Vector(0,0,0), Vector dir = Vector(0,0,-1))
		: _width(width), _height(height), _pos(pos), _dir(dir), _right(_right = {tanf(angle / 360 * 2 * M_PI), 0, 0}) {}
	~Camera() {};

	void strafe(float velocity, float dt);
	void move(float velocity, float dt);
	void lookAt(float x, float y);

	const Vector pos() const   { return _pos; };
	const Vector dir() const   { return _dir; };
	const Vector up() const    { return _right % _dir * _height / float(_width);  };
	const Vector right() const { return _right; };
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
	Vector _right;
};

#endif /* CAMERA_H_ */
