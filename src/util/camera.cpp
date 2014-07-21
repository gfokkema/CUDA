#include "camera.h"

void Camera::strafe(float velocity, float dt) {
	_pos += velocity * dt * _right;
}

void Camera::move(float velocity, float dt) {
	_pos += velocity * dt * _dir;
}

void Camera::lookAt(float x, float y) {
	float invwidth = 1.f / _width;
	float invheight = 1.f / _height;
	x = (x + .5) * invwidth - 0.5;
	y = (y + .5) * invheight - 0.5;
	Vector up = this->up();
	Vector newDir = (x * _right + y * up + _dir).normalize();
	Vector newRight  = (_right - (_right * newDir) * newDir).normalize() * _fovx;
	_dir = newDir, _right = newRight;
}
