#include "camera.h"

void Camera::strafe(float velocity, float dt) {
	_pos += velocity * dt * (_dir % _up);
}

void Camera::move(float velocity, float dt) {
	_pos += velocity * dt * _dir;
}

void Camera::lookAt(float x, float y) {
	float invwidth = 1.f / _width;
	float invheight = 1.f / _height;
	x = (x + .5) * invwidth - 0.5;
	y = (y + .5) * invheight - 0.5;
	Vector right = this->right();
	Vector newDir = (x * right + y * _up + _dir).normalize();
	Vector newUp  = (_up - (_up * newDir) * newDir).normalize();
	_dir = newDir, _up = newUp;
}
