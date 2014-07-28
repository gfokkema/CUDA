#include "camera.h"

void Camera::move(Vector dir, float dt) {
	_pos += dt * (	dir[0] * _dir % _up +
					dir[1] * _up +
					dir[2] * _dir);
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
