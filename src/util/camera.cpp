#include "camera.h"

void Camera::strafe(float velocity, float dt) {
	_pos += velocity * dt * (_dir % _up);
}

void Camera::move(float velocity, float dt) {
	_pos += velocity * dt * _dir;
}
