#include "util/camera.h"
#include "util/vector.h"

int main() {
	Vector vec1(1,2,3);
	Vector vec2(1,2,3);
	Vector vec = (vec1 + vec2).normalize();
	std::cout << vec << " length: " << vec.length() << std::endl;

	Camera cam(512, 512);
	std::cout << "pos:\t" << cam.pos() << std::endl;
	std::cout << "dir:\t" << cam.dir() << std::endl;
	std::cout << "up:\t" << cam.up() << std::endl;
	std::cout << "right:\t" << cam.right() << std::endl;
}
