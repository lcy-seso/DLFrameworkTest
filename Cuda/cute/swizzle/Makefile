all: build

build:
	@mkdir -p build
	cd build && cmake ..
	cd build && make

clean:
	rm -rf build/*

run: build
	./build/swizzle_mode
	
.PHONY: all build clean run
