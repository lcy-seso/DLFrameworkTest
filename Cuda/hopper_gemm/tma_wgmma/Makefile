.PHONY: all clean run

all:
	mkdir -p build
	cd build && cmake .. && make

run: all
	cd build && ./tma_wgmma

clean:
	rm -rf build
