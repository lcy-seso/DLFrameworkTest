.PHONY: all clean run

all:
	mkdir -p build
	cd build && cmake .. && make

run: all
	cd build && ./tma_load_pipeline

clean:
	rm -rf build 
