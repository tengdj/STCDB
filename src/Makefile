CC = gcc
CXX = g++
NVCC = nvcc

SRCS := $(wildcard */*.cpp)
OBJS := $(patsubst %.cpp,%.o,$(SRCS))

INCLUDES	= -I /usr/include -I ./src -I /usr/local/cuda/targets/x86_64-linux/include/
LIBS		= -L/usr/lib/x86_64-linux-gnu -lboost_program_options -lpthread -fopenmp -pg -L/usr/local/lib -I/usr/local/include  -lrocksdb -ltbb

CPPFLAGS	= -g -std=c++17 -Wno-cpp -fopenmp

CUDALIBS	= -L/usr/local/cuda/lib64 -lcudart

GEOMETRY_SRCS := $(wildcard geometry/*.cpp)
GEOMETRY_OBJS := $(patsubst %.cpp,%.o,$(GEOMETRY_SRCS))

TRACING_SRCS := $(wildcard tracing/*.cpp)
TRACING_OBJS := $(patsubst %.cpp,%.o,$(TRACING_SRCS))

#compile all the cu files
CU_SRCS := $(wildcard cuda/*.cu)
CU_OBJS := $(patsubst %.cu,%_cu.o,$(CU_SRCS))


ifdef USE_GPU
	CPPFLAGS += -DUSE_GPU
	LIBS  += ${CUDALIBS}
	GEOMETRY_OBJS += ${CU_OBJS}
endif

%_cu.o: %.cu
	$(NVCC) -c -g -std=c++17 -arch=sm_60 -DCUB_IGNORE_DEPRECATED_CPP_DIALECT --extended-lambda $(INCLUDES) -o $@ $<

%.o:	%.cpp
	$(CXX) -c $(CPPFLAGS) $(INCLUDES) -o $@ $<

all:	 pipeline datagen load_search compaction points_to_csv meetings_to_csv trace rocksdb raid_test transfer meta_search batchsize_ex


# for macro queries
pipeline:		tools/pipeline.o $(GEOMETRY_OBJS) $(TRACING_OBJS)
	$(CXX) -o ../build/$@ $^ $(LIBS)

datagen:	tools/datagen.o $(GEOMETRY_OBJS) $(TRACING_OBJS)
	$(CXX) -o ../build/$@ $^ $(LIBS)

trace:	tools/trace.o $(GEOMETRY_OBJS) $(TRACING_OBJS)
	$(CXX) -o ../build/$@ $^ $(LIBS)

load_search:		tools/load_search.o $(GEOMETRY_OBJS) $(TRACING_OBJS)
	$(CXX) -o ../build/$@ $^ $(LIBS)

compaction:		tools/compaction.o $(GEOMETRY_OBJS) $(TRACING_OBJS)
	$(CXX) -o ../build/$@ $^ $(LIBS)

points_to_csv:		tools/points_to_csv.o $(GEOMETRY_OBJS) $(TRACING_OBJS)
	$(CXX) -o ../build/$@ $^ $(LIBS)

meetings_to_csv:		tools/meetings_to_csv.o $(GEOMETRY_OBJS) $(TRACING_OBJS)
	$(CXX) -o ../build/$@ $^ $(LIBS)

rocksdb:		tools/rocksdb.o $(GEOMETRY_OBJS) $(TRACING_OBJS)
	$(CXX) -o ../build/$@ $^ $(LIBS)

raid_test:		tools/ssd_test.o $(GEOMETRY_OBJS) $(TRACING_OBJS)
	$(CXX) -o ../build/$@ $^ $(LIBS)

transfer:		tools/transfer.o $(GEOMETRY_OBJS) $(TRACING_OBJS)
	$(CXX) -o ../build/$@ $^ $(LIBS)

meta_search:		tools/meta_change_search.o $(GEOMETRY_OBJS) $(TRACING_OBJS)
	$(CXX) -o ../build/$@ $^ $(LIBS)

batchsize_ex:		tools/batchsize_ex.o $(GEOMETRY_OBJS) $(TRACING_OBJS)
	$(CXX) -o ../build/$@ $^ $(LIBS)

#searchsst:	tools/searchsst.o $(GEOMETRY_OBJS) $(TRACING_OBJS)
#	$(CXX) -o ../build/$@ $^ $(LIBS)

printtrace:	tools/printtrace.o $(GEOMETRY_OBJS) $(TRACING_OBJS)
	$(CXX) -o ../build/$@ $^ $(LIBS)

parse:	tools/parsetweets.o $(GEOMETRY_OBJS) $(TRACING_OBJS)
	$(CXX) -o ../build/$@ $^ $(LIBS)

combiner:	tools/combiner.o $(GEOMETRY_OBJS) $(TRACING_OBJS)
	$(CXX) -o ../build/$@ $^ $(LIBS)

cudatest:	tools/cudatest.o cuda/test_cu.o
	$(CXX) -o ../build/$@ $^ $(LIBS) $(CUDALIBS)

hashtest:	tools/hashtest.o cuda/hash_cu.o
	$(CXX) -o ../build/$@ $^ $(LIBS) $(CUDALIBS)

clean:
	rm -fr ../build/* $(OBJS) $(CU_OBJS)