BIN_DIR = bin

LINKCC = $(CXX)

CXX = g++
CXXFLAGS = -Wall -I./ -DHAVE_INLINE
LDFLAGS = -lgsl -lgslcblas

ifeq ($(CFG),debug)
  CXXFLAGS += -O0 -g -DDEBUG=true
else 
  CXXFLAGS += -O3 -DDEBUG=false
  ifeq ($(CFG),static)
    LDFLAGS += --static
  endif
endif

OBJS_UTILS := $(patsubst %.cpp,%.o,$(wildcard Utils/*.cpp))
OBJS_DATAS := $(patsubst %.cpp,%.o,$(wildcard Datas/*.cpp))
OBJS_CLASSIFIERS := $(patsubst %.cpp,%.o,$(wildcard Classifiers/*.cpp))
OBJS_LEARNERS := $(patsubst %.cpp,%.o,$(wildcard Learners/*.cpp))

all: PbscAlign PbscNonAlign PbscClassify

PbscAlign: $(OBJS_UTILS) $(OBJS_DATAS) $(OBJS_CLASSIFIERS) $(OBJS_LEARNERS) main_PbscAlign.o
	$(LINKCC) -o $(BIN_DIR)/pbsc_align $(OBJS_UTILS) $(OBJS_DATAS) $(OBJS_CLASSIFIERS) $(OBJS_LEARNERS) main_PbscAlign.o  $(LDFLAGS)

PbscNonAlign: $(OBJS_UTILS) $(OBJS_DATAS) $(OBJS_CLASSIFIERS) $(OBJS_LEARNERS) main_PbscNonAlign.o
	$(LINKCC) -o $(BIN_DIR)/pbsc_nonalign $(OBJS_UTILS) $(OBJS_DATAS) $(OBJS_CLASSIFIERS) $(OBJS_LEARNERS) main_PbscNonAlign.o  $(LDFLAGS)

PbscClassify: $(OBJS_UTILS) $(OBJS_DATAS) $(OBJS_CLASSIFIERS)  main_classify.o
	$(LINKCC) -o $(BIN_DIR)/pbsc_classify $(OBJS_UTILS) $(OBJS_DATAS) $(OBJS_CLASSIFIERS)  main_classify.o  $(LDFLAGS)

clean:
	-rm */*.o main_*.o $(BIN_DIR)/pbsc_align $(BIN_DIR)/pbsc_nonalign $(BIN_DIR)/pbsc_classify

