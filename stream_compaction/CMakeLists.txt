set(SOURCE_FILES
  "compact.cu"
  "compact.hpp"
  )

cuda_add_library(stream_compaction
    ${SOURCE_FILES}
    OPTIONS -arch=sm_20
    )
