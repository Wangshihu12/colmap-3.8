set(FOLDER_NAME "glomap")

COLMAP_ADD_SOURCES(
    global_pipeline.h global_pipeline.cc
    global_mapper.h global_mapper.cc
    observation_manager.h observation_manager.cc
    pose_graph.h pose_graph.cc
    poselib_relpose.h poselib_relpose.cc
    view_graph_calibration.h view_graph_calibration.cc
)
