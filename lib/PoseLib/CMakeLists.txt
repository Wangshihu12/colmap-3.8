set(POSELIB_SOURCES
    solvers/gp3p.cc
    solvers/gp4ps.cc
    solvers/p1p2ll.cc
    solvers/p2p1ll.cc
    solvers/p2p2pl.cc
    solvers/p3ll.cc
    solvers/p3p.cc
    solvers/p3p_lambdatwist.cc
    solvers/p4pf.cc
    solvers/p35pf.cc
    solvers/p5pf.cc
    solvers/p5pfr.cc
    solvers/p3p1llf.cc
    solvers/p2p2llf.cc
    solvers/p1p3llf.cc
    solvers/p4llf.cc
    solvers/p5lp_radial.cc
    solvers/p6lp.cc
    solvers/ugp2p.cc
    solvers/ugp3ps.cc
    solvers/up1p2pl.cc
    solvers/up2p.cc
    solvers/up1p1ll.cc
    solvers/up4pl.cc
    solvers/ugp4pl.cc
    solvers/relpose_upright_3pt.cc
    solvers/relpose_upright_planar_2pt.cc
    solvers/relpose_upright_planar_3pt.cc
    solvers/relpose_8pt.cc
    solvers/relpose_5pt.cc
    solvers/relpose_monodepth_3pt.cc
    solvers/relpose_7pt.cc
    solvers/relpose_6pt_focal.cc
    solvers/relpose_k2Fk1_10pt.cc
    solvers/relpose_kFk_9pt.cc
    solvers/relpose_monodepth_3pt_shared_focal.cc
    solvers/relpose_monodepth_3pt_varying_focal.cc
    solvers/gen_relpose_upright_4pt.cc
    solvers/gen_relpose_5p1pt.cc
    solvers/gen_relpose_6pt.cc
    solvers/homography_4pt.cc
    misc/qep.cc
    misc/univariate.cc
    misc/essential.cc
    misc/re3q3.cc
    misc/camera_models.cc
    misc/decompositions.cc
    robust.cc
    robust/ransac.cc
    robust/robust_loss.cc
    robust/bundle.cc
    robust/utils.cc
    robust/sampling.cc
    robust/recalibrator.cc
    robust/estimators/absolute_pose.cc
    robust/estimators/relative_pose.cc
    robust/estimators/hybrid_pose.cc
    robust/estimators/homography.cc
    robust/optim/lm_impl.cc
    robust/hybrid_ransac.cc
    robust/hybrid_estimators/hybrid_point_line_absolute_pose.cc
)

add_library(PoseLib ${POSELIB_SOURCES})
target_link_libraries(PoseLib PUBLIC Eigen3::Eigen)
target_include_directories(PoseLib PUBLIC ${CMAKE_SOURCE_DIR}/lib)
target_compile_features(PoseLib PUBLIC cxx_std_17)

if(MSVC)
    target_compile_options(PoseLib PRIVATE /bigobj)
else()
    target_compile_options(PoseLib PRIVATE
        -O3
        -Wall
        -Wno-sign-compare
        -Wno-unused-variable
        -Wno-maybe-uninitialized
        -Wno-deprecated-declarations
        -Wno-unknown-pragmas
        -fPIC
    )
endif()
