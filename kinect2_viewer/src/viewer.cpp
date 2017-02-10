/**
 * Copyright 2014 University of Bremen, Institute for Artificial Intelligence
 * Author: Thiemo Wiedemeyer <wiedemeyer@cs.uni-bremen.de>
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#include <stdlib.h>
#include <stdio.h>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>
#include <cmath>
#include <mutex>
#include <thread>
#include <chrono>

#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <pcl/io/pcd_io.h>
#include <pcl/visualization/cloud_viewer.h>
#include <pcl_conversions/pcl_conversions.h>
//#include <pcl/filters/voxel_grid.h>
#include <pcl/filters/extract_indices.h>
#include <pcl/filters/passthrough.h>
#include <pcl/filters/crop_box.h>
//#include <pcl/filters/boost.h>
#include <pcl/common/transforms.h>
#include <pcl/common/common_headers.h>
#include <pcl/kdtree/kdtree_flann.h>

#include <opencv2/opencv.hpp>

#include <ros/ros.h>
#include <ros/spinner.h>
#include <sensor_msgs/CameraInfo.h>
#include <sensor_msgs/Image.h>

#include <cv_bridge/cv_bridge.h>

#include <image_transport/image_transport.h>
#include <image_transport/subscriber_filter.h>

#include <message_filters/subscriber.h>
#include <message_filters/synchronizer.h>
#include <message_filters/sync_policies/exact_time.h>
#include <message_filters/sync_policies/approximate_time.h>

#include <kinect2_bridge/kinect2_definitions.h>

#include <tf/transform_listener.h>
#include <tf/transform_broadcaster.h>
#include <tf/transform_datatypes.h>
#include <tf_conversions/tf_eigen.h>

// MoveIt!
#include <moveit/robot_model_loader/robot_model_loader.h>
#include <moveit/robot_model/robot_model.h>
#include <moveit/robot_state/robot_state.h>
//#include <moveit/robot_state/joint_state_group.h>
#include <moveit/robot_state/conversions.h>
#include <moveit/move_group_interface/move_group.h>

#include <geometry_msgs/PointStamped.h>
#include <sensor_msgs/JointState.h>
//#include <tf2/convert.h>
//#include <Eigen/Geometry>

class Receiver
{
public:
  enum Mode
  {
    IMAGE = 0,
    CLOUD,
    BOTH
  };

private:
  std::mutex lock;

  const std::string topicColor, topicDepth;
  const bool useExact, useCompressed;

  bool updateImage, updateCloud;
  bool save;
  bool running;
  size_t frame;
  const size_t queueSize;

  cv::Mat color, depth;
  cv::Mat cameraMatrixColor, cameraMatrixDepth;
  cv::Mat lookupX, lookupY;

  typedef message_filters::sync_policies::ExactTime<sensor_msgs::Image, sensor_msgs::Image, sensor_msgs::CameraInfo, sensor_msgs::CameraInfo> ExactSyncPolicy;
  typedef message_filters::sync_policies::ApproximateTime<sensor_msgs::Image, sensor_msgs::Image, sensor_msgs::CameraInfo, sensor_msgs::CameraInfo> ApproximateSyncPolicy;

  ros::NodeHandle nh;
  ros::AsyncSpinner spinner;
  image_transport::ImageTransport it;
  image_transport::SubscriberFilter *subImageColor, *subImageDepth;
  message_filters::Subscriber<sensor_msgs::CameraInfo> *subCameraInfoColor, *subCameraInfoDepth;

  message_filters::Synchronizer<ExactSyncPolicy> *syncExact;
  message_filters::Synchronizer<ApproximateSyncPolicy> *syncApproximate;

  std::thread imageViewerThread;
  Mode mode;

  pcl::PointCloud<pcl::PointXYZRGB>::Ptr cloud;
  pcl::PointCloud<pcl::PointXYZRGB>::Ptr cloud_filtered;
  pcl::PointCloud<pcl::PointXYZRGB>::Ptr cloud_sub1;
  pcl::PointCloud<pcl::PointXYZRGB>::Ptr cloud_sub2;
  pcl::PointCloud<pcl::PointXYZRGB>::Ptr cloud_sub3;
   pcl::PointCloud<pcl::PointXYZRGB>::Ptr cloud_sub4;
  pcl::PCDWriter writer;
  std::ostringstream oss;
  std::vector<int> params;

  //add new
  robot_model_loader::RobotModelLoader robot_model_loader;
  robot_model::RobotModelPtr kinematic_model;
  moveit::core::RobotStatePtr kinematic_state;
  tf::TransformListener tf_listener_;
  std::map<std::string, double> joints;
  ros::Subscriber js_sub;
  Eigen::Vector4f minPoint, maxPoint;

public:
  Receiver(const std::string &topicColor, const std::string &topicDepth, const bool useExact, const bool useCompressed)
    : topicColor(topicColor), topicDepth(topicDepth), useExact(useExact), useCompressed(useCompressed),
      updateImage(false), updateCloud(false), save(false), running(false), frame(0), queueSize(5),
      nh("~"), spinner(0), it(nh), mode(CLOUD)
  {
    cameraMatrixColor = cv::Mat::zeros(3, 3, CV_64F);
    cameraMatrixDepth = cv::Mat::zeros(3, 3, CV_64F);
    params.push_back(cv::IMWRITE_JPEG_QUALITY);
    params.push_back(100);
    params.push_back(cv::IMWRITE_PNG_COMPRESSION);
    params.push_back(1);
    params.push_back(cv::IMWRITE_PNG_STRATEGY);
    params.push_back(cv::IMWRITE_PNG_STRATEGY_RLE);
    params.push_back(0);

    //    for(int i = 0; i < joints.size(); ++i) {
    //        Map::iterator it = joints.begin();
    //        //std::advance(it, i);
    //        ROS_INFO_STREAM("%s " << it->first);
    //        ROS_INFO_STREAM("%f " << it->second);}

    //    std::map<string, float>::iterator it;

    //    for(it=joints.begin();it!=joints.end();it++)
    //    {
    //        std::cout<<it->first<<std::endl;
    //        std::cout<<it->second<<std::endl;
    //    }
    //add
    js_sub = nh.subscribe("/joint_states", 10, &Receiver::js_callback, this);
    robot_model_loader = robot_model_loader::RobotModelLoader ("robot_description");
    kinematic_model = robot_model_loader.getModel();
    ROS_INFO("Model frame: %s", kinematic_model->getModelFrame().c_str());
    kinematic_state.reset(new robot_state::RobotState(kinematic_model));
    kinematic_state->setToDefaultValues();
    //const robot_state::JointModelGroup* joint_model_group1 = kinematic_model->getJointModelGroup("Robot");
    kinematic_model->getJointModelGroup("Robot");
    kinematic_model->getJointModelGroup("Left_SideGripper");

//    joints["left_elbow_joint"] = 1.362766;
//    joints["left_shoulder_lift_joint"] = -0.323989;
//    joints["left_shoulder_pan_joint"] = -1.721952;
//    joints["left_wrist_1_joint"] = -2.609463;
//    joints["left_wrist_2_joint"] = -1.570665;
//    joints["left_wrist_3_joint"] = -0.151294;
//    //joints["left_joint_vaccum_ee"] = 0.000000;

//    joints["left_pi4_gripper_prismatic_joint"] = 0.003000;
//    joints["left_pi4_gripper_finger1_joint"] = 0.009200;
//    joints["left_pi4_gripper_finger2_joint"] = -0.009200;

//    joints["right_elbow_joint"] = -1.163788;
//    joints["right_shoulder_lift_joint"] = -2.888825;
//    joints["right_shoulder_pan_joint"] = 1.762168;
//    joints["right_wrist_1_joint"] = -0.659665;
//    joints["right_wrist_2_joint"] = 1.570607;
//    joints["right_wrist_3_joint"] = 0.191639;
    // joints["right_joint_vaccum_ee"] = 0.000000;

    //kinematic_state->setVariablePositions (joints);

  }

  ~Receiver()
  {
  }

  void run(const Mode mode)
  {
    start(mode);
    stop();
  }

private:
  void start(const Mode mode)
  {
    this->mode = mode;
    running = true;

    std::string topicCameraInfoColor = topicColor.substr(0, topicColor.rfind('/')) + "/camera_info";
    std::string topicCameraInfoDepth = topicDepth.substr(0, topicDepth.rfind('/')) + "/camera_info";

    image_transport::TransportHints hints(useCompressed ? "compressed" : "raw");
    subImageColor = new image_transport::SubscriberFilter(it, topicColor, queueSize, hints);
    subImageDepth = new image_transport::SubscriberFilter(it, topicDepth, queueSize, hints);
    subCameraInfoColor = new message_filters::Subscriber<sensor_msgs::CameraInfo>(nh, topicCameraInfoColor, queueSize);
    subCameraInfoDepth = new message_filters::Subscriber<sensor_msgs::CameraInfo>(nh, topicCameraInfoDepth, queueSize);


    if(useExact)
      {
        syncExact = new message_filters::Synchronizer<ExactSyncPolicy>(ExactSyncPolicy(queueSize), *subImageColor, *subImageDepth, *subCameraInfoColor, *subCameraInfoDepth);
        syncExact->registerCallback(boost::bind(&Receiver::callback, this, _1, _2, _3, _4));
      }
    else
      {
        syncApproximate = new message_filters::Synchronizer<ApproximateSyncPolicy>(ApproximateSyncPolicy(queueSize), *subImageColor, *subImageDepth, *subCameraInfoColor, *subCameraInfoDepth);
        syncApproximate->registerCallback(boost::bind(&Receiver::callback, this, _1, _2, _3, _4));
      }

    spinner.start();

    std::chrono::milliseconds duration(1);
    while(!updateImage || !updateCloud)
      {
        if(!ros::ok())
          {
            return;
          }
        std::this_thread::sleep_for(duration);
      }
    cloud = pcl::PointCloud<pcl::PointXYZRGB>::Ptr(new pcl::PointCloud<pcl::PointXYZRGB>());
    cloud->height = color.rows;
    cloud->width = color.cols;
    cloud->is_dense = false;
    cloud->points.resize(cloud->height * cloud->width);
    createLookup(this->color.cols, this->color.rows);

    cloud_filtered = pcl::PointCloud<pcl::PointXYZRGB>::Ptr(new pcl::PointCloud<pcl::PointXYZRGB>());
    cloud_filtered->height = color.rows;
    cloud_filtered->width = color.cols;
    cloud_filtered->is_dense = false;
    cloud_filtered->points.resize(cloud_filtered->height * cloud_filtered->width);

    cloud_sub1 = pcl::PointCloud<pcl::PointXYZRGB>::Ptr(new pcl::PointCloud<pcl::PointXYZRGB>());
    cloud_sub2 = pcl::PointCloud<pcl::PointXYZRGB>::Ptr(new pcl::PointCloud<pcl::PointXYZRGB>());
    cloud_sub3 = pcl::PointCloud<pcl::PointXYZRGB>::Ptr(new pcl::PointCloud<pcl::PointXYZRGB>());
    cloud_sub4 = pcl::PointCloud<pcl::PointXYZRGB>::Ptr(new pcl::PointCloud<pcl::PointXYZRGB>());


    switch(mode)
      {
      case CLOUD:
        cloudViewer();
        break;
      case IMAGE:
        imageViewer();
        break;
      case BOTH:
        imageViewerThread = std::thread(&Receiver::imageViewer, this);
        cloudViewer();
        break;
      }
  }

  void stop()
  {
    spinner.stop();

    if(useExact)
      {
        delete syncExact;
      }
    else
      {
        delete syncApproximate;
      }

    delete subImageColor;
    delete subImageDepth;
    delete subCameraInfoColor;
    delete subCameraInfoDepth;

    running = false;
    if(mode == BOTH)
      {
        imageViewerThread.join();
      }
  }

  void callback(const sensor_msgs::Image::ConstPtr imageColor, const sensor_msgs::Image::ConstPtr imageDepth,
                const sensor_msgs::CameraInfo::ConstPtr cameraInfoColor, const sensor_msgs::CameraInfo::ConstPtr cameraInfoDepth)
  {
    cv::Mat color, depth;

    readCameraInfo(cameraInfoColor, cameraMatrixColor);
    readCameraInfo(cameraInfoDepth, cameraMatrixDepth);
    readImage(imageColor, color);
    readImage(imageDepth, depth);

    // IR image input
    if(color.type() == CV_16U)
      {
        cv::Mat tmp;
        color.convertTo(tmp, CV_8U, 0.02);
        cv::cvtColor(tmp, color, CV_GRAY2BGR);
      }

    lock.lock();
    this->color = color;
    this->depth = depth;
    updateImage = true;
    updateCloud = true;
    lock.unlock();
  }

  void imageViewer()
  {
    cv::Mat color, depth, depthDisp, combined;
    std::chrono::time_point<std::chrono::high_resolution_clock> start, now;
    double fps = 0;
    size_t frameCount = 0;
    std::ostringstream oss;
    const cv::Point pos(5, 15);
    const cv::Scalar colorText = CV_RGB(255, 255, 255);
    const double sizeText = 0.5;
    const int lineText = 1;
    const int font = cv::FONT_HERSHEY_SIMPLEX;

    cv::namedWindow("Image Viewer");
    oss << "starting...";

    start = std::chrono::high_resolution_clock::now();
    for(; running && ros::ok();)
      {
        if(updateImage)
          {
            lock.lock();
            color = this->color;
            depth = this->depth;
            updateImage = false;
            lock.unlock();

            ++frameCount;
            now = std::chrono::high_resolution_clock::now();
            double elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(now - start).count() / 1000.0;
            if(elapsed >= 1.0)
              {
                fps = frameCount / elapsed;
                oss.str("");
                oss << "fps: " << fps << " ( " << elapsed / frameCount * 1000.0 << " ms)";
                start = now;
                frameCount = 0;
              }

            dispDepth(depth, depthDisp, 12000.0f);
            combine(color, depthDisp, combined);
            //combined = color;

            cv::putText(combined, oss.str(), pos, font, sizeText, colorText, lineText, CV_AA);
            cv::imshow("Image Viewer", combined);
          }

        int key = cv::waitKey(1);
        switch(key & 0xFF)
          {
          case 27:
          case 'q':
            running = false;
            break;
          case ' ':
          case 's':
            if(mode == IMAGE)
              {
                createCloud(depth, color, cloud);
                saveCloudAndImages(cloud, color, depth, depthDisp);
              }
            else
              {
                save = true;
              }
            break;
          }
      }
    cv::destroyAllWindows();
    cv::waitKey(100);
  }

  void processMyCloud(pcl::visualization::PCLVisualizer::Ptr visualizer)
  {
      ROS_ERROR("get the global link transform");
      const Eigen::Affine3d &joint_state0 = kinematic_state->getGlobalLinkTransform("left_base_link");
      const Eigen::Affine3d &joint_state1 = kinematic_state->getGlobalLinkTransform("left_shoulder_link");
      const Eigen::Affine3d &joint_state2 = kinematic_state->getGlobalLinkTransform("left_upper_arm_link");
      const Eigen::Affine3d &joint_state3 = kinematic_state->getGlobalLinkTransform("left_forearm_link");
      const Eigen::Affine3d &joint_state4 = kinematic_state->getGlobalLinkTransform("left_wrist_1_link");
      const Eigen::Affine3d &joint_state5 = kinematic_state->getGlobalLinkTransform("left_wrist_2_link");
      const Eigen::Affine3d &joint_state6 = kinematic_state->getGlobalLinkTransform("left_wrist_3_link");
      const Eigen::Affine3d &joint_state7 = kinematic_state->getGlobalLinkTransform("left_ee_link");
      const Eigen::Affine3d &joint_state8 = kinematic_state->getGlobalLinkTransform("left_base_link_gripper");
      const Eigen::Affine3d &joint_state9 = kinematic_state->getGlobalLinkTransform("left_ee_gripper_link");
      const Eigen::Affine3d &joint_state10 = kinematic_state->getGlobalLinkTransform("left_pi4_gripper_fixed_link");
      const Eigen::Affine3d &joint_state11 = kinematic_state->getGlobalLinkTransform("left_pi4_gripper_prismatic_link");
      const Eigen::Affine3d &joint_state110 = kinematic_state->getGlobalLinkTransform("left_pi4_gripper_finger1_link");
      const Eigen::Affine3d &joint_state111 = kinematic_state->getGlobalLinkTransform("left_pi4_gripper_finger2_link");
      const Eigen::Affine3d &joint_state112 = kinematic_state->getGlobalLinkTransform("left_ee_pi4_gripper_link");



      const Eigen::Affine3d &joint_state12 = kinematic_state->getGlobalLinkTransform("right_base_link");
      const Eigen::Affine3d &joint_state14 = kinematic_state->getGlobalLinkTransform("right_shoulder_link");
      const Eigen::Affine3d &joint_state15 = kinematic_state->getGlobalLinkTransform("right_upper_arm_link");
      const Eigen::Affine3d &joint_state16 = kinematic_state->getGlobalLinkTransform("right_forearm_link");
      const Eigen::Affine3d &joint_state17 = kinematic_state->getGlobalLinkTransform("right_wrist_1_link");
      const Eigen::Affine3d &joint_state18 = kinematic_state->getGlobalLinkTransform("right_wrist_2_link");
      const Eigen::Affine3d &joint_state19 = kinematic_state->getGlobalLinkTransform("right_wrist_3_link");
      const Eigen::Affine3d &joint_state20 = kinematic_state->getGlobalLinkTransform("right_ee_link");
      const Eigen::Affine3d &joint_state21 = kinematic_state->getGlobalLinkTransform("right_base_link_gripper");
      const Eigen::Affine3d &joint_state22 = kinematic_state->getGlobalLinkTransform("right_ee_gripper_link");
      const Eigen::Affine3d &joint_state23 = kinematic_state->getGlobalLinkTransform("Left_stereoCam_link");
      const Eigen::Affine3d &joint_state24 = kinematic_state->getGlobalLinkTransform("stereoCam_link");

      ROS_ERROR("store the translation data");

      double x0 = joint_state0.translation().x();
      double y0 = joint_state0.translation().y();
      double z0 = joint_state0.translation().z();

      double x1 = joint_state1.translation().x();
      double y1 = joint_state1.translation().y();
      double z1 = joint_state1.translation().z();

      double x2 = joint_state2.translation().x();
      double y2 = joint_state2.translation().y();
      double z2 = joint_state2.translation().z();

      double x3 = joint_state3.translation().x();
      double y3 = joint_state3.translation().y();
      double z3 = joint_state3.translation().z();

      double x4 = joint_state4.translation().x();
      double y4 = joint_state4.translation().y();
      double z4 = joint_state4.translation().z();

      double x5 = joint_state5.translation().x();
      double y5 = joint_state5.translation().y();
      double z5 = joint_state5.translation().z();

      double x6 = joint_state6.translation().x();
      double y6 = joint_state6.translation().y();
      double z6 = joint_state6.translation().z();

      double x7 = joint_state7.translation().x();
      double y7 = joint_state7.translation().y();
      double z7 = joint_state7.translation().z();

      double x8 = joint_state8.translation().x();
      double y8 = joint_state8.translation().y();
      double z8 = joint_state8.translation().z();

      double x9 = joint_state9.translation().x();
      double y9 = joint_state9.translation().y();
      double z9 = joint_state9.translation().z();

      double x10 = joint_state10.translation().x();
      double y10 = joint_state10.translation().y();
      double z10 = joint_state10.translation().z();

      double x11 = joint_state11.translation().x();
      double y11 = joint_state11.translation().y();
      double z11 = joint_state11.translation().z();

      double x110 = joint_state110.translation().x();
      double y110 = joint_state110.translation().y();
      double z110 = joint_state110.translation().z();

      double x111 = joint_state111.translation().x();
      double y111 = joint_state111.translation().y();
      double z111 = joint_state111.translation().z();

      double x112 = joint_state112.translation().x();
      double y112 = joint_state112.translation().y();
      double z112 = joint_state112.translation().z();

      double x12 = joint_state12.translation().x();
      double y12 = joint_state12.translation().y();
      double z12 = joint_state12.translation().z();

      double x14 = joint_state14.translation().x();
      double y14 = joint_state14.translation().y();
      double z14 = joint_state14.translation().z();

      double x15 = joint_state15.translation().x();
      double y15 = joint_state15.translation().y();
      double z15 = joint_state15.translation().z();

      double x16 = joint_state16.translation().x();
      double y16 = joint_state16.translation().y();
      double z16 = joint_state16.translation().z();

      double x17 = joint_state17.translation().x();
      double y17 = joint_state17.translation().y();
      double z17 = joint_state17.translation().z();

      double x18 = joint_state18.translation().x();
      double y18 = joint_state18.translation().y();
      double z18 = joint_state18.translation().z();

      double x19 = joint_state19.translation().x();
      double y19 = joint_state19.translation().y();
      double z19 = joint_state19.translation().z();

      double x20 = joint_state20.translation().x();
      double y20 = joint_state20.translation().y();
      double z20 = joint_state20.translation().z();

      double x21 = joint_state21.translation().x();
      double y21 = joint_state21.translation().y();
      double z21 = joint_state21.translation().z();

      double x22 = joint_state22.translation().x();
      double y22 = joint_state22.translation().y();
      double z22 = joint_state22.translation().z(); //hard coded because values from vision are not always reliable


//        ROS_INFO_STREAM("Rotation left_pose: " << left_pose3.rotation());
//        ROS_INFO_STREAM("Translation: " << joint_state20.translation());
//        ROS_INFO_STREAM("Rotation: " << joint_state20.rotation());
//        ROS_INFO_STREAM("Translation: " << l_q);




      //      Eigen::Matrix4d matrix23 = joint_state23.matrix();
      //      Eigen::Matrix4d matrix24 = joint_state24.matrix();


      geometry_msgs::PointStamped pt1, pt1_, pt2, pt2_, pt3, pt3_,
          pt0 , pt0_, pt4, pt4_, pt5, pt5_, pt6, pt6_,pt7, pt7_, pt8, pt8_, pt9, pt9_,pt10, pt10_, pt11, pt11_, pt110, pt110_, pt111, pt111_, pt112, pt112_,
          pt12, pt12_, pt14, pt14_, pt15, pt15_, pt16, pt16_, pt17, pt17_, pt18, pt18_, pt19, pt19_, pt20, pt20_, pt21, pt21_, pt22, pt22_, pt23, pt23_, pt24, pt24_;


      pt0.header.frame_id = "/world";
      pt1.header.frame_id = "/world";
      pt2.header.frame_id = "/world";
      pt3.header.frame_id = "/world";
      pt4.header.frame_id = "/world";
      pt5.header.frame_id = "/world";
      pt6.header.frame_id = "/world";
      pt7.header.frame_id = "/world";
      pt8.header.frame_id = "/world";
      pt9.header.frame_id = "/world";
      pt10.header.frame_id = "/world";
      pt11.header.frame_id = "/world";
      pt110.header.frame_id = "/world";
      pt111.header.frame_id = "/world";
      pt112.header.frame_id = "/world";

      pt12.header.frame_id = "/world";
      pt14.header.frame_id = "/world";
      pt15.header.frame_id = "/world";
      pt16.header.frame_id = "/world";
      pt17.header.frame_id = "/world";
      pt18.header.frame_id = "/world";
      pt19.header.frame_id = "/world";
      pt20.header.frame_id = "/world";
      pt21.header.frame_id = "/world";
      pt22.header.frame_id = "/world";
      pt23.header.frame_id = "/world";
      pt24.header.frame_id = "/world";

      pt0.point.x = x0;
      pt0.point.y = y0;
      pt0.point.z = z0;

      pt1.point.x = x1;
      pt1.point.y = y1;
      pt1.point.z = z1;

      pt2.point.x = x2;
      pt2.point.y = y2;
      pt2.point.z = z2;

      pt3.point.x = x3;
      pt3.point.y = y3;
      pt3.point.z = z3;

      pt4.point.x = x4;
      pt4.point.y = y4;
      pt4.point.z = z4;

      pt5.point.x = x5;
      pt5.point.y = y5;
      pt5.point.z = z5;

      pt6.point.x = x6;
      pt6.point.y = y6;
      pt6.point.z = z6;

      pt7.point.x = x7;
      pt7.point.y = y7;
      pt7.point.z = z7;

      pt8.point.x = x8;
      pt8.point.y = y8;
      pt8.point.z = z8;

      pt9.point.x = x9;
      pt9.point.y = y9;
      pt9.point.z = z9;

      pt10.point.x = x10;
      pt10.point.y = y10;
      pt10.point.z = z10;


      pt11.point.x = x11;
      pt11.point.y = y11;
      pt11.point.z = z11;

      pt110.point.x = x110;
      pt110.point.y = y110;
      pt110.point.z = z110;

      pt111.point.x = x111;
      pt111.point.y = y111;
      pt111.point.z = z111;

      pt112.point.x = x112;
      pt112.point.y = y112;
      pt112.point.z = z112;


      pt12.point.x = x12;
      pt12.point.y = y12;
      pt12.point.z = z12;

      pt14.point.x = x14;
      pt14.point.y = y14;
      pt14.point.z = z14;

      pt15.point.x = x15;
      pt15.point.y = y15;
      pt15.point.z = z15;

      pt16.point.x = x16;
      pt16.point.y = y16;
      pt16.point.z = z16;

      pt17.point.x = x17;
      pt17.point.y = y17;
      pt17.point.z = z17;

      pt18.point.x = x18;
      pt18.point.y = y18;
      pt18.point.z = z18;

      pt19.point.x = x19;
      pt19.point.y = y19;
      pt19.point.z = z19;

      pt20.point.x = x20;
      pt20.point.y = y20;
      pt20.point.z = z20;

      pt21.point.x = x21;
      pt21.point.y = y21;
      pt21.point.z = z21;

      pt22.point.x = x22;
      pt22.point.y = y22;
      pt22.point.z = z22;

      //            pt23.point.x = matrix23(0, 3);
      //            pt23.point.y = matrix23(1, 3);
      //            pt23.point.z = matrix23(2, 3);

      //            pt24.point.x = matrix24(0, 3);
      //            pt24.point.y = matrix24(1, 3);
      //            pt24.point.z = matrix24(2, 3);
      ROS_ERROR("Do the transform");
      tf_listener_.transformPoint("/kinect2_frame", pt0, pt0_);
      tf_listener_.transformPoint("/kinect2_frame", pt1, pt1_);
      tf_listener_.transformPoint("/kinect2_frame", pt2, pt2_);
      tf_listener_.transformPoint("/kinect2_frame", pt3, pt3_);
      tf_listener_.transformPoint("/kinect2_frame", pt4, pt4_);
      tf_listener_.transformPoint("/kinect2_frame", pt5, pt5_);
      tf_listener_.transformPoint("/kinect2_frame", pt6, pt6_);
      tf_listener_.transformPoint("/kinect2_frame", pt7, pt7_);
      tf_listener_.transformPoint("/kinect2_frame", pt8, pt8_);
      tf_listener_.transformPoint("/kinect2_frame", pt9, pt9_);
      tf_listener_.transformPoint("/kinect2_frame", pt10, pt10_);
      tf_listener_.transformPoint("/kinect2_frame", pt11, pt11_);
      tf_listener_.transformPoint("/kinect2_frame", pt110, pt110_);
      tf_listener_.transformPoint("/kinect2_frame", pt111, pt111_);
      tf_listener_.transformPoint("/kinect2_frame", pt112, pt112_);

      tf_listener_.transformPoint("/kinect2_frame", pt12, pt12_);
      tf_listener_.transformPoint("/kinect2_frame", pt14, pt14_);
      tf_listener_.transformPoint("/kinect2_frame", pt15, pt15_);
      tf_listener_.transformPoint("/kinect2_frame", pt16, pt16_);
      tf_listener_.transformPoint("/kinect2_frame", pt17, pt17_);
      tf_listener_.transformPoint("/kinect2_frame", pt18, pt18_);
      tf_listener_.transformPoint("/kinect2_frame", pt19, pt19_);
      tf_listener_.transformPoint("/kinect2_frame", pt20, pt20_);
      tf_listener_.transformPoint("/kinect2_frame", pt21, pt21_);
      tf_listener_.transformPoint("/kinect2_frame", pt22, pt22_);
      tf_listener_.transformPoint("/kinect2_frame", pt23, pt23_);
      tf_listener_.transformPoint("/kinect2_frame", pt24, pt24_);


      pcl::PointXYZRGB point0, point1, point2, point3, point4, point5, point6, point7, point8, point9, point10, point11, point110, point111, point112,
          point12, point14, point15, point16, point17, point18, point19, point20, point21, point22, point23, point24;

      point0.x = pt0_.point.x;
      point0.y = pt0_.point.y;
      point0.z = pt0_.point.z;

      point1.x = pt1_.point.x;
      point1.y = pt1_.point.y;
      point1.z = pt1_.point.z;

      point2.x = pt2_.point.x;
      point2.y = pt2_.point.y;
      point2.z = pt2_.point.z;

      point3.x = pt3_.point.x;
      point3.y = pt3_.point.y;
      point3.z = pt3_.point.z;

      point4.x = pt4_.point.x;
      point4.y = pt4_.point.y;
      point4.z = pt4_.point.z;

      point5.x = pt5_.point.x;
      point5.y = pt5_.point.y;
      point5.z = pt5_.point.z;

      point6.x = pt6_.point.x;
      point6.y = pt6_.point.y;
      point6.z = pt6_.point.z;

      point7.x = pt7_.point.x;
      point7.y = pt7_.point.y;
      point7.z = pt7_.point.z;

      point8.x = pt8_.point.x;
      point8.y = pt8_.point.y;
      point8.z = pt8_.point.z;

      point9.x = pt9_.point.x;
      point9.y = pt9_.point.y;
      point9.z = pt9_.point.z;

      point10.x = pt10_.point.x;
      point10.y = pt10_.point.y;
      point10.z = pt10_.point.z;

      point11.x = pt11_.point.x;
      point11.y = pt11_.point.y;
      point11.z = pt11_.point.z;

      point110.x = pt110_.point.x;
      point110.y = pt110_.point.y;
      point110.z = pt110_.point.z;

      point111.x = pt111_.point.x;
      point111.y = pt111_.point.y;
      point111.z = pt111_.point.z;

      point112.x = pt112_.point.x;
      point112.y = pt112_.point.y;
      point112.z = pt112_.point.z;


      point12.x = pt12_.point.x;
      point12.y = pt12_.point.y;
      point12.z = pt12_.point.z;

      point14.x = pt14_.point.x;
      point14.y = pt14_.point.y;
      point14.z = pt14_.point.z;

      point15.x = pt15_.point.x;
      point15.y = pt15_.point.y;
      point15.z = pt15_.point.z;

      point16.x = pt16_.point.x;
      point16.y = pt16_.point.y;
      point16.z = pt16_.point.z;

      point17.x = pt17_.point.x;
      point17.y = pt17_.point.y;
      point17.z = pt17_.point.z;

      point18.x = pt18_.point.x;
      point18.y = pt18_.point.y;
      point18.z = pt18_.point.z;

      point19.x = pt19_.point.x;
      point19.y = pt19_.point.y;
      point19.z = pt19_.point.z;

      point20.x = pt20_.point.x;
      point20.y = pt20_.point.y;
      point20.z = pt20_.point.z;

      point21.x = pt21_.point.x;
      point21.y = pt21_.point.y;
      point21.z = pt21_.point.z;

      point22.x = pt22_.point.x;
      point22.y = pt22_.point.y;
      point22.z = pt22_.point.z;

      point23.x = pt23_.point.x;
      point23.y = pt23_.point.y;
      point23.z = pt23_.point.z;

      point24.x = pt24_.point.x;
      point24.y = pt24_.point.y;
      point24.z = pt24_.point.z;

      //        double radius = 0.02;
//              double r = 255.0;
//              double g = 15.0;
//              double b = 15.0;
//              double r1 = 0.0;
//              double g1 = 255.0;
//              double b1 = 0.0;
//              double r2 = 255.0;
//              double g2 = 0.0;
//              double b2 = 0.0;
            int viewport = 0;

      pcl::PassThrough<pcl::PointXYZRGB> pass;
      pass.setInputCloud(cloud);
      pass.setFilterFieldName ("z");
      pass.setFilterLimits (point3.z, point22.z-0.2);
      pass.setFilterLimitsNegative (true);
      pass.filter(*cloud_filtered);

      pcl::KdTreeFLANN<pcl::PointXYZRGB> kdtree;
      kdtree.setInputCloud(cloud_filtered);
      pcl::PointXYZRGB searchPoint;
      searchPoint.x = point22.x+0.04;
      searchPoint.y = point22.y-0.08;
      searchPoint.z = point22.z-0.08;

      std::vector<int> pointIdxRadiusSearch;
      std::vector<float> pointRadiusSquaredDistance;
      float radius = 0.14;
      if ( kdtree.radiusSearch (searchPoint, radius, pointIdxRadiusSearch, pointRadiusSquaredDistance) > 0 )
        {
//          for (size_t i = 0; i < pointIdxRadiusSearch.size (); ++i)
//            std::cout << "    "  <<   cloud_filtered->points[ pointIdxRadiusSearch[i] ].x
//                      << " " << cloud_filtered->points[ pointIdxRadiusSearch[i] ].y
//                      << " " << cloud_filtered->points[ pointIdxRadiusSearch[i] ].z
//                      << " (squared distance: " << pointRadiusSquaredDistance[i] << ")" << std::endl
//                      << "indices size: " << pointIdxRadiusSearch.size() << std::endl;
        }

      pcl::PointIndices::Ptr ids (new pcl::PointIndices());
      ids->indices = pointIdxRadiusSearch;

      pcl::ExtractIndices<pcl::PointXYZRGB> extract;
      extract.setInputCloud (cloud_filtered);
      extract.setIndices (ids);
      extract.setNegative (true);
      extract.filter (*cloud_sub1);
      cloud_filtered.swap(cloud_sub1);


      pcl::KdTreeFLANN<pcl::PointXYZRGB> kdtree2;
      kdtree2.setInputCloud(cloud_filtered);
      pcl::PointXYZRGB searchPoint2;
      searchPoint2.x = point112.x+0.07;
      searchPoint2.y = point112.y-0.14;
      searchPoint2.z = point112.z-0.14;

      std::vector<int> pointIdxRadiusSearch2;
      std::vector<float> pointRadiusSquaredDistance2;
      float radius2 = 0.15;
      if ( kdtree2.radiusSearch (searchPoint2, radius2, pointIdxRadiusSearch2, pointRadiusSquaredDistance2) > 0 )
        {
//          for (size_t i = 0; i < pointIdxRadiusSearch2.size (); ++i)
//            std::cout << "    "  <<   cloud_filtered->points[ pointIdxRadiusSearch2[i] ].x
//                      << " " << cloud_filtered->points[ pointIdxRadiusSearch2[i] ].y
//                      << " " << cloud_filtered->points[ pointIdxRadiusSearch2[i] ].z
//                      << " (squared distance: " << pointRadiusSquaredDistance2[i] << ")" << std::endl
//                      << "indices size: " << pointIdxRadiusSearch2.size() << std::endl;
        }

      pcl::PointIndices::Ptr ids2 (new pcl::PointIndices());
      ids2->indices = pointIdxRadiusSearch2;

      pcl::ExtractIndices<pcl::PointXYZRGB> extract2;
      extract2.setInputCloud (cloud_filtered);
      extract2.setIndices (ids2);
      extract2.setNegative (true);
      extract2.filter (*cloud_sub2);
      cloud_filtered.swap(cloud_sub2);


      pcl::KdTreeFLANN<pcl::PointXYZRGB> kdtree3;
      kdtree3.setInputCloud(cloud_filtered);
      pcl::PointXYZRGB searchPoint3;
      searchPoint3.x = point112.x -0.01;
      searchPoint3.y = point112.y-0.30;
      searchPoint3.z = point112.z-0.02;

      std::vector<int> pointIdxRadiusSearch3;
      std::vector<float> pointRadiusSquaredDistance3;
      float radius3 = 0.15;
      if ( kdtree3.radiusSearch (searchPoint3, radius3, pointIdxRadiusSearch3, pointRadiusSquaredDistance3) > 0 )
        {
//          for (size_t i = 0; i < pointIdxRadiusSearch3.size (); ++i)
//            std::cout << "    "  <<   cloud_filtered->points[ pointIdxRadiusSearch3[i] ].x
//                      << " " << cloud_filtered->points[ pointIdxRadiusSearch3[i] ].y
//                      << " " << cloud_filtered->points[ pointIdxRadiusSearch3[i] ].z
//                      << " (squared distance: " << pointRadiusSquaredDistance3[i] << ")" << std::endl
//                      << "indices size: " << pointIdxRadiusSearch3.size() << std::endl;
        }

      pcl::PointIndices::Ptr ids3 (new pcl::PointIndices());
      ids3->indices = pointIdxRadiusSearch3;

      pcl::ExtractIndices<pcl::PointXYZRGB> extract3;
      extract3.setInputCloud (cloud_filtered);
      extract3.setIndices (ids3);
      extract3.setNegative (true);
      extract3.filter (*cloud_sub3);
      cloud_filtered.swap(cloud_sub3);


      pcl::KdTreeFLANN<pcl::PointXYZRGB> kdtree4;
      kdtree4.setInputCloud(cloud_filtered);
      pcl::PointXYZRGB searchPoint4;
      searchPoint4.x = point21.x +0.1;
      searchPoint4.y = point21.y-0.07;
      searchPoint4.z = point21.z+0.2;

      std::vector<int> pointIdxRadiusSearch4;
      std::vector<float> pointRadiusSquaredDistance4;
      float radius4 = 0.2;
      if ( kdtree4.radiusSearch (searchPoint4, radius4, pointIdxRadiusSearch4, pointRadiusSquaredDistance4) > 0 )
        {
//          for (size_t i = 0; i < pointIdxRadiusSearch4.size (); ++i)
//            std::cout << "    "  <<   cloud_filtered->points[ pointIdxRadiusSearch4[i] ].x
//                      << " " << cloud_filtered->points[ pointIdxRadiusSearch4[i] ].y
//                      << " " << cloud_filtered->points[ pointIdxRadiusSearch4[i] ].z
//                      << " (squared distance: " << pointRadiusSquaredDistance4[i] << ")" << std::endl
//                      << "indices size: " << pointIdxRadiusSearch4.size() << std::endl;
        }

      pcl::PointIndices::Ptr ids4 (new pcl::PointIndices());
      ids4->indices = pointIdxRadiusSearch4;

      pcl::ExtractIndices<pcl::PointXYZRGB> extract4;
      extract4.setInputCloud (cloud_filtered);
      extract4.setIndices (ids4);
      extract4.setNegative (true);
      extract4.filter (*cloud_sub4);
      cloud_filtered.swap(cloud_sub4);


//      pass.setInputCloud(cloud_filtered);
//      pass.setFilterFieldName ("x");
//      pass.setFilterLimits (minPoint[0], maxPoint[0]);
//      pass.setFilterLimitsNegative (true);
//      pass.filter(*cloud_filtered);

//      pass.setInputCloud(cloud_filtered);
//      pass.setFilterFieldName ("y");
//      pass.setFilterLimits (minPoint[1], maxPoint[1]);
//      pass.setFilterLimitsNegative (true);
//      pass.filter(*cloud_filtered);

//      pcl::CropBox<pcl::PointXYZRGB> boxFilter;
//      //Eigen::Vector4f minPoint, maxPoint;
//      Eigen::Vector3f boxTranslatation;
//            boxTranslatation[0]=minPoint[0] +(maxPoint[0]-minPoint[0])/2;
//            boxTranslatation[1]=minPoint[1] +(maxPoint[1]-minPoint[1])/2;
//            boxTranslatation[2]=minPoint[2] +(maxPoint[2]-minPoint[2])/2;
//      Eigen::Vector3f boxRotation;
//            boxRotation[0]=0 * (M_PI/180);  // rotation around x-axis
//            boxRotation[1]=0 * (M_PI/180);  // rotation around y-axis
//            boxRotation[2]=0 * (M_PI/180);  //in radians rotation around z-axis. this rotates your cube 45deg around z-axis.
//      Eigen::Affine3f transform = Eigen::Affine3f::Identity();

//      boxFilter.setInputCloud (cloud);
//      boxFilter.setMin(point50.getVector4fMap());
//      boxFilter.setMax(point57.getVector4fMap());
//      boxFilter.setTranslation(boxTranslatation);
//      boxFilter.setRotation(boxRotation);
//      boxFilter.setTransform(transform);
//      boxFilter.filter(*cloud_filtered);
//      ROS_ERROR("Done the filter");
//      ROS_INFO("point 22 x: %1.2f", point22.x);
//      ROS_INFO("point 22 y: %1.2f", point22.y);
//      ROS_INFO("point 22 z: %1.2f", point22.z);

//      ROS_INFO("point 112 x: %1.2f", point112.x);
//      ROS_INFO("point 112 y: %1.2f", point112.y);
//      ROS_INFO("point 112 z: %1.2f", point112.z);

//      ROS_INFO_STREAM("Msg: " << cloud->header.frame_id.c_str());
//      ROS_INFO("%s", cloud->header.frame_id.c_str());
//      ROS_INFO("%s", cloud_filtered->header.frame_id.c_str());
      //pcl::copyPointCloud(*cloud_filtered, *cloud);


//          pcl::copyPointCloud(*cloud_filtered, *cloud);
//          *cloud = *cloud_filtered;




//        const std::string &id0 = "sphere0";
//        const std::string &id1 = "sphere1";
//        const std::string &id2 = "sphere2";
//        const std::string &id3 = "sphere3";
//        const std::string &id4 = "sphere4";
//        const std::string &id5 = "sphere5";
//        const std::string &id6 = "sphere6";
//        const std::string &id7 = "sphere7";
//        const std::string &id8 = "sphere8";
//        const std::string &id9 = "sphere9";
//        const std::string &id10 = "sphere10";
//        const std::string &id11 = "sphere11";
//        const std::string &id110 = "sphere110";
//        const std::string &id111 = "sphere111";
//        const std::string &id112 = "sphere112";

//        const std::string &id12 = "sphere12";
//        const std::string &id14 = "sphere14";
//        const std::string &id15 = "sphere15";
//        const std::string &id16 = "sphere16";
//        const std::string &id17 = "sphere17";
//        const std::string &id18 = "sphere18";
//        const std::string &id19 = "sphere19";
//        const std::string &id20 = "sphere20";
//        const std::string &id21 = "sphere21";
//        const std::string &id22 = "sphere22";
//        const std::string &id23 = "sphere23";
//        const std::string &id24 = "sphere24";
      //      boxFilter.setTranslation(boxTranslatation);

      visualizer->removeAllShapes();

//      visualizer->addLine(point50, point51, r2, g2, b2, "line1", viewport);
//      visualizer->addLine(point52, point53, r1, g1, b1, "line2", viewport);
//      visualizer->addLine(point50, point52, r1, g1, b1, "line3", viewport);
//      visualizer->addLine(point51, point53, r1, g1, b1, "line4", viewport);
//      visualizer->addLine(point50, point54, r1, g1, b1, "line5", viewport);
//      visualizer->addLine(point51, point55, r1, g1, b1, "line6", viewport);
//      visualizer->addLine(point52, point56, r1, g1, b1, "line7", viewport);
//      visualizer->addLine(point53, point57, r1, g1, b1, "line8", viewport);

//      visualizer->addLine(point54, point55, r1, g1, b1, "line9", viewport);
//      visualizer->addLine(point56, point57, r1, g1, b1, "line10", viewport);
//      visualizer->addLine(point54, point56, r1, g1, b1, "line11", viewport);
//      visualizer->addLine(point55, point57, r1, g1, b1, "line12", viewport);

      //visualizer->addLine(searchPoint4, point21, r2, g2, b2, "line1", viewport);

//        visualizer->addSphere(point0, radius, r, g, b, id0, viewport);
//        visualizer->addSphere(point1, radius, r, g, b, id1, viewport);
//        visualizer->addSphere(point2, radius, r, g, b, id2, viewport);
//        visualizer->addSphere(point3, radius, r, g, b, id3, viewport);
//        visualizer->addSphere(point4, radius, r, g, b, id4, viewport);
//        visualizer->addSphere(point5, radius, r, g, b, id5, viewport);
//        visualizer->addSphere(point6, radius, r, g, b, id6, viewport);
//        visualizer->addSphere(point7, radius, r, g, b, id7, viewport);
//        //visualizer->addSphere(point8, radius, r, g, b, id8, viewport);
//        visualizer->addSphere(point9, radius, r, g, b, id9, viewport);
//        visualizer->addSphere(point10, radius, r, g, b, id10, viewport);
//        visualizer->addSphere(point11, radius, r, g, b, id11, viewport);
//        visualizer->addSphere(point110, radius, r, g, b, id110, viewport);
//        visualizer->addSphere(point111, radius, r, g, b, id111, viewport);
//        visualizer->addSphere(point112, radius, r, g, b, id112, viewport);

//        visualizer->addLine(point0, point1, r1, g1, b1, "line1", viewport);
//        visualizer->addLine(point1, point2, r1, g1, b1, "line2", viewport);
//        visualizer->addLine(point2, point3, r1, g1, b1, "line3", viewport);
//        visualizer->addLine(point3, point4, r1, g1, b1, "line4", viewport);
//        visualizer->addLine(point4, point5, r1, g1, b1, "line5", viewport);
//        visualizer->addLine(point5, point6, r1, g1, b1, "line6", viewport);
//        visualizer->addLine(point6, point7, r1, g1, b1, "line7", viewport);
//        visualizer->addLine(point7, point8, r1, g1, b1, "line8", viewport);
//        visualizer->addLine(point8, point11, r1, g1, b1, "line9", viewport);

//        visualizer->addSphere(point12, radius, r, g, b, id12, viewport);
//        visualizer->addSphere(point14, radius, r, g, b, id14, viewport);
//        visualizer->addSphere(point15, radius, r, g, b, id15, viewport);
//        visualizer->addSphere(point16, radius, r, g, b, id16, viewport);
//        visualizer->addSphere(point17, radius, r, g, b, id17, viewport);
//        visualizer->addSphere(point18, radius, r, g, b, id18, viewport);
//        visualizer->addSphere(point19, radius, r, g, b, id19, viewport);
//        visualizer->addSphere(point20, radius, r, g, b, id20, viewport);
//        //      visualizer->addSphere(point21, radius, r, g, b, id21, viewport);
//        visualizer->addSphere(point22, radius, r, g, b, id22, viewport);
//        visualizer->addSphere(point23, radius, r, g, b, id23, viewport);
//        //      visualizer->addSphere(point24, radius, r, g, b, id24, viewport);

//        visualizer->addLine(point12, point14, r2, g2, b2, "line10", viewport);
//        visualizer->addLine(point14, point15, r2, g2, b2, "line11", viewport);
//        visualizer->addLine(point15, point16, r2, g2, b2, "line12", viewport);
//        visualizer->addLine(point16, point17, r2, g2, b2, "line14", viewport);
//        visualizer->addLine(point17, point18, r2, g2, b2, "line15", viewport);
//        visualizer->addLine(point18, point19, r2, g2, b2, "line16", viewport);
//        visualizer->addLine(point19, point20, r2, g2, b2, "line17", viewport);
//        visualizer->addLine(point20, point22, r2, g2, b2, "line18", viewport);

      ROS_ERROR("ADD THE CINLYNDER TO POINT CLOUD");
      double r = 0.055;
      pcl::ModelCoefficients cylinder_coeff1;
      cylinder_coeff1.values.resize(7);
      cylinder_coeff1.values[0] = point0.x;
      cylinder_coeff1.values[1] = point0.y;
      cylinder_coeff1.values[2] = point0.z;
      cylinder_coeff1.values[3] = point1.x - point0.x;
      cylinder_coeff1.values[4] = point1.y - point0.y;
      cylinder_coeff1.values[5] = point1.z - point0.z;
      cylinder_coeff1.values[6] = r;
      visualizer->addCylinder (cylinder_coeff1, "cylinder1", viewport);

      pcl::ModelCoefficients cylinder_coeff2;
      cylinder_coeff2.values.resize(7);
      cylinder_coeff2.values[0] = point1.x;
      cylinder_coeff2.values[1] = point1.y;
      cylinder_coeff2.values[2] = point1.z;
      cylinder_coeff2.values[3] = point2.x - point1.x;
      cylinder_coeff2.values[4] = point2.y - point1.y;
      cylinder_coeff2.values[5] = point2.z - point1.z;
      cylinder_coeff2.values[6] = r;
      visualizer->addCylinder (cylinder_coeff2, "cylinder2", viewport);

      pcl::ModelCoefficients cylinder_coeff3;
      cylinder_coeff3.values.resize(7);
      cylinder_coeff3.values[0] = point2.x;
      cylinder_coeff3.values[1] = point2.y;
      cylinder_coeff3.values[2] = point2.z;
      cylinder_coeff3.values[3] = point3.x - point2.x;
      cylinder_coeff3.values[4] = point3.y - point2.y;
      cylinder_coeff3.values[5] = point3.z - point2.z;
      cylinder_coeff3.values[6] = r;
      visualizer->addCylinder (cylinder_coeff3, "cylinder3", viewport);

      pcl::ModelCoefficients cylinder_coeff4;
      cylinder_coeff4.values.resize(7);
      cylinder_coeff4.values[0] = point3.x;
      cylinder_coeff4.values[1] = point3.y;
      cylinder_coeff4.values[2] = point3.z;
      cylinder_coeff4.values[3] = point4.x - point3.x;
      cylinder_coeff4.values[4] = point4.y - point3.y;
      cylinder_coeff4.values[5] = point4.z - point3.z;
      cylinder_coeff4.values[6] = r;
      visualizer->addCylinder (cylinder_coeff4, "cylinder4", viewport);


//      pcl::ModelCoefficients cylinder_coeff40;
//      cylinder_coeff40.values.resize(7);
//      cylinder_coeff40.values[0] = point3.x;
//      cylinder_coeff40.values[1] = point3.y - 0.1;
//      cylinder_coeff40.values[2] = point3.z;
//      cylinder_coeff40.values[3] = point4.x - point3.x;
//      cylinder_coeff40.values[4] = point4.y - point3.y;
//      cylinder_coeff40.values[5] = point4.z - point3.z;
//      cylinder_coeff40.values[6] = 0.04;
//      visualizer->addCylinder (cylinder_coeff40, "cylinder40", viewport);


      pcl::ModelCoefficients cylinder_coeff5;
      cylinder_coeff5.values.resize(7);
      cylinder_coeff5.values[0] = point4.x;
      cylinder_coeff5.values[1] = point4.y;
      cylinder_coeff5.values[2] = point4.z;
      cylinder_coeff5.values[3] = point5.x - point4.x;
      cylinder_coeff5.values[4] = point5.y - point4.y;
      cylinder_coeff5.values[5] = point5.z - point4.z;
      cylinder_coeff5.values[6] = r;
      visualizer->addCylinder (cylinder_coeff5, "cylinder5", viewport);

      pcl::ModelCoefficients cylinder_coeff6;
      cylinder_coeff6.values.resize(7);
      cylinder_coeff6.values[0] = point5.x;
      cylinder_coeff6.values[1] = point5.y;
      cylinder_coeff6.values[2] = point5.z;
      cylinder_coeff6.values[3] = point6.x - point5.x;
      cylinder_coeff6.values[4] = point6.y - point5.y;
      cylinder_coeff6.values[5] = point6.z - point5.z;
      cylinder_coeff6.values[6] = r;
      visualizer->addCylinder (cylinder_coeff6, "cylinder6", viewport);

      pcl::ModelCoefficients cylinder_coeff7;
      cylinder_coeff7.values.resize(7);
      cylinder_coeff7.values[0] = point6.x;
      cylinder_coeff7.values[1] = point6.y;
      cylinder_coeff7.values[2] = point6.z;
      cylinder_coeff7.values[3] = point7.x - point6.x;
      cylinder_coeff7.values[4] = point7.y - point6.y;
      cylinder_coeff7.values[5] = point7.z - point6.z;
      cylinder_coeff7.values[6] = r;
      visualizer->addCylinder (cylinder_coeff7, "cylinder7", viewport);

      pcl::ModelCoefficients cylinder_coeff8;
      cylinder_coeff8.values.resize(7);
      cylinder_coeff8.values[0] = point7.x;
      cylinder_coeff8.values[1] = point7.y;
      cylinder_coeff8.values[2] = point7.z;
      cylinder_coeff8.values[3] = point8.x - point7.x;
      cylinder_coeff8.values[4] = point8.y - point7.y;
      cylinder_coeff8.values[5] = point8.z - point7.z;
      cylinder_coeff8.values[6] = r;
      visualizer->addCylinder (cylinder_coeff8, "cylinder8", viewport);

      pcl::ModelCoefficients cylinder_coeff9;
      cylinder_coeff9.values.resize(7);
      cylinder_coeff9.values[0] = point8.x;
      cylinder_coeff9.values[1] = point8.y;
      cylinder_coeff9.values[2] = point8.z;
      cylinder_coeff9.values[3] = point11.x - point8.x;
      cylinder_coeff9.values[4] = point11.y - point8.y;
      cylinder_coeff9.values[5] = point11.z - point8.z;
      cylinder_coeff9.values[6] = r;
      visualizer->addCylinder (cylinder_coeff9, "cylinder9", viewport);

      pcl::ModelCoefficients cylinder_coeff10;
      cylinder_coeff10.values.resize(7);
      cylinder_coeff10.values[0] = point11.x;
      cylinder_coeff10.values[1] = point11.y;
      cylinder_coeff10.values[2] = point11.z;
      cylinder_coeff10.values[3] = point110.x - point11.x;
      cylinder_coeff10.values[4] = point110.y - point11.y;
      cylinder_coeff10.values[5] = point110.z - point11.z;
      cylinder_coeff10.values[6] = r;
      visualizer->addCylinder (cylinder_coeff10, "cylinder10", viewport);



      pcl::ModelCoefficients cylinder_coeff14;
      cylinder_coeff14.values.resize(7);
      cylinder_coeff14.values[0] = point12.x;
      cylinder_coeff14.values[1] = point12.y;
      cylinder_coeff14.values[2] = point12.z;
      cylinder_coeff14.values[3] = point14.x - point12.x;
      cylinder_coeff14.values[4] = point14.y - point12.y;
      cylinder_coeff14.values[5] = point14.z - point12.z;
      cylinder_coeff14.values[6] = r;
      visualizer->addCylinder (cylinder_coeff14, "cylinder14", viewport);

      pcl::ModelCoefficients cylinder_coeff15;
      cylinder_coeff15.values.resize(7);
      cylinder_coeff15.values[0] = point14.x;
      cylinder_coeff15.values[1] = point14.y;
      cylinder_coeff15.values[2] = point14.z;
      cylinder_coeff15.values[3] = point15.x - point14.x;
      cylinder_coeff15.values[4] = point15.y - point14.y;
      cylinder_coeff15.values[5] = point15.z - point14.z;
      cylinder_coeff15.values[6] = r;
      visualizer->addCylinder (cylinder_coeff15, "cylinder15", viewport);

      pcl::ModelCoefficients cylinder_coeff16;
      cylinder_coeff16.values.resize(7);
      cylinder_coeff16.values[0] = point15.x;
      cylinder_coeff16.values[1] = point15.y;
      cylinder_coeff16.values[2] = point15.z;
      cylinder_coeff16.values[3] = point16.x - point15.x;
      cylinder_coeff16.values[4] = point16.y - point15.y;
      cylinder_coeff16.values[5] = point16.z - point15.z;
      cylinder_coeff16.values[6] = r;
      visualizer->addCylinder (cylinder_coeff16, "cylinder16", viewport);

      pcl::ModelCoefficients cylinder_coeff17;
      cylinder_coeff17.values.resize(7);
      cylinder_coeff17.values[0] = point16.x;
      cylinder_coeff17.values[1] = point16.y;
      cylinder_coeff17.values[2] = point16.z;
      cylinder_coeff17.values[3] = point17.x - point16.x;
      cylinder_coeff17.values[4] = point17.y - point16.y;
      cylinder_coeff17.values[5] = point17.z - point16.z;
      cylinder_coeff17.values[6] = r;
      visualizer->addCylinder (cylinder_coeff17, "cylinder17", viewport);

      pcl::ModelCoefficients cylinder_coeff18;
      cylinder_coeff18.values.resize(7);
      cylinder_coeff18.values[0] = point17.x;
      cylinder_coeff18.values[1] = point17.y;
      cylinder_coeff18.values[2] = point17.z;
      cylinder_coeff18.values[3] = point18.x - point17.x;
      cylinder_coeff18.values[4] = point18.y - point17.y;
      cylinder_coeff18.values[5] = point18.z - point17.z;
      cylinder_coeff18.values[6] = r;
      visualizer->addCylinder (cylinder_coeff18, "cylinder18", viewport);

      pcl::ModelCoefficients cylinder_coeff19;
      cylinder_coeff19.values.resize(7);
      cylinder_coeff19.values[0] = point18.x;
      cylinder_coeff19.values[1] = point18.y;
      cylinder_coeff19.values[2] = point18.z;
      cylinder_coeff19.values[3] = point19.x - point18.x;
      cylinder_coeff19.values[4] = point19.y - point18.y;
      cylinder_coeff19.values[5] = point19.z - point18.z;
      cylinder_coeff19.values[6] = r;
      visualizer->addCylinder (cylinder_coeff19, "cylinder19", viewport);

      pcl::ModelCoefficients cylinder_coeff20;
      cylinder_coeff20.values.resize(7);
      cylinder_coeff20.values[0] = point19.x;
      cylinder_coeff20.values[1] = point19.y;
      cylinder_coeff20.values[2] = point19.z;
      cylinder_coeff20.values[3] = point20.x - point19.x;
      cylinder_coeff20.values[4] = point20.y - point19.y;
      cylinder_coeff20.values[5] = point20.z - point19.z;
      cylinder_coeff20.values[6] = r;
      visualizer->addCylinder (cylinder_coeff20, "cylinder20", viewport);

      pcl::ModelCoefficients cylinder_coeff21;
      cylinder_coeff21.values.resize(7);
      cylinder_coeff21.values[0] = point20.x;
      cylinder_coeff21.values[1] = point20.y;
      cylinder_coeff21.values[2] = point20.z;
      cylinder_coeff21.values[3] = point22.x - point20.x;
      cylinder_coeff21.values[4] = point22.y - point20.y;
      cylinder_coeff21.values[5] = point22.z - point20.z;
      cylinder_coeff21.values[6] = r;
      visualizer->addCylinder (cylinder_coeff21, "cylinder21", viewport);

  }

  void cloudViewer()
  {
    cv::Mat color, depth;
    pcl::visualization::PCLVisualizer::Ptr visualizer(new pcl::visualization::PCLVisualizer("Cloud Viewer"));
    const std::string cloudName = "rendered";

    lock.lock();
    color = this->color;
    depth = this->depth;
    updateCloud = false;
    lock.unlock();

    createCloud(depth, color, cloud);

    visualizer->addPointCloud(cloud, cloudName);
    visualizer->setPointCloudRenderingProperties(pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 1, cloudName);
    visualizer->initCameraParameters();
    visualizer->setBackgroundColor(0, 0, 0);
    visualizer->setPosition(mode == BOTH ? color.cols : 0, 0);
    visualizer->setSize(color.cols, color.rows);
    visualizer->setShowFPS(true);
    visualizer->setCameraPosition(0, 0, 0, 0, -1, 0);
    visualizer->registerKeyboardCallback(&Receiver::keyboardEvent, *this);
    ROS_ERROR("THIS LINE IS EXECUTED1");

    for(; running && ros::ok();)
      {
        ROS_ERROR("THIS LINE IS EXECUTED333");

        if(updateCloud)
          {
            lock.lock();
            color = this->color;
            depth = this->depth;
            updateCloud = false;
            lock.unlock();

            createCloud(depth, color, cloud);
            //

            processMyCloud(visualizer);

            visualizer->updatePointCloud(cloud_filtered, cloudName);
          }
        if(save)
          {
            save = false;
            cv::Mat depthDisp;
            dispDepth(depth, depthDisp, 12000.0f);
            saveCloudAndImages(cloud, color, depth, depthDisp);
          }

        visualizer->spinOnce(10);
      }
    visualizer->close();
  }

  void keyboardEvent(const pcl::visualization::KeyboardEvent &event, void *)
  {
    if(event.keyUp())
      {
        switch(event.getKeyCode())
          {
          case 27:
          case 'q':
            running = false;
            break;
          case ' ':
          case 's':
            save = true;
            break;
          }
      }
  }

  void readImage(const sensor_msgs::Image::ConstPtr msgImage, cv::Mat &image) const
  {
    cv_bridge::CvImageConstPtr pCvImage;
    pCvImage = cv_bridge::toCvShare(msgImage, msgImage->encoding);
    pCvImage->image.copyTo(image);
  }

  void readCameraInfo(const sensor_msgs::CameraInfo::ConstPtr cameraInfo, cv::Mat &cameraMatrix) const
  {
    double *itC = cameraMatrix.ptr<double>(0, 0);
    for(size_t i = 0; i < 9; ++i, ++itC)
      {
        *itC = cameraInfo->K[i];
      }
  }

  void dispDepth(const cv::Mat &in, cv::Mat &out, const float maxValue)
  {
    cv::Mat tmp = cv::Mat(in.rows, in.cols, CV_8U);
    const uint32_t maxInt = 255;

#pragma omp parallel for
    for(int r = 0; r < in.rows; ++r)
      {
        const uint16_t *itI = in.ptr<uint16_t>(r);
        uint8_t *itO = tmp.ptr<uint8_t>(r);

        for(int c = 0; c < in.cols; ++c, ++itI, ++itO)
          {
            *itO = (uint8_t)std::min((*itI * maxInt / maxValue), 255.0f);
          }
      }

    cv::applyColorMap(tmp, out, cv::COLORMAP_JET);
  }

  void combine(const cv::Mat &inC, const cv::Mat &inD, cv::Mat &out)
  {
    out = cv::Mat(inC.rows, inC.cols, CV_8UC3);

#pragma omp parallel for
    for(int r = 0; r < inC.rows; ++r)
      {
        const cv::Vec3b
            *itC = inC.ptr<cv::Vec3b>(r),
            *itD = inD.ptr<cv::Vec3b>(r);
        cv::Vec3b *itO = out.ptr<cv::Vec3b>(r);

        for(int c = 0; c < inC.cols; ++c, ++itC, ++itD, ++itO)
          {
            itO->val[0] = (itC->val[0] + itD->val[0]) >> 1;
            itO->val[1] = (itC->val[1] + itD->val[1]) >> 1;
            itO->val[2] = (itC->val[2] + itD->val[2]) >> 1;
          }
      }
  }

  void createCloud(const cv::Mat &depth, const cv::Mat &color, pcl::PointCloud<pcl::PointXYZRGB>::Ptr &cloud) const
  {
    const float badPoint = std::numeric_limits<float>::quiet_NaN();

#pragma omp parallel for
    for(int r = 0; r < depth.rows; ++r)
      {
        pcl::PointXYZRGB *itP = &cloud->points[r * depth.cols];
        const uint16_t *itD = depth.ptr<uint16_t>(r);
        const cv::Vec3b *itC = color.ptr<cv::Vec3b>(r);
        const float y = lookupY.at<float>(0, r);
        const float *itX = lookupX.ptr<float>();

        for(size_t c = 0; c < (size_t)depth.cols; ++c, ++itP, ++itD, ++itC, ++itX)
          {
            register const float depthValue = *itD / 1000.0f;
            // Check for invalid measurements
            if(*itD == 0)
              {
                // not valid
                itP->x = itP->y = itP->z = badPoint;
                itP->rgba = 0;
                continue;
              }
            itP->z = depthValue;
            itP->x = *itX * depthValue;
            itP->y = y * depthValue;
            itP->b = itC->val[0];
            itP->g = itC->val[1];
            itP->r = itC->val[2];
            itP->a = 255;
          }
      }
  }

  void saveCloudAndImages(const pcl::PointCloud<pcl::PointXYZRGB>::ConstPtr cloud, const cv::Mat &color, const cv::Mat &depth, const cv::Mat &depthColored)
  {
    oss.str("");
    oss << "./" << std::setfill('0') << std::setw(4) << frame;
    const std::string baseName = oss.str();
    const std::string cloudName = baseName + "_cloud.pcd";
    const std::string colorName = baseName + "_color.jpg";
    const std::string depthName = baseName + "_depth.png";
    const std::string depthColoredName = baseName + "_depth_colored.png";

    OUT_INFO("saving cloud: " << cloudName);
    writer.writeBinary(cloudName, *cloud);
    OUT_INFO("saving color: " << colorName);
    cv::imwrite(colorName, color, params);
    OUT_INFO("saving depth: " << depthName);
    cv::imwrite(depthName, depth, params);
    OUT_INFO("saving depth: " << depthColoredName);
    cv::imwrite(depthColoredName, depthColored, params);
    OUT_INFO("saving complete!");
    ++frame;
  }

  void createLookup(size_t width, size_t height)
  {
    const float fx = 1.0f / cameraMatrixColor.at<double>(0, 0);
    const float fy = 1.0f / cameraMatrixColor.at<double>(1, 1);
    const float cx = cameraMatrixColor.at<double>(0, 2);
    const float cy = cameraMatrixColor.at<double>(1, 2);
    float *it;

    lookupY = cv::Mat(1, height, CV_32F);
    it = lookupY.ptr<float>();
    for(size_t r = 0; r < height; ++r, ++it)
      {
        *it = (r - cy) * fy;
      }

    lookupX = cv::Mat(1, width, CV_32F);
    it = lookupX.ptr<float>();
    for(size_t c = 0; c < width; ++c, ++it)
      {
        *it = (c - cx) * fx;
      }
  }

  void js_callback(const sensor_msgs::JointState::ConstPtr& msg)
  {
    joints.clear();

    joints.insert (std::pair<std::string,double>(msg->name[0],msg->position[0]));
    joints.insert (std::pair<std::string,double>(msg->name[1],msg->position[1]));
    joints.insert (std::pair<std::string,double>(msg->name[2],msg->position[2]));
    joints.insert (std::pair<std::string,double>(msg->name[3],msg->position[3]));
    joints.insert (std::pair<std::string,double>(msg->name[4],msg->position[4]));
    joints.insert (std::pair<std::string,double>(msg->name[5],msg->position[5]));
    //joints.insert (std::pair<std::string,double>(msg->name[6],msg->position[6]));
    joints.insert (std::pair<std::string,double>(msg->name[7],msg->position[7]));
    joints.insert (std::pair<std::string,double>(msg->name[8],msg->position[8]));
    joints.insert (std::pair<std::string,double>(msg->name[9],msg->position[9]));
    joints.insert (std::pair<std::string,double>(msg->name[10],msg->position[10]));
    joints.insert (std::pair<std::string,double>(msg->name[11],msg->position[11]));
    joints.insert (std::pair<std::string,double>(msg->name[12],msg->position[12]));
    joints.insert (std::pair<std::string,double>(msg->name[13],msg->position[13]));
    joints.insert (std::pair<std::string,double>(msg->name[14],msg->position[14]));
    joints.insert (std::pair<std::string,double>(msg->name[15],msg->position[15]));
    //joints.insert (std::pair<std::string,double>(msg->name[16],msg->position[16]));
    kinematic_state->setVariablePositions (joints);

//    const Eigen::Affine3d &joint_state0 = kinematic_state->getGlobalLinkTransform("left_base_link");
//    const Eigen::Affine3d &joint_state1 = kinematic_state->getGlobalLinkTransform("left_shoulder_link");
//    const Eigen::Affine3d &joint_state2 = kinematic_state->getGlobalLinkTransform("left_upper_arm_link");
//    const Eigen::Affine3d &joint_state3 = kinematic_state->getGlobalLinkTransform("left_forearm_link");
//    const Eigen::Affine3d &joint_state4 = kinematic_state->getGlobalLinkTransform("left_wrist_1_link");
//    const Eigen::Affine3d &joint_state5 = kinematic_state->getGlobalLinkTransform("left_wrist_2_link");
//    const Eigen::Affine3d &joint_state6 = kinematic_state->getGlobalLinkTransform("left_wrist_3_link");
//    const Eigen::Affine3d &joint_state7 = kinematic_state->getGlobalLinkTransform("left_ee_link");
//    const Eigen::Affine3d &joint_state8 = kinematic_state->getGlobalLinkTransform("left_base_link_gripper");
//    const Eigen::Affine3d &joint_state9 = kinematic_state->getGlobalLinkTransform("left_ee_gripper_link");
//    const Eigen::Affine3d &joint_state10 = kinematic_state->getGlobalLinkTransform("left_pi4_gripper_fixed_link");
//    const Eigen::Affine3d &joint_state11 = kinematic_state->getGlobalLinkTransform("left_pi4_gripper_prismatic_link");
//    const Eigen::Affine3d &joint_state110 = kinematic_state->getGlobalLinkTransform("left_pi4_gripper_finger1_link");
//    const Eigen::Affine3d &joint_state111 = kinematic_state->getGlobalLinkTransform("left_pi4_gripper_finger2_link");
//    const Eigen::Affine3d &joint_state112 = kinematic_state->getGlobalLinkTransform("left_ee_pi4_gripper_link");



//    const Eigen::Affine3d &joint_state12 = kinematic_state->getGlobalLinkTransform("right_base_link");
//    const Eigen::Affine3d &joint_state14 = kinematic_state->getGlobalLinkTransform("right_shoulder_link");
//    const Eigen::Affine3d &joint_state15 = kinematic_state->getGlobalLinkTransform("right_upper_arm_link");
//    const Eigen::Affine3d &joint_state16 = kinematic_state->getGlobalLinkTransform("right_forearm_link");
//    const Eigen::Affine3d &joint_state17 = kinematic_state->getGlobalLinkTransform("right_wrist_1_link");
//    const Eigen::Affine3d &joint_state18 = kinematic_state->getGlobalLinkTransform("right_wrist_2_link");
//    const Eigen::Affine3d &joint_state19 = kinematic_state->getGlobalLinkTransform("right_wrist_3_link");
//    const Eigen::Affine3d &joint_state20 = kinematic_state->getGlobalLinkTransform("right_ee_link");
//    const Eigen::Affine3d &joint_state21 = kinematic_state->getGlobalLinkTransform("right_base_link_gripper");
//    const Eigen::Affine3d &joint_state22 = kinematic_state->getGlobalLinkTransform("right_ee_gripper_link");
//    const Eigen::Affine3d &joint_state23 = kinematic_state->getGlobalLinkTransform("Left_stereoCam_link");
//    const Eigen::Affine3d &joint_state24 = kinematic_state->getGlobalLinkTransform("stereoCam_link");

//    ROS_ERROR("store the translation data");

//    double x0 = joint_state0.translation().x();
//    double y0 = joint_state0.translation().y();
//    double z0 = joint_state0.translation().z();

//    double x1 = joint_state1.translation().x();
//    double y1 = joint_state1.translation().y();
//    double z1 = joint_state1.translation().z();

//    double x2 = joint_state2.translation().x();
//    double y2 = joint_state2.translation().y();
//    double z2 = joint_state2.translation().z();

//    double x3 = joint_state3.translation().x();
//    double y3 = joint_state3.translation().y();
//    double z3 = joint_state3.translation().z();

//    double x4 = joint_state4.translation().x();
//    double y4 = joint_state4.translation().y();
//    double z4 = joint_state4.translation().z();

//    double x5 = joint_state5.translation().x();
//    double y5 = joint_state5.translation().y();
//    double z5 = joint_state5.translation().z();

//    double x6 = joint_state6.translation().x();
//    double y6 = joint_state6.translation().y();
//    double z6 = joint_state6.translation().z();

//    double x7 = joint_state7.translation().x();
//    double y7 = joint_state7.translation().y();
//    double z7 = joint_state7.translation().z();

//    double x8 = joint_state8.translation().x();
//    double y8 = joint_state8.translation().y();
//    double z8 = joint_state8.translation().z();

//    double x9 = joint_state9.translation().x();
//    double y9 = joint_state9.translation().y();
//    double z9 = joint_state9.translation().z();

//    double x10 = joint_state10.translation().x();
//    double y10 = joint_state10.translation().y();
//    double z10 = joint_state10.translation().z();

//    double x11 = joint_state11.translation().x();
//    double y11 = joint_state11.translation().y();
//    double z11 = joint_state11.translation().z();

//    double x110 = joint_state110.translation().x();
//    double y110 = joint_state110.translation().y();
//    double z110 = joint_state110.translation().z();

//    double x111 = joint_state111.translation().x();
//    double y111 = joint_state111.translation().y();
//    double z111 = joint_state111.translation().z();

//    double x112 = joint_state112.translation().x();
//    double y112 = joint_state112.translation().y();
//    double z112 = joint_state112.translation().z();

//    double x12 = joint_state12.translation().x();
//    double y12 = joint_state12.translation().y();
//    double z12 = joint_state12.translation().z();

//    double x14 = joint_state14.translation().x();
//    double y14 = joint_state14.translation().y();
//    double z14 = joint_state14.translation().z();

//    double x15 = joint_state15.translation().x();
//    double y15 = joint_state15.translation().y();
//    double z15 = joint_state15.translation().z();

//    double x16 = joint_state16.translation().x();
//    double y16 = joint_state16.translation().y();
//    double z16 = joint_state16.translation().z();

//    double x17 = joint_state17.translation().x();
//    double y17 = joint_state17.translation().y();
//    double z17 = joint_state17.translation().z();

//    double x18 = joint_state18.translation().x();
//    double y18 = joint_state18.translation().y();
//    double z18 = joint_state18.translation().z();

//    double x19 = joint_state19.translation().x();
//    double y19 = joint_state19.translation().y();
//    double z19 = joint_state19.translation().z();

//    double x20 = joint_state20.translation().x();
//    double y20 = joint_state20.translation().y();
//    double z20 = joint_state20.translation().z();

//    double x21 = joint_state21.translation().x();
//    double y21 = joint_state21.translation().y();
//    double z21 = joint_state21.translation().z();

//    double x22 = joint_state22.translation().x();
//    double y22 = joint_state22.translation().y();
//    double z22 = joint_state22.translation().z(); //hard coded because values from vision are not always reliable

//    double roll3, pitch3, yaw3;
//    tf::Matrix3x3 matrix3;
//    Eigen::Matrix3d link_orientation3 = joint_state3.rotation();
//    tf::matrixEigenToTF(link_orientation3, matrix3);

//    matrix3.getRPY(roll3, pitch3, yaw3);
//    ROS_INFO("roll 3: %1.2f", roll3);
//    ROS_INFO("pitch 3: %1.2f", pitch3);
//    ROS_INFO("raw 3: %1.2f", yaw3);

//    double l_roll_deg3=195;//180
//    double l_pitch_deg3=35;//30.37
//    double l_yaw_deg3=180;//171.31

//    double l_roll_rad3 = l_roll_deg3 * (M_PI/180);
//    double l_pitch_rad3 = l_pitch_deg3 * (M_PI/180);
//    double l_yaw_rad3 = l_yaw_deg3 * (M_PI/180);

//    tf::Quaternion l_q3;
//    l_q3.setRPY(l_roll_rad3, l_pitch_rad3, l_yaw_rad3);
//    Eigen::Affine3d left_pose3 = Eigen::Translation3d(x3, y3, z3)
//        * Eigen::Quaterniond(l_q3);
//    kinematic_state->updateStateWithLinkAt("left_forearm_link", left_pose3);

////-----------------------------

//    double roll5, pitch5, yaw5;
//    tf::Matrix3x3 matrix5;
//    Eigen::Matrix3d link_orientation5 = joint_state5.rotation();
//    tf::matrixEigenToTF(link_orientation5, matrix5);

//    matrix5.getRPY(roll5, pitch5, yaw5);
//    ROS_INFO("roll 5: %1.2f", roll5);//-143.81
//    ROS_INFO("pitch 5: %1.2f", pitch5);//-12.03
//    ROS_INFO("raw 5: %1.2f", yaw5);//-98.55

//    double l_roll_deg5=-84.80;//-84.80
//    double l_pitch_deg5=-15.03;//-12.03
//    double l_yaw_deg5=-90.55;//-98.55

//    double l_roll_rad5 = l_roll_deg5 * (M_PI/180);
//    double l_pitch_rad5 = l_pitch_deg5 * (M_PI/180);
//    double l_yaw_rad5 = l_yaw_deg5 * (M_PI/180);

//    tf::Quaternion l_q5;
//    l_q5.setRPY(l_roll_rad5, l_pitch_rad5, l_yaw_rad5);
//    Eigen::Affine3d left_pose5 = Eigen::Translation3d(x5, y5, z5)
//        * Eigen::Quaterniond(l_q5);
//    kinematic_state->updateStateWithLinkAt("left_wrist_2_link", left_pose5);



//    //-------------------------
//    double roll16, pitch16, yaw16;
//    tf::Matrix3x3 matrix16;
//    Eigen::Matrix3d link_orientation16 = joint_state16.rotation();
//    tf::matrixEigenToTF(link_orientation16, matrix16);

//    matrix16.getRPY(roll16, pitch16, yaw16);
//    ROS_INFO("roll 16: %1.2f", roll16);
//    ROS_INFO("pitch 16: %1.2f", pitch16);
//    ROS_INFO("raw 16: %1.2f", yaw16);

//    double l_roll_deg16= 180;//180
//    double l_pitch_deg16=-42.39;//-38.39
//    double l_yaw_deg16=15.47;//15.47

//    double l_roll_rad16 = l_roll_deg16 * (M_PI/180);
//    double l_pitch_rad16 = l_pitch_deg16 * (M_PI/180);
//    double l_yaw_rad16 = l_yaw_deg16 * (M_PI/180);

//    tf::Quaternion l_q16;
//    l_q16.setRPY(l_roll_rad16, l_pitch_rad16, l_yaw_rad16);
//    Eigen::Affine3d left_pose16 = Eigen::Translation3d(x16, y16, z16)
//        * Eigen::Quaterniond(l_q16);
//    kinematic_state->updateStateWithLinkAt("right_forearm_link", left_pose16);

//    //--------------------------------
//    double roll18, pitch18, yaw18;
//    tf::Matrix3x3 matrix18;
//    Eigen::Matrix3d link_orientation18 = joint_state18.rotation();
//    tf::matrixEigenToTF(link_orientation18, matrix18);

//    matrix18.getRPY(roll18, pitch18, yaw18);
//    ROS_INFO("roll 18: %1.2f", roll18);
//    ROS_INFO("pitch 18: %1.2f", pitch18);
//    ROS_INFO("raw 18: %1.2f", yaw18);

//    double l_roll_deg18= -89.38;//-89.38
//    double l_pitch_deg18=-0.00;//-0.00
//    double l_yaw_deg18=-87.50;//-78.50

//    double l_roll_rad18 = l_roll_deg18 * (M_PI/180);
//    double l_pitch_rad18 = l_pitch_deg18 * (M_PI/180);
//    double l_yaw_rad18 = l_yaw_deg18 * (M_PI/180);

//    tf::Quaternion l_q18;
//    l_q18.setRPY(l_roll_rad18, l_pitch_rad18, l_yaw_rad18);
//    Eigen::Affine3d left_pose18 = Eigen::Translation3d(x18, y18, z18)
//        * Eigen::Quaterniond(l_q18);
//    kinematic_state->updateStateWithLinkAt("right_wrist_2_link", left_pose18);

    kinematic_state->update();

  }
};


void help(const std::string &path)
{
  std::cout << path << FG_BLUE " [options]" << std::endl
            << FG_GREEN "  name" NO_COLOR ": " FG_YELLOW "'any string'" NO_COLOR " equals to the kinect2_bridge topic base name" << std::endl
            << FG_GREEN "  mode" NO_COLOR ": " FG_YELLOW "'qhd'" NO_COLOR ", " FG_YELLOW "'hd'" NO_COLOR ", " FG_YELLOW "'sd'" NO_COLOR " or " FG_YELLOW "'ir'" << std::endl
            << FG_GREEN "  visualization" NO_COLOR ": " FG_YELLOW "'image'" NO_COLOR ", " FG_YELLOW "'cloud'" NO_COLOR " or " FG_YELLOW "'both'" << std::endl
            << FG_GREEN "  options" NO_COLOR ":" << std::endl
            << FG_YELLOW "    'compressed'" NO_COLOR " use compressed instead of raw topics" << std::endl
            << FG_YELLOW "    'approx'" NO_COLOR " use approximate time synchronization" << std::endl;
}

int main(int argc, char **argv)
{
#if EXTENDED_OUTPUT
  ROSCONSOLE_AUTOINIT;
  if(!getenv("ROSCONSOLE_FORMAT"))
    {
      ros::console::g_formatter.tokens_.clear();
      ros::console::g_formatter.init("[${severity}] ${message}");
    }
#endif

  ros::init(argc, argv, "kinect2_viewer", ros::init_options::AnonymousName);

  if(!ros::ok())
    {
      return 0;
    }

  std::string ns = K2_DEFAULT_NS;
  std::string topicColor = K2_TOPIC_QHD K2_TOPIC_IMAGE_COLOR K2_TOPIC_IMAGE_RECT;
  std::string topicDepth = K2_TOPIC_QHD K2_TOPIC_IMAGE_DEPTH K2_TOPIC_IMAGE_RECT;
  bool useExact = true;
  bool useCompressed = false;
  Receiver::Mode mode = Receiver::CLOUD;

  for(size_t i = 1; i < (size_t)argc; ++i)
    {
      std::string param(argv[i]);

      if(param == "-h" || param == "--help" || param == "-?" || param == "--?")
        {
          help(argv[0]);
          ros::shutdown();
          return 0;
        }
      else if(param == "qhd")
        {
          topicColor = K2_TOPIC_QHD K2_TOPIC_IMAGE_COLOR K2_TOPIC_IMAGE_RECT;
          topicDepth = K2_TOPIC_QHD K2_TOPIC_IMAGE_DEPTH K2_TOPIC_IMAGE_RECT;
        }
      else if(param == "hd")
        {
          topicColor = K2_TOPIC_HD K2_TOPIC_IMAGE_COLOR K2_TOPIC_IMAGE_RECT;
          topicDepth = K2_TOPIC_HD K2_TOPIC_IMAGE_DEPTH K2_TOPIC_IMAGE_RECT;
        }
      else if(param == "ir")
        {
          topicColor = K2_TOPIC_SD K2_TOPIC_IMAGE_IR K2_TOPIC_IMAGE_RECT;
          topicDepth = K2_TOPIC_SD K2_TOPIC_IMAGE_DEPTH K2_TOPIC_IMAGE_RECT;
        }
      else if(param == "sd")
        {
          topicColor = K2_TOPIC_SD K2_TOPIC_IMAGE_COLOR K2_TOPIC_IMAGE_RECT;
          topicDepth = K2_TOPIC_SD K2_TOPIC_IMAGE_DEPTH K2_TOPIC_IMAGE_RECT;
        }
      else if(param == "approx")
        {
          useExact = false;
        }

      else if(param == "compressed")
        {
          useCompressed = true;
        }
      else if(param == "image")
        {
          mode = Receiver::IMAGE;
        }
      else if(param == "cloud")
        {
          mode = Receiver::CLOUD;
        }
      else if(param == "both")
        {
          mode = Receiver::BOTH;
        }
      else
        {
          ns = param;
        }
    }

  topicColor = "/" + ns + topicColor;
  topicDepth = "/" + ns + topicDepth;
  OUT_INFO("topic color: " FG_CYAN << topicColor << NO_COLOR);
  OUT_INFO("topic depth: " FG_CYAN << topicDepth << NO_COLOR);

  Receiver receiver(topicColor, topicDepth, useExact, useCompressed);

  OUT_INFO("starting receiver...");
  receiver.run(mode);

  ros::shutdown();
  return 0;
}
