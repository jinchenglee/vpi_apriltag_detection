/*
* Copyright (c) 2025, NVIDIA CORPORATION. All rights reserved.
*
* Redistribution and use in source and binary forms, with or without
* modification, are permitted provided that the following conditions
* are met:
*  * Redistributions of source code must retain the above copyright
*    notice, this list of conditions and the following disclaimer.
*  * Redistributions in binary form must reproduce the above copyright
*    notice, this list of conditions and the following disclaimer in the
*    documentation and/or other materials provided with the distribution.
*  * Neither the name of NVIDIA CORPORATION nor the names of its
*    contributors may be used to endorse or promote products derived
*    from this software without specific prior written permission.
*
* THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS ``AS IS'' AND ANY
* EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
* IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR
* PURPOSE ARE DISCLAIMED.  IN NO EVENT SHALL THE COPYRIGHT OWNER OR
* CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
* EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
* PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
* PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY
* OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
* (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
* OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
*/

#include <opencv2/core/version.hpp>
#if CV_MAJOR_VERSION >= 3
#    include <opencv2/imgcodecs.hpp>
#else
#    include <opencv2/contrib/contrib.hpp> // for applyColorMap
#    include <opencv2/highgui/highgui.hpp>
#endif

#include <opencv2/imgproc/imgproc.hpp>
#include <vpi/OpenCVInterop.hpp>

#include <vpi/Array.h>
#include <vpi/Image.h>
#include <vpi/Status.h>
#include <vpi/Stream.h>
#include <vpi/algo/ConvertImageFormat.h>
#include <vpi/algo/AprilTags.h>

#include <cstdio>
#include <cstring> // for memset
#include <iostream>
#include <sstream>

#define CHECK_STATUS(STMT)                                    \
    do                                                        \
    {                                                         \
        VPIStatus status = (STMT);                            \
        if (status != VPI_SUCCESS)                            \
        {                                                     \
            char buffer[VPI_MAX_STATUS_MESSAGE_LENGTH];       \
            vpiGetLastStatusMessage(buffer, sizeof(buffer));  \
            std::ostringstream ss;                            \
            ss << vpiStatusGetName(status) << ": " << buffer; \
            throw std::runtime_error(ss.str());               \
        }                                                     \
    } while (0);

static cv::Mat DrawAprilTags(cv::Mat img, VPIAprilTagDetection *detections, VPIPose *poses, int numDetections)
{
    cv::Mat out;
    cv::cvtColor(img, out, cv::COLOR_GRAY2BGR);

    if (numDetections == 0)
    {
        return out;
    }

    for (int i = 0; i < numDetections; ++i) {
        const VPIAprilTagDetection &det = detections[i];
        const VPIPose &pose = poses[i];

        // Print detection information
        printf("\nTag ID: %d\n", det.id);
        printf("Decision Margin: %.2f\n", det.decisionMargin);
        printf("Corrected Bits: %d\n", det.correctedBits);
        printf("Center: (%.2f, %.2f)\n", det.center.x, det.center.y);
        
        // Print pose information
        printf("Pose Error: %.4f\n", pose.error);
        printf("Transform Matrix:\n");
        for (int row = 0; row < 3; ++row) {
            printf("[ ");
            for (int col = 0; col < 4; ++col) {
                printf("%.4f ", pose.transform[row][col]);
            }
            printf("]\n");
        }

        // Draw the tag corners
        for (int j = 0; j < 4; ++j) {
            int next = (j + 1) % 4;
            cv::line(out,
                    cv::Point(det.corners[j].x, det.corners[j].y),
                    cv::Point(det.corners[next].x, det.corners[next].y),
                    cv::Scalar(0, 255, 0), 2);
        }

        // Draw the tag center
        cv::circle(out, cv::Point(det.center.x, det.center.y), 5, cv::Scalar(0, 0, 255), -1);

        // Draw the tag ID
        cv::putText(out, std::to_string(det.id),
                   cv::Point(det.center.x + 10, det.center.y),
                   cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(255, 0, 0), 2);
    }

    return out;
}

int main(int argc, char *argv[])
{
    // OpenCV image that will be wrapped by a VPIImage.
    // Define it here so that it's destroyed *after* wrapper is destroyed
    cv::Mat cvImage;

    // VPI objects that will be used
    VPIImage imgInput     = NULL;
    VPIImage imgGrayscale    = NULL;
    VPIArray detections   = NULL;
    VPIArray poses        = NULL;
    VPIStream stream      = NULL;
    VPIPayload payload    = NULL;

    // AprilTagDecode parameters
    const int maxHamming           = 1;
    const VPIAprilTagFamily family = VPI_APRILTAG_36H11;
    VPIAprilTagDecodeParams apritagDecodeParams = {NULL, 0, maxHamming, family};

    int retval = 0;

    try
    {
        // =============================
        // Parse command line parameters

        if (argc != 3)
        {
            throw std::runtime_error(std::string("Usage: ") + argv[0] + " <cpu|pva> <input image>");
        }

        std::string strBackend       = argv[1];
        std::string strInputFileName = argv[2];

        // Now parse the backend
        VPIBackend backend;

        if (strBackend == "cpu")
        {
            backend = VPI_BACKEND_CPU;
        }
        else if (strBackend == "pva")
        {
            backend = VPI_BACKEND_PVA;
        }
        else
        {
            throw std::runtime_error("Backend '" + strBackend +
                                     "' not recognized, it must be either cpu or pva.");
        }

        // =====================
        // Load the input image

        cvImage = cv::imread(strInputFileName);
        if (cvImage.empty())
        {
            throw std::runtime_error("Can't open '" + strInputFileName + "'");
        }

        // =================================
        // Allocate all VPI resources needed

        // Create the stream where processing will happen
        CHECK_STATUS(vpiStreamCreate(0, &stream));

        // We now wrap the loaded image into a VPIImage object to be used by VPI.
        // VPI won't make a copy of it, so the original
        // image must be in scope at all times.
        CHECK_STATUS(vpiImageCreateWrapperOpenCVMat(cvImage, 0, &imgInput));
        CHECK_STATUS(vpiImageCreate(cvImage.cols, cvImage.rows, VPI_IMAGE_FORMAT_U8, 0, &imgGrayscale));

        const int maxDetections = 64;
        // Create the output detection and poses array
        CHECK_STATUS(vpiArrayCreate(maxDetections, VPI_ARRAY_TYPE_APRILTAG_DETECTION, VPI_BACKEND_CPU | VPI_BACKEND_PVA, &detections));
        // Pose detection is only implemented on CPU
        CHECK_STATUS(vpiArrayCreate(maxDetections, VPI_ARRAY_TYPE_POSE, VPI_BACKEND_CPU | VPI_BACKEND_PVA, &poses));

        // Create the payload for AprilTag Detector algorithm
        CHECK_STATUS(vpiCreateAprilTagDetector(backend, cvImage.cols, cvImage.rows, &apritagDecodeParams, &payload));

        // AprilTagPoseEstimation parameters
        const VPICameraIntrinsic intrinsics = {{cvImage.cols / 3.5f, 0.0f, cvImage.cols / 2.f}, {0.0f, cvImage.rows / 3.6f, cvImage.rows / 2.f}};
        const float tagSize       = 0.2f;


        // ================
        // Processing stage

        // First convert input to grayscale
        CHECK_STATUS(vpiSubmitConvertImageFormat(stream, VPI_BACKEND_CPU, imgInput, imgGrayscale, NULL));

        // Then get AprilTag detections
        CHECK_STATUS(
            vpiSubmitAprilTagDetector(stream, backend, payload, maxDetections, imgGrayscale, detections)
        );

        // Then get AprilTag poses
        CHECK_STATUS(vpiSubmitAprilTagPoseEstimation(stream, VPI_BACKEND_CPU, detections, intrinsics, tagSize, poses));

        // Wait until the algorithm finishes processing
        CHECK_STATUS(vpiStreamSync(stream));

        // =======================================
        // Output processing and saving it to disk

        // Lock output keypoints and poses to retrieve its data on cpu memory
        VPIArrayData outDetectionsData;
        VPIArrayData outPosesData;
        VPIImageData imgData;
        CHECK_STATUS(vpiArrayLockData(detections, VPI_LOCK_READ, VPI_ARRAY_BUFFER_HOST_AOS, &outDetectionsData));
        CHECK_STATUS(vpiArrayLockData(poses, VPI_LOCK_READ, VPI_ARRAY_BUFFER_HOST_AOS, &outPosesData));
        CHECK_STATUS(vpiImageLockData(imgGrayscale, VPI_LOCK_READ, VPI_IMAGE_BUFFER_HOST_PITCH_LINEAR, &imgData));

        VPIAprilTagDetection *outDetections = (VPIAprilTagDetection *)outDetectionsData.buffer.aos.data;
        VPIPose *outPoses = (VPIPose *)outPosesData.buffer.aos.data;
        int numDetections = *outDetectionsData.buffer.aos.sizePointer;

        printf("\n%d AprilTags detected\n", numDetections);

        // Convert the grayscale image to BGR for visualization
        cv::Mat img;
        CHECK_STATUS(vpiImageDataExportOpenCVMat(imgData, &img));

        cv::Mat outImage = DrawAprilTags(img, outDetections, outPoses, numDetections);
        imwrite("apriltag_detections_" + strBackend + ".png", outImage);

        // Done handling outputs, don't forget to unlock them.
        CHECK_STATUS(vpiImageUnlock(imgGrayscale));
        CHECK_STATUS(vpiArrayUnlock(poses));
        CHECK_STATUS(vpiArrayUnlock(detections));
    }
    catch (std::exception &e)
    {
        std::cerr << e.what() << std::endl;
        retval = 1;
    }

    // ========
    // Clean up

    // Make sure stream is synchronized before destroying the objects
    // that might still be in use.
    if (stream != NULL)
    {
        vpiStreamSync(stream);
    }

    vpiImageDestroy(imgInput);
    vpiImageDestroy(imgGrayscale);
    vpiArrayDestroy(detections);
    vpiArrayDestroy(poses);
    vpiPayloadDestroy(payload);
    vpiStreamDestroy(stream);
    return retval;
}
