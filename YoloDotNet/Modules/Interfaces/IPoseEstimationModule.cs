﻿namespace YoloDotNet.Modules.Interfaces
{
    internal interface IPoseEstimationModule : IModule
    {
        List<PoseEstimation> ProcessImage(SKImage image, double confidence, double iou);
        Dictionary<int, List<PoseEstimation>> ProcessVideo(VideoOptions options, double confidence, double iou);
    }
}