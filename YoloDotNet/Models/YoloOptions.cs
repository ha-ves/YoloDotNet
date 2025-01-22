﻿namespace YoloDotNet.Models
{
    /// <summary>
    /// Represents options for configuring a Yolo object.
    /// </summary>
    public class YoloOptions
    {
        /// <summary>
        /// Gets or sets the file path to the ONNX model.
        /// </summary>
        public string OnnxModel { get; set; } = default!;

        /// <summary>
        /// Gets or sets the type of the model (e.g., detection, classification).
        /// </summary>
        public ModelType ModelType { get; set; }

        /// <summary>
        /// Gets or sets a value indicating whether to use CUDA for GPU acceleration (default is true).
        /// </summary>
        public bool Cuda { get; set; } = true;

        /// <summary>
        /// Gets or sets a value indicating whether to prime the GPU before inference (default is false).
        /// </summary>
        public bool PrimeGpu { get; set; } = false;

        /// <summary>
        /// Gets or sets the ID of the GPU to use (default is 0).
        /// </summary>
        public int GpuId { get; set; } = 0;

        /// <summary>
        /// Custom <see cref="Microsoft.ML.OnnxRuntime.SessionOptions"/> overriding <see cref="Cuda"/> and <see cref="GpuId"/>.
        /// </summary>
        /// <remarks>
        /// !! Mutually Exclusive with <see cref="Cuda"/> and <see cref="GpuId"/>.
        /// <br/>
        /// If you set this property, this will get used regardless of <see cref="Cuda"/> and <see cref="GpuId"/>.
        /// </remarks>
        public SessionOptions? SessionOptions { get; set; } = null;
    }
}
